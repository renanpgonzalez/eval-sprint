"""`target_retries` (contract C18): a target attempt that errors is called again inside the run, and the retried calls are paid.

The fixture target logs every call (`input|attempt`) and fails a chosen number of times before it answers, so a test can say
exactly how often the target was called, what the record keeps and what the summary counts.
"""

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from evalsprint.cli import main
from evalsprint.config import SuiteConfig
from evalsprint.runner import _call_target, load_run, run_suite
from evalsprint.types import Case, TargetOutput, Usage

TARGET = '''\
import json
import os


def answer(input, params):
    key = f"{input}|{params.get('_attempt')}"
    log = os.environ["TR_CALLS"]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(key + "\\n")
    with open(log, encoding="utf-8") as fh:
        seen = sum(1 for line in fh if line.strip() == key)
    failures = json.loads(os.environ.get("TR_FAILS", "{}")).get(input, 0)  # the first N calls of this input (and attempt) fail
    if seen <= failures:
        if os.environ.get("TR_MODE") == "paid":  # the provider answered with an error AFTER charging for the tokens
            return {"text": "", "error": "HTTP 529 overloaded", "model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 20}}
        raise TimeoutError("upstream timeout")
    return {"text": "answer to " + input, "model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 20}}
'''

JUDGE = '''\
def picky(case, output):
    if "boom" in case.input:
        raise RuntimeError("judge exploded")
    return "answer" in output.text
'''

EVAL_YAML = """\
suite: retries
dataset: cases.jsonl
concurrency: 1
runs_dir: runs
{extra}target:
  callable: target.py:answer
  params:
    model: claude-haiku-4-5
judges:
  - type: contains
    name: says_answer
    values: ["answer"]
"""

CASES = [{"id": f"c{i}", "input": f"q{i}", "tags": ["t"]} for i in range(4)]


class RetryFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "suite"
        self.dir.mkdir()
        (self.dir / "target.py").write_text(TARGET, encoding="utf-8")
        (self.dir / "my_judge.py").write_text(JUDGE, encoding="utf-8")
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
        self.calls_file = self.dir / "calls.log"
        env = mock.patch.dict(os.environ, {"TR_CALLS": str(self.calls_file)})
        env.start()
        self.addCleanup(env.stop)
        for name in ("TR_FAILS", "TR_MODE"):
            os.environ.pop(name, None)
        self.write_config()

    def write_config(self, extra: str = ""):
        (self.dir / "eval.yaml").write_text(EVAL_YAML.format(extra=extra), encoding="utf-8")

    def run_it(self, overrides=None, fails=None, mode=None, **kw):
        if fails is not None:
            os.environ["TR_FAILS"] = json.dumps(fails)
        if mode:
            os.environ["TR_MODE"] = mode
        kw.setdefault("progress", False)
        kw.setdefault("out_dir", self.dir / "out")
        return run_suite(SuiteConfig.load(self.dir / "eval.yaml", overrides), **kw)

    def calls(self, prefix: str | None = None) -> list[str]:
        if not self.calls_file.exists():
            return []
        lines = self.calls_file.read_text(encoding="utf-8").splitlines()
        return [line for line in lines if prefix is None or line.startswith(prefix + "|")]


class TransientCrashTests(RetryFixture):
    def test_a_target_that_fails_once_is_rescued_inside_the_run(self):
        run = self.run_it(fails={"q1": 1})
        record = run.by_case()["c1"]
        self.assertEqual((record["outcome"], record["passed"], record["error_source"]), ("pass", True, None))
        self.assertEqual(record["target_retries_used"], 1)
        self.assertEqual(self.calls("q1"), ["q1|None", "q1|None"])
        self.assertEqual(self.calls("q0"), ["q0|None"])  # the healthy cases are called once
        s = run.summary
        self.assertEqual((s["n_errors"], s["n_target_errors"], s["n_target_retries"]), (0, 0, 1))
        self.assertEqual(s["pass_rate"], 1.0)

    def test_the_retry_is_on_by_default_and_zero_turns_it_off(self):
        self.assertEqual(SuiteConfig.load(self.dir / "eval.yaml").target_retries, 1)
        run = self.run_it(overrides=["target_retries=0"], fails={"q1": 1})
        record = run.by_case()["c1"]
        self.assertEqual((record["outcome"], record["error_source"], record["target_retries_used"]), ("error", "target", 0))
        self.assertEqual(self.calls("q1"), ["q1|None"])
        self.assertEqual((run.summary["n_target_errors"], run.summary["n_target_retries"]), (1, 0))

    def test_a_crash_that_outlasts_the_retries_is_still_an_error(self):
        run = self.run_it(fails={"q1": 9})
        record = run.by_case()["c1"]
        self.assertEqual((record["outcome"], record["error_source"], record["target_retries_used"]), ("error", "target", 1))
        self.assertEqual(len(self.calls("q1")), 2)  # the first call and one retry
        self.assertIn("TimeoutError", record["output"]["error"])
        self.assertEqual((run.summary["n_target_errors"], run.summary["n_target_retries"]), (1, 1))

    def test_the_number_of_retries_follows_the_config(self):
        run = self.run_it(overrides=["target_retries=3"], fails={"q1": 2, "q2": 9})
        by_case = run.by_case()
        self.assertEqual((by_case["c1"]["outcome"], by_case["c1"]["target_retries_used"]), ("pass", 2))
        self.assertEqual((by_case["c2"]["outcome"], by_case["c2"]["target_retries_used"]), ("error", 3))
        self.assertEqual((len(self.calls("q1")), len(self.calls("q2"))), (3, 4))
        self.assertEqual(run.summary["n_target_retries"], 5)

    def test_the_yaml_key_is_read(self):
        self.write_config("target_retries: 2\n")
        run = self.run_it(fails={"q1": 2})
        self.assertEqual(run.by_case()["c1"]["outcome"], "pass")
        self.assertEqual(run.by_case()["c1"]["target_retries_used"], 2)

    def test_a_retried_record_is_written_to_disk_and_reloads(self):
        run = self.run_it(fails={"q1": 1})
        loaded = load_run(run.run_dir)
        self.assertEqual(loaded.by_case()["c1"]["target_retries_used"], 1)
        self.assertEqual(loaded.summary["n_target_retries"], 1)
        partial_lines = (run.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum(json.loads(line)["target_retries_used"] for line in partial_lines), 1)


class ConfiguredValueInSummaryTests(RetryFixture):
    """C22: `summary.target_retries` is the CONFIGURED value, always written; `n_target_retries` is how many calls were repeated.

    The gate compares the configured value between baseline and candidate: a retry hides a transient crash from the pass rate,
    so raising it changes what a healthy run means. The two numbers answer different questions and both stay in the summary.
    """

    def test_the_default_is_written_even_when_nothing_was_retried(self):
        run = self.run_it()
        self.assertEqual((run.summary["target_retries"], run.summary["n_target_retries"]), (1, 0))
        self.assertIs(type(run.summary["target_retries"]), int)

    def test_zero_is_written_not_left_out(self):
        run = self.run_it(overrides=["target_retries=0"])
        self.assertIn("target_retries", run.summary)
        self.assertEqual(run.summary["target_retries"], 0)  # falsy, but a recorded value: "retries off" differs from "not recorded"

    def test_the_configured_value_is_not_the_number_of_calls_used(self):
        run = self.run_it(overrides=["target_retries=3"], fails={"q1": 2, "q2": 9})
        self.assertEqual((run.summary["target_retries"], run.summary["n_target_retries"]), (3, 5))
        self.assertEqual(run.summary["n_attempts"], 4)  # unchanged: retried calls are not attempts

    def test_the_value_follows_the_config_and_the_yaml_key(self):
        self.write_config("target_retries: 2\n")
        self.assertEqual(self.run_it().summary["target_retries"], 2)
        self.assertEqual(self.run_it(overrides=["target_retries=5"]).summary["target_retries"], 5)  # --set wins over the yaml

    def test_a_repeated_run_records_the_per_call_value_not_the_sum(self):
        run = self.run_it(repeat=3, fails={"q1": 1})
        self.assertEqual((run.summary["target_retries"], run.summary["n_target_retries"]), (1, 3))

    def test_it_reaches_summary_json_and_reloads(self):
        run = self.run_it(overrides=["target_retries=4"])
        on_disk = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["target_retries"], 4)
        self.assertEqual(load_run(run.run_dir).summary["target_retries"], 4)

    def test_a_resumed_run_records_the_value_of_the_resuming_config(self):
        finished = self.run_it(overrides=["target_retries=2"], out_dir=self.dir / "first")
        resumed = self.run_it(overrides=["target_retries=2"], out_dir=self.dir / "second", resume_from=finished.run_dir)
        self.assertEqual(resumed.summary["resumed_from"]["reused_records"], 4)
        self.assertEqual(resumed.summary["target_retries"], 2)
        # a changed value does not change the records' fingerprint (reused attempts keep what they used), but it is what the new summary says
        again = self.run_it(overrides=["target_retries=0"], out_dir=self.dir / "third", resume_from=finished.run_dir)
        self.assertEqual(again.summary["target_retries"], 0)

    def test_the_cli_run_summary_has_it(self):
        out_dir = self.dir / "cli-out"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["run", "-c", str(self.dir / "eval.yaml"), "--out", str(out_dir), "--quiet", "--no-report", "--set", "target_retries=3"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["target_retries"], 3)


class PaidRetryTests(RetryFixture):
    def test_the_failed_call_was_paid_and_is_counted_in_the_cost_and_the_tokens(self):
        run = self.run_it(fails={"q1": 1}, mode="paid")
        by_case = run.by_case()
        clean, retried = by_case["c0"]["output"], by_case["c1"]["output"]
        self.assertEqual(retried["usage"]["input_tokens"], 2 * clean["usage"]["input_tokens"])
        self.assertEqual(retried["usage"]["output_tokens"], 2 * clean["usage"]["output_tokens"])
        self.assertAlmostEqual(retried["cost_usd"], 2 * clean["cost_usd"])
        self.assertEqual(retried["meta"]["target_retry_errors"], ["HTTP 529 overloaded"])
        self.assertEqual(retried["text"], "answer to q1")  # what is kept is the answer that worked
        self.assertIsNone(retried["error"])
        total = run.summary["cost"]["system_total_usd"]
        self.assertGreater(total, 0)
        self.assertAlmostEqual(total, sum(r["output"]["cost_usd"] + sum(j["cost_usd"] or 0 for j in r["judges"]) for r in run.results))
        self.assertEqual(run.summary["n_target_retries"], 1)

    def test_a_raise_costs_nothing_and_leaves_the_clean_cost(self):
        run = self.run_it(fails={"q1": 1})
        by_case = run.by_case()
        self.assertEqual(by_case["c1"]["output"]["usage"], by_case["c0"]["output"]["usage"])
        self.assertAlmostEqual(by_case["c1"]["output"]["cost_usd"], by_case["c0"]["output"]["cost_usd"])
        self.assertEqual(len(by_case["c1"]["output"]["meta"]["target_retry_errors"]), 1)
        self.assertIn("TimeoutError", by_case["c1"]["output"]["meta"]["target_retry_errors"][0])

    def test_an_error_that_outlasts_the_retries_keeps_the_paid_calls(self):
        run = self.run_it(fails={"q1": 9}, mode="paid")
        output = run.by_case()["c1"]["output"]
        self.assertEqual(output["error"], "HTTP 529 overloaded")
        self.assertEqual(output["usage"]["input_tokens"], 200)  # both calls were charged
        self.assertEqual(output["meta"]["target_retry_errors"], ["HTTP 529 overloaded"])


class ScopeTests(RetryFixture):
    def test_judge_errors_are_not_retried_here(self):
        (self.dir / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in [{"id": "c0", "input": "q0"}, {"id": "c1", "input": "boom"}]), encoding="utf-8")
        (self.dir / "eval.yaml").write_text(
            EVAL_YAML.format(extra="").replace("type: contains\n    name: says_answer\n    values: [\"answer\"]", "type: custom\n    name: picky\n    callable: my_judge.py:picky"),
            encoding="utf-8",
        )
        run = self.run_it()
        record = run.by_case()["c1"]
        self.assertEqual((record["outcome"], record["error_source"]), ("error", "judge"))
        self.assertEqual(record["target_retries_used"], 0)
        self.assertEqual(self.calls("boom"), ["boom|None"])  # the target was not called again for a judge failure
        self.assertEqual(run.summary["n_target_retries"], 0)

    def test_each_attempt_of_a_repeated_case_is_retried_on_its_own(self):
        run = self.run_it(repeat=3, fails={"q1": 1})
        record = run.by_case()["c1"]
        self.assertEqual(record["outcome"], "pass")
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["pass"] * 3)
        self.assertEqual([a["target_retries_used"] for a in record["attempts"]], [1, 1, 1])  # the fixture fails the first call of each (input, attempt)
        self.assertEqual(record["target_retries_used"], 3)  # the record adds them up
        self.assertEqual(run.summary["n_target_retries"], 3)
        self.assertEqual(run.summary["n_attempt_errors"], 0)
        self.assertEqual(len(self.calls("q1")), 6)

    def test_a_retry_that_rescues_the_attempt_keeps_the_attempt_error_rate_at_zero(self):
        run = self.run_it(repeat=3, fails={"q1": 1})
        self.assertEqual(run.summary["attempt_error_rate"], 0.0)

    def test_the_retry_loop_is_bounded_and_returns_the_last_output(self):
        outputs = [TargetOutput(text="", error=f"e{i}", cost_usd=0.01, usage=Usage(10, 1)) for i in range(5)]
        target = mock.Mock(side_effect=outputs)
        output, used = _call_target(target, Case(id="c", input="q"), None, 2, threading.Event())
        self.assertEqual((used, target.call_count, output.error), (2, 3, "e2"))
        self.assertAlmostEqual(output.cost_usd, 0.03)
        self.assertEqual(output.usage.input_tokens, 30)
        self.assertEqual(output.meta["target_retry_errors"], ["e0", "e1"])

    def test_no_retry_starts_after_a_stop_was_requested(self):
        stop = threading.Event()
        stop.set()  # Ctrl-C or SIGTERM arrived while the call was in flight: do not pay for another one
        target = mock.Mock(return_value=TargetOutput(text="", error="boom"))
        output, used = _call_target(target, Case(id="c", input="q"), None, 3, stop)
        self.assertEqual((used, target.call_count, output.error), (0, 1, "boom"))

    def test_a_healthy_call_is_never_repeated(self):
        target = mock.Mock(return_value=TargetOutput(text="fine"))
        output, used = _call_target(target, Case(id="c", input="q"), None, 5, threading.Event())
        self.assertEqual((used, target.call_count, output.meta), (0, 1, {}))


class CliTests(RetryFixture):
    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_run_prints_the_retries_and_a_rescued_run_is_healthy(self):
        os.environ["TR_FAILS"] = json.dumps({"q1": 1})
        code, out, err = self.cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "cli-out", "--quiet", "--no-report")
        self.assertEqual((code, err), (0, ""))
        self.assertRegex(out, r"Target retries:\s+1 call\(s\) repeated after a crash \(target_retries: 1\)")

    def test_without_retries_the_same_flaky_target_makes_the_run_unhealthy(self):
        os.environ["TR_FAILS"] = json.dumps({"q1": 1})
        code, out, err = self.cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "cli-out", "--quiet", "--no-report", "--set", "target_retries=0")
        self.assertEqual(code, 3, err)
        self.assertIn("1 target error(s)", err)
        self.assertNotIn("Target retries", out)

    def test_progress_says_which_case_needed_a_retry(self):
        os.environ["TR_FAILS"] = json.dumps({"q1": 1})
        _, _, err = self.cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "cli-out", "--no-report")
        self.assertIn("PASS c1", err)
        self.assertIn("(after 1 retry)", err)

    def test_describe_shows_the_setting(self):
        code, out, _ = self.cli("describe", "-c", self.dir / "eval.yaml")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"Target retries:\s+1")
        self.write_config("target_retries: 0\n")
        code, out, _ = self.cli("describe", "-c", self.dir / "eval.yaml")
        self.assertRegex(out, r"Target retries:\s+0 \(off\)")


if __name__ == "__main__":
    unittest.main()
