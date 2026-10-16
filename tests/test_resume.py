"""`run --resume` and robust writes (contract C12): a run that stops half-way never loses what was paid for.

The fixture target logs every call to a file (`RESUME_TEST_CALLS`), so a test can say exactly WHICH cases a resumed run
paid for again, and can stop the run at a chosen input (`RESUME_TEST_STOP_AT`, raises KeyboardInterrupt, the way Ctrl-C does).
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from evalsprint import runner
from evalsprint.cli import main
from evalsprint.config import SuiteConfig
from evalsprint.runner import PARTIAL_NAME, RUN_META_NAME, load_run, run_suite, save_run

TARGET = '''\
import os
import time


def answer(input, params):
    with open(os.environ["RESUME_TEST_CALLS"], "a", encoding="utf-8") as fh:
        fh.write(f"{input}|{params.get('_attempt')}\\n")
    key = f"{input}|{params.get('_attempt')}"
    stops, crashes = os.environ.get("RESUME_TEST_STOP_AT", "").split(","), os.environ.get("RESUME_TEST_CRASH", "").split(",")
    if input in stops or key in stops:
        time.sleep(float(os.environ.get("RESUME_TEST_STOP_DELAY", "0")))
        raise KeyboardInterrupt
    if input in crashes or key in crashes:
        raise RuntimeError("upstream outage")
    time.sleep(float(os.environ.get("RESUME_TEST_SLEEP", "0")))
    if "bad" in input:
        return {"text": "nope", "model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 20}}
    return {"text": "answer to " + input, "model": "claude-haiku-4-5", "usage": {"input_tokens": 100, "output_tokens": 20}}
'''

EVAL_YAML = """\
suite: resume-fixture
dataset: cases.jsonl
concurrency: {concurrency}
runs_dir: runs
target:
  callable: target.py:answer
  params:
    model: claude-haiku-4-5
    prompt: v1
judges:
  - type: contains
    name: says_answer
    values: ["answer"]
"""


def cases_text(n: int = 6, edited: tuple = ()) -> str:
    rows = []
    for i in range(n):
        name = f"bad-{i}" if i == 2 else f"q{i}"
        row = {"id": f"c{i}", "input": name, "tags": ["t"]}
        if f"c{i}" in edited:
            row["reference"] = "an edited reference"
        rows.append(json.dumps(row))
    return "\n".join(rows) + "\n"


class ResumeFixture(unittest.TestCase):
    concurrency = 1

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "suite"
        self.dir.mkdir()
        (self.dir / "target.py").write_text(TARGET, encoding="utf-8")
        (self.dir / "eval.yaml").write_text(EVAL_YAML.format(concurrency=self.concurrency), encoding="utf-8")
        (self.dir / "cases.jsonl").write_text(cases_text(), encoding="utf-8")
        self.calls_file = self.dir / "calls.log"
        env = mock.patch.dict(os.environ, {"RESUME_TEST_CALLS": str(self.calls_file)})
        env.start()
        self.addCleanup(env.stop)
        for name in ("RESUME_TEST_STOP_AT", "RESUME_TEST_CRASH", "RESUME_TEST_SLEEP", "RESUME_TEST_STOP_DELAY"):
            os.environ.pop(name, None)

    # ---- helpers
    def config(self, overrides=None) -> SuiteConfig:
        return SuiteConfig.load(self.dir / "eval.yaml", overrides)

    def run_it(self, overrides=None, **kw):
        kw.setdefault("progress", False)
        return run_suite(self.config(overrides), **kw)

    def calls(self) -> list[str]:
        """Inputs the target was called with since the last `reset_calls`, in call order (`input|attempt`)."""
        if not self.calls_file.exists():
            return []
        return self.calls_file.read_text(encoding="utf-8").splitlines()

    def reset_calls(self):
        self.calls_file.unlink(missing_ok=True)

    def interrupted(self, stop_at: str, overrides=None, **kw) -> Path:
        """Run until the target raises KeyboardInterrupt at `stop_at`; return the run directory that was left behind."""
        os.environ["RESUME_TEST_STOP_AT"] = stop_at
        try:
            with self.assertRaises(KeyboardInterrupt) as ctx:
                self.run_it(overrides, **kw)
        finally:
            os.environ.pop("RESUME_TEST_STOP_AT", None)
        run_dir = Path(ctx.exception.run_dir)
        self.assertTrue((run_dir / PARTIAL_NAME).is_file())
        self.assertFalse((run_dir / "summary.json").exists())
        self.reset_calls()
        return run_dir

    @staticmethod
    def partial_lines(run_dir: Path) -> list[dict]:
        return [json.loads(line) for line in (run_dir / PARTIAL_NAME).read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def comparable(run):
        """What must be identical between a clean run and an interrupted-then-resumed one (everything but ids, times, files)."""
        return [
            (r["case_id"], r["case_sha"], r["outcome"], r["output"]["text"], r["output"]["cost_usd"], r["n_attempt_errors"], r.get("attempts"))
            for r in run.results
        ]


class InterruptThenResumeTests(ResumeFixture):
    def test_only_the_missing_cases_run_and_the_result_equals_a_clean_run(self):
        clean = self.run_it()
        self.reset_calls()
        partial_dir = self.interrupted("q3")  # c0..c2 finished, c3 was in flight when Ctrl-C arrived
        self.assertEqual([r["case_id"] for r in self.partial_lines(partial_dir)], ["c0", "c1", "c2"])

        resumed = self.run_it(resume_from=partial_dir)
        self.assertEqual(self.calls(), ["q3|None", "q4|None", "q5|None"])  # nothing that was paid for is paid for again
        self.assertEqual(self.comparable(resumed), self.comparable(clean))
        for key in ("n_cases", "n_passed", "n_failed", "pass_rate", "pass_rate_ci", "n_errors", "cost", "tokens", "by_tag", "by_judge", "n_attempts"):
            self.assertEqual(resumed.summary[key], clean.summary[key], key)
        self.assertNotEqual(resumed.run_id, clean.run_id)

    def test_summary_names_the_run_it_continued(self):
        partial_dir = self.interrupted("q3")
        old_id = json.loads((partial_dir / RUN_META_NAME).read_text(encoding="utf-8"))["run_id"]
        resumed = self.run_it(resume_from=partial_dir)
        info = resumed.summary["resumed_from"]
        self.assertEqual((info["run_id"], info["reused_records"]), (old_id, 3))
        self.assertNotEqual(resumed.run_id, old_id)
        self.assertGreater(info["reused_cost_usd"], 0)  # the reused cost is kept as spent ...
        self.assertAlmostEqual(resumed.summary["cost"]["system_total_usd"], self.run_it().summary["cost"]["system_total_usd"])  # ... and stays in the total

    def test_a_run_that_was_not_resumed_has_no_resumed_from(self):
        self.assertNotIn("resumed_from", self.run_it().summary)

    def test_the_new_run_gets_its_own_directory_and_the_old_one_is_left_alone(self):
        partial_dir = self.interrupted("q3")
        before = (partial_dir / PARTIAL_NAME).read_bytes()
        resumed = self.run_it(resume_from=partial_dir)
        self.assertNotEqual(Path(resumed.run_dir).resolve(), partial_dir.resolve())
        self.assertEqual((partial_dir / PARTIAL_NAME).read_bytes(), before)  # a resume only reads the old run
        self.assertEqual(sorted(p.name for p in Path(resumed.run_dir).iterdir()), ["results.jsonl", "run_meta.json", "summary.json"])

    def test_run_meta_holds_the_fingerprint_and_the_run_id(self):
        run = self.run_it()
        meta = json.loads((Path(run.run_dir) / RUN_META_NAME).read_text(encoding="utf-8"))
        self.assertEqual(meta["run_id"], run.run_id)
        self.assertEqual(set(meta["fingerprint"]), {"judges_sha", "target", "repeat"})
        self.assertEqual(meta["fingerprint"]["judges_sha"], run.summary["judges_sha"])
        self.assertEqual(meta["fingerprint"]["target"], run.summary["target"])
        self.assertEqual(meta["fingerprint"]["repeat"], {"n": 1, "per_tag": {}})
        self.assertRegex(meta["fingerprint"]["target"]["source_sha"], r"^[0-9a-f]{12}$")

    def test_run_meta_is_written_before_the_first_case_starts(self):
        seen = []
        real = runner.run_judges

        def spy(judges, case, output):
            seen.append(sorted(p.name for p in Path(run_dirs[0]).iterdir()))
            return real(judges, case, output)

        run_dirs = [self.dir / "out"]
        with mock.patch("evalsprint.runner.run_judges", spy):
            self.run_it(out_dir=run_dirs[0])
        self.assertIn(RUN_META_NAME, seen[0])
        self.assertIn(PARTIAL_NAME, seen[0])

    def test_resuming_twice_keeps_every_paid_case(self):
        first = self.interrupted("q3")
        os.environ["RESUME_TEST_STOP_AT"] = "q4"
        with self.assertRaises(KeyboardInterrupt) as ctx:
            self.run_it(resume_from=first)
        os.environ.pop("RESUME_TEST_STOP_AT")
        second = Path(ctx.exception.run_dir)
        self.assertEqual([r["case_id"] for r in self.partial_lines(second)], ["c0", "c1", "c2", "c3"])  # the reused lines were carried over
        self.reset_calls()
        final = self.run_it(resume_from=second)
        self.assertEqual(self.calls(), ["q4|None", "q5|None"])
        self.assertEqual(final.summary["resumed_from"]["reused_records"], 4)
        self.assertEqual(final.summary["n_cases"], 6)

    def test_out_dir_can_be_the_directory_that_is_resumed(self):
        clean = self.run_it()
        self.reset_calls()
        partial_dir = self.interrupted("q3")
        resumed = self.run_it(resume_from=partial_dir, out_dir=partial_dir)
        self.assertEqual(Path(resumed.run_dir), partial_dir)
        self.assertEqual(self.calls(), ["q3|None", "q4|None", "q5|None"])
        self.assertEqual(self.comparable(resumed), self.comparable(clean))
        self.assertEqual(sorted(p.name for p in partial_dir.iterdir()), ["results.jsonl", "run_meta.json", "summary.json"])
        self.assertEqual(load_run(partial_dir).summary["resumed_from"]["reused_records"], 3)

    def test_resuming_a_finished_run_into_itself_never_drops_its_records(self):
        finished = self.run_it()
        self.reset_calls()
        seen = []
        real = runner._clear_previous_run

        def spy(run_dir):
            seen.append(len((Path(run_dir) / PARTIAL_NAME).read_text(encoding="utf-8").splitlines()))  # the reused lines are already safe
            return real(run_dir)

        with mock.patch("evalsprint.runner._clear_previous_run", spy):
            again = self.run_it(resume_from=finished.run_dir, out_dir=finished.run_dir)
        self.assertEqual(seen, [6])
        self.assertEqual(self.calls(), [])
        self.assertEqual(again.summary["resumed_from"]["reused_records"], 6)

    def test_a_torn_last_line_is_ignored_and_that_attempt_runs_again(self):
        partial_dir = self.interrupted("q4")  # c0..c3 finished
        with open(partial_dir / PARTIAL_NAME, "a", encoding="utf-8") as fh:
            fh.write('{"case_id": "c4", "case_sha": "abc", "outcome": "pa')  # the crash cut this line short
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            resumed = self.run_it(resume_from=partial_dir)
        self.assertIn("1 unreadable line", err.getvalue())
        self.assertEqual(self.calls(), ["q4|None", "q5|None"])
        self.assertEqual(resumed.summary["resumed_from"]["reused_records"], 4)

    def test_progress_counts_only_what_is_left(self):
        partial_dir = self.interrupted("q3")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.run_it(resume_from=partial_dir, progress=True)
        self.assertIn("resuming", err.getvalue())
        self.assertIn("3 record(s) reused", err.getvalue())
        self.assertIn("[   3/3]", err.getvalue())


class ResumeFromAFinishedRunTests(ResumeFixture):
    def test_every_record_is_reused_and_nothing_is_called(self):
        finished = self.run_it()
        self.reset_calls()
        again = self.run_it(resume_from=finished.run_dir)
        self.assertEqual(self.calls(), [])
        self.assertEqual(again.summary["resumed_from"], {"run_id": finished.run_id, "reused_records": 6, "reused_cost_usd": again.summary["resumed_from"]["reused_cost_usd"]})
        self.assertEqual(self.comparable(again), self.comparable(finished))

    def test_an_edited_case_runs_again_and_the_others_are_reused(self):
        finished = self.run_it()
        self.reset_calls()
        (self.dir / "cases.jsonl").write_text(cases_text(edited=("c4",)), encoding="utf-8")
        again = self.run_it(resume_from=finished.run_dir)
        self.assertEqual(self.calls(), ["q4|None"])  # only the case whose case_sha changed
        self.assertEqual(again.summary["resumed_from"]["reused_records"], 5)
        self.assertEqual(again.by_case()["c4"]["case"]["reference"], "an edited reference")

    def test_a_case_that_was_removed_is_not_carried_over(self):
        finished = self.run_it()
        (self.dir / "cases.jsonl").write_text(cases_text(n=4), encoding="utf-8")
        again = self.run_it(resume_from=finished.run_dir)
        self.assertEqual([r["case_id"] for r in again.results], ["c0", "c1", "c2", "c3"])
        self.assertEqual(again.summary["resumed_from"]["reused_records"], 4)

    def test_a_case_added_since_runs(self):
        (self.dir / "cases.jsonl").write_text(cases_text(n=4), encoding="utf-8")
        finished = self.run_it()
        self.reset_calls()
        (self.dir / "cases.jsonl").write_text(cases_text(n=6), encoding="utf-8")
        again = self.run_it(resume_from=finished.run_dir)
        self.assertEqual(self.calls(), ["q4|None", "q5|None"])
        self.assertEqual(again.summary["n_cases"], 6)

    def test_subset_flags_only_take_the_selected_cases(self):
        finished = self.run_it()
        self.reset_calls()
        again = self.run_it(resume_from=finished.run_dir, ids=["c1", "c2"])
        self.assertEqual((self.calls(), [r["case_id"] for r in again.results], again.summary["resumed_from"]["reused_records"]), ([], ["c1", "c2"], 2))

    def test_a_partial_file_wins_over_a_stale_results_file(self):
        finished = self.run_it()
        run_dir = Path(finished.run_dir)
        partial = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()][:2]
        (run_dir / PARTIAL_NAME).write_text("".join(json.dumps(r) + "\n" for r in partial), encoding="utf-8")
        self.reset_calls()
        self.run_it(resume_from=run_dir)
        self.assertEqual(self.calls(), ["bad-2|None", "q3|None", "q4|None", "q5|None"])


class FingerprintTests(ResumeFixture):
    """Records are reused only when the judges, the target and the repeat protocol are the ones that produced them."""

    def assertNothingReused(self, run, differs, stderr, n_calls=6):
        self.assertEqual(len(self.calls()), n_calls)  # every case ran again
        self.assertEqual(run.summary["resumed_from"]["reused_records"], 0)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], differs)
        self.assertIn(f"{', '.join(differs)} differ", stderr)

    def resume_with(self, finished, overrides=None):
        self.reset_calls()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            run = self.run_it(overrides, resume_from=finished.run_dir)
        return run, err.getvalue()

    def test_a_changed_judge_reuses_nothing(self):
        finished = self.run_it()
        run, err = self.resume_with(finished, ["judges.0.values=[answer, to]"])
        self.assertNothingReused(run, ["judges_sha"], err)

    def test_changed_target_params_reuse_nothing(self):
        finished = self.run_it()
        run, err = self.resume_with(finished, ["target.params.prompt=v2"])
        self.assertNothingReused(run, ["target"], err)

    def test_a_changed_target_callable_reuses_nothing(self):
        finished = self.run_it()
        shutil.copy(self.dir / "target.py", self.dir / "target2.py")
        run, err = self.resume_with(finished, ["target.callable=target2.py:answer"])
        self.assertNothingReused(run, ["target"], err)

    def test_a_changed_repeat_protocol_reuses_nothing(self):
        finished = self.run_it()
        run, err = self.resume_with(finished, ["repeat={n: 2, tags: [t]}"])
        self.assertNothingReused(run, ["repeat"], err, n_calls=12)

    def test_the_cli_repeat_override_is_part_of_the_protocol(self):
        finished = self.run_it()
        self.reset_calls()
        with contextlib.redirect_stderr(io.StringIO()):
            run = self.run_it(resume_from=finished.run_dir, repeat=2)
        self.assertEqual(run.summary["resumed_from"]["fingerprint_mismatch"], ["repeat"])
        self.assertEqual(len(self.calls()), 12)

    def test_a_secret_in_target_params_is_masked_and_a_rotated_secret_still_matches(self):
        finished = self.run_it(["target.params.api_key=sk-first-secret-0123456789"])
        meta = (Path(finished.run_dir) / RUN_META_NAME).read_text(encoding="utf-8")
        self.assertNotIn("sk-first-secret", meta)
        self.reset_calls()
        again = self.run_it(["target.params.api_key=sk-second-secret-9876543210"], resume_from=finished.run_dir)
        self.assertEqual((self.calls(), again.summary["resumed_from"]["reused_records"]), ([], 6))

    def test_an_edited_custom_judge_reuses_nothing(self):
        judge = self.dir / "my_judge.py"
        judge.write_text("def check(case, output):\n    return len(output.text) > 3\n", encoding="utf-8")
        with open(self.dir / "eval.yaml", "a", encoding="utf-8") as fh:
            fh.write("  - type: custom\n    name: mine\n    callable: my_judge.py:check\n")
        overrides = None
        finished = self.run_it(overrides)
        self.reset_calls()
        same = self.run_it(overrides, resume_from=finished.run_dir)
        self.assertEqual(same.summary["resumed_from"]["reused_records"], 6)  # untouched code: reused
        judge.write_text("def check(case, output):\n    return True  # softened by a pull request\n", encoding="utf-8")
        self.reset_calls()
        with contextlib.redirect_stderr(io.StringIO()):
            changed = self.run_it(overrides, resume_from=finished.run_dir)
        self.assertEqual(changed.summary["resumed_from"]["fingerprint_mismatch"], ["judges_sha"])
        self.assertEqual(len(self.calls()), 6)
        self.assertNotEqual(changed.summary["judges_sha"], finished.summary["judges_sha"])


class ResumeInputErrorTests(ResumeFixture):
    def test_not_a_directory(self):
        with self.assertRaisesRegex(ValueError, "is not a directory"):
            self.run_it(resume_from=self.dir / "nope")

    def test_nothing_to_resume(self):
        empty = self.dir / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "nothing to resume"):
            self.run_it(resume_from=empty)
        self.assertEqual(self.calls(), [])

    def test_a_run_without_run_meta_or_summary_cannot_prove_its_fingerprint(self):
        old = self.dir / "old"
        old.mkdir()
        (old / PARTIAL_NAME).write_text(json.dumps({"case_id": "c0", "case_sha": "x"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no run_meta.json"):
            self.run_it(resume_from=old)
        self.assertEqual(self.calls(), [])  # refused before any money was spent

    def test_a_finished_run_of_evalsprint_0_1_resumes_from_its_summary(self):
        finished = self.run_it()
        run_dir = Path(finished.run_dir)
        (run_dir / RUN_META_NAME).unlink()  # 0.1 wrote no run_meta.json
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary.pop("repeat")  # ... and no `repeat` when n == 1
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.reset_calls()
        again = self.run_it(resume_from=run_dir)
        self.assertEqual((self.calls(), again.summary["resumed_from"]["reused_records"]), ([], 6))

    def test_retry_errors_needs_resume(self):
        with self.assertRaisesRegex(ValueError, "retry_errors only applies"):
            self.run_it(retry_errors=True)


class RetryErrorsTests(ResumeFixture):
    def crashed_run(self):
        os.environ["RESUME_TEST_CRASH"] = "q1,q3"
        try:
            run = self.run_it()
        finally:
            os.environ.pop("RESUME_TEST_CRASH")
        self.assertEqual(run.summary["n_target_errors"], 2)
        self.reset_calls()
        return run

    def test_errors_are_kept_by_default(self):
        crashed = self.crashed_run()
        again = self.run_it(resume_from=crashed.run_dir)
        self.assertEqual(self.calls(), [])
        self.assertEqual(again.summary["n_target_errors"], 2)

    def test_retry_errors_runs_only_the_errored_records_again(self):
        crashed = self.crashed_run()
        again = self.run_it(resume_from=crashed.run_dir, retry_errors=True)
        self.assertEqual(self.calls(), ["q1|None", "q3|None"])
        self.assertEqual((again.summary["n_errors"], again.summary["n_scored"]), (0, 6))
        info = again.summary["resumed_from"]
        self.assertEqual((info["reused_records"], info["retried_errors"]), (4, 2))
        self.assertIn("discarded_cost_usd", info)


class RepeatResumeTests(ResumeFixture):
    """With `repeat`, the unit of paid work is one attempt: an interrupted case keeps the attempts that finished."""

    REPEAT = ["repeat={n: 3, tags: [t]}"]

    def test_finished_attempts_of_the_interrupted_case_are_kept(self):
        clean = self.run_it(self.REPEAT)
        self.reset_calls()
        os.environ["RESUME_TEST_STOP_AT"] = "q1"
        with self.assertRaises(KeyboardInterrupt) as ctx:
            self.run_it(self.REPEAT)  # c0 x3 finished, then q1 raised on its FIRST attempt
        os.environ.pop("RESUME_TEST_STOP_AT")
        partial_dir = Path(ctx.exception.run_dir)
        self.assertEqual([(r["case_id"], r["attempt"]) for r in self.partial_lines(partial_dir)], [("c0", 0), ("c0", 1), ("c0", 2)])
        self.reset_calls()
        resumed = self.run_it(self.REPEAT, resume_from=partial_dir)
        self.assertEqual(sorted(set(self.calls())), sorted({f"{name}|{k}" for name in ("q1", "bad-2", "q3", "q4", "q5") for k in range(3)}))
        self.assertEqual(len(self.calls()), 15)  # c0's three attempts are not paid again
        self.assertEqual(resumed.summary["resumed_from"]["reused_records"], 3)
        self.assertEqual(self.comparable(resumed), self.comparable(clean))

    def test_a_partly_finished_repeated_case_only_runs_its_missing_attempts(self):
        clean = self.run_it(self.REPEAT)
        self.reset_calls()
        partial_dir = self.interrupted("q1|1", self.REPEAT)  # c0 x3 and attempt 0 of c1 finished; attempt 1 of c1 was running
        self.assertEqual([(r["case_id"], r["attempt"]) for r in self.partial_lines(partial_dir)], [("c0", 0), ("c0", 1), ("c0", 2), ("c1", 0)])
        resumed = self.run_it(self.REPEAT, resume_from=partial_dir)
        self.assertEqual(self.calls()[:2], ["q1|1", "q1|2"])  # the two attempts of c1 that are missing, not the finished one
        self.assertEqual(len(self.calls()), 2 + 4 * 3)
        self.assertEqual(resumed.summary["resumed_from"]["reused_records"], 4)
        self.assertEqual(self.comparable(resumed), self.comparable(clean))

    def test_a_finished_repeated_run_is_reused_as_combined_records(self):
        clean = self.run_it(self.REPEAT)
        self.reset_calls()
        again = self.run_it(self.REPEAT, resume_from=clean.run_dir)
        self.assertEqual(self.calls(), [])
        self.assertEqual(again.summary["resumed_from"]["reused_records"], 6)
        self.assertEqual(self.comparable(again), self.comparable(clean))
        self.assertEqual(again.summary["stability"], clean.summary["stability"])
        self.assertEqual(again.summary["cost"], clean.summary["cost"])
        self.assertTrue(all(len(r["attempts"]) == 3 for r in again.results))

    def test_combined_records_carry_into_the_next_partial_file(self):
        clean = self.run_it(self.REPEAT)
        (self.dir / "cases.jsonl").write_text(cases_text(edited=("c4",)), encoding="utf-8")
        reference = self.run_it(self.REPEAT)  # a clean run of the edited test set
        self.reset_calls()
        partial_dir = self.interrupted("q4", self.REPEAT, resume_from=clean.run_dir)  # c0..c3 reused, c4 (edited) stops the run
        seeded = self.partial_lines(partial_dir)
        self.assertEqual([(r["case_id"], len(r["attempts"])) for r in seeded], [("c0", 3), ("c1", 3), ("c2", 3), ("c3", 3), ("c5", 3)])  # c5 is reusable too
        final = self.run_it(self.REPEAT, resume_from=partial_dir)
        self.assertEqual(self.calls(), ["q4|0", "q4|1", "q4|2"])  # only the edited case, three attempts
        self.assertEqual(final.summary["resumed_from"]["reused_records"], 5)
        self.assertEqual(self.comparable(final), self.comparable(reference))

    def test_a_case_that_was_not_repeated_before_runs_all_its_attempts(self):
        finished = self.run_it()
        with contextlib.redirect_stderr(io.StringIO()):
            self.reset_calls()
            again = self.run_it(self.REPEAT, resume_from=finished.run_dir)
        self.assertEqual(again.summary["resumed_from"]["fingerprint_mismatch"], ["repeat"])
        self.assertEqual(len(self.calls()), 18)


class InFlightAttemptsTests(ResumeFixture):
    concurrency = 3

    def test_attempts_running_when_ctrl_c_arrives_are_saved(self):
        (self.dir / "cases.jsonl").write_text(cases_text(n=8), encoding="utf-8")
        os.environ["RESUME_TEST_STOP_AT"] = "q0"
        os.environ["RESUME_TEST_STOP_DELAY"] = "0.15"  # the interrupt comes while c1 and c2 are still running
        os.environ["RESUME_TEST_SLEEP"] = "0.4"
        try:
            with self.assertRaises(KeyboardInterrupt) as ctx:
                self.run_it()
        finally:
            for name in ("RESUME_TEST_STOP_AT", "RESUME_TEST_STOP_DELAY", "RESUME_TEST_SLEEP"):
                os.environ.pop(name, None)
        run_dir = Path(ctx.exception.run_dir)
        saved = sorted(r["case_id"] for r in self.partial_lines(run_dir))
        self.assertEqual(saved, ["c1", "c2"])  # paid for, so kept; nothing new started after the interrupt
        self.assertEqual(sorted(c.split("|")[0] for c in self.calls()), ["bad-2", "q0", "q1"])  # nothing else was started
        self.reset_calls()
        resumed = self.run_it(resume_from=run_dir)
        self.assertEqual(sorted(c.split("|")[0] for c in self.calls()), sorted(["q0", "q3", "q4", "q5", "q6", "q7"]))
        self.assertEqual(resumed.summary["n_cases"], 8)

    def test_a_second_ctrl_c_abandons_the_running_attempts(self):
        (self.dir / "cases.jsonl").write_text(cases_text(n=4), encoding="utf-8")
        release = threading.Event()
        real = runner.run_judges

        def slow_judges(judges, case, output):
            release.wait(5)
            return real(judges, case, output)

        os.environ["RESUME_TEST_STOP_AT"] = "q0"
        os.environ["RESUME_TEST_STOP_DELAY"] = "0.1"
        real_shutdown = runner.ThreadPoolExecutor.shutdown
        state = {"n": 0}

        def shutdown(self, wait=True, *, cancel_futures=False):
            state["n"] += 1
            if wait:
                release.set()  # let the workers go, then behave as if the user pressed Ctrl-C again
                raise KeyboardInterrupt
            return real_shutdown(self, wait=wait, cancel_futures=cancel_futures)

        try:
            with mock.patch("evalsprint.runner.run_judges", slow_judges), mock.patch.object(runner.ThreadPoolExecutor, "shutdown", shutdown):
                with self.assertRaises(KeyboardInterrupt) as ctx:
                    self.run_it()
        finally:
            release.set()
            for name in ("RESUME_TEST_STOP_AT", "RESUME_TEST_STOP_DELAY"):
                os.environ.pop(name, None)
        self.assertTrue(hasattr(ctx.exception, "run_dir"))  # the original interrupt is what propagates


class CliResumeTests(ResumeFixture):
    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_interrupt_message_says_how_to_continue_and_keeps_the_fingerprint_options(self):
        os.environ["RESUME_TEST_STOP_AT"] = "q3"
        code, _, err = self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--set", "target.params.prompt=v two", "--repeat", "1", "--label", "nightly")
        os.environ.pop("RESUME_TEST_STOP_AT")
        self.assertEqual(code, 130)
        lines = err.splitlines()
        self.assertEqual(lines[0], "interrupted")
        self.assertIn(PARTIAL_NAME, err)
        hint = lines[-1].strip()
        self.assertTrue(hint.startswith("evalsprint run -c "))
        self.assertIn("--set", hint)
        self.assertIn("prompt=v two", hint)  # quoted: it is one word
        self.assertIn("--repeat 1", hint)
        self.assertIn("--label nightly", hint)
        self.assertIn("--resume", hint)

    def test_cli_resume_end_to_end(self):
        os.environ["RESUME_TEST_STOP_AT"] = "q3"
        code, _, err = self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report")
        os.environ.pop("RESUME_TEST_STOP_AT")
        self.assertEqual(code, 130)
        run_dir = next((self.dir / "runs").iterdir())
        self.reset_calls()
        code, out, err = self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--resume", run_dir)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.calls(), ["q3|None", "q4|None", "q5|None"])
        self.assertIn("Resumed:", out)
        self.assertIn("3 record(s) of run", out)
        self.assertEqual(len(list((self.dir / "runs").iterdir())), 2)  # the interrupted directory and the new run

    def test_the_resume_command_quotes_for_the_shell_it_will_be_pasted_into(self):
        import types

        from evalsprint import cli

        with mock.patch("evalsprint.cli.os", types.SimpleNamespace(name="posix")):
            self.assertEqual(cli._quote("a b"), "'a b'")
            self.assertEqual(cli._quote("plain"), "plain")
            self.assertEqual(cli._quote("it's"), "'it'\"'\"'s'")
        with mock.patch("evalsprint.cli.os", types.SimpleNamespace(name="nt")):
            self.assertEqual(cli._quote("C:\\Users\\me\\my runs\\r1"), '"C:\\Users\\me\\my runs\\r1"')
            self.assertEqual(cli._quote("C:\\runs\\r1"), "C:\\runs\\r1")
            self.assertEqual(cli._quote('say "hi"'), '"say \\"hi\\""')
            self.assertEqual(cli._quote(""), '""')

    def test_retry_errors_without_resume_is_a_usage_error(self):
        code, _, err = self.cli("run", "-c", self.dir / "eval.yaml", "--retry-errors")
        self.assertEqual(code, 2)
        self.assertIn("--retry-errors only applies together with --resume", err)
        self.assertEqual(self.calls(), [])

    def test_resume_of_a_missing_directory_is_a_usage_error(self):
        code, _, err = self.cli("run", "-c", self.dir / "eval.yaml", "--resume", self.dir / "nope")
        self.assertEqual(code, 2)
        self.assertIn("is not a directory", err)

    def test_an_unfinished_run_directory_points_at_resume(self):
        os.environ["RESUME_TEST_STOP_AT"] = "q3"
        self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report")
        os.environ.pop("RESUME_TEST_STOP_AT")
        run_dir = next((self.dir / "runs").iterdir())
        code, _, err = self.cli("report", run_dir)
        self.assertEqual(code, 2)
        self.assertIn("run it again", err)
        self.assertIn(f"--resume {run_dir}", err)

    def test_attempt_errors_are_shown_when_a_repeated_case_hides_a_crash(self):
        # 3 attempts per case; the target crashes on attempt 1 of q1 only: q1 still passes by majority, the crash must not vanish.
        # 1 of 18 attempts is 5.6%, above the default 5% limit (C16): the run exits 3, and with a raised limit it only warns.
        os.environ["RESUME_TEST_CRASH"] = "q1|1"
        try:
            code, out, err = self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--set", "repeat={n: 3, tags: [t]}", "--set", "gate.max_error_rate=0.1")
        finally:
            os.environ.pop("RESUME_TEST_CRASH")
        self.assertEqual(code, 0, err)
        self.assertRegex(out, r"Errors:\s+0 \(0\.0%")
        self.assertRegex(out, r"Attempt errors:\s+1 of 18 attempts \(5\.6%; 1 target, 0 judge\)")
        self.assertIn("1 attempt(s) of repeated cases errored but did not change their case's majority verdict", err)

    def test_a_hidden_attempt_error_rate_above_the_limit_exits_3(self):
        os.environ["RESUME_TEST_CRASH"] = "q1|1"
        try:
            code, out, err = self.cli("run", "-c", self.dir / "eval.yaml", "--quiet", "--no-report", "--set", "repeat={n: 3, tags: [t]}")
        finally:
            os.environ.pop("RESUME_TEST_CRASH")
        self.assertEqual(code, 3, err)
        self.assertIn("error: unhealthy run:", err)
        self.assertIn("1 of 18 attempts errored (5.6% of the attempts, limit 5.0%)", err)
        self.assertIn("which the majority vote of the repeated cases hides", err)
        run_dir = next(p for p in (self.dir / "runs").iterdir())
        self.assertTrue((run_dir / "summary.json").is_file())  # a paid run is never lost


class RetryHelperTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        patcher = mock.patch("evalsprint.runner._SLEEP", self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def flaky(fail_times, exc=lambda: PermissionError(13, "Permission denied")):
        state = {"n": 0}

        def action():
            state["n"] += 1
            if state["n"] <= fail_times:
                raise exc()
            return "done"

        return action, state

    def test_succeeds_after_transient_permission_errors_with_doubling_backoff(self):
        action, state = self.flaky(3)
        self.assertEqual(runner._with_retry(action), "done")
        self.assertEqual(state["n"], 4)
        self.assertEqual(self.sleeps, [0.2, 0.4, 0.8])

    def test_five_retries_then_the_error_is_raised(self):
        action, state = self.flaky(99)
        with self.assertRaises(PermissionError):
            runner._with_retry(action)
        self.assertEqual(state["n"], 6)  # the first try and 5 retries
        self.assertEqual(self.sleeps, [0.2, 0.4, 0.8, 1.6, 3.2])

    def test_the_fifth_retry_can_still_win(self):
        action, state = self.flaky(5)
        self.assertEqual(runner._with_retry(action), "done")
        self.assertEqual(len(self.sleeps), 5)

    def test_plain_oserror_is_retried_too(self):
        action, state = self.flaky(2, lambda: OSError(5, "Input/output error"))
        self.assertEqual(runner._with_retry(action), "done")
        self.assertEqual(state["n"], 3)

    def test_errors_no_wait_can_fix_are_raised_at_once(self):
        import errno

        for make in (lambda: FileNotFoundError(2, "missing"), lambda: OSError(errno.ENOSPC, "No space left on device"), lambda: OSError(errno.EROFS, "Read-only file system"), lambda: IsADirectoryError(21, "dir")):
            action, state = self.flaky(99, make)
            with self.assertRaises(OSError):
                runner._with_retry(action)
            self.assertEqual(state["n"], 1)
        self.assertEqual(self.sleeps, [])

    def test_other_exceptions_pass_through(self):
        def action():
            raise ValueError("bug")

        with self.assertRaises(ValueError):
            runner._with_retry(action)
        self.assertEqual(self.sleeps, [])

    def test_save_run_survives_a_destination_held_open_for_a_moment(self):
        real = os.replace
        state = {"n": 0}

        def replace(src, dst):
            state["n"] += 1
            if state["n"] <= 2:
                raise PermissionError(13, "The process cannot access the file because it is being used by another process")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as tmp, mock.patch("evalsprint.runner.os.replace", replace):
            run_dir = save_run(Path(tmp) / "run", {"run_id": "r"}, [{"case_id": "a", "outcome": "pass"}])
            self.assertEqual(json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["run_id"], "r")
            self.assertEqual(sorted(p.name for p in run_dir.iterdir()), ["results.jsonl", "summary.json"])  # no temp file left behind
        self.assertEqual(self.sleeps, [0.2, 0.4])

    def test_save_run_gives_up_after_the_retries_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("evalsprint.runner.os.replace", side_effect=PermissionError(13, "locked")):
                with self.assertRaises(PermissionError):
                    save_run(Path(tmp) / "run", {"run_id": "r"}, [{"case_id": "a"}])
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "run").iterdir()), [])
        self.assertEqual(self.sleeps, [0.2, 0.4, 0.8, 1.6, 3.2])  # results.jsonl never lands, so summary.json is never attempted

    def test_unlink_with_retry_ignores_a_missing_file_and_retries_a_locked_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner.unlink_with_retry(Path(tmp) / "absent")
            target = Path(tmp) / "held"
            target.write_text("x", encoding="utf-8")
            real = Path.unlink
            state = {"n": 0}

            def unlink(self, missing_ok=False):
                state["n"] += 1
                if state["n"] == 1:
                    raise PermissionError(13, "locked")
                return real(self, missing_ok=missing_ok)

            with mock.patch.object(Path, "unlink", unlink):
                runner.unlink_with_retry(target)
            self.assertFalse(target.exists())
        self.assertEqual(self.sleeps, [0.2])


class LeftoverPartialFileTests(ResumeFixture):
    def test_a_partial_file_that_cannot_be_removed_does_not_fail_a_finished_run(self):
        real = runner.unlink_with_retry

        def locked(path, *a, **kw):
            if Path(path).name == PARTIAL_NAME:
                raise PermissionError(13, "held open by an antivirus scan")
            return real(path, *a, **kw)

        err = io.StringIO()
        with mock.patch("evalsprint.runner.unlink_with_retry", locked), contextlib.redirect_stderr(err):
            run = self.run_it()
        self.assertIn("could not be removed", err.getvalue())
        loaded = load_run(run.run_dir)  # the leftover is older than summary.json: the run still reads as finished
        self.assertEqual(loaded.summary["n_cases"], 6)
        self.reset_calls()
        again = self.run_it(resume_from=run.run_dir)  # and resuming from it still works (the partial file holds the same records)
        self.assertEqual((self.calls(), again.summary["resumed_from"]["reused_records"]), ([], 6))


class BaselineCopyRetryTests(unittest.TestCase):
    """The baseline copy (`evalsprint baseline`) is an atomic replace as well: same retry."""

    def test_install_files_retries_a_locked_destination(self):
        from evalsprint.cli import _install_files

        real = os.replace
        state = {"n": 0}
        sleeps = []

        def replace(src, dst):
            state["n"] += 1
            if state["n"] <= 3:
                raise PermissionError(13, "locked by an antivirus scan")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "run", Path(tmp) / "baseline"
            source.mkdir()
            dest.mkdir()
            for name in ("results.jsonl", "summary.json"):
                (source / name).write_text(name, encoding="utf-8")
            with mock.patch("evalsprint.runner.os.replace", replace), mock.patch("evalsprint.runner._SLEEP", sleeps.append):
                _install_files(source, dest, ["results.jsonl", "summary.json"])
            self.assertEqual((dest / "summary.json").read_text(encoding="utf-8"), "summary.json")
            self.assertEqual((dest / "results.jsonl").read_text(encoding="utf-8"), "results.jsonl")
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["results.jsonl", "summary.json"])
        self.assertEqual(sleeps, [0.2, 0.4, 0.8])


if __name__ == "__main__":
    unittest.main()
