"""Placeholder runs (contract C18): a run whose target is the scaffold's `offline_stub`, or whose LLM judge runs on the mock provider,
proves the pipeline and measures nothing. `summary.placeholder` says so; `run` warns, `baseline` refuses without `--force`.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint.cli import main
from evalsprint.config import SuiteConfig
from evalsprint.runner import placeholder_reason, run_suite

STUB_TARGET = '''\
def offline_stub(input, params):
    return {"text": "canned answer to " + input}


def call_http(input, params):
    return {"text": "a real answer to " + input}
'''

JUDGE_PY = '''\
import json


def responder(system, messages, model):
    payload = {"criteria": {"helpful": {"reasoning": "fine", "pass": True}}}
    return {"text": json.dumps(payload), "usage": {"input_tokens": 10, "output_tokens": 5}}
'''

LLM_JUDGE = {
    "type": "llm",
    "name": "quality",
    "provider": "${JUDGE_PROVIDER:-mock}",
    "provider_options": {"responder": "judge_sim.py:responder"},
    "model": "claude-haiku-4-5",
    "rubric": {"name": "r", "task": "Support answers.", "criteria": [{"id": "helpful", "question": "Does the output help?"}]},
}
CONTAINS = {"type": "contains", "name": "says_answer", "values": ["answer"]}


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class PlaceholderFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "suite"
        self.dir.mkdir()
        (self.dir / "target.py").write_text(STUB_TARGET, encoding="utf-8")
        (self.dir / "judge_sim.py").write_text(JUDGE_PY, encoding="utf-8")
        (self.dir / "cases.jsonl").write_text("".join(json.dumps({"id": f"c{i}", "input": f"q{i}"}) + "\n" for i in range(3)), encoding="utf-8")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("JUDGE_PROVIDER", None)

    def config(self, callable_="target.py:call_http", judges=None, **extra) -> Path:
        path = self.dir / "eval.yaml"
        path.write_text(
            yaml.safe_dump({"suite": "ph", "dataset": "cases.jsonl", "target": {"callable": callable_}, "judges": judges or [CONTAINS], **extra}),
            encoding="utf-8",
        )
        return path

    def run_dir(self, name="run", *extra) -> Path:
        out = self.dir / name
        code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--out", out, "--quiet", "--no-report", *extra)
        self.assertEqual(code, 0, err)
        return out

    @staticmethod
    def summary(run_dir: Path) -> dict:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


class DetectionTests(PlaceholderFixture):
    def reason(self, **kw):
        return placeholder_reason(SuiteConfig.load(self.config(**kw)))

    def test_a_real_target_with_a_deterministic_judge_is_not_a_placeholder(self):
        self.assertIsNone(self.reason())

    def test_the_scaffold_stub_is_detected_by_its_name(self):
        for spec in ("target.py:offline_stub", "evals/target_adapter.py:offline_stub", "adapters/x.py: offline_stub"):
            reason = self.reason(callable_=spec)
            self.assertIsNotNone(reason, spec)
            self.assertIn("offline_stub", reason)

    def test_a_similarly_named_callable_is_not_the_stub(self):
        self.assertIsNone(self.reason(callable_="target.py:offline_stub_v2"))
        self.assertIsNone(self.reason(callable_="offline_stub.py:call_http"))

    def test_an_llm_judge_on_the_mock_provider_is_detected(self):
        reason = self.reason(judges=[CONTAINS, {**LLM_JUDGE, "provider": "mock"}])
        self.assertIn("mock provider", reason)
        self.assertIn("'quality'", reason)
        self.assertIn("mock", self.reason(judges=[{**LLM_JUDGE, "provider": "MOCK"}]))  # the provider name is case-insensitive

    def test_a_real_provider_is_not(self):
        self.assertIsNone(self.reason(judges=[{**LLM_JUDGE, "provider": "anthropic"}]))
        self.assertIsNone(self.reason(judges=[{k: v for k, v in LLM_JUDGE.items() if k != "provider"}]))  # the default provider is anthropic

    def test_a_non_llm_judge_named_mock_is_not_a_provider(self):
        self.assertIsNone(self.reason(judges=[{"type": "contains", "name": "mock", "values": ["answer"], "provider": "mock"}]))

    def test_the_env_variable_the_scaffold_uses_decides(self):
        judges = [{**LLM_JUDGE, "provider": "${JUDGE_PROVIDER:-anthropic}"}]
        self.assertIsNone(self.reason(judges=judges))
        os.environ["JUDGE_PROVIDER"] = "mock"
        self.assertIn("mock provider", self.reason(judges=judges))

    def test_a_set_override_is_seen(self):
        self.config(judges=[{**LLM_JUDGE, "provider": "anthropic"}])
        config = SuiteConfig.load(self.dir / "eval.yaml", ["judges.0.provider=mock"])
        self.assertIn("mock provider", placeholder_reason(config))


class SummaryTests(PlaceholderFixture):
    def test_the_key_is_always_written_and_a_real_run_is_false(self):
        self.config()
        run = run_suite(SuiteConfig.load(self.dir / "eval.yaml"), progress=False, out_dir=self.dir / "run")
        self.assertIs(run.summary["placeholder"], False)
        self.assertIs(self.summary(self.dir / "run")["placeholder"], False)

    def test_a_stub_run_is_true(self):
        self.config(callable_="target.py:offline_stub")
        run = run_suite(SuiteConfig.load(self.dir / "eval.yaml"), progress=False, out_dir=self.dir / "run")
        self.assertIs(run.summary["placeholder"], True)

    def test_a_mock_judge_run_is_true(self):
        self.config(judges=[{**LLM_JUDGE, "provider": "mock"}])
        run = run_suite(SuiteConfig.load(self.dir / "eval.yaml"), progress=False, out_dir=self.dir / "run")
        self.assertIs(run.summary["placeholder"], True)


class RunCommandTests(PlaceholderFixture):
    def test_a_placeholder_run_warns_even_with_quiet(self):
        self.config(callable_="target.py:offline_stub")
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "run", "--quiet", "--no-report")
        self.assertEqual(code, 0)  # a placeholder is not an unhealthy run: the smoke test must pass
        self.assertIn("warning: placeholder run:", err)
        self.assertIn("offline_stub", err)
        self.assertIn("`evalsprint baseline` refuses it without --force", err)

    def test_a_real_run_prints_no_warning(self):
        self.config()
        code, out, err = cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "run", "--quiet", "--no-report")
        self.assertEqual((code, err), (0, ""))

    def test_the_mock_provider_from_the_environment_is_the_documented_smoke_test(self):
        self.config(judges=[{**LLM_JUDGE, "provider": "${JUDGE_PROVIDER:-anthropic}"}])
        os.environ["JUDGE_PROVIDER"] = "mock"
        code, _, err = cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "run", "--quiet", "--no-report")
        self.assertEqual(code, 0, err)
        self.assertIn("mock provider", err)
        self.assertTrue(self.summary(self.dir / "run")["placeholder"])

    def test_the_report_carries_the_banner(self):
        self.config(callable_="target.py:offline_stub")
        cli("run", "-c", self.dir / "eval.yaml", "--out", self.dir / "run", "--quiet")
        html = (self.dir / "run" / "report.html").read_text(encoding="utf-8")
        self.assertIn("Placeholder run", html)


class BaselineGuardTests(PlaceholderFixture):
    def placeholder_run(self) -> Path:
        self.config(callable_="target.py:offline_stub")
        return self.run_dir("placeholder")

    def test_baseline_refuses_a_placeholder_run_and_copies_nothing(self):
        dest = self.dir / "baseline"
        code, out, err = cli("baseline", self.placeholder_run(), "--to", dest)
        self.assertEqual(code, 3)
        self.assertFalse(dest.exists())
        self.assertEqual(out, "")
        self.assertIn("refusing to use run", err)
        self.assertIn("placeholder run", err)
        self.assertIn("offline_stub", err)
        self.assertIn("--force", err)
        self.assertIn("exit 3", err)

    def test_force_copies_it_with_a_warning(self):
        dest = self.dir / "baseline"
        code, out, err = cli("baseline", self.placeholder_run(), "--to", dest, "--force")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["results.jsonl", "summary.json"])
        self.assertIn("warning: --force: this run is a placeholder", err)
        self.assertIn("copied to", out)

    def test_a_mock_judge_run_is_refused_too(self):
        self.config(judges=[{**LLM_JUDGE, "provider": "mock"}])
        run = self.run_dir("mock")
        code, _, err = cli("baseline", run, "--to", self.dir / "baseline")
        self.assertEqual(code, 3)
        self.assertIn("its LLM judge 'quality' runs on the mock provider", err)

    def test_a_real_run_is_copied_without_noise(self):
        self.config()
        code, out, err = cli("baseline", self.run_dir("real"), "--to", self.dir / "baseline")
        self.assertEqual((code, err), (0, ""))

    def test_a_summary_from_before_c18_has_no_key_and_is_accepted(self):
        self.config()
        run = self.run_dir("old")
        summary = self.summary(run)
        summary.pop("placeholder")
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, _, err = cli("baseline", run, "--to", self.dir / "baseline")
        self.assertEqual((code, err), (0, ""))

    def test_only_a_literal_true_counts(self):
        self.config()
        run = self.run_dir("odd")
        summary = self.summary(run)
        summary["placeholder"] = "yes"
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, _, _ = cli("baseline", run, "--to", self.dir / "baseline")
        self.assertEqual(code, 0)

    def test_an_unhealthy_placeholder_run_is_refused_with_both_reasons(self):
        self.config(callable_="target.py:offline_stub")
        run = self.run_dir("both")
        summary = self.summary(run)
        summary.update(n_errors=3, n_target_errors=3, error_rate=1.0, n_scored=0)
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, _, err = cli("baseline", run, "--to", self.dir / "baseline")
        self.assertEqual(code, 3)
        self.assertEqual(err.count("refusing to use run"), 2)  # one line per cause
        self.assertIn("cases errored", err)
        self.assertIn("placeholder", err)


class CompareTests(PlaceholderFixture):
    def test_comparing_against_a_placeholder_baseline_warns(self):
        self.config(callable_="target.py:offline_stub")
        baseline = self.run_dir("baseline")
        candidate = self.run_dir("candidate")
        code, out, err = cli("compare", baseline, candidate)
        self.assertIn("placeholder", out + err)
        self.assertIn("baseline", (out + err).lower())


if __name__ == "__main__":
    unittest.main()
