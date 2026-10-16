"""Tests for the `evalsprint` command line (evalsprint/cli.py).

A tiny suite is written to a temp directory: a mock target, a mock judge
responder (per the mock responder contract in docs/ARCHITECTURE.md) and six
cases. Commands that need compare/calibration/labeling are skipped when those
modules are not present.
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evalsprint.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_COMPARE = importlib.util.find_spec("evalsprint.compare") is not None
HAS_CALIBRATION = importlib.util.find_spec("evalsprint.calibration") is not None
HAS_LABELING = importlib.util.find_spec("evalsprint.labeling") is not None

TARGET_PY = '''
def answer(input, params):
    text = str(input)
    if params.get("variant") == "bad" or "parcel" in text:
        reply = "BAD: I cannot help with that."
    else:
        reply = "Here is the answer to: " + text[:40]
    return {
        "text": reply,
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 120, "output_tokens": 30},
        "latency_s": 0.2,
    }
'''

JUDGE_PY = '''
import json
import re


def judge_responder(system, messages, model):
    # the judge wraps the graded output in nonce delimiters: <ai_output_7f3a9c> ... </ai_output_7f3a9c>
    match = re.search(r"<ai_output(?:_[0-9a-f]+)?>(.*?)</ai_output(?:_[0-9a-f]+)?>", messages[-1]["content"], re.S)
    verdict = "BAD" not in match.group(1)
    payload = {"criteria": {"helpful": {"reasoning": "checked the answer", "pass": verdict}}}
    return {"text": json.dumps(payload), "usage": {"input_tokens": 300, "output_tokens": 50}}
'''

DOWN_JUDGE_PY = '''
def responder(system, messages, model):
    raise RuntimeError("HTTP 401 invalid x-api-key")
'''

INTERRUPT_JUDGE_PY = '''
def responder(system, messages, model):
    raise KeyboardInterrupt  # the user pressed Ctrl-C while the judge call was in flight
'''

# the target dies on the two "parcel" cases (2 of 6 = 33% errors) and answers the rest
FLAKY_TARGET_PY = '''
def answer(input, params):
    if "parcel" in str(input):
        raise ConnectionError("upstream timed out")
    return {"text": "Here is the answer to: " + str(input)[:40], "model": "claude-haiku-4-5", "usage": {"input_tokens": 120, "output_tokens": 30}, "latency_s": 0.2}
'''

CASES = [
    {"id": "billing-001", "input": "How do I get a refund?", "tags": ["billing"]},
    {"id": "billing-002", "input": "Can I change my invoice email?", "tags": ["billing"]},
    {"id": "billing-003", "input": "Where do I download receipts?", "tags": ["billing", "core"]},
    {"id": "shipping-001", "input": "Where is my parcel?", "tags": ["shipping"]},
    {"id": "shipping-002", "input": "<script>alert(1)</script> parcel status?", "tags": ["shipping"]},
    {"id": "misc-001", "input": "Hello there"},
]  # the two "parcel" cases always fail: 4 of 6 pass (66.7%)

CONFIG = {
    "suite": "cli-test",
    "dataset": "cases.jsonl",
    "concurrency": 2,
    "target": {"callable": "target.py:answer", "params": {"model": "claude-haiku-4-5"}},
    "judges": [
        {
            "type": "llm",
            "name": "quality",
            "provider": "mock",
            "provider_options": {"responder": "judge_sim.py:judge_responder"},
            "model": "claude-haiku-4-5",
            "rubric": {"name": "r", "task": "Support answers.", "criteria": [{"id": "helpful", "question": "Does the output help?"}]},
        }
    ],
    "gate": {"max_pass_rate_drop": 0.05},
}


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def as_a_real_run(run_dir):
    """Every run this file makes judges with the mock provider, so `summary.placeholder` is true (C18). Tests that need a
    legitimate baseline declare it the way a real judge would have left it."""
    path = Path(run_dir) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["placeholder"] = False
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(run_dir)


def without_placeholder_note(err: str) -> str:
    """stderr minus the placeholder warning every mock-judge run prints (see TestPlaceholderGuard for the warning itself)."""
    return "".join(line for line in err.splitlines(keepends=True) if not line.startswith("warning: placeholder run:"))


JUDGE_INTERRUPTED = ["--set", "judges.0.provider_options.responder=interrupt_judge.py:responder"]
JUDGE_DOWN = ["--set", "judges.0.provider_options.responder=down_judge.py:responder"]  # every judge call fails
TARGET_FLAKY = ["--set", "target.callable=flaky_target.py:answer"]  # the two "parcel" cases crash the target


class CliTestCase(unittest.TestCase):
    """Writes the suite once and produces a good and a bad run to work with."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.suite = cls.root / "suite"
        cls.suite.mkdir()
        (cls.suite / "target.py").write_text(TARGET_PY, encoding="utf-8")
        (cls.suite / "judge_sim.py").write_text(JUDGE_PY, encoding="utf-8")
        (cls.suite / "down_judge.py").write_text(DOWN_JUDGE_PY, encoding="utf-8")
        (cls.suite / "flaky_target.py").write_text(FLAKY_TARGET_PY, encoding="utf-8")
        (cls.suite / "interrupt_judge.py").write_text(INTERRUPT_JUDGE_PY, encoding="utf-8")
        (cls.suite / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8")
        cls.config = cls.suite / "eval.yaml"
        cls.config.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
        cls.good_run = cls.root / "good"
        cls.bad_run = cls.root / "bad"
        assert run_cli("run", "-c", cls.config, "--out", cls.good_run, "--label", "good", "--quiet")[0] == 0
        assert run_cli("run", "-c", cls.config, "--out", cls.bad_run, "--label", "bad", "--quiet", "--set", "target.params.variant=bad")[0] == 0
        as_a_real_run(cls.good_run), as_a_real_run(cls.bad_run)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class TestRun(CliTestCase):
    def test_writes_run_directory_and_report(self):
        out_dir = self.tmp / "run"
        code, out, err = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet")
        self.assertEqual(code, 0, err)
        for name in ("results.jsonl", "summary.json", "report.html"):
            self.assertTrue((out_dir / name).is_file(), name)
        self.assertEqual(without_placeholder_note(err), "")  # --quiet
        for expected in ("Pass rate:", "66.7%", "95% CI", "Cost / 1k requests:", "Latency p95:", "Judge cost:", str(out_dir), str(out_dir / "report.html")):
            self.assertIn(expected, out)
        self.assertNotIn("—", out)  # terminal output stays plain

        report = (out_dir / "report.html").read_text(encoding="utf-8")
        self.assertIn("66.7%", report)
        for case_id in ("shipping-001", "shipping-002"):  # failing cases are listed with an anchor
            self.assertIn(f'id="case-{case_id}"', report)

    def test_untrusted_input_is_escaped_in_the_report(self):
        out_dir = self.tmp / "run"
        run_cli("run", "-c", self.config, "--out", out_dir, "--quiet")
        report = (out_dir / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertNotIn("<script", report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)

    def test_progress_goes_to_stderr_unless_quiet(self):
        code, out, err = run_cli("run", "-c", self.config, "--out", self.tmp / "run")
        self.assertEqual(code, 0)
        self.assertIn("PASS billing-001", err)
        self.assertIn("FAIL shipping-001", err)
        self.assertNotIn("PASS billing-001", out)

    def test_no_report_flag(self):
        out_dir = self.tmp / "run"
        code, out, _ = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", "--no-report")
        self.assertEqual(code, 0)
        self.assertFalse((out_dir / "report.html").exists())
        self.assertIn("--no-report", out)

    def test_tags_and_limit(self):
        out_dir = self.tmp / "run"
        code, *_ = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", "--tags", "billing,core", "--limit", "2")
        self.assertEqual(code, 0)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["n_cases"], 2)
        self.assertEqual(set(summary["by_tag"]), {"billing"})  # the first two cases carrying billing or core

    def test_set_overrides_and_label(self):
        summary = json.loads((self.bad_run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["pass_rate"], 0.0)
        self.assertEqual(summary["label"], "bad")
        self.assertTrue(summary["run_id"].endswith("-bad"))
        self.assertEqual(summary["target"]["params"]["variant"], "bad")

    def test_default_run_directory_is_under_runs_next_to_the_config(self):
        suite = self.tmp / "suite"
        suite.mkdir()
        for path in self.suite.iterdir():
            if path.is_file():
                (suite / path.name).write_bytes(path.read_bytes())
        code, out, _ = run_cli("run", "-c", suite / "eval.yaml", "--quiet", "--label", "smoke")
        self.assertEqual(code, 0)
        [run_dir] = list((suite / "runs").iterdir())
        self.assertTrue(run_dir.name.endswith("-smoke"))
        self.assertTrue((run_dir / "report.html").is_file())


class TestCompare(CliTestCase):
    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_regression_exits_1_and_writes_files(self):
        code, out, err = run_cli("compare", self.good_run, self.bad_run, "-c", self.config, "--md", self.tmp / "out" / "c.md", "--json", self.tmp / "c.json")
        self.assertEqual((code, err), (1, ""))
        self.assertIn("Regression", out)
        self.assertIn("Regression", (self.tmp / "out" / "c.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((self.tmp / "c.json").read_text(encoding="utf-8"))["verdict"], "regression")

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_identical_runs_pass_with_exit_0(self):
        code, out, _ = run_cli("compare", self.good_run, self.good_run)
        self.assertEqual(code, 0)
        self.assertNotIn("Regression detected", out)

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_gate_from_config_overrides_the_run_gate(self):
        # good -> bad is a regression under any pass-rate gate, so relax everything: the config wins
        relaxed = self.tmp / "relaxed.yaml"
        relaxed.write_text(yaml.safe_dump({**CONFIG, "dataset": str(self.suite / "cases.jsonl"), "gate": {"max_pass_rate_drop": 1.0, "max_error_rate": 1.0}}), encoding="utf-8")
        code, *_ = run_cli("compare", self.good_run, self.bad_run, "-c", relaxed)
        self.assertEqual(code, 0)

    def test_compare_needs_finished_runs(self):
        code, out, err = run_cli("compare", self.tmp / "missing", self.good_run)
        self.assertEqual(code, 2)
        self.assertIn("not a run directory", err)
        self.assertEqual(out, "")


class TestReportCommand(CliTestCase):
    def test_report_for_a_single_run(self):
        code, out, _ = run_cli("report", self.good_run)
        self.assertEqual(code, 0)
        self.assertTrue((self.good_run / "report.html").is_file())
        self.assertIn(str(self.good_run / "report.html"), out)

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_report_with_baseline_shows_the_verdict(self):
        target = self.tmp / "custom" / "r.html"
        code, out, _ = run_cli("report", self.bad_run, "--baseline", self.good_run, "-o", target)
        self.assertEqual(code, 0)  # a report is not a gate
        html = target.read_text(encoding="utf-8")
        self.assertIn("Regression detected", html)
        self.assertIn("vs baseline", html)
        self.assertIn("regression", out)

    @unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
    def test_report_with_labels_adds_calibration(self):
        labels = self._write_labels()
        target = self.tmp / "r.html"
        code, _, err = run_cli("report", self.good_run, "--labels", labels, "-o", target)
        self.assertEqual((code, err), (0, ""))
        html = target.read_text(encoding="utf-8")
        self.assertIn("Calibration", html)
        self.assertIn("Calibrated pass rate", html)

    def test_judge_and_config_need_their_companion_flags(self):
        for argv in (["--judge", "quality"], ["-c", self.config]):
            code, _, err = run_cli("report", self.good_run, *argv)
            self.assertEqual(code, 2, argv)
            self.assertEqual(len(err.strip().splitlines()), 1)

    def _write_labels(self):
        records = [json.loads(line) for line in (self.good_run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        path = self.tmp / "labels.jsonl"
        rows = [{"case_id": r["case_id"], "label": "pass" if r["passed"] else "fail", "output_sha": r["output"]["sha"], "labeler": "test"} for r in records]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    @unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
    def test_calibrate_command(self):
        labels = self._write_labels()
        code, out, err = run_cli("calibrate", self.good_run, "--labels", labels, "--md", self.tmp / "cal.md", "--json", self.tmp / "cal.json")
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(out.strip())
        self.assertEqual(json.loads((self.tmp / "cal.json").read_text(encoding="utf-8"))["n"], 6)
        self.assertTrue((self.tmp / "cal.md").read_text(encoding="utf-8").strip())

    @unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
    def test_calibrate_reports_a_missing_labels_file_as_a_usage_error(self):
        code, _, err = run_cli("calibrate", self.good_run, "--labels", self.tmp / "nope.jsonl")
        self.assertEqual(code, 2)
        self.assertIn("file not found", err)


class TestLabel(CliTestCase):
    @unittest.skipUnless(HAS_LABELING, "evalsprint.labeling is not available")
    def test_html_by_default_and_csv_on_request(self):
        code, *_ = run_cli("label", self.good_run)
        self.assertEqual(code, 0)
        self.assertTrue((self.good_run / "labeling.html").is_file())
        code, *_ = run_cli("label", self.good_run, "--csv", "--sample", "3", "--seed", "1", "--no-stratify", "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "sheet.csv").is_file())

    def test_sample_must_be_positive(self):
        code, _, err = run_cli("label", self.good_run, "--sample", "0")
        self.assertEqual(code, 2)


class TestBaseline(CliTestCase):
    def test_copies_results_and_summary(self):
        dest = self.tmp / "evals" / "baseline"
        code, out, _ = run_cli("baseline", self.good_run, "--to", dest)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["results.jsonl", "summary.json"])  # report/labeling files are not copied
        for name in ("results.jsonl", "summary.json"):
            self.assertEqual((dest / name).read_bytes(), (self.good_run / name).read_bytes())
            self.assertIn(name, out)

    def test_overwrites_an_existing_baseline(self):
        dest = self.tmp / "baseline"
        run_cli("baseline", self.bad_run, "--to", dest)
        run_cli("baseline", self.good_run, "--to", dest)
        self.assertEqual((dest / "summary.json").read_bytes(), (self.good_run / "summary.json").read_bytes())

    def test_a_copied_baseline_is_a_valid_run(self):
        dest = self.tmp / "baseline"
        run_cli("baseline", self.good_run, "--to", dest)
        code, *_ = run_cli("report", dest, "-o", self.tmp / "r.html")
        self.assertEqual(code, 0)

    def test_refuses_to_copy_onto_itself(self):
        code, _, err = run_cli("baseline", self.good_run, "--to", self.good_run)
        self.assertEqual(code, 2)
        self.assertIn("different directory", err)


class TestDescribe(CliTestCase):
    def test_lists_dataset_statistics_and_judges(self):
        code, out, _ = run_cli("describe", "-c", self.config)
        self.assertEqual(code, 0)
        for expected in ("cli-test", "Cases:", "6", "billing", "shipping", "quality", "llm", "mock/claude-haiku-4-5", "target.py:answer", "max_pass_rate_drop=0.05"):
            self.assertIn(expected, out)

    def test_shows_the_judges_sha_a_run_will_store(self):
        code, out, _ = run_cli("describe", "-c", self.config)
        stored = json.loads((self.good_run / "summary.json").read_text(encoding="utf-8"))["judges_sha"]
        self.assertRegex(out, rf"Judges sha:\s+{stored}\b")

    def test_warns_when_the_code_of_a_custom_judge_cannot_be_hashed(self):
        code, out, err = run_cli("describe", "-c", self.config)
        self.assertNotIn("will NOT change judges_sha", out)
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["judges"].append({"type": "custom", "name": "floor", "callable": "math:floor"})  # a compiled module: no source file
        path = self.suite / "eval_custom.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.addCleanup(path.unlink)  # the suite directory is shared by the whole class
        code, out, err = run_cli("describe", "-c", path)
        self.assertEqual(code, 0, err)
        self.assertIn("custom judge 'floor'", out)
        self.assertIn("will NOT change judges_sha", out)


class TestUsageErrors(CliTestCase):
    def assertUsageError(self, *argv, contains=None):
        code, out, err = run_cli(*argv)
        self.assertEqual(code, 2, (argv, err))
        lines = err.strip().splitlines()
        self.assertEqual(len(lines), 1, err)  # one line, no traceback
        self.assertTrue(lines[0].startswith("error: "), err)
        self.assertNotIn("Traceback", err)
        if contains:
            self.assertIn(contains, err)

    def config_with(self, text, name="bad.yaml"):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_config_file(self):
        self.assertUsageError("run", "-c", self.tmp / "nope.yaml", contains="file not found")

    def test_malformed_yaml(self):
        self.assertUsageError("run", "-c", self.config_with("suite: [unclosed"), contains="invalid YAML")

    def test_config_missing_required_key(self):
        self.assertUsageError("describe", "-c", self.config_with("suite: x\ntarget: {callable: 'a.py:b'}\n"), contains="missing 'dataset'")

    def test_dataset_file_missing(self):
        config = self.config_with(yaml.safe_dump({**CONFIG, "dataset": "nope.jsonl"}))
        self.assertUsageError("run", "-c", config, "--quiet", contains="file not found")

    def test_bad_override(self):
        self.assertUsageError("run", "-c", self.config, "--set", "no-equals-sign", contains="override")

    def test_missing_required_argument(self):
        self.assertUsageError("run", contains="-c/--config")

    def test_no_command_and_unknown_command(self):
        self.assertUsageError()
        self.assertUsageError("frobnicate", contains="invalid choice")

    def test_invalid_numbers(self):
        self.assertUsageError("run", "-c", self.config, "--limit", "many", contains="--limit")
        self.assertUsageError("run", "-c", self.config, "--limit", "0", contains="positive")

    def test_tags_matching_nothing(self):
        self.assertUsageError("run", "-c", self.config, "--tags", "nonexistent", contains="no cases match")
        self.assertFalse((self.suite / "runs").exists())  # nothing was run

    def test_unknown_run_directory(self):
        self.assertUsageError("report", self.tmp / "no-such-run", contains="not a run directory")
        self.assertUsageError("baseline", self.tmp, "--to", self.tmp / "b", contains="not a run directory")

    def test_debug_shows_the_traceback(self):
        with self.assertRaises(FileNotFoundError):
            run_cli("run", "-c", self.tmp / "nope.yaml", "--debug")
        with self.assertRaises(FileNotFoundError):
            run_cli("--debug", "run", "-c", self.tmp / "nope.yaml")

    def test_help_and_version_exit_0(self):
        code, out, _ = run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn("evalsprint", out)
        code, out, _ = run_cli("run", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--tags", out)


class TestUnhealthyRun(CliTestCase):
    """Exit 3: the run finished but errors make its numbers meaningless. Files are still written."""

    def run_broken(self, *extra, name="run"):
        out_dir = self.tmp / name
        code, out, err = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", *extra)
        return code, out, err, out_dir

    def test_judge_outage_exits_3_and_writes_everything(self):
        code, out, err, out_dir = self.run_broken(*JUDGE_DOWN)
        self.assertEqual(code, 3, err)
        for name in ("results.jsonl", "summary.json"):
            self.assertTrue((out_dir / name).is_file(), name)  # a paid run is never lost
        s = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((s["n_errors"], s["n_judge_errors"], s["n_target_errors"], s["n_scored"]), (6, 6, 0, 0))
        self.assertEqual((s["error_rate"], s["pass_rate"]), (1.0, None))  # a dead judge is not a 0% pass rate
        self.assertIn("Errors", out)
        self.assertIn("6 (100.0%", out)

    def test_the_unhealthy_message_is_one_line_naming_the_cause(self):
        _, _, err, _ = self.run_broken(*JUDGE_DOWN)
        lines = without_placeholder_note(err).strip().splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("error: unhealthy run: "), err)
        self.assertIn("6 judge error(s)", err)
        self.assertIn("no case could be scored", err)
        self.assertIn("exit 3", err)

    def test_target_errors_above_the_limit_exit_3_and_are_out_of_the_pass_rate(self):
        code, out, err, out_dir = self.run_broken(*TARGET_FLAKY)
        self.assertEqual(code, 3, err)
        s = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((s["n_target_errors"], s["n_judge_errors"], s["n_scored"], s["n_passed"]), (2, 0, 4, 4))
        self.assertEqual(s["pass_rate"], 1.0)  # 4 of the 4 that could be scored; the crashes are not failures
        self.assertIn("2 target error(s)", err)
        self.assertIn("limit 5.0%", err)

    def test_a_raised_limit_makes_the_same_run_healthy(self):
        code, _, err, _ = self.run_broken(*TARGET_FLAKY, "--set", "gate.max_error_rate=0.5")
        self.assertEqual((code, without_placeholder_note(err)), (0, ""))

    def test_max_error_rate_null_disables_the_check(self):
        config = self.tmp / "no-limit.yaml"
        config.write_text(yaml.safe_dump({**CONFIG, "dataset": str(self.suite / "cases.jsonl"), "gate": {"max_error_rate": None}}), encoding="utf-8")
        args = ["run", "-c", config, "--quiet", "--set", "target.callable=flaky_target.py:answer"]
        # the config lives outside the suite directory: point the target and judge files at it explicitly
        for name in ("flaky_target.py", "judge_sim.py"):
            (self.tmp / name).write_bytes((self.suite / name).read_bytes())
        code, _, err = run_cli(*args, "--out", self.tmp / "run")
        self.assertEqual((code, without_placeholder_note(err)), (0, ""))

    def test_report_failure_is_only_a_warning_and_keeps_the_exit_code(self):
        with mock.patch("evalsprint.cli.write_report", side_effect=RuntimeError("template exploded")):
            code, out, err = run_cli("run", "-c", self.config, "--out", self.tmp / "ok", "--quiet")
            self.assertEqual(code, 0)
            self.assertIn("report.html was not written: RuntimeError: template exploded", err)
            self.assertIn("not written (see warning)", out)
            self.assertTrue((self.tmp / "ok" / "summary.json").is_file())  # the results are safe
            code, _, err = run_cli("run", "-c", self.config, "--out", self.tmp / "bad", "--quiet", *JUDGE_DOWN)
            self.assertEqual(code, 3)  # an unhealthy run stays exit 3 even when the report fails too
            self.assertIn("template exploded", err)
            self.assertIn("unhealthy run", err)

    def test_files_are_written_before_anything_is_printed(self):
        out_dir = self.tmp / "run"
        seen = []

        def spy(text=""):
            seen.append(((out_dir / "summary.json").is_file(), (out_dir / "results.jsonl").is_file(), (out_dir / "report.html").is_file()))

        with mock.patch("evalsprint.cli._say", spy):
            run_cli("run", "-c", self.config, "--out", out_dir, "--quiet")
        self.assertTrue(seen)
        self.assertTrue(all(flags == (True, True, True) for flags in seen), seen)

    def test_compare_writes_md_and_json_before_printing(self):
        md, js = self.tmp / "c.md", self.tmp / "c.json"
        seen = []

        def spy(text=""):
            seen.append((md.is_file(), js.is_file()))

        with mock.patch("evalsprint.cli._say", spy):
            run_cli("compare", self.good_run, self.bad_run, "--md", md, "--json", js)
        self.assertEqual(seen[0], (True, True))

    def test_interrupted_run_leaves_no_summary_and_compare_says_so(self):
        run_dir = self.tmp / "half"
        run_dir.mkdir()
        (run_dir / "results.partial.jsonl").write_text("{}\n", encoding="utf-8")
        code, _, err = run_cli("report", run_dir)
        self.assertEqual(code, 2)
        self.assertIn("unfinished run", err)
        self.assertIn("run it again", err)


class TestStaleRunDirectory(CliTestCase):
    """--out onto a finished run, then an interruption: nothing may read the old run as the new one."""

    def rerun_interrupted_into_a_copy_of_the_good_run(self):
        out = self.tmp / "reused"
        shutil_copy_run(self.good_run, out)
        code, _, err = run_cli("run", "-c", self.config, "--out", out, "--quiet", *JUDGE_INTERRUPTED)
        self.assertEqual((code, err.splitlines()[0]), (130, "interrupted"))
        self.assertIn(f"--resume {out}", err)  # and the message says how to continue without paying for the finished cases again
        return out

    def test_the_old_summary_and_results_are_gone_after_an_interruption(self):
        out = self.rerun_interrupted_into_a_copy_of_the_good_run()
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["results.partial.jsonl", "run_meta.json"])

    def test_every_consumer_refuses_the_interrupted_directory(self):
        out = self.rerun_interrupted_into_a_copy_of_the_good_run()
        for argv in (
            ["compare", self.good_run, out],
            ["compare", out, self.good_run],
            ["report", out],
            ["baseline", out, "--to", self.tmp / "bl"],
            ["calibrate", out, "--labels", self.tmp / "labels.jsonl"],
            ["label", out],
        ):
            with self.subTest(command=argv[0]):
                code, stdout, err = run_cli(*argv)
                self.assertEqual(code, 2, err)
                self.assertIn("unfinished run", err)
                self.assertEqual(stdout, "")
        self.assertFalse((self.tmp / "bl").exists())

    def test_a_stale_summary_next_to_a_newer_partial_file_is_refused(self):
        # a directory as an older evalsprint (or a crash at the wrong moment) could leave it: finished files + a newer partial file
        out = self.tmp / "stale"
        shutil_copy_run(self.good_run, out)
        partial = out / "results.partial.jsonl"
        partial.write_text('{"case_id": "billing-001"}\n', encoding="utf-8")
        later = (out / "summary.json").stat().st_mtime + 60
        os.utime(partial, (later, later))
        for argv in (["compare", self.good_run, out], ["report", out], ["baseline", out, "--to", self.tmp / "bl"], ["label", out]):
            with self.subTest(command=argv[0]):
                code, stdout, err = run_cli(*argv)
                self.assertEqual(code, 2, err)
                self.assertIn("results.partial.jsonl is newer than summary.json", err)
                self.assertEqual(len(err.strip().splitlines()), 1)
                self.assertEqual(stdout, "")
        self.assertFalse((self.tmp / "bl").exists())

    def test_a_rerun_that_completes_is_readable_again(self):
        out = self.rerun_interrupted_into_a_copy_of_the_good_run()
        self.assertEqual(run_cli("run", "-c", self.config, "--out", out, "--quiet")[0], 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["report.html", "results.jsonl", "run_meta.json", "summary.json"])
        self.assertEqual(run_cli("report", out)[0], 0)

    def test_out_onto_a_directory_with_a_foreign_summary_is_a_usage_error(self):
        out = self.tmp / "foreign"
        out.mkdir()
        (out / "summary.json").write_text('{"revenue": 1}', encoding="utf-8")
        code, _, err = run_cli("run", "-c", self.config, "--out", out, "--quiet")
        self.assertEqual(code, 2)
        self.assertIn("not an evalsprint run file", err)
        self.assertEqual((out / "summary.json").read_text(encoding="utf-8"), '{"revenue": 1}')


class TestBaselineGuards(CliTestCase):
    def broken_run(self):
        out_dir = self.tmp / "broken"
        self.assertEqual(run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", *JUDGE_DOWN)[0], 3)
        return as_a_real_run(out_dir)  # unhealthy, but not a placeholder: the refusal below is about the errors alone

    def test_an_unhealthy_run_is_refused_as_a_baseline(self):
        dest = self.tmp / "baseline"
        code, out, err = run_cli("baseline", self.broken_run(), "--to", dest)
        self.assertEqual(code, 3)
        self.assertFalse(dest.exists())  # nothing was copied
        self.assertEqual(out, "")
        self.assertIn("refusing to use run", err)
        self.assertIn("--force", err)
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_force_copies_it_anyway_with_a_warning(self):
        dest = self.tmp / "baseline"
        code, out, err = run_cli("baseline", self.broken_run(), "--to", dest, "--force")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["results.jsonl", "summary.json"])
        self.assertIn("warning: --force", err)
        self.assertIn("unhealthy", err)
        self.assertIn("copied to", out)

    def test_a_partially_broken_run_within_the_limit_is_accepted(self):
        run_dir = self.tmp / "tolerated"
        run_cli("run", "-c", self.config, "--out", run_dir, "--quiet", *TARGET_FLAKY, "--set", "gate.max_error_rate=0.5")
        as_a_real_run(run_dir)
        code, _, err = run_cli("baseline", run_dir, "--to", self.tmp / "baseline")
        self.assertEqual((code, err), (0, ""))  # the run's own gate says 33% errors are acceptable

    def test_calibration_json_travels_with_the_baseline(self):
        source = self.tmp / "run"
        shutil_copy_run(self.good_run, source)
        (source / "calibration.json").write_text(json.dumps({"n": 6, "kappa": 0.9}), encoding="utf-8")
        dest = self.tmp / "baseline"
        code, out, _ = run_cli("baseline", source, "--to", dest)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["calibration.json", "results.jsonl", "summary.json"])
        self.assertEqual(json.loads((dest / "calibration.json").read_text(encoding="utf-8"))["kappa"], 0.9)
        self.assertIn("calibration.json", out)

    def test_a_stale_calibration_is_not_left_next_to_a_new_baseline(self):
        dest = self.tmp / "baseline"
        source = self.tmp / "run"
        shutil_copy_run(self.good_run, source)
        (source / "calibration.json").write_text(json.dumps({"n": 6}), encoding="utf-8")
        run_cli("baseline", source, "--to", dest)
        code, _, err = run_cli("baseline", self.bad_run, "--to", dest)  # a new baseline without labels
        self.assertEqual(code, 0)
        self.assertFalse((dest / "calibration.json").exists())  # it described the previous baseline's labels
        self.assertIn("removed the previous baseline's calibration.json", err)

    def previous_baseline(self):
        dest = self.tmp / "baseline"
        self.assertEqual(run_cli("baseline", self.bad_run, "--to", dest)[0], 0)
        return dest

    def test_a_failure_while_swapping_files_never_leaves_new_results_next_to_the_old_summary(self):
        dest = self.previous_baseline()
        real_replace = os.replace

        def summary_is_locked(src, dst):  # Windows: another process (antivirus, an editor) holds summary.json open
            if Path(dst).name == "summary.json":
                raise PermissionError(13, "The process cannot access the file because it is being used by another process", str(dst))
            return real_replace(src, dst)

        with mock.patch("evalsprint.cli.os.replace", summary_is_locked):
            code, _, err = run_cli("baseline", self.good_run, "--to", dest)
        self.assertEqual(code, 2)
        self.assertIn("being used by another process", err)
        self.assertFalse((dest / "summary.json").exists())  # torn state = no summary: it reads as "not a run directory", never as half old, half new
        self.assertEqual([p.name for p in dest.iterdir() if ".tmp" in p.name], [])
        code, _, err = run_cli("report", dest)
        self.assertEqual(code, 2)
        self.assertIn("not a run directory", err)
        self.assertEqual(run_cli("baseline", self.good_run, "--to", dest)[0], 0)  # and the next attempt repairs it
        self.assertEqual((dest / "summary.json").read_bytes(), (self.good_run / "summary.json").read_bytes())

    def test_a_failure_while_staging_changes_nothing(self):
        dest = self.previous_baseline()
        before = {p.name: p.read_bytes() for p in dest.iterdir()}
        real_copy = shutil.copyfile

        def disk_full(src, dst, **kw):
            if Path(dst).name.startswith("summary.json"):
                raise OSError(28, "No space left on device")
            return real_copy(src, dst, **kw)

        with mock.patch("evalsprint.cli.shutil.copyfile", disk_full):
            code, _, err = run_cli("baseline", self.good_run, "--to", dest)
        self.assertEqual(code, 2)
        self.assertIn("No space left", err)
        self.assertEqual({p.name: p.read_bytes() for p in dest.iterdir()}, before)  # the previous baseline is intact, no temp files

    def test_copies_are_atomic_no_temp_files_remain(self):
        dest = self.tmp / "baseline"
        run_cli("baseline", self.good_run, "--to", dest)
        self.assertFalse([p for p in dest.iterdir() if ".tmp" in p.name])


def shutil_copy_run(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "summary.json"):
        (dest / name).write_bytes((source / name).read_bytes())


@unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
class TestCompareGate(CliTestCase):
    """C7: `--gate-config` (gate only from that file) and the baseline's calibration.json."""

    def gate_file(self, gate, name="gate.yaml"):
        path = self.tmp / name
        path.write_text(yaml.safe_dump({"gate": gate}), encoding="utf-8")
        return path

    def spy_compare(self, *argv):
        from evalsprint import compare

        with mock.patch("evalsprint.compare.compare_runs", wraps=compare.compare_runs) as spy:
            result = run_cli(*argv)
        self.assertEqual(spy.call_count, 1)
        return result, spy.call_args

    def test_gate_config_replaces_the_gate_and_missing_keys_take_the_defaults(self):
        from evalsprint.config import DEFAULT_GATE

        _, call = self.spy_compare("compare", self.good_run, self.bad_run, "--gate-config", self.gate_file({"max_pass_rate_drop": 0.2}))
        self.assertEqual(call.kwargs["gate"], {**DEFAULT_GATE, "max_pass_rate_drop": 0.2})

    def test_a_candidate_cannot_loosen_its_own_gate(self):
        # the bad run was produced under a gate that tolerates anything (what a malicious or careless PR would commit)
        loose = self.tmp / "loose-run"
        run_cli("run", "-c", self.config, "--out", loose, "--quiet", "--set", "target.params.variant=bad", "--set", "gate.max_pass_rate_drop=1.0", "--set", "gate.max_error_rate=1.0")
        code, *_ = run_cli("compare", self.good_run, loose)
        self.assertEqual(code, 0)  # without --gate-config the run's own (loosened) gate applies: this is the hole
        code, out, _ = run_cli("compare", self.good_run, loose, "--gate-config", self.gate_file({"max_pass_rate_drop": 0.05}))
        self.assertEqual(code, 1)  # the base branch's gate applies instead
        self.assertIn("Regression", out)

    def test_gate_config_wins_over_config(self):
        strict = self.gate_file({"max_pass_rate_drop": 0.05})
        relaxed = self.tmp / "relaxed.yaml"
        relaxed.write_text(yaml.safe_dump({**CONFIG, "dataset": str(self.suite / "cases.jsonl"), "gate": {"max_pass_rate_drop": 1.0, "max_error_rate": 1.0}}), encoding="utf-8")
        self.assertEqual(run_cli("compare", self.good_run, self.bad_run, "-c", relaxed)[0], 0)
        self.assertEqual(run_cli("compare", self.good_run, self.bad_run, "-c", relaxed, "--gate-config", strict)[0], 1)

    def test_gate_config_may_be_a_whole_eval_yaml(self):
        code, *_ = run_cli("compare", self.good_run, self.bad_run, "--gate-config", self.config)
        self.assertEqual(code, 1)  # only its gate: block is read; the rest (dataset, target, judges) is ignored

    def test_gate_config_errors_are_usage_errors(self):
        code, _, err = run_cli("compare", self.good_run, self.bad_run, "--gate-config", self.tmp / "missing.yaml")
        self.assertEqual(code, 2)
        self.assertIn("file not found", err)
        code, _, err = run_cli("compare", self.good_run, self.bad_run, "--gate-config", self.gate_file({"max_pas_rate_drop": 0.1}))
        self.assertEqual(code, 2)
        self.assertIn("did you mean 'max_pass_rate_drop'", err)
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_no_gate_flags_uses_the_candidate_runs_gate(self):
        _, call = self.spy_compare("compare", self.good_run, self.bad_run)
        self.assertIsNone(call.kwargs["gate"])

    def test_the_baselines_calibration_json_is_loaded(self):
        baseline = self.tmp / "baseline"
        shutil_copy_run(self.good_run, baseline)
        calibration = {"n": 12, "kappa": 0.81, "judge": "overall"}
        (baseline / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
        (_, _, _), call = self.spy_compare("compare", baseline, self.good_run)
        self.assertEqual(call.kwargs["calibration"], calibration)

    def test_no_calibration_json_means_no_calibration_argument(self):
        _, call = self.spy_compare("compare", self.good_run, self.bad_run)
        self.assertNotIn("calibration", call.kwargs)

    def test_a_broken_calibration_json_is_a_usage_error(self):
        baseline = self.tmp / "baseline"
        shutil_copy_run(self.good_run, baseline)
        (baseline / "calibration.json").write_text("{not json", encoding="utf-8")
        code, out, err = run_cli("compare", baseline, self.good_run)
        self.assertEqual(code, 2)
        self.assertIn("calibration.json is not valid JSON", err)
        self.assertEqual(out, "")
        (baseline / "calibration.json").write_text("[1, 2]", encoding="utf-8")
        code, _, err = run_cli("compare", baseline, self.good_run)
        self.assertEqual(code, 2)
        self.assertIn("JSON object", err)

    @unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
    def test_calibrate_json_into_the_baseline_feeds_compare(self):
        baseline = self.tmp / "baseline"
        shutil_copy_run(self.good_run, baseline)
        records = [json.loads(line) for line in (self.good_run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        labels = self.tmp / "labels.jsonl"
        labels.write_text("\n".join(json.dumps({"case_id": r["case_id"], "label": "pass" if r["passed"] else "fail", "output_sha": r["output"]["sha"], "labeler": "t"}) for r in records) + "\n", encoding="utf-8")
        self.assertEqual(run_cli("calibrate", baseline, "--labels", labels, "--json", baseline / "calibration.json")[0], 0)
        (_, _, err), call = self.spy_compare("compare", baseline, self.good_run)
        self.assertEqual(err, "")
        self.assertEqual(call.kwargs["calibration"]["n"], 6)


class TestLabelJudge(CliTestCase):
    @unittest.skipUnless(HAS_LABELING, "evalsprint.labeling is not available")
    def test_judge_reaches_the_export(self):
        from evalsprint import labeling

        with mock.patch("evalsprint.labeling.export_labeling_html", wraps=labeling.export_labeling_html) as spy:
            code, *_ = run_cli("label", self.good_run, "--judge", "quality", "-o", self.tmp / "l.html")
            self.assertEqual(code, 0)
            self.assertEqual(spy.call_args.kwargs["judge"], "quality")
            run_cli("label", self.good_run, "-o", self.tmp / "l2.html")
            self.assertNotIn("judge", spy.call_args.kwargs)  # not passed unless asked
        with mock.patch("evalsprint.labeling.export_labeling_csv", wraps=labeling.export_labeling_csv) as spy:
            self.assertEqual(run_cli("label", self.good_run, "--csv", "--judge", "quality", "-o", self.tmp / "l.csv")[0], 0)
            self.assertEqual(spy.call_args.kwargs["judge"], "quality")

    @unittest.skipUnless(HAS_LABELING, "evalsprint.labeling is not available")
    def test_judge_reaches_select_sample_from_both_exports(self):
        from evalsprint import labeling

        for flags, out in ((["--judge", "quality"], "l.html"), (["--csv", "--judge", "quality"], "l.csv")):
            with self.subTest(out=out), mock.patch("evalsprint.labeling.select_sample", wraps=labeling.select_sample) as spy:
                code, _, err = run_cli("label", self.good_run, *flags, "--sample", "4", "--seed", "3", "-o", self.tmp / out)
                self.assertEqual((code, err), (0, ""))
                call = spy.call_args
                judge = call.kwargs["judge"] if "judge" in call.kwargs else call.args[4]  # (run, sample, seed, stratify, judge)
                self.assertEqual(judge, "quality")
        with mock.patch("evalsprint.labeling.select_sample", wraps=labeling.select_sample) as spy:
            run_cli("label", self.good_run, "-o", self.tmp / "none.html")
            call = spy.call_args
            self.assertIsNone(call.kwargs["judge"] if "judge" in call.kwargs else call.args[4])  # overall verdict when not asked
        csv_text = (self.tmp / "l.csv").read_text(encoding="utf-8-sig")
        self.assertIn("quality", csv_text.splitlines()[1])  # the sheet records what the sample was stratified on

    @unittest.skipUnless(HAS_LABELING, "evalsprint.labeling is not available")
    def test_unknown_judge_is_a_usage_error_with_a_suggestion(self):
        code, _, err = run_cli("label", self.good_run, "--judge", "qualty")
        self.assertEqual(code, 2)
        self.assertIn("did you mean 'quality'", err)
        self.assertIn("judges: quality", err)


class TestReportOptions(CliTestCase):
    def spy_report(self, *argv):
        from evalsprint.report import write_report

        with mock.patch("evalsprint.cli.write_report", wraps=write_report) as spy:
            result = run_cli(*argv)
        return result, spy.call_args

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_baseline_run_is_passed_to_the_report(self):
        (code, _, err), call = self.spy_report("report", self.bad_run, "--baseline", self.good_run, "-o", self.tmp / "r.html")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(call.kwargs["baseline"].run_id, json.loads((self.good_run / "summary.json").read_text(encoding="utf-8"))["run_id"])
        self.assertNotIn("monthly_requests", call.kwargs)

    def test_monthly_requests_reaches_the_report(self):
        (code, _, err), call = self.spy_report("report", self.good_run, "--monthly-requests", "50000", "-o", self.tmp / "r.html")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(call.kwargs["monthly_requests"], 50000)
        self.assertNotIn("baseline", call.kwargs)

    def test_monthly_requests_must_be_positive(self):
        for bad in ("0", "-5"):
            code, _, err = run_cli("report", self.good_run, "--monthly-requests", bad)
            self.assertEqual(code, 2, bad)
            self.assertIn("--monthly-requests", err)

    def test_gate_config_needs_a_baseline(self):
        code, _, err = run_cli("report", self.good_run, "--gate-config", self.config)
        self.assertEqual(code, 2)
        self.assertIn("--baseline", err)


class TestRepeatFlag(CliTestCase):
    def test_repeat_flag_runs_every_case_n_times(self):
        out_dir = self.tmp / "run"
        code, out, err = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", "--repeat", "3")
        self.assertEqual((code, without_placeholder_note(err)), (0, ""))
        s = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(s["repeat"], {"n": 3, "per_tag": {}})
        self.assertEqual(s["stability"]["n_repeated"], 6)
        self.assertIn("Stability", out)
        self.assertIn("flip rate 0.0%", out)
        records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(len(r["attempts"]) == 3 for r in records))

    def test_repeat_must_be_positive(self):
        code, _, err = run_cli("run", "-c", self.config, "--repeat", "0")
        self.assertEqual(code, 2)
        self.assertIn("--repeat", err)


NON_ASCII_CASES = [
    {"id": "café-001", "input": "Où est mon colis ? 你好", "tags": ["facturação", "日本語"]},
    {"id": "日本-002", "input": "Comment changer l’e-mail — “vite” ✓", "tags": ["日本語"]},
    {"id": "ok-003", "input": "Hello", "tags": ["facturação"]},
    {"id": "parcel-004 ✓", "input": "Où est mon parcel ?", "tags": ["facturação"]},  # "parcel" always fails in TARGET_PY
]


class TestLegacyConsole(unittest.TestCase):
    """A redirected Windows console encodes stdout as cp1252 (or ascii in a bare CI container): a non-ASCII
    tag or case id must never turn `compare`'s exit 1 into a UnicodeEncodeError crash (exit 2), and the
    files the command was asked to write must be complete UTF-8."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        suite = cls.root / "suite"
        suite.mkdir()
        (suite / "target.py").write_text(TARGET_PY, encoding="utf-8")
        (suite / "judge_sim.py").write_text(JUDGE_PY, encoding="utf-8")
        (suite / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in NON_ASCII_CASES) + "\n", encoding="utf-8")
        cls.config = suite / "eval.yaml"
        cls.config.write_text(yaml.safe_dump({**CONFIG, "suite": "sessão-日本"}, allow_unicode=True), encoding="utf-8")
        cls.good, cls.bad = cls.root / "good", cls.root / "bad"
        assert run_cli("run", "-c", cls.config, "--out", cls.good, "--quiet")[0] == 0
        assert run_cli("run", "-c", cls.config, "--out", cls.bad, "--quiet", "--set", "target.params.variant=bad")[0] == 0
        records = [json.loads(line) for line in (cls.good / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.labels = cls.root / "labels.jsonl"
        rows = [  # the human disagrees with the judge on the Japanese-id case, so the calibration lists it
            {"case_id": r["case_id"], "label": ("fail" if r["case_id"] == "日本-002" else "pass" if r["passed"] else "fail"), "output_sha": r["output"]["sha"], "labeler": "t"}
            for r in records
        ]
        cls.labels.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def cli(self, encoding, *argv):
        env = {**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUTF8": "0"}
        proc = subprocess.run([sys.executable, "-m", "evalsprint", *map(str, argv)], cwd=REPO_ROOT, env=env, capture_output=True, timeout=120)
        return proc.returncode, proc.stdout, proc.stderr.decode(encoding, "replace")

    def assertNoCrash(self, err):
        self.assertNotIn("UnicodeEncodeError", err)
        self.assertNotIn("Traceback", err)

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_compare_keeps_exit_1_and_writes_both_files(self):
        for encoding in ("cp1252", "ascii"):
            with self.subTest(encoding=encoding):
                md, js = self.tmp / f"{encoding}.md", self.tmp / f"{encoding}.json"
                code, out, err = self.cli(encoding, "compare", self.good, self.bad, "--md", md, "--json", js)
                self.assertEqual(code, 1, err)  # the verdict, not 2 (crash)
                self.assertNoCrash(err)
                self.assertIn(b"Regression", out)
                self.assertIn("日本語", md.read_text(encoding="utf-8"))  # the files keep the real characters
                self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["verdict"], "regression")

    @unittest.skipUnless(HAS_COMPARE, "evalsprint.compare is not available")
    def test_identical_runs_exit_0(self):
        code, out, err = self.cli("cp1252", "compare", self.good, self.good)
        self.assertEqual(code, 0, err)
        self.assertNoCrash(err)

    @unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
    def test_calibrate_writes_its_files_and_exits_0(self):
        for encoding in ("cp1252", "ascii"):
            with self.subTest(encoding=encoding):
                md, js = self.tmp / f"{encoding}.md", self.tmp / f"{encoding}.json"
                code, out, err = self.cli(encoding, "calibrate", self.good, "--labels", self.labels, "--md", md, "--json", js)
                self.assertEqual(code, 0, err)
                self.assertNoCrash(err)
                self.assertTrue(out.strip())
                self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["n"], 4)
                self.assertTrue(md.read_text(encoding="utf-8").strip())

    def test_describe_replaces_what_the_console_cannot_show(self):
        for encoding in ("cp1252", "ascii"):
            with self.subTest(encoding=encoding):
                code, out, err = self.cli(encoding, "describe", "-c", self.config)
                self.assertEqual(code, 0, err)
                self.assertNoCrash(err)
                self.assertIn(b"Config and test set are valid", out)
                self.assertIn(b"???", out)  # the Japanese tag became question marks instead of a crash

    def test_run_prints_progress_for_non_ascii_case_ids(self):
        out_dir = self.tmp / "run"
        code, out, err = self.cli("cp1252", "run", "-c", self.config, "--out", out_dir)
        self.assertEqual(code, 0, err)
        self.assertNoCrash(err)
        self.assertTrue((out_dir / "report.html").is_file())
        self.assertIn("parcel-004", err)  # progress line: FAIL parcel-004 ?


class TestStreams(unittest.TestCase):
    def test_configure_streams_makes_a_strict_console_lenient(self):
        from evalsprint.cli import _configure_streams

        with self.assertRaises(UnicodeEncodeError):  # the premise: a strict cp1252 stream does crash on this text
            io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict").write("日本語")
        strict = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="\n")  # newline="\n": the default writes \r\n on Windows, and this test compares bytes
        with mock.patch.object(sys, "stdout", strict), mock.patch.object(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="\n")):
            _configure_streams()
            print("olá 日本語 ✓ — “x”")
            print("olá 日本語", file=sys.stderr)
            sys.stdout.flush()
            self.assertEqual(sys.stdout.buffer.getvalue().decode("cp1252"), "olá ??? ? — “x”\n")

    def test_configure_streams_ignores_streams_that_cannot_be_reconfigured(self):
        from evalsprint.cli import _configure_streams

        with mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(sys, "stderr", io.StringIO()):
            _configure_streams()  # StringIO has no reconfigure: nothing to do, nothing raised

    def test_a_closed_pipe_does_not_change_the_exit_code(self):
        class ClosedPipe(io.StringIO):
            def write(self, text):
                raise BrokenPipeError("consumer closed the pipe")  # `evalsprint compare ... | head -1`

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "target.py").write_text(TARGET_PY, encoding="utf-8")
            (tmp / "judge_sim.py").write_text(JUDGE_PY, encoding="utf-8")
            (tmp / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8")
            (tmp / "eval.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
            self.assertEqual(run_cli("run", "-c", tmp / "eval.yaml", "--out", tmp / "good", "--quiet")[0], 0)
            self.assertEqual(run_cli("run", "-c", tmp / "eval.yaml", "--out", tmp / "bad", "--quiet", "--set", "target.params.variant=bad")[0], 0)
            with mock.patch.object(sys, "stdout", ClosedPipe()):
                code = main(["compare", str(tmp / "good"), str(tmp / "bad"), "--md", str(tmp / "c.md")])
            self.assertEqual(code, 1 if HAS_COMPARE else 2)
            if HAS_COMPARE:
                self.assertIn("Regression", (tmp / "c.md").read_text(encoding="utf-8"))  # the file was written before printing

    def test_say_stops_printing_after_the_first_broken_pipe(self):
        from evalsprint import cli

        calls = []

        class ClosedPipe(io.StringIO):
            def write(self, text):
                calls.append(text)
                raise BrokenPipeError

        try:
            with mock.patch.object(sys, "stdout", ClosedPipe()):
                cli._say("one")
                cli._say("two")
            self.assertEqual(len(calls), 1)  # the second line never touches the dead pipe
        finally:
            cli._stdout_broken = False


class TestHistory(CliTestCase):
    """`evalsprint history RUNS_DIR [-o FILE] [--suite NAME]` hands every finished run to `report.write_history(path, runs)`."""

    def setUp(self):
        super().setUp()
        from evalsprint import report

        self.report = report
        self.runs = self.tmp / "runs"
        shutil_copy_run(self.good_run, self.runs / "20260101-000000-good")
        shutil_copy_run(self.bad_run, self.runs / "20260201-000000-bad")
        self.seen = []

        def fake_write_history(path, runs):
            self.seen.append((Path(path), list(runs)))
            Path(path).write_text("<html>history</html>", encoding="utf-8")
            return Path(path)

        patch = mock.patch.object(report, "write_history", fake_write_history, create=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_writes_runs_dir_history_html_by_default(self):
        code, out, err = run_cli("history", self.runs)
        self.assertEqual((code, err), (0, ""))
        path, runs = self.seen[0]
        self.assertEqual(path, self.runs / "history.html")
        self.assertEqual([r.summary["label"] for r in runs], ["good", "bad"])  # oldest first, whatever the directory names sort like
        self.assertIn("History of 2 run(s)", out)
        self.assertIn(str(self.runs / "history.html"), out)

    def test_runs_are_ordered_by_created_at(self):
        for name, stamp in (("20260201-000000-bad", "2026-01-01T00:00:00+00:00"), ("20260101-000000-good", "2026-03-01T00:00:00+00:00")):
            summary = json.loads((self.runs / name / "summary.json").read_text(encoding="utf-8"))
            summary["created_at"] = stamp
            (self.runs / name / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        run_cli("history", self.runs)
        self.assertEqual([r.summary["label"] for r in self.seen[0][1]], ["bad", "good"])

    def test_output_option(self):
        target = self.tmp / "page" / "trend.html"
        target.parent.mkdir()
        code, _, err = run_cli("history", self.runs, "-o", target)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.seen[0][0], target)
        self.assertTrue(target.is_file())

    def test_suite_filter(self):
        summary = json.loads((self.runs / "20260201-000000-bad" / "summary.json").read_text(encoding="utf-8"))
        summary["suite"] = "another-feature"
        (self.runs / "20260201-000000-bad" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        code, _, _ = run_cli("history", self.runs, "--suite", "cli-test")
        self.assertEqual(code, 0)
        self.assertEqual([r.summary["label"] for r in self.seen[0][1]], ["good"])
        code, _, err = run_cli("history", self.runs, "--suite", "nothing-like-it")
        self.assertEqual(code, 2)
        self.assertIn("no finished run of suite 'nothing-like-it'", err)

    def test_unfinished_runs_and_other_folders_are_skipped(self):
        half = self.runs / "20260301-000000-half"
        half.mkdir()
        (half / "results.partial.jsonl").write_text("{}\n", encoding="utf-8")
        (half / "run_meta.json").write_text("{}", encoding="utf-8")
        (self.runs / "scratch").mkdir()
        (self.runs / "notes.txt").write_text("x", encoding="utf-8")
        torn = self.runs / "20260302-000000-torn"
        shutil_copy_run(self.good_run, torn)
        (torn / "results.partial.jsonl").write_text("{}\n", encoding="utf-8")  # a newer run started in this directory
        later = (torn / "summary.json").stat().st_mtime + 60  # both files were written within one clock tick (about 16 ms on Windows): say which is newer
        os.utime(torn / "results.partial.jsonl", (later, later))
        code, _, err = run_cli("history", self.runs)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.seen[0][1]), 2)
        self.assertIn("skipped run 20260302-000000-torn", err)
        self.assertNotIn(str(self.runs / "scratch"), err)  # the stray directory is not reported; asserting on a bare word would depend on TMPDIR's own name
        self.assertNotIn("skipped run scratch", err)
        self.assertNotIn("20260301-000000-half", err)  # only a partial file and a meta file: not a finished run, nothing to say

    def test_no_runs_is_a_usage_error(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        code, _, err = run_cli("history", empty)
        self.assertEqual(code, 2)
        self.assertIn("no finished run", err)
        self.assertEqual(self.seen, [])

    def test_a_missing_directory_is_a_usage_error(self):
        code, _, err = run_cli("history", self.tmp / "nope")
        self.assertEqual(code, 2)
        self.assertIn("is not a directory", err)

    def test_history_is_listed_in_help(self):
        self.assertIn("history", run_cli("--help")[1])
        code, out, _ = run_cli("history", "--help")
        self.assertEqual(code, 0)
        for expected in ("RUNS_DIR", "--suite", "-o"):
            self.assertIn(expected, out)

    def test_a_kit_without_write_history_says_so(self):
        with mock.patch.object(self.report, "write_history", None, create=True):
            code, _, err = run_cli("history", self.runs)
        self.assertEqual(code, 2)
        self.assertIn("report.write_history", err)


@unittest.skipUnless(importlib.util.find_spec("evalsprint.report") is not None and hasattr(__import__("evalsprint.report", fromlist=["x"]), "write_history"), "report.write_history is not available yet")
class TestHistoryWithTheRealReport(CliTestCase):
    def test_a_real_page_is_written(self):
        runs = self.tmp / "runs"
        shutil_copy_run(self.good_run, runs / "a")
        shutil_copy_run(self.bad_run, runs / "b")
        code, out, err = run_cli("history", runs)
        self.assertEqual((code, err), (0, ""))
        html = (runs / "history.html").read_text(encoding="utf-8")
        self.assertIn("<html", html.lower())
        self.assertIn("good", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))  # no external requests


@unittest.skipUnless(HAS_CALIBRATION, "evalsprint.calibration is not available")
class TestCalibrateFewshot(CliTestCase):
    """`calibrate --export-fewshot PATH [--fewshot-max N] [--holdout-frac F]` (+ `--exclude-holdout FILE`)."""

    def labels(self, flip=("billing-001", "shipping-001")):
        """Human labels equal to the judge's verdicts, except for `flip` (those are the disagreements)."""
        records = [json.loads(line) for line in (self.good_run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = []
        for r in records:
            verdict = "pass" if r["passed"] else "fail"
            if r["case_id"] in flip:
                verdict = "fail" if verdict == "pass" else "pass"
            rows.append({"case_id": r["case_id"], "label": verdict, "output_sha": r["output"]["sha"], "labeler": "test", "note": "the human explains" if r["case_id"] in flip else ""})
        path = self.tmp / "labels.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_export_writes_examples_and_a_holdout_list(self):
        import yaml as yaml_module

        target = self.tmp / "out" / "fewshot.yaml"
        code, out, err = run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", target, "--holdout-frac", "0.5")
        self.assertEqual((code, err), (0, ""))
        data = yaml_module.safe_load(target.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"examples", "holdout_case_ids"})
        self.assertEqual(len(data["holdout_case_ids"]), 3)  # half of the 6 labeled cases
        for example in data["examples"]:
            self.assertIn(example["case_id"], data["holdout_case_ids"])  # the examples come from the held-out pool
            self.assertIn(example["verdict"], ("pass", "fail"))
        self.assertIn("Few-shot examples written to", out)
        self.assertIn("--exclude-holdout", out)

    def test_the_arguments_reach_export_fewshot(self):
        from evalsprint import calibration

        real = calibration.export_fewshot
        calls = []

        def spy(cal, run, out_path, **kw):
            calls.append((cal["run_id"] == run.run_id, str(out_path), kw))
            return real(cal, run, out_path, **kw)

        with mock.patch.object(calibration, "export_fewshot", spy):
            run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", self.tmp / "a.yaml")
            run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", self.tmp / "b.yaml", "--fewshot-max", "2", "--holdout-frac", "0.4")
        self.assertEqual(calls[0], (True, str(self.tmp / "a.yaml"), {"max_examples": 6, "holdout_frac": 0.3, "seed": 0}))
        self.assertEqual(calls[1], (True, str(self.tmp / "b.yaml"), {"max_examples": 2, "holdout_frac": 0.4, "seed": 0}))

    def test_the_report_of_calibrate_is_still_written_and_printed(self):
        code, out, _ = run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", self.tmp / "f.yaml", "--md", self.tmp / "cal.md", "--json", self.tmp / "cal.json")
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "cal.md").read_text(encoding="utf-8").strip())
        self.assertEqual(json.loads((self.tmp / "cal.json").read_text(encoding="utf-8"))["n"], 6)
        self.assertIn("Judge calibration", out)

    def test_without_the_flag_nothing_is_exported(self):
        code, out, _ = run_cli("calibrate", self.good_run, "--labels", self.labels())
        self.assertEqual(code, 0)
        self.assertNotIn("Few-shot", out)

    def test_the_companion_flags_need_export_fewshot(self):
        for extra in (["--fewshot-max", "3"], ["--holdout-frac", "0.5"]):
            code, _, err = run_cli("calibrate", self.good_run, "--labels", self.labels(), *extra)
            self.assertEqual(code, 2, extra)
            self.assertIn("only apply together with --export-fewshot", err)

    def test_invalid_values_are_usage_errors_before_anything_is_written(self):
        target = self.tmp / "f.yaml"
        for extra, message in ((["--holdout-frac", "0"], "strictly between 0 and 1"), (["--holdout-frac", "1"], "strictly between 0 and 1"), (["--holdout-frac", "1.5"], "strictly between 0 and 1"), (["--fewshot-max", "-1"], "zero or a positive")):
            code, _, err = run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", target, *extra)
            self.assertEqual(code, 2, extra)
            self.assertIn(message, err)
        self.assertFalse(target.exists())

    def test_zero_examples_is_allowed(self):
        import yaml as yaml_module

        code, _, err = run_cli("calibrate", self.good_run, "--labels", self.labels(), "--export-fewshot", self.tmp / "z.yaml", "--fewshot-max", "0")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(yaml_module.safe_load((self.tmp / "z.yaml").read_text(encoding="utf-8"))["examples"], [])

    def test_exclude_holdout_leaves_those_cases_out_of_the_next_calibration(self):
        import yaml as yaml_module

        labels = self.labels()
        fewshot = self.tmp / "f.yaml"
        run_cli("calibrate", self.good_run, "--labels", labels, "--export-fewshot", fewshot, "--holdout-frac", "0.5")
        held = yaml_module.safe_load(fewshot.read_text(encoding="utf-8"))["holdout_case_ids"]
        code, _, err = run_cli("calibrate", self.good_run, "--labels", labels, "--exclude-holdout", fewshot, "--json", self.tmp / "next.json")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads((self.tmp / "next.json").read_text(encoding="utf-8"))["n"], 6 - len(held))

    def test_a_missing_holdout_file_is_a_usage_error(self):
        code, _, err = run_cli("calibrate", self.good_run, "--labels", self.labels(), "--exclude-holdout", self.tmp / "nope.yaml")
        self.assertEqual(code, 2)
        self.assertIn("nope.yaml", err)


class TestEntryPoint(unittest.TestCase):
    def test_python_dash_m_evalsprint(self):
        proc = subprocess.run([sys.executable, "-m", "evalsprint", "--version"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("evalsprint", proc.stdout)

    def test_exit_code_2_for_usage_errors(self):
        proc = subprocess.run([sys.executable, "-m", "evalsprint", "run"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
