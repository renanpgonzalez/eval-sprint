"""CLI behaviour of cycle 4 (C16, C18, C20): run health per ATTEMPT, incremental relabelling, several `--labels` files and the warning
for a calibration made for other judges.

Builds on the suite of tests/test_cli.py (a mock target, a mock judge responder, six cases; the two "parcel" cases always fail).
"""

import csv
import json
import unittest
from unittest import mock

from evalsprint.cli import _health
from tests import test_cli
from tests.test_cli import as_a_real_run, run_cli, shutil_copy_run, without_placeholder_note

# Crashes on attempts 0 and 1 of every case when the case runs 5 times (40% of the attempts), whatever the retries: the crash is
# a function of the attempt index, so the in-run retry hits it again.
ATTEMPT_FLAKY_TARGET_PY = '''
def answer(input, params):
    if params.get("_attempt") in (0, 1):
        raise ConnectionError("upstream timed out")
    text = str(input)
    reply = "BAD: I cannot help with that." if "parcel" in text else "Here is the answer to: " + text[:40]
    return {"text": reply, "model": "claude-haiku-4-5", "usage": {"input_tokens": 120, "output_tokens": 30}, "latency_s": 0.2}
'''

ATTEMPT_FLAKY = ["--set", "target.callable=attempt_flaky.py:answer"]


class SuiteWithExtras(test_cli.CliTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.suite / "attempt_flaky.py").write_text(ATTEMPT_FLAKY_TARGET_PY, encoding="utf-8")


class AttemptHealthTests(SuiteWithExtras):
    """C16: a repeated case whose majority stands hides crashed attempts: `run` exits 3 and `baseline` refuses when too many crash."""

    def flaky_run(self, *extra, name="flaky"):
        out_dir = self.tmp / name
        code, out, err = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", "--repeat", "5", *ATTEMPT_FLAKY, *extra)
        return code, out, err, out_dir

    def summary(self, run_dir):
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    def test_forty_percent_of_the_attempts_crashing_makes_the_run_unhealthy(self):
        code, out, err, run_dir = self.flaky_run()
        self.assertEqual(code, 3, err)
        s = self.summary(run_dir)
        self.assertEqual((s["n_attempts"], s["n_attempt_errors"], s["n_errors"]), (30, 12, 0))  # no CASE errored: the majority of each stands
        self.assertAlmostEqual(s["attempt_error_rate"], 0.4)
        self.assertEqual(s["error_rate"], 0.0)
        self.assertIn("error: unhealthy run:", err)
        self.assertIn("12 of 30 attempts errored (40.0% of the attempts, limit 5.0%)", err)
        self.assertIn("which the majority vote of the repeated cases hides: a target or judge that crashes at random", err)
        self.assertIn("exit 3", err)
        for name in ("results.jsonl", "summary.json", "report.html"):
            self.assertTrue((run_dir / name).is_file(), name)  # a paid run is never lost

    def test_the_run_output_shows_the_attempt_errors(self):
        _, out, _, _ = self.flaky_run()
        self.assertRegex(out, r"Attempt errors:\s+12 of 30 attempts \(40\.0%; 12 target, 0 judge\)")
        self.assertRegex(out, r"Errors:\s+0 \(0\.0%")

    def test_baseline_refuses_such_a_run_and_force_copies_it(self):
        _, _, _, run_dir = self.flaky_run()
        as_a_real_run(run_dir)  # the mock judge makes every run of this file a placeholder: isolate the attempt errors
        dest = self.tmp / "baseline"
        code, out, err = run_cli("baseline", run_dir, "--to", dest)
        self.assertEqual(code, 3)
        self.assertFalse(dest.exists())
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("refusing to use run", err)
        self.assertIn("12 of 30 attempts errored", err)
        self.assertIn("--force", err)
        code, out, err = run_cli("baseline", run_dir, "--to", dest, "--force")
        self.assertEqual(code, 0)
        self.assertIn("warning: --force: this run is unhealthy", err)
        self.assertIn("12 of 30 attempts errored", err)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["results.jsonl", "summary.json"])

    def test_a_raised_limit_accepts_it_and_only_the_hidden_crash_warning_remains(self):
        code, out, err, _ = self.flaky_run("--set", "gate.max_error_rate=0.5")
        self.assertEqual(code, 0, err)
        err = without_placeholder_note(err)
        self.assertIn("warning: 12 attempt(s) of repeated cases errored but did not change their case's majority verdict", err)
        self.assertNotIn("unhealthy", err)

    def test_a_null_limit_turns_the_check_off(self):
        code, _, err, _ = self.flaky_run("--set", "gate.max_error_rate=null")
        self.assertEqual(code, 0, err)
        self.assertNotIn("unhealthy", err)

    def test_the_same_crashes_without_repeat_count_once_per_case(self):
        out_dir = self.tmp / "once"
        code, _, err = run_cli("run", "-c", self.config, "--out", out_dir, "--quiet", *ATTEMPT_FLAKY)
        self.assertEqual(code, 0, err)  # a single attempt has index None: the target answers
        s = self.summary(out_dir)
        self.assertEqual((s["n_attempt_errors"], s["n_attempts"]), (0, 6))

    def test_one_hidden_crash_in_a_large_run_stays_under_the_limit(self):
        # 1 of 30 attempts is 3.3%: below the 5% default, so the run is healthy and the warning says why it still deserves a look
        s = {"n_cases": 6, "n_errors": 0, "error_rate": 0.0, "n_attempts": 30, "n_attempt_errors": 1, "attempt_error_rate": 1 / 30, "n_scored": 6}
        self.assertEqual(_health(s, {"max_error_rate": 0.05})[0], False)


class HealthUnitTests(unittest.TestCase):
    BASE = {"n_cases": 10, "n_errors": 0, "error_rate": 0.0, "n_scored": 10, "n_judge_errors": 0, "n_target_errors": 0}

    def test_cases_only(self):
        self.assertEqual(_health({**self.BASE, "n_errors": 1, "error_rate": 0.1}, {"max_error_rate": 0.05})[0], True)
        self.assertEqual(_health(self.BASE, {"max_error_rate": 0.05})[0], False)

    def test_attempts_only(self):
        s = {**self.BASE, "n_attempts": 30, "n_attempt_errors": 6, "attempt_error_rate": 0.2}
        unhealthy, why = _health(s, {"max_error_rate": 0.05})
        self.assertTrue(unhealthy)
        self.assertIn("6 of 30 attempts errored (20.0% of the attempts, limit 5.0%)", why)
        self.assertIn("hides", why)

    def test_both_do_not_repeat_the_same_count_when_every_case_ran_once(self):
        s = {**self.BASE, "n_errors": 2, "error_rate": 0.2, "n_attempts": 10, "n_attempt_errors": 2, "attempt_error_rate": 0.2}
        unhealthy, why = _health(s, {"max_error_rate": 0.05})
        self.assertTrue(unhealthy)
        self.assertNotIn("attempts errored", why)  # one attempt per case: the case count already says it
        self.assertIn("2 of 10 cases errored", why)

    def test_a_case_problem_does_not_hide_behind_the_attempt_one(self):
        s = {**self.BASE, "n_errors": 3, "error_rate": 0.3, "n_attempts": 30, "n_attempt_errors": 6, "attempt_error_rate": 0.2}
        unhealthy, why = _health(s, {"max_error_rate": 0.05})
        self.assertTrue(unhealthy)
        self.assertIn("3 of 10 cases errored", why)
        self.assertIn("6 of 30 attempts errored", why)
        self.assertNotIn("which the majority vote", why)  # the cases are already flagged: no need to explain what hides

    def test_the_limit_is_inclusive_and_none_disables(self):
        s = {**self.BASE, "n_attempts": 20, "n_attempt_errors": 1, "attempt_error_rate": 0.05}
        self.assertFalse(_health(s, {"max_error_rate": 0.05})[0])  # at the limit is fine: only above it is not
        s["attempt_error_rate"] = 0.06
        self.assertTrue(_health(s, {"max_error_rate": 0.05})[0])
        self.assertFalse(_health(s, {"max_error_rate": None})[0])
        self.assertFalse(_health(s, {"max_error_rate": 0.5})[0])

    def test_a_summary_written_before_attempts_were_counted(self):
        old = {**self.BASE}  # no n_attempts, no attempt_error_rate
        self.assertFalse(_health(old, {"max_error_rate": 0.05})[0])
        self.assertTrue(_health({**old, "n_errors": 5, "error_rate": 0.5}, {"max_error_rate": 0.05})[0])

    def test_the_rate_is_derived_from_the_counts_when_it_is_missing(self):
        s = {**self.BASE, "n_attempts": 10, "n_attempt_errors": 3}
        self.assertTrue(_health(s, {"max_error_rate": 0.05})[0])

    def test_nothing_scored_is_unhealthy_whatever_the_limit(self):
        s = {**self.BASE, "n_scored": 0, "n_errors": 10, "error_rate": 1.0}
        unhealthy, why = _health(s, {"max_error_rate": None})
        self.assertTrue(unhealthy)
        self.assertIn("no case could be scored", why)

    def test_the_default_limit_is_the_runs_own_gate_when_none_is_passed(self):
        s = {**self.BASE, "n_attempts": 10, "n_attempt_errors": 4, "attempt_error_rate": 0.4, "gate": {"max_error_rate": 0.5}}
        self.assertFalse(_health(s)[0])
        s["gate"] = {"max_error_rate": 0.1}
        self.assertTrue(_health(s)[0])


class LabelsFixture(SuiteWithExtras):
    def records(self):
        return [json.loads(line) for line in (self.good_run / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    def write_labels(self, name, case_ids=None, other_output=False):
        rows = []
        for r in self.records():
            if case_ids is not None and r["case_id"] not in case_ids:
                continue
            rows.append({"case_id": r["case_id"], "label": "pass" if r["passed"] else "fail", "output_sha": ("0" * 12 if other_output else r["output"]["sha"]), "labeler": "t"})
        path = self.tmp / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path


@unittest.skipUnless(test_cli.HAS_LABELING, "evalsprint.labeling is not available")
class PreviousLabelsTests(LabelsFixture):
    IDS = ["billing-001", "billing-002", "billing-003", "shipping-001"]

    def sheet_ids(self, path):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return sorted(row["case_id"] for row in csv.DictReader(fh))

    def test_only_the_cases_without_a_label_for_their_current_output_are_exported(self):
        labels = self.write_labels("prev.jsonl", self.IDS)
        code, out, err = run_cli("label", self.good_run, "--csv", "--previous-labels", labels, "-o", self.tmp / "sheet.csv")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.sheet_ids(self.tmp / "sheet.csv"), ["misc-001", "shipping-002"])
        self.assertIn("Previous labels: 4 of 6 labelable case(s) already have a label for their current output, 2 do not; 2 to label now.", out)
        self.assertIn(f"--labels {labels} --labels <the new labels file>", out)

    def test_an_html_page_is_written_too(self):
        labels = self.write_labels("prev.jsonl", self.IDS)
        code, out, _ = run_cli("label", self.good_run, "--previous-labels", labels, "-o", self.tmp / "page.html")
        self.assertEqual(code, 0)
        page = (self.tmp / "page.html").read_text(encoding="utf-8")
        self.assertIn("misc-001", page)
        self.assertNotIn("billing-002", page)

    def test_nothing_to_label_is_not_an_error_and_writes_no_page(self):
        labels = self.write_labels("all.jsonl")
        out_path = self.tmp / "none.html"
        code, out, err = run_cli("label", self.good_run, "--previous-labels", labels, "-o", out_path)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("Nothing to label", out)
        self.assertFalse(out_path.exists())
        self.assertIn(f"Next: evalsprint calibrate {self.good_run} --labels {labels}", out)

    def test_a_changed_output_is_exported_again(self):
        labels = self.write_labels("stale.jsonl", other_output=True)  # the humans saw some other output: the shas do not match
        code, out, _ = run_cli("label", self.good_run, "--csv", "--previous-labels", labels, "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.sheet_ids(self.tmp / "sheet.csv")), 6)
        self.assertIn("0 of 6 labelable case(s) already have a label", out)

    def test_a_label_without_a_sha_cannot_be_checked_so_it_does_not_count(self):
        rows = [{"case_id": r["case_id"], "label": "pass"} for r in self.records()]
        path = self.tmp / "nosha.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        code, out, _ = run_cli("label", self.good_run, "--csv", "--previous-labels", path, "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.sheet_ids(self.tmp / "sheet.csv")), 6)

    def test_several_previous_files_are_read_as_one_list(self):
        first = self.write_labels("a.jsonl", self.IDS[:2])
        second = self.write_labels("b.jsonl", self.IDS[2:])
        code, out, _ = run_cli("label", self.good_run, "--csv", "--previous-labels", first, "--previous-labels", second, "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertEqual(self.sheet_ids(self.tmp / "sheet.csv"), ["misc-001", "shipping-002"])
        self.assertIn(f"--labels {first} --labels {second} --labels <the new labels file>", out)

    def test_the_sample_cap_applies_to_what_is_left(self):
        labels = self.write_labels("prev.jsonl", self.IDS[:1])
        code, out, _ = run_cli("label", self.good_run, "--csv", "--previous-labels", labels, "--sample", "2", "--seed", "1", "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.sheet_ids(self.tmp / "sheet.csv")), 2)
        self.assertNotIn("billing-001", self.sheet_ids(self.tmp / "sheet.csv"))
        self.assertIn("5 do not; 2 to label now", out)

    def test_without_the_flag_nothing_changes(self):
        code, out, _ = run_cli("label", self.good_run, "--csv", "-o", self.tmp / "sheet.csv")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.sheet_ids(self.tmp / "sheet.csv")), 6)
        self.assertNotIn("Previous labels", out)
        self.assertIn("--labels <labels file>", out)

    def test_a_missing_previous_file_is_a_usage_error(self):
        code, _, err = run_cli("label", self.good_run, "--previous-labels", self.tmp / "nope.jsonl")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error:"), err)

    def test_the_flag_is_in_the_help(self):
        code, out, _ = run_cli("label", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--previous-labels", out)


@unittest.skipUnless(test_cli.HAS_CALIBRATION, "evalsprint.calibration is not available")
class SeveralLabelsFilesTests(LabelsFixture):
    def test_calibrate_reads_repeated_labels_as_one_list(self):
        ids = [r["case_id"] for r in self.records()]
        a, b = self.write_labels("a.jsonl", ids[:3]), self.write_labels("b.jsonl", ids[3:])
        code, _, err = run_cli("calibrate", self.good_run, "--labels", a, "--labels", b, "--json", self.tmp / "cal.json")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads((self.tmp / "cal.json").read_text(encoding="utf-8"))["n"], 6)
        code, _, _ = run_cli("calibrate", self.good_run, "--labels", a, "--json", self.tmp / "half.json")
        self.assertEqual(json.loads((self.tmp / "half.json").read_text(encoding="utf-8"))["n"], 3)

    def test_both_files_reach_load_labels_in_the_order_given(self):
        first, second = self.write_labels("a.jsonl", ["billing-001"]), self.write_labels("b.jsonl", ["billing-002"])
        with mock.patch("evalsprint.calibration.load_labels", wraps=__import__("evalsprint.calibration", fromlist=["x"]).load_labels) as spy:
            code, *_ = run_cli("calibrate", self.good_run, "--labels", first, "--labels", second)
        self.assertEqual(code, 0)
        self.assertEqual([str(p) for p in spy.call_args.args[0]], [str(first), str(second)])

    def test_report_takes_several_files_too(self):
        ids = [r["case_id"] for r in self.records()]
        a, b = self.write_labels("a.jsonl", ids[:3]), self.write_labels("b.jsonl", ids[3:])
        code, out, err = run_cli("report", self.good_run, "--labels", a, "--labels", b, "-o", self.tmp / "r.html")
        self.assertEqual((code, err), (0, ""))
        self.assertTrue((self.tmp / "r.html").is_file())

    def test_a_single_labels_file_keeps_working_as_a_string(self):
        labels = self.write_labels("one.jsonl")
        with mock.patch("evalsprint.calibration.load_labels", wraps=__import__("evalsprint.calibration", fromlist=["x"]).load_labels) as spy:
            code, *_ = run_cli("calibrate", self.good_run, "--labels", labels)
        self.assertEqual(code, 0)
        self.assertEqual([str(p) for p in spy.call_args.args[0]], [str(labels)])

    def test_the_calibration_records_the_judges_sha_of_the_run(self):
        labels = self.write_labels("one.jsonl")
        run_cli("calibrate", self.good_run, "--labels", labels, "--json", self.tmp / "cal.json")
        cal = json.loads((self.tmp / "cal.json").read_text(encoding="utf-8"))
        summary = json.loads((self.good_run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(cal["judges_sha"], summary["judges_sha"])

    def test_the_flag_is_repeatable_in_the_help(self):
        for command in ("calibrate", "report"):
            code, out, _ = run_cli(command, "--help")
            self.assertEqual(code, 0)
            self.assertIn("--labels", out)


@unittest.skipUnless(test_cli.HAS_CALIBRATION and test_cli.HAS_COMPARE, "compare and calibration are needed")
class StaleCalibrationTests(LabelsFixture):
    TEXT = "calibration was made for a different judge configuration"

    def baseline_with(self, calibration):
        baseline = self.tmp / "baseline"
        shutil_copy_run(self.good_run, baseline)
        (baseline / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
        return baseline

    def sha(self):
        return json.loads((self.good_run / "summary.json").read_text(encoding="utf-8"))["judges_sha"]

    def test_compare_warns_when_the_calibration_was_made_for_other_judges(self):
        baseline = self.baseline_with({"n": 6, "kappa": 0.9, "judges_sha": "0" * 12})
        code, out, err = run_cli("compare", baseline, self.good_run)
        self.assertIn(self.TEXT, out)
        self.assertIn("0" * 12, out)

    def test_compare_is_quiet_when_the_shas_match(self):
        baseline = self.baseline_with({"n": 6, "kappa": 0.9, "judges_sha": self.sha()})
        _, out, _ = run_cli("compare", baseline, self.good_run)
        self.assertNotIn(self.TEXT, out)

    def test_a_calibration_from_before_c20_has_no_sha_and_says_nothing(self):
        baseline = self.baseline_with({"n": 6, "kappa": 0.9})
        _, out, _ = run_cli("compare", baseline, self.good_run)
        self.assertNotIn(self.TEXT, out)

    def test_the_warning_reaches_the_json_and_the_markdown_file(self):
        baseline = self.baseline_with({"n": 6, "kappa": 0.9, "judges_sha": "f" * 12})
        run_cli("compare", baseline, self.good_run, "--md", self.tmp / "c.md", "--json", self.tmp / "c.json")
        self.assertIn(self.TEXT, (self.tmp / "c.md").read_text(encoding="utf-8"))
        self.assertTrue(any(self.TEXT in w for w in json.loads((self.tmp / "c.json").read_text(encoding="utf-8"))["warnings"]))

    def test_the_warning_is_added_once(self):
        baseline = self.baseline_with({"n": 6, "kappa": 0.9, "judges_sha": "f" * 12})
        _, out, _ = run_cli("compare", baseline, self.good_run)
        self.assertEqual(out.count(self.TEXT), 1)

    def test_baseline_warns_when_it_copies_a_stale_calibration(self):
        source = self.tmp / "run"
        shutil_copy_run(self.good_run, source)
        (source / "calibration.json").write_text(json.dumps({"n": 6, "judges_sha": "a" * 12}), encoding="utf-8")
        dest = self.tmp / "dest"
        code, out, err = run_cli("baseline", source, "--to", dest)
        self.assertEqual(code, 0)
        self.assertIn(f"warning: {source / 'calibration.json'}: ", err)
        self.assertIn(self.TEXT, err)
        self.assertTrue((dest / "calibration.json").is_file())  # still copied: the warning says what to do about it

    def test_baseline_is_quiet_for_a_matching_calibration(self):
        source = self.tmp / "run"
        shutil_copy_run(self.good_run, source)
        (source / "calibration.json").write_text(json.dumps({"n": 6, "judges_sha": self.sha()}), encoding="utf-8")
        code, _, err = run_cli("baseline", source, "--to", self.tmp / "dest")
        self.assertEqual((code, err), (0, ""))

    def test_baseline_copes_with_an_unreadable_calibration_file(self):
        source = self.tmp / "run"
        shutil_copy_run(self.good_run, source)
        (source / "calibration.json").write_text("{not json", encoding="utf-8")
        code, _, err = run_cli("baseline", source, "--to", self.tmp / "dest")
        self.assertEqual((code, err), (0, ""))  # copied as it is; `compare` reports it when it reads it


if __name__ == "__main__":
    unittest.main()
