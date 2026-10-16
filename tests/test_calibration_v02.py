"""Cycle 3, contract C14: the calibration story of v0.2.

* ``tnr_reliable`` is monotone in judge quality (a closed-form interval, no resampling noise), covers the run's true
  TNR when it says "reliable", and lets a perfect judge be certified with enough human fails;
* the stratum token of the labeling files is an HMAC keyed with the run's ``.label_secret``; legacy tokens still decode;
* warnings: a run_id that is not this run's, labels of one kind only (the advice is to label the other kind);
* ``export_fewshot`` writes the rubric examples and the cases to keep out of the next calibration.

The measured coverage / power tables behind the constants live in
orchestration/2026-09-29_kit-v0.2/cycle3/eval-scientist-judge/ (tnr_tables_final.txt); the tests here are the fast,
seeded, offline guards of the same properties.
"""

import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from evalsprint import calibration
from evalsprint.judges.llm_judge import Rubric, _format_examples
from evalsprint.runner import RunResult, summarize
from tests.test_calibration import FAST, confusion_scenario, label, make_record, make_run, with_unlabeled

QUICK = {"iters": 60}  # the flags under test do not depend on the bootstrap: keep the end-to-end sweeps fast


def saved_run(records, run_dir, run_id="run-keyed"):
    """A run that lives in a folder (the run's `.label_secret` goes there)."""
    return RunResult(run_dir=Path(run_dir), summary={"run_id": run_id, **summarize(records)}, results=records)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)


# ------------------------------------------------------------------------------------------ the closed-form TNR interval
def design(kind, n_pass, n_fail, lab_pass, lab_fail, fp, tn):
    """(frame, labeled) of a sample stratified on the judge's verdict ("judge") or on an overall verdict ("overall": both judge
    verdicts live in both strata). The humans failed ``fp`` cases the judge passed and ``tn`` it failed."""
    if kind == "judge":
        frame = [("judge_pass", True)] * n_pass + [("judge_fail", False)] * n_fail
        labeled = {
            "judge_pass": [(False, True)] * fp + [(True, True)] * (lab_pass - fp),
            "judge_fail": [(False, False)] * tn + [(True, False)] * (lab_fail - tn),
        }
        return frame, labeled
    fa, fb = fp - fp // 2, fp // 2
    ta, tb = tn - tn // 2, tn // 2
    frame = (
        [("judge_pass", True)] * (n_pass // 2)
        + [("judge_pass", False)] * (n_fail // 2)
        + [("judge_fail", True)] * (n_pass - n_pass // 2)
        + [("judge_fail", False)] * (n_fail - n_fail // 2)
    )
    labeled = {
        "judge_pass": [(False, True)] * fa + [(False, False)] * ta + [(True, True)] * max(0, lab_pass - fa - ta),
        "judge_fail": [(False, True)] * fb + [(False, False)] * tb + [(True, False)] * max(0, lab_fail - fb - tb),
    }
    return frame, labeled


def point_tnr(frame, labeled):
    sizes = Counter(s for s, _ in frame)
    caught = missed = 0.0
    for stratum, size in sizes.items():
        pairs = labeled.get(stratum) or []
        if not pairs:
            return None
        weight = size / len(pairs)
        caught += weight * sum(1 for h, j in pairs if not h and not j)
        missed += weight * sum(1 for h, j in pairs if not h and j)
    return None if caught + missed <= 0 else caught / (caught + missed)


def evaluate(kind, *args):
    frame, labeled = design(kind, *args)
    ci = calibration._tnr_interval(frame, labeled)
    point = point_tnr(frame, labeled)
    if ci is None or point is None:
        return None
    return point, (ci[1] - ci[0]) / 2


def reliable_flag(result, human_fails):
    return (
        result is not None
        and human_fails >= calibration.MIN_HUMAN_FAILS
        and result[0] >= calibration.TNR_MIN_RELIABLE
        and result[1] <= calibration.TNR_MAX_HALF_WIDTH
    )


def scan(kind, seed, n_designs):
    """Walk random designs along paths on which the judge only gets better (one missed bad output becomes a caught one,
    the human fails held fixed) and count the paths on which the flag goes from True to False."""
    rng = random.Random(seed)
    guard = calibration.TNR_MIN_RELIABLE
    stats = Counter()
    for _ in range(n_designs):
        total = rng.randint(60, 400)
        n_pass = max(8, int(total * rng.uniform(0.5, 0.97)))
        n_fail = max(8, total - n_pass)
        lab_pass = rng.randint(4, min(150, n_pass))
        lab_fail = rng.randint(4, min(150, n_fail))
        if lab_pass + lab_fail < calibration.MIN_HUMAN_FAILS:
            continue
        hf = rng.randint(calibration.MIN_HUMAN_FAILS, min(lab_pass + lab_fail, 90))
        for move in ("labeled counts fixed", "run shifts too", "census swap"):
            hi_fp = min(hf, lab_pass)
            lo_fp = max(0, hf - lab_fail) if move != "census swap" else 0
            if hi_fp < lo_fp:
                continue
            previous = None
            flag_violation = width_violation = False
            for fp in range(hi_fp, lo_fp - 1, -1):  # the judge improves along this loop
                moved = hi_fp - fp
                if move == "labeled counts fixed":
                    args = (n_pass, n_fail, lab_pass, lab_fail, fp, hf - fp)
                elif move == "run shifts too":
                    shift = min(int(round(moved * n_pass / max(1, lab_pass))), n_pass - lab_pass)
                    args = (n_pass - shift, n_fail + shift, lab_pass, lab_fail, fp, hf - fp)
                else:
                    if lab_pass - moved < 1 or lab_fail + moved < 1 or hf - fp > lab_fail + moved:
                        break
                    shift = max(moved, min(int(round(moved * n_pass / max(1, lab_pass))), n_pass - (lab_pass - moved)))
                    if n_pass - shift < lab_pass - moved or n_pass - shift < 1:
                        break
                    args = (n_pass - shift, n_fail + shift, lab_pass - moved, lab_fail + moved, fp, hf - fp)
                result = evaluate(kind, *args)
                if result is None:
                    previous = None
                    continue
                flag = reliable_flag(result, hf)
                if previous is not None:
                    flag_violation |= previous[0] and not flag
                    # a wider interval matters only near the threshold: a rise from 0.029 to 0.032 flips nothing
                    if previous[1][0] >= guard and result[0] >= guard and result[1] > max(previous[1][1], 0.10) + 1e-12:
                        width_violation = True
                previous = (flag, result)
            stats[(move, "paths")] += 1
            stats[(move, "flag")] += flag_violation
            stats[(move, "width")] += width_violation
    return stats


class TnrIntervalTests(unittest.TestCase):
    def test_it_contains_the_point_estimate_and_stays_in_the_unit_interval(self):
        rng = random.Random(11)
        for _ in range(300):
            kind = rng.choice(["judge", "overall"])
            n_pass, n_fail = rng.randint(8, 300), rng.randint(8, 100)
            lab_pass, lab_fail = rng.randint(2, n_pass), rng.randint(2, n_fail)
            hf = rng.randint(1, lab_pass + lab_fail)
            fp = rng.randint(max(0, hf - lab_fail), min(hf, lab_pass))
            frame, labeled = design(kind, n_pass, n_fail, lab_pass, lab_fail, fp, hf - fp)
            ci, point = calibration._tnr_interval(frame, labeled), point_tnr(frame, labeled)
            self.assertIsNotNone(ci)
            self.assertTrue(0.0 <= ci[0] <= point <= ci[1] <= 1.0, (kind, ci, point))

    def test_undefined_without_a_human_fail_or_with_an_unlabeled_stratum(self):
        frame, labeled = design("judge", 100, 40, 30, 30, 0, 0)
        self.assertIsNone(calibration._tnr_interval(frame, labeled))  # no human fail at all
        frame, labeled = design("judge", 100, 40, 30, 30, 2, 5)
        labeled["judge_fail"] = []
        self.assertIsNone(calibration._tnr_interval(frame, labeled))  # nothing labeled where the judge failed

    def test_it_is_a_deterministic_function_of_the_counts(self):
        frame, labeled = design("judge", 300, 60, 60, 40, 6, 30)
        first = calibration._tnr_interval(frame, labeled)
        for _ in range(3):
            self.assertEqual(calibration._tnr_interval(frame, labeled), first)
        self.assertEqual(calibration._tnr_interval(list(reversed(frame)), labeled), first)  # not even the frame order matters

    def test_no_observed_miss_is_uncertainty_not_certainty(self):
        # a perfect judge on a stratified sample: the misses hiding among the judge-passes are unobserved, so the upper
        # bound reaches 1 but the lower bound stays clearly below it, and more labels of judge-passes raise it
        lows = []
        for lab_pass in (10, 20, 40, 80):
            frame, labeled = design("judge", 400, 40, lab_pass, 40, 0, 40)
            lo, hi = calibration._tnr_interval(frame, labeled)
            self.assertEqual(hi, 1.0)
            self.assertLess(lo, 0.95)
            lows.append(lo)
        self.assertEqual(lows, sorted(lows))
        self.assertLess(lows[0], lows[-1] - 0.05)

    def test_more_caught_bad_outputs_narrow_the_interval_of_a_perfect_judge(self):
        widths = []
        for hf in range(10, 61, 5):  # the judge failed hf bad outputs and every one of them is labeled: a census of that stratum
            frame, labeled = design("judge", 400, hf, 60, hf, 0, hf)
            lo, hi = calibration._tnr_interval(frame, labeled)
            widths.append(hi - lo)
        for wider, narrower in zip(widths, widths[1:]):
            self.assertLessEqual(narrower, wider + 1e-12)
        self.assertGreater(widths[0], widths[-1] + 0.2)

    def test_a_judge_that_passes_nothing_has_no_uncertainty_to_add(self):
        frame = [("judge_fail", False)] * 30
        labeled = {"judge_fail": [(False, False)] * 12 + [(True, False)] * 18}
        self.assertEqual(calibration._tnr_interval(frame, labeled), [1.0, 1.0])


class TnrMonotoneTests(unittest.TestCase):
    """A judge with fewer misses is never less reliable: the flag never goes from True to False as one missed bad output
    becomes a caught one (the number of human fails, the sample and the run size held as the move allows)."""

    def test_flag_is_monotone_on_judge_stratified_designs(self):
        stats = scan("judge", seed=5100001, n_designs=70)
        for move in ("labeled counts fixed", "run shifts too", "census swap"):
            self.assertGreater(stats[(move, "paths")], 30, move)  # the scan is not vacuous
            self.assertEqual(stats[(move, "flag")], 0, move)
            self.assertEqual(stats[(move, "width")], 0, move)

    def test_flag_is_monotone_on_overall_stratified_designs_up_to_a_documented_residual(self):
        # Both judge verdicts live in both strata here, so a labeled case can change the weights while it changes sides:
        # about 0.1% of random paths were found to flip in the 1000-path scan of cycle 3. The guard is 1%.
        stats = scan("overall", seed=5100002, n_designs=70)
        for move in ("labeled counts fixed", "run shifts too"):
            self.assertGreater(stats[(move, "paths")], 30, move)
            self.assertLessEqual(stats[(move, "flag")], 0.01 * stats[(move, "paths")], move)

    def test_flag_is_monotone_through_calibrate_when_the_judge_improves(self):
        for human_fails, tp, extra in ((20, 70, (60, 30)), (40, 50, (0, 0)), (12, 60, (100, 50))):
            flags, tnrs = [], []
            for fp in range(human_fails, -1, -1):  # the judge catches one more of the same human fails each step
                run, labels = confusion_scenario(tp=tp, fn=2, fp=fp, tn=human_fails - fp)
                if extra != (0, 0):
                    run = with_unlabeled(run, *extra)
                cal = calibration.calibrate(run, labels, **QUICK)
                flags.append(cal["tnr_reliable"])
                tnrs.append(cal["tnr"])
            self.assertEqual(flags, sorted(flags), (human_fails, flags))  # False ... False True ... True
            self.assertEqual(tnrs, sorted(tnrs))
            self.assertFalse(flags[0])  # a judge that misses every bad output is never reliable
            self.assertTrue(flags[-1])  # one that catches them all, with this many, is

    def test_the_flag_does_not_depend_on_seed_or_bootstrap_iterations(self):
        run, labels = confusion_scenario(tp=60, fn=2, fp=3, tn=25)
        run = with_unlabeled(run, 200, 30)
        outputs = [calibration.calibrate(run, labels, seed=seed, iters=iters) for seed, iters in ((0, 60), (7, 300), (99, 1000))]
        self.assertEqual(len({(tuple(c["tnr_ci"]), c["tnr_reliable"], c["tnr_reliable_reason"]) for c in outputs}), 1)
        self.assertIn("tnr_ci_bootstrap", outputs[0])  # the plain bootstrap interval stays available, for transparency
        self.assertIsNotNone(outputs[0]["tnr_ci_bootstrap"])


def draw_run(rng, size, theta, tpr, tnr):
    """A run of ``size`` cases: (human_pass, judge_pass) per case."""
    cases = []
    for _ in range(size):
        human_pass = rng.random() < theta
        judge_pass = rng.random() < tpr if human_pass else rng.random() >= tnr
        cases.append((human_pass, judge_pass))
    return cases


def stratified_labels(rng, cases, n_labels):
    """The `evalsprint label` sample: half of the labels per judge verdict (topped up from the other one), no replacement."""
    passed = [c for c in cases if c[1]]
    failed = [c for c in cases if not c[1]]
    n_fail = min(len(failed), n_labels // 2 + n_labels % 2)
    n_pass = min(len(passed), n_labels - n_fail)
    n_fail = min(len(failed), n_labels - n_pass)
    return rng.sample(passed, n_pass), rng.sample(failed, n_fail)


class TnrCoverageTests(unittest.TestCase):
    """When the flag says "reliable", the interval contains the run's true TNR (the census value of this finite run)."""

    PROFILES = ((0.85, 0.97, 0.90, 200, 100), (0.85, 0.97, 0.80, 200, 150), (0.85, 0.99, 1.00, 200, 100), (0.70, 0.95, 0.85, 400, 100))

    def test_coverage_given_reliable_is_at_least_ninety_percent(self):
        rng = random.Random(4800001)
        reliable = covered = intervals = intervals_covered = 0
        for theta, tpr, tnr, size, n_labels in self.PROFILES:
            for _ in range(250):
                cases = draw_run(rng, size, theta, tpr, tnr)
                bad = [c for c in cases if not c[0]]
                if not bad:
                    continue
                truth = sum(1 for c in bad if not c[1]) / len(bad)
                passed, failed = stratified_labels(rng, cases, n_labels)
                frame = [("judge_pass", True)] * sum(1 for c in cases if c[1]) + [("judge_fail", False)] * sum(1 for c in cases if not c[1])
                labeled = {"judge_pass": passed, "judge_fail": failed}
                ci = calibration._tnr_interval(frame, labeled)
                if ci is None:
                    continue
                intervals += 1
                intervals_covered += ci[0] - 1e-9 <= truth <= ci[1] + 1e-9
                human_fails = sum(1 for h, _ in passed + failed if not h)
                point = point_tnr(frame, labeled)
                if reliable_flag((point, (ci[1] - ci[0]) / 2), human_fails):
                    reliable += 1
                    covered += ci[0] - 1e-9 <= truth <= ci[1] + 1e-9
        self.assertGreater(reliable, 150)  # enough replications say "reliable" for the check to mean something
        self.assertGreaterEqual(covered / reliable, 0.90)
        self.assertGreaterEqual(intervals_covered / intervals, 0.90)  # and the interval is honest whether or not it is called reliable


class PerfectJudgeTests(unittest.TestCase):
    def test_a_perfect_judge_is_certified_from_ten_human_fails_on_a_census(self):
        for human_fails in range(0, 14):
            run, labels = confusion_scenario(tp=60, fn=0, fp=0, tn=human_fails)
            cal = calibration.calibrate(run, labels, **QUICK)
            self.assertEqual(cal["tnr_reliable"], human_fails >= calibration.MIN_HUMAN_FAILS, human_fails)
            if human_fails < calibration.MIN_HUMAN_FAILS:
                self.assertNotEqual(cal["kappa_decision"], "trustworthy", human_fails)
        # with plenty of labels on both sides the verdict follows
        cal = calibration.calibrate(*confusion_scenario(tp=60, fn=0, fp=0, tn=40), **FAST)
        self.assertEqual(cal["kappa_decision"], "trustworthy")

    def test_a_perfect_judge_needs_more_human_fails_when_the_labels_are_a_sample(self):
        # 400 cases: the judge failed 60 (all of them bad, all labeled) and passed 340, of which some labels tell how many bad
        # outputs it missed. 60 human fails alone do not certify the TNR: the misses hiding among the passes must be bounded too.
        for labeled_passes, reliable, reason in ((100, True, "ok"), (12, False, "wide_ci")):
            run, labels = confusion_scenario(tp=labeled_passes, fn=0, fp=0, tn=60)
            cal = calibration.calibrate(with_unlabeled(run, 340 - labeled_passes, 0), labels, **QUICK)
            self.assertEqual(cal["tnr"], 1.0)
            self.assertEqual((cal["tnr_reliable"], cal["tnr_reliable_reason"]), (reliable, reason), labeled_passes)


# ---------------------------------------------------------------------------------------------------- stratum tokens
class TokenPrimitivesTests(TempDirCase):
    def test_hmac_token_shape_and_key_dependence(self):
        secret = bytes(range(32))
        token = calibration.stratum_token("run-1", "overall", "c1", "judge_pass", secret)
        self.assertRegex(token, r"^s-[0-9a-f]{16}$")
        self.assertEqual(token, calibration.stratum_token("run-1", "overall", "c1", "judge_pass", secret))
        others = {
            calibration.stratum_token("run-1", "overall", "c1", "judge_pass", bytes(range(1, 33))),  # another secret
            calibration.stratum_token("run-1", "overall", "c1", "judge_fail", secret),  # another stratum
            calibration.stratum_token("run-1", "overall", "c2", "judge_pass", secret),  # another case
            calibration.stratum_token("run-2", "overall", "c1", "judge_pass", secret),  # another run
            calibration.stratum_token("run-1", "quality", "c1", "judge_pass", secret),  # another design
        }
        self.assertNotIn(token, others)
        self.assertEqual(len(others), 5)

    def test_the_keyed_token_is_not_the_legacy_token_and_cannot_be_decoded_without_the_secret(self):
        secret = os.urandom(32)
        for stratum in calibration.STRATA:
            keyed = calibration.stratum_token("run-1", "overall", "c1", stratum, secret)
            legacy = calibration.stratum_token("run-1", "overall", "c1", stratum)
            self.assertRegex(legacy, r"^s-[0-9a-f]{10}$")
            self.assertNotEqual(keyed[:12], legacy)
            # someone holding the source code but not the secret recomputes the unkeyed codes: they match nothing
            self.assertEqual(calibration.stratum_status(keyed, "run-1", "overall", "c1"), (None, "unverifiable"))
            self.assertEqual(calibration.decode_stratum(keyed, "run-1", "overall", "c1"), None)
            self.assertEqual(calibration.stratum_status(keyed, "run-1", "overall", "c1", secret), (stratum, "hmac"))

    def test_status_of_each_kind_of_recorded_value(self):
        secret = os.urandom(32)
        status = lambda value, **kw: calibration.stratum_status(value, "run-1", "overall", "c1", **kw)  # noqa: E731
        self.assertEqual(status(None), (None, "none"))
        self.assertEqual(status(""), (None, "none"))
        self.assertEqual(status("judge_fail"), ("judge_fail", "plain"))
        legacy = calibration.stratum_token("run-1", "overall", "c1", "judge_pass")
        self.assertEqual(status(legacy), ("judge_pass", "legacy"))
        self.assertEqual(status(legacy, secret=secret), ("judge_pass", "legacy"))  # old labels still decode next to a secret
        self.assertEqual(status("s-0123456789"), (None, "invalid"))  # right shape, matches nothing
        self.assertEqual(status("banana"), (None, "invalid"))
        keyed = calibration.stratum_token("run-1", "overall", "c1", "judge_fail", secret)
        self.assertEqual(status(keyed, secret=secret), ("judge_fail", "hmac"))
        self.assertEqual(status(keyed, secret=os.urandom(32)), (None, "invalid"))  # the wrong secret
        self.assertEqual(calibration.stratum_status(keyed, "run-2", "overall", "c1", secret), (None, "invalid"))  # another run
        self.assertEqual(calibration.stratum_status(keyed, "run-1", "overall", "c2", secret), (None, "invalid"))  # another case

    def test_ensure_label_secret_creates_a_private_file_and_reuses_it(self):
        run_dir = self.dir / "run"
        self.assertIsNone(calibration.read_label_secret(run_dir))
        secret = calibration.ensure_label_secret(run_dir)
        path = calibration.label_secret_path(run_dir)
        self.assertEqual(path.name, ".label_secret")
        self.assertEqual(len(secret), 32)
        self.assertEqual(calibration.read_label_secret(run_dir), secret)
        self.assertEqual(calibration.ensure_label_secret(run_dir), secret)  # earlier labels stay verifiable
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotEqual(calibration.ensure_label_secret(self.dir / "other"), secret)  # one secret per run

    def test_an_unusable_secret_file_is_replaced_never_followed(self):
        run_dir = self.dir / "run"
        run_dir.mkdir()
        path = run_dir / ".label_secret"
        for junk in ("", "not hex at all\n", "abcd\n", "\xff"):  # empty, garbled, too short (2 bytes), not ASCII
            path.write_bytes(junk.encode("latin-1"))
            self.assertIsNone(calibration.read_label_secret(run_dir), junk)
            secret = calibration.ensure_label_secret(run_dir)
            self.assertEqual(calibration.read_label_secret(run_dir), secret)
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.unlink()
        if hasattr(os, "symlink") and os.name == "posix":
            victim = self.dir / "victim.txt"
            victim.write_text("keep me", encoding="utf-8")
            path.symlink_to(victim)
            secret = calibration.ensure_label_secret(run_dir)
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")  # the symlink target is left alone
            self.assertFalse(path.is_symlink())
            self.assertEqual(calibration.read_label_secret(run_dir), secret)

    def test_a_missing_run_folder_and_a_none_run_dir(self):
        self.assertIsNone(calibration.read_label_secret(None))
        self.assertIsNone(calibration.read_label_secret(self.dir / "does-not-exist"))
        with self.assertRaises(OSError):
            blocker = self.dir / "a-file"
            blocker.write_text("x", encoding="utf-8")
            calibration.ensure_label_secret(blocker / "run")  # its parent is a file: the caller falls back to legacy tokens


class KeyedCalibrationTests(TempDirCase):
    """calibrate reads the secret from ``run.run_dir``; the strata always come from the run's own verdicts."""

    def setUp(self):
        super().setUp()
        self.run_dir = self.dir / "run"
        self.run_dir.mkdir()
        records = [make_record(f"p{i:02d}", True) for i in range(60)] + [make_record(f"f{i:02d}", False) for i in range(15)]
        self.run = saved_run(records, self.run_dir)
        self.secret = calibration.ensure_label_secret(self.run_dir)
        self.ids = [f"p{i:02d}" for i in range(30)] + [f"f{i:02d}" for i in range(15)]  # 30 of 60 passes, all 15 fails

    def rows(self, secret="keyed", run_id=None, flip=False):
        rows = []
        for cid in self.ids:
            judge_pass = cid.startswith("p")
            stratum = "judge_pass" if judge_pass != flip else "judge_fail"
            row = label(cid, "pass" if judge_pass else "fail")
            row["run_id"] = run_id
            row["stratified_on"] = "overall"
            row["stratum"] = calibration.stratum_token(self.run.run_id, "overall", cid, stratum, self.secret if secret == "keyed" else None)
            rows.append(row)
        return rows

    def cal(self, rows, run=None):
        return calibration.calibrate(run or self.run, rows, **QUICK)

    def test_keyed_tokens_verify_with_the_runs_secret(self):
        cal = self.cal(self.rows())
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertEqual(cal["strata"]["n_unverifiable_tokens"], 0)
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 2.0, "judge_fail": 1.0})
        self.assertFalse(any(".label_secret" in w or "does not match" in w for w in cal["warnings"]))

    def test_without_the_secret_the_strata_are_recomputed_and_a_warning_says_so(self):
        with_secret = self.cal(self.rows())
        (self.run_dir / ".label_secret").unlink()
        cal = self.cal(self.rows())
        self.assertEqual(cal["strata"]["source"], "recomputed")
        self.assertEqual(cal["strata"]["n_unverifiable_tokens"], 45)
        warning = next(w for w in cal["warnings"] if ".label_secret" in w)
        self.assertIn("45 labels", warning)
        self.assertIn("recomputed from this run's own verdicts", warning)
        for key in ("tpr", "tnr", "kappa_weighted", "corrected_pass_rate", "agreement_weighted"):
            self.assertEqual(cal[key], with_secret[key], key)  # the numbers are the same: the weights never came from the token
        self.assertEqual(cal["strata"]["weights"], with_secret["strata"]["weights"])

    def test_a_run_with_no_folder_warns_the_same_way(self):
        run = make_run(self.run.results, run_id=self.run.run_id)  # run_dir None
        cal = self.cal(self.rows(), run=run)
        self.assertEqual(cal["strata"]["source"], "recomputed")
        self.assertTrue(any("no folder on disk" in w and ".label_secret" in w for w in cal["warnings"]))

    def test_a_wrong_secret_verifies_nothing_and_the_weights_are_still_the_runs_own(self):
        (self.run_dir / ".label_secret").write_text(os.urandom(32).hex() + "\n", encoding="ascii")
        cal = self.cal(self.rows())
        self.assertEqual(cal["strata"]["source"], "recomputed")
        self.assertTrue(any("does not match this run" in w for w in cal["warnings"]))
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 2.0, "judge_fail": 1.0})

    def test_a_stratum_claimed_for_the_other_verdict_is_caught_and_ignored(self):
        cal = self.cal(self.rows(flip=True))  # valid tokens, but for the wrong stratum of every case
        self.assertTrue(any("does not match this run" in w for w in cal["warnings"]))
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 2.0, "judge_fail": 1.0})
        self.assertEqual(cal["strata"]["source"], "recomputed")

    def test_legacy_sha256_tokens_are_still_accepted(self):
        for secret_present in (True, False):
            if not secret_present:
                (self.run_dir / ".label_secret").unlink()
            cal = self.cal(self.rows(secret="legacy"))
            self.assertEqual(cal["strata"]["source"], "recorded", secret_present)
            self.assertEqual(cal["strata"]["n_unverifiable_tokens"], 0)
            self.assertFalse(any("cannot be verified" in w or "does not match" in w for w in cal["warnings"]))

    def test_plain_strata_of_older_label_files_are_still_accepted(self):
        rows = self.rows()
        for row in rows:
            row["stratum"] = "judge_pass" if row["case_id"].startswith("p") else "judge_fail"
        cal = self.cal(rows)
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 2.0, "judge_fail": 1.0})

    def test_tokens_of_another_run_do_not_verify(self):
        other = saved_run(self.run.results, self.run_dir, run_id="another-run")
        cal = self.cal(self.rows(), run=other)  # tokens were made for run-keyed, the calibrated run is another-run
        self.assertTrue(any("does not match this run" in w for w in cal["warnings"]))
        self.assertEqual(cal["strata"]["source"], "recomputed")


# ------------------------------------------------------------------------------------------------------- warnings
class RunIdWarningTests(unittest.TestCase):
    def rows(self, run_ids):
        run, labels = confusion_scenario(tp=25, fn=2, fp=3, tn=15)
        for row, run_id in zip(labels, run_ids * len(labels)):
            row["run_id"] = run_id
        return run, labels

    def test_labels_of_another_run_are_matched_but_flagged(self):
        run, labels = self.rows(["run-test", "run-test", "run-test", "run-OLD"])
        cal = calibration.calibrate(run, labels, **QUICK)
        self.assertEqual(cal["n"], 45)  # matched by case id and output sha, not dropped
        self.assertEqual(cal["n_foreign_run_labels"], 11)
        self.assertEqual(cal["foreign_run_ids"], ["run-OLD"])
        warning = next(w for w in cal["warnings"] if "run_id" in w)
        self.assertIn("11 label rows", warning)
        self.assertIn("'run-OLD'", warning)
        self.assertIn("'run-test'", warning)
        self.assertIn("evalsprint label", warning)

    def test_the_warning_lists_a_few_ids_and_counts_the_rest(self):
        run, labels = self.rows([f"old-{i}" for i in range(5)])
        cal = calibration.calibrate(run, labels, **QUICK)
        self.assertEqual(cal["foreign_run_ids"], [f"old-{i}" for i in range(5)])
        self.assertIn("and 2 more", next(w for w in cal["warnings"] if "run_id" in w))

    def test_the_run_id_of_this_run_and_a_missing_one_are_not_flagged(self):
        for run_ids in (["run-test"], [None], [""], ["run-test", None]):
            run, labels = self.rows(run_ids)
            cal = calibration.calibrate(run, labels, **QUICK)
            self.assertEqual(cal["n_foreign_run_labels"], 0, run_ids)
            self.assertEqual(cal["foreign_run_ids"], [])
            self.assertFalse(any("run_id" in w for w in cal["warnings"]), run_ids)


class OneSidedLabelsTests(unittest.TestCase):
    def mixed_run(self, n_pass=60, n_fail=20):
        return make_run([make_record(f"p{i:02d}", True) for i in range(n_pass)] + [make_record(f"f{i:02d}", False) for i in range(n_fail)])

    def test_labels_of_judge_passes_only_ask_for_cases_the_judge_failed(self):
        labels = [label(f"p{i:02d}", "fail" if i % 6 == 0 else "pass") for i in range(40)]
        cal = calibration.calibrate(self.mixed_run(), labels, **FAST)
        self.assertEqual(cal["one_sided_labels"], "pass")
        self.assertIn("humans failed 7 of them", cal["recommendation"])  # the disagreements are evidence, and the text says so
        for text in [cal["recommendation"]] + cal["warnings"]:
            self.assertNotIn("revise the rubric", text.lower())
        self.assertTrue(any("label cases the judge failed" in w for w in cal["warnings"]))
        self.assertIn("label cases the judge failed", cal["recommendation"])
        self.assertNotIn("trustworthy for gating", cal["recommendation"])

    def test_labels_of_judge_failures_only_ask_for_cases_the_judge_passed(self):
        labels = [label(f"f{i:02d}", "pass" if i % 5 == 0 else "fail") for i in range(20)]
        cal = calibration.calibrate(self.mixed_run(), labels, **FAST)
        self.assertEqual(cal["one_sided_labels"], "fail")
        self.assertTrue(any("label cases the judge passed" in w for w in cal["warnings"]))
        for text in [cal["recommendation"]] + cal["warnings"]:
            self.assertNotIn("revise the rubric", text.lower())

    def test_a_judge_that_passes_everything_is_asked_for_known_bad_outputs_not_a_new_rubric(self):
        run, labels = confusion_scenario(tp=28, fn=0, fp=12, tn=0)  # the judge passed all 40, humans failed 12
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(cal["one_sided_labels"], "pass")
        text = " ".join([cal["recommendation"]] + cal["warnings"])
        self.assertIn("known-bad outputs", text)
        self.assertIn("label cases the judge failed", text)
        self.assertNotIn("revise the rubric", text.lower())
        self.assertEqual(cal["kappa_decision"], "not_trustworthy")

    def test_labels_of_both_kinds_are_not_one_sided(self):
        cal = calibration.calibrate(*confusion_scenario(tp=25, fn=2, fp=3, tn=15), **FAST)
        self.assertIsNone(cal["one_sided_labels"])
        self.assertFalse(any("labeled cases are ones the judge" in w for w in cal["warnings"]))

    def test_a_single_label(self):
        cal = calibration.calibrate(self.mixed_run(), [label("p00", "pass")], **FAST)
        self.assertEqual(cal["one_sided_labels"], "pass")
        self.assertIn("The only labeled case is one the judge passed", cal["recommendation"])


class ExcludeCaseIdsTests(unittest.TestCase):
    def test_excluded_cases_leave_the_calibration_but_not_the_run(self):
        run, labels = confusion_scenario(tp=30, fn=2, fp=3, tn=25)
        full = calibration.calibrate(run, labels, **QUICK)
        excluded = {"tp-000", "tp-001", "tn-000", "fp-000", "not-a-case"}
        cal = calibration.calibrate(run, labels, exclude_case_ids=excluded, **QUICK)
        self.assertEqual(cal["n"], full["n"] - 4)
        self.assertEqual(cal["n_excluded_holdout"], 4)
        self.assertEqual(full["n_excluded_holdout"], 0)
        self.assertEqual(cal["n_unmatched"], 0)  # an unknown id in the exclusion list is not a label problem
        self.assertFalse({"tp-000", "tp-001", "tn-000", "fp-000"} & set(cal["labeled_case_ids"]))
        self.assertEqual(cal["confusion"]["human_pass_judge_pass"], 28)
        self.assertEqual(cal["confusion"]["human_fail_judge_pass"], 2)
        self.assertEqual(cal["strata"]["run_cases"], full["strata"]["run_cases"])  # the run's strata (the weights' N_h) are unchanged
        self.assertTrue(any("held out" in line for line in calibration.to_markdown(cal).splitlines()))

    def test_labeled_case_ids_are_the_cases_used_in_run_order(self):
        run, labels = confusion_scenario(tp=3, fn=1, fp=1, tn=3)
        labels = list(reversed(labels))
        cal = calibration.calibrate(run, labels, **QUICK)
        self.assertEqual(cal["labeled_case_ids"], [r["case_id"] for r in run.results])


# ------------------------------------------------------------------------------------------------------ few-shot loop
LONG = "word " * 900  # 4500 characters


def fewshot_setup(n_pass=160, n_fail=40, label_pass=30, label_fail=30):
    """A run of n_pass + n_fail cases labeled on a sample stratified on the judge's verdict, with a few notes."""
    records = [make_record(f"p{i:03d}", True, input=f"question p{i}", text=f"answer p{i}") for i in range(n_pass)]
    records += [make_record(f"f{i:03d}", False, input=f"question f{i}", text=f"answer f{i}") for i in range(n_fail)]
    labels = []
    for i in range(label_pass):
        human = "fail" if i % 4 == 0 else "pass"  # the judge was too lenient on these
        labels.append(label(f"p{i:03d}", human, note="invented a refund policy" if human == "fail" and i % 8 == 0 else ""))
    for i in range(label_fail):
        human = "pass" if i % 5 == 0 else "fail"  # ...and too harsh on these
        labels.append(label(f"f{i:03d}", human, note="the answer is fine, just terse" if human == "pass" and i % 10 == 0 else ""))
    return make_run(records), labels


class FewshotExportTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.run, self.labels = fewshot_setup()
        self.cal = calibration.calibrate(self.run, self.labels, **FAST)
        self.path = self.dir / "out" / "fewshot.yaml"

    def export(self, **kw):
        out = calibration.export_fewshot(self.cal, self.run, self.path, **kw)
        self.assertEqual(out, self.path)
        return yaml.safe_load(self.path.read_text(encoding="utf-8"))

    def test_the_yaml_has_examples_and_holdout_case_ids_with_the_human_verdict_first(self):
        data = self.export()
        self.assertEqual(list(data), ["examples", "holdout_case_ids"])
        self.assertTrue(data["examples"])
        human = {row["case_id"]: row["label"] for row in self.labels}
        by_case = self.run.by_case()
        for example in data["examples"]:
            self.assertEqual(list(example), ["verdict", "input", "output", "reasoning", "case_id"])
            self.assertEqual(example["verdict"], human[example["case_id"]])  # the human's verdict, not the judge's
            self.assertNotEqual(example["verdict"], "pass" if by_case[example["case_id"]]["passed"] else "fail")  # a disagreement
            self.assertEqual(example["input"], by_case[example["case_id"]]["case"]["input"])
            self.assertEqual(example["output"], by_case[example["case_id"]]["output"]["text"])
            self.assertIn(example["case_id"], data["holdout_case_ids"])
        self.assertIn("# Few-shot examples", self.path.read_text(encoding="utf-8").splitlines()[0])

    def test_the_holdout_is_a_random_pool_that_leaves_the_calibrating_remainder_unbiased(self):
        data = self.export(holdout_frac=0.3)
        pool = data["holdout_case_ids"]
        labeled = self.cal["labeled_case_ids"]
        self.assertEqual(len(set(pool)), len(pool))
        self.assertTrue(set(pool) <= set(labeled))
        self.assertEqual(len(pool), 18)  # 30% of 30 labels in each of the two strata
        strata = Counter("p" if cid.startswith("p") else "f" for cid in pool)
        self.assertEqual(strata, {"p": 9, "f": 9})  # drawn within each sampling stratum, never a whole stratum
        # the pool is not just the disagreements: it also holds agreements, or the remainder would be biased toward agreement
        by_case = self.run.by_case()
        human = {row["case_id"]: row["label"] for row in self.labels}
        agreements = [c for c in pool if human[c] == ("pass" if by_case[c]["passed"] else "fail")]
        self.assertTrue(agreements)
        # the next calibration, without the pool, keeps a stratified sample of both kinds
        rest = calibration.calibrate(self.run, self.labels, exclude_case_ids=calibration.load_holdout_ids(self.path), **FAST)
        self.assertEqual(rest["n"], len(labeled) - len(pool))
        self.assertEqual(rest["n_excluded_holdout"], len(pool))
        self.assertFalse(set(rest["labeled_case_ids"]) & set(pool))
        self.assertEqual(set(rest["strata"]["labeled"]), {"judge_pass", "judge_fail"})
        self.assertTrue(all(rest["strata"]["labeled"][s] > 0 for s in rest["strata"]["labeled"]))
        self.assertEqual(rest["strata"]["run_cases"], self.cal["strata"]["run_cases"])

    def test_it_is_deterministic_for_a_seed_and_the_seed_changes_the_pool(self):
        first = self.export(seed=3)
        text = self.path.read_text(encoding="utf-8")
        self.export(seed=3)
        self.assertEqual(self.path.read_text(encoding="utf-8"), text)
        other = self.export(seed=4)
        self.assertNotEqual(other["holdout_case_ids"], first["holdout_case_ids"])
        self.assertEqual(len(other["holdout_case_ids"]), len(first["holdout_case_ids"]))

    def test_max_examples_bounds_the_examples_and_balances_the_two_kinds(self):
        data = self.export(max_examples=4, holdout_frac=0.9)
        self.assertEqual(len(data["examples"]), 4)
        verdicts = [e["verdict"] for e in data["examples"]]
        self.assertEqual(sorted(verdicts), ["fail", "fail", "pass", "pass"])
        self.assertEqual(verdicts, sorted(verdicts))  # human-fail examples (the judge was too lenient) come first
        self.assertEqual(len(self.export(max_examples=1, holdout_frac=0.9)["examples"]), 1)
        empty = self.export(max_examples=0)
        self.assertEqual(empty["examples"], [])
        self.assertTrue(empty["holdout_case_ids"])  # the pool is still written: the holdout is the point of the file

    def test_examples_with_a_human_note_come_first_and_the_note_is_the_reasoning(self):
        data = self.export(max_examples=20, holdout_frac=0.9)
        for side in ("fail", "pass"):
            reasons = [bool(e["reasoning"]) for e in data["examples"] if e["verdict"] == side]
            self.assertEqual(reasons, sorted(reasons, reverse=True), side)  # noted ones first
        noted = [e for e in data["examples"] if e["reasoning"]]
        self.assertTrue(noted)
        self.assertTrue(all(e["reasoning"] in ("invented a refund policy", "the answer is fine, just terse") for e in noted))
        self.assertIn("empty `reasoning`", self.path.read_text(encoding="utf-8"))  # the header says what to fill in

    def test_a_confident_error_ranks_before_a_hesitant_one(self):
        run, labels = fewshot_setup(n_pass=40, n_fail=10, label_pass=20, label_fail=8)
        for row in labels:
            row["note"] = ""  # no human note: the judge's own confidence in its mistake decides the order
        for record in run.results:  # the judge was sure about every case except p004
            record["judges"][0]["score"] = 0.55 if record["case_id"] == "p004" else (0.99 if record["passed"] else 0.01)
        cal = calibration.calibrate(run, labels, **FAST)
        checked = 0
        for seed in range(6):
            out = calibration.export_fewshot(cal, run, self.dir / f"ranked{seed}.yaml", max_examples=20, holdout_frac=0.95, seed=seed)
            data = yaml.safe_load(out.read_text(encoding="utf-8"))
            fails = [e["case_id"] for e in data["examples"] if e["verdict"] == "fail"]
            if "p004" in fails:
                checked += 1
                self.assertEqual(fails[-1], "p004", (seed, fails))  # the hesitant mistake is the least instructive: last
        self.assertGreater(checked, 0)

    def test_long_and_multiline_texts_round_trip(self):
        records = [make_record(f"c{i:02d}", i < 20, input="line one\nline two: with a colon\n- and a dash", text=LONG if i == 20 else "plain") for i in range(40)]
        odd = [
            "Multi\nline\n\ntext with trailing newline\n",
            "  leading and trailing spaces  ",
            "- starts like a YAML list\nkey: value\n---\n# not a comment",
            "acao: revisao, 'aspas' e \"duplas\" – café 中文",
            "tab\tinside and a trailing space \n",
        ]
        for i, text in enumerate(odd):
            records[20 + i]["output"]["text"] = text
        labels = [label(r["case_id"], "pass" if r["case_id"] >= "c20" else "fail") for r in records[10:35]]
        labels += [label(r["case_id"], "fail" if int(r["case_id"][1:]) % 2 else "pass") for r in records[:10]]
        run = make_run(records)
        cal = calibration.calibrate(run, labels, **QUICK)
        out = calibration.export_fewshot(cal, run, self.dir / "odd.yaml", max_examples=40, holdout_frac=0.95)
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        by_case = run.by_case()
        self.assertTrue(data["examples"])
        for example in data["examples"]:
            record = by_case[example["case_id"]]
            expected_out = record["output"]["text"]
            if len(expected_out) > calibration.FEWSHOT_MAX_FIELD_CHARS:
                self.assertTrue(example["output"].startswith(expected_out[: calibration.FEWSHOT_MAX_FIELD_CHARS]))
                self.assertIn("[... truncated 2500 chars ...]", example["output"])
                self.assertLess(len(example["output"]), 2100)
            else:
                self.assertEqual(example["output"], expected_out)
            self.assertEqual(example["input"], record["case"]["input"])

    def test_no_disagreement_means_no_examples_and_says_why(self):
        run, labels = confusion_scenario(tp=25, fn=0, fp=0, tn=15)
        cal = calibration.calibrate(run, labels, **QUICK)
        data = calibration.export_fewshot(cal, run, self.dir / "none.yaml")
        parsed = yaml.safe_load(data.read_text(encoding="utf-8"))
        self.assertEqual(parsed["examples"], [])
        self.assertTrue(parsed["holdout_case_ids"])
        self.assertIn("No disagreement fell inside the pool", data.read_text(encoding="utf-8"))

    def test_a_named_judge_pools_within_the_strata_of_that_judge(self):
        records = []
        for i in range(100):
            quality, safety = i < 70, i % 3 != 0
            judges = [
                {"judge": "quality", "passed": quality, "score": 1.0 if quality else 0.0, "reasoning": "r", "details": {}, "cost_usd": 0.0, "usage": {}, "error": None},
                {"judge": "safety", "passed": safety, "score": 1.0 if safety else 0.0, "reasoning": "r", "details": {}, "cost_usd": 0.0, "usage": {}, "error": None},
            ]
            records.append(make_record(f"c{i:03d}", quality and safety, judges=judges, passed=quality and safety))
        labels = []
        for i in list(range(0, 70, 3)) + list(range(70, 100, 2)):
            row = label(f"c{i:03d}", "fail" if i % 7 == 0 else "pass")
            row["stratified_on"] = "quality"
            labels.append(row)
        run = make_run(records)
        cal = calibration.calibrate(run, labels, judge="quality", **QUICK)
        self.assertEqual(cal["strata"]["stratified_on"], "quality")
        data = yaml.safe_load(calibration.export_fewshot(cal, run, self.dir / "named.yaml", holdout_frac=0.4).read_text(encoding="utf-8"))
        pool = data["holdout_case_ids"]
        quality_pass = {f"c{i:03d}" for i in range(70)}
        self.assertTrue(set(pool) & quality_pass)
        self.assertTrue(set(pool) - quality_pass)  # both quality strata are represented
        labeled = set(cal["labeled_case_ids"])
        self.assertLess(len(set(pool) & quality_pass), len(labeled & quality_pass))  # never a whole stratum

    def test_the_header_cannot_be_broken_by_a_judge_name(self):
        judges = [{"judge": "evil\nexamples: [pwned]", "passed": True, "score": 1.0, "reasoning": "r", "details": {}, "cost_usd": 0.0, "usage": {}, "error": None}]
        records = [make_record(f"c{i}", i % 3 != 0, judges=[dict(judges[0], passed=i % 3 != 0)], passed=i % 3 != 0) for i in range(30)]
        labels = [label(f"c{i}", "pass" if i % 4 else "fail") for i in range(30)]
        run = make_run(records)
        cal = calibration.calibrate(run, labels, judge="evil\nexamples: [pwned]", **QUICK)
        out = calibration.export_fewshot(cal, run, self.dir / "evil.yaml")
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual(list(data), ["examples", "holdout_case_ids"])
        self.assertNotEqual(data["examples"], ["pwned"])

    def test_the_examples_load_into_a_rubric_and_reach_the_judge_prompt(self):
        data = self.export(max_examples=2, holdout_frac=0.6)
        rubric = Rubric.load({"name": "r", "criteria": [{"id": "grounded", "question": "Is it grounded?"}], "examples": data["examples"]})
        text = _format_examples(rubric.examples, nonce="abc123")
        self.assertEqual(text.count("Human verdict:"), 2)
        self.assertIn("question ", text)
        self.assertNotIn("case_id", text)  # the extra key is ignored by the judge

    def test_arguments_are_validated(self):
        for kw in ({"holdout_frac": 0.0}, {"holdout_frac": 1.0}, {"holdout_frac": -0.1}, {"holdout_frac": 1.5}, {"max_examples": -1}):
            with self.assertRaises(ValueError, msg=str(kw)):
                calibration.export_fewshot(self.cal, self.run, self.path, **kw)
        self.assertFalse(self.path.exists())
        old = {k: v for k, v in self.cal.items() if k != "labeled_case_ids"}
        with self.assertRaisesRegex(ValueError, "labeled_case_ids"):
            calibration.export_fewshot(old, self.run, self.path)
        other = make_run(list(self.run.results), run_id="some-other-run")
        with self.assertRaisesRegex(ValueError, "some-other-run"):
            calibration.export_fewshot(self.cal, other, self.path)


class LoadHoldoutIdsTests(TempDirCase):
    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_the_exported_yaml(self):
        run, labels = fewshot_setup()
        cal = calibration.calibrate(run, labels, **FAST)
        path = calibration.export_fewshot(cal, run, self.dir / "fs.yaml")
        ids = calibration.load_holdout_ids(path)
        self.assertEqual(ids, yaml.safe_load(path.read_text(encoding="utf-8"))["holdout_case_ids"])
        self.assertTrue(all(isinstance(i, str) for i in ids))

    def test_other_formats(self):
        self.assertEqual(calibration.load_holdout_ids(self.write("a.yaml", "- c1\n- c2\n")), ["c1", "c2"])
        self.assertEqual(calibration.load_holdout_ids(self.write("b.json", '["c1", "c2", 3]')), ["c1", "c2", "3"])
        self.assertEqual(calibration.load_holdout_ids(self.write("c.txt", "# ids to hold out\nc1\n\n  c2  \n")), ["c1", "c2"])
        self.assertEqual(calibration.load_holdout_ids(self.write("d.yaml", "holdout_case_ids:\n- 7\n- 8\nexamples: []\n")), ["7", "8"])
        self.assertEqual(calibration.load_holdout_ids(self.write("e.yaml", "examples: []\nholdout_case_ids: []\n")), [])
        self.assertEqual(calibration.load_holdout_ids(self.write("f.txt", "")), [])
        self.assertEqual(calibration.load_holdout_ids(self.write("g.txt", "﻿c1\nc2\n")), ["c1", "c2"])

    def test_a_yaml_without_the_key_or_with_a_scalar_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "holdout_case_ids"):
            calibration.load_holdout_ids(self.write("h.yaml", "examples: []\n"))
        with self.assertRaises(ValueError):
            calibration.load_holdout_ids(self.write("i.yaml", "holdout_case_ids: c1\n"))


if __name__ == "__main__":
    unittest.main()
