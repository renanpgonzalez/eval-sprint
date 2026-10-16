"""Cycle 4, contract C20: iterating on a rubric must not cost a full labeling pass, and every calibration message stays honest.

* ``previous_labels`` on the labeling exports (only the cases without a label for their CURRENT output are exported);
* ``calibrate`` takes several label files; labels made on an earlier run whose output is unchanged are carried over and
  counted as a fully labeled part of the run, the labels made on this run stand for the rest (an exact estimator, unlike pooling);
* ``judges_sha`` in the calibration and ``stale_calibration``;
* ``labels_budget_table``;
* ``tnr_reliable`` is phrased as a screening flag;
* ``case_id`` is formula-neutralised in the labeling CSV and round-trips through ``load_labels``.

The bias / coverage tables behind the design live in orchestration/2026-09-29_kit-v0.2/cycle4/eval-scientist-judge/; the tests here are the
fast, seeded, offline guards of the same properties.
"""

import csv
import json
import random
import unittest
from pathlib import Path

from evalsprint import calibration, labeling
from tests.test_calibration import confusion_scenario, label, make_record, make_run, sha_of
from tests.test_labeling import TempDirCase, embedded_payload

FAST = {"iters": 200}


# ------------------------------------------------------------------------------------------------- a two-run world
def world_run(run_id, n, *, changed=frozenset(), verdict=None):
    """``n`` cases ``c000..``; the output text of the cases in ``changed`` differs from the default one (a new sha).
    ``verdict(i)`` is the judge's pass/fail for case ``i`` (default: every third case fails)."""
    verdict = verdict or (lambda i: i % 3 != 0)
    records = [
        make_record(f"c{i:03d}", verdict(i), text=f"answer {i} v2" if i in changed else f"answer {i}") for i in range(n)
    ]
    return make_run(records, run_id=run_id)


def label_rows(run, human, cids=None, *, labeler="ann", run_id=None):
    """Label rows as the labeling tool writes them (case id, label, output sha, run id); ``human(i)`` is pass/fail of case ``i``."""
    by_case = run.by_case()
    rows = []
    for cid in cids if cids is not None else list(by_case):
        i = int(cid[1:])
        rows.append(
            {
                "case_id": cid,
                "label": "pass" if human(i) else "fail",
                "output_sha": by_case[cid]["output"]["sha"],
                "labeler": labeler,
                "run_id": run_id or run.run_id,
                "note": "",
            }
        )
    return rows


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Path(path)


# --------------------------------------------------------------------------------------------- previous_labels / export
class CoveredCasesTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.run1 = world_run("run-1", 60)
        self.human = lambda i: i % 4 != 0
        self.rows1 = label_rows(self.run1, self.human)

    def test_a_label_covers_its_case_only_while_the_output_is_the_same(self):
        changed = frozenset(range(0, 60, 5))  # 12 outputs changed
        run2 = world_run("run-2", 60, changed=changed)
        covered = labeling.covered_case_ids(run2, self.rows1)
        self.assertEqual(covered, frozenset(f"c{i:03d}" for i in range(60) if i not in changed))

    def test_the_run_a_label_was_made_on_does_not_matter(self):
        run2 = world_run("run-2", 60)
        rows = [dict(r, run_id=None) for r in self.rows1[:10]] + [dict(r, run_id="run-xyz") for r in self.rows1[10:20]]
        self.assertEqual(len(labeling.covered_case_ids(run2, rows)), 20)

    def test_a_label_without_a_sha_cannot_be_checked_so_it_does_not_cover(self):
        run2 = world_run("run-2", 60)
        rows = [dict(r, output_sha=None) for r in self.rows1[:10]] + self.rows1[10:12]
        self.assertEqual(labeling.covered_case_ids(run2, rows), frozenset(r["case_id"] for r in self.rows1[10:12]))

    def test_unlabeled_unknown_and_unreadable_rows_cover_nothing(self):
        run2 = world_run("run-2", 60)
        rows = [dict(self.rows1[0], label=""), dict(self.rows1[1], label="maybe"), dict(self.rows1[2], case_id="nope")]
        self.assertEqual(labeling.covered_case_ids(run2, rows), frozenset())

    def test_a_full_length_sha_matches_the_short_one_in_the_run(self):
        run2 = world_run("run-2", 60)
        full = [dict(r, output_sha=sha_of(f"answer {int(r['case_id'][1:])}")) for r in self.rows1[:5]]
        self.assertTrue(all(len(r["output_sha"]) >= len(run2.by_case()[r["case_id"]]["output"]["sha"]) for r in full))
        self.assertEqual(len(labeling.covered_case_ids(run2, full)), 5)

    def test_takes_rows_a_path_or_a_list_of_paths(self):
        run2 = world_run("run-2", 60)
        a = write_jsonl(self.dir / "a.jsonl", self.rows1[:10])
        b = write_jsonl(self.dir / "b.jsonl", self.rows1[10:25])
        self.assertEqual(len(labeling.covered_case_ids(run2, a)), 10)
        self.assertEqual(len(labeling.covered_case_ids(run2, [a, b])), 25)
        self.assertEqual(len(labeling.covered_case_ids(run2, [a, self.rows1[10:15]])), 15)  # a path and loaded rows, mixed


class PreviousLabelsExportTests(TempDirCase):
    """The 30%-changed re-run of the contract: the export holds exactly the changed outputs."""

    def setUp(self):
        super().setUp()
        self.human = lambda i: i % 4 != 0
        self.run1 = world_run("run-1", 60)
        self.rows1 = label_rows(self.run1, self.human)
        self.changed = frozenset(range(2, 60, 3)) - frozenset(range(0, 60, 9))  # about a third of the outputs
        self.run2 = world_run("run-2", 60, changed=self.changed)
        self.changed_ids = {f"c{i:03d}" for i in self.changed}

    def test_html_export_holds_only_the_changed_outputs(self):
        out = labeling.export_labeling_html(self.run2, self.dir / "l.html", previous_labels=self.rows1)
        ids = {item["case_id"] for item in embedded_payload(out.read_text(encoding="utf-8"))["items"]}
        self.assertEqual(ids, self.changed_ids)
        self.assertGreaterEqual(len(ids) / 60, 0.25)

    def test_csv_export_holds_only_the_changed_outputs(self):
        out = labeling.export_labeling_csv(self.run2, self.dir / "l.csv", previous_labels=self.rows1)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            ids = {row["case_id"] for row in csv.DictReader(fh)}
        self.assertEqual(ids, self.changed_ids)

    def test_previous_labels_can_be_files(self):
        path = write_jsonl(self.dir / "labels-run1.jsonl", self.rows1)
        out = labeling.export_labeling_csv(self.run2, self.dir / "l.csv", previous_labels=path)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            self.assertEqual({row["case_id"] for row in csv.DictReader(fh)}, self.changed_ids)

    def test_sample_counts_new_labels_among_the_cases_without_one(self):
        chosen = labeling.select_sample(self.run2, sample=10, seed=3, previous_labels=self.rows1[:30])
        self.assertEqual(len(chosen), 10)
        covered = labeling.covered_case_ids(self.run2, self.rows1[:30])
        self.assertTrue(all(r["case_id"] not in covered for r in chosen))
        verdicts = [r["passed"] for r in chosen]
        self.assertEqual(sum(verdicts), 5)  # still balanced on the verdict: 5 passes, 5 fails

    def test_without_previous_labels_the_draw_is_what_it_was(self):
        plain = [r["case_id"] for r in labeling.select_sample(self.run2, sample=20, seed=5)]
        self.assertEqual(plain, [r["case_id"] for r in labeling.select_sample(self.run2, sample=20, seed=5, previous_labels=None)])
        nothing_covered = [r["case_id"] for r in labeling.select_sample(self.run2, sample=20, seed=5, previous_labels=[])]
        self.assertEqual(plain, nothing_covered)

    def test_every_case_labeled_is_nothing_to_export_and_says_so(self):
        run_same = world_run("run-2", 60)
        with self.assertRaises(labeling.NothingToLabel) as caught:
            labeling.export_labeling_html(run_same, self.dir / "l.html", previous_labels=self.rows1)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIn("already have a label", str(caught.exception))
        self.assertFalse((self.dir / "l.html").exists())
        with self.assertRaises(labeling.NothingToLabel):
            labeling.export_labeling_csv(run_same, self.dir / "l.csv", previous_labels=self.rows1)

    def test_plan_relabel_counts_without_writing(self):
        plan = labeling.plan_relabel(self.run2, self.rows1)
        self.assertEqual(plan["eligible"], 60)
        self.assertEqual(plan["uncovered"], len(self.changed_ids))
        self.assertEqual(plan["covered"], 60 - len(self.changed_ids))
        self.assertEqual(plan["to_label"], len(self.changed_ids))
        self.assertEqual(set(plan["case_ids"]), self.changed_ids)
        capped = labeling.plan_relabel(self.run2, self.rows1, sample=5)
        self.assertEqual((capped["to_label"], len(capped["case_ids"]), capped["uncovered"]), (5, 5, len(self.changed_ids)))
        done = labeling.plan_relabel(world_run("run-2", 60), self.rows1)
        self.assertEqual((done["uncovered"], done["to_label"], done["case_ids"]), (0, 0, []))

    def test_only_changed_outputs_when_the_earlier_labels_covered_every_unchanged_case_else_never_labeled_ones_too(self):
        # full earlier pass: the export is exactly the changed outputs (the contract's 30% case)
        full = labeling.plan_relabel(self.run2, self.rows1)
        self.assertEqual(set(full["case_ids"]), self.changed_ids)
        # sampled earlier pass (half of the cases): the unchanged cases that never had a label come with them
        half = self.rows1[:30]
        labeled_unchanged = {r["case_id"] for r in half} - self.changed_ids
        never_labeled_unchanged = {f"c{i:03d}" for i in range(60)} - self.changed_ids - labeled_unchanged
        self.assertTrue(never_labeled_unchanged)
        plan = labeling.plan_relabel(self.run2, half)
        self.assertEqual(set(plan["case_ids"]), self.changed_ids | never_labeled_unchanged)
        self.assertGreater(plan["to_label"], len(self.changed_ids))
        out = labeling.export_labeling_csv(self.run2, self.dir / "half.csv", previous_labels=half)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            self.assertEqual({row["case_id"] for row in csv.DictReader(fh)}, self.changed_ids | never_labeled_unchanged)

    def test_the_docs_of_the_exports_say_when_only_changed_outputs_need_a_label(self):
        for doc in (labeling.__doc__, labeling.select_sample.__doc__, labeling.plan_relabel.__doc__, labeling.export_labeling_html.__doc__,
                    labeling.export_labeling_csv.__doc__, calibration.__doc__, calibration.calibrate.__doc__):
            flat = " ".join(doc.split())
            self.assertIn("every unchanged case", flat)
            self.assertIn("never-labeled unchanged", flat)

    def test_the_labeler_still_sees_no_verdict(self):
        out = labeling.export_labeling_html(self.run2, self.dir / "l.html", previous_labels=self.rows1)
        page = out.read_text(encoding="utf-8")
        for item in embedded_payload(page)["items"]:
            self.assertEqual(set(item) & {"passed", "judges", "verdict"}, set())
        self.assertNotIn("judge_pass", page)

    def test_the_new_labels_carry_this_runs_id_and_stratum_code(self):
        self.assertEqual(labeling.select_sample(self.run2, 4, previous_labels=self.rows1)[0]["case_id"] in self.changed_ids, True)
        out = labeling.export_labeling_csv(self.run2, self.dir / "l.csv", previous_labels=self.rows1)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(all(r["run_id"] == "run-2" and r["stratum"].startswith("s-") for r in rows))


# ---------------------------------------------------------------------------------------------- several label files
class MultipleLabelFilesTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.run = world_run("run-1", 50)
        self.rows = label_rows(self.run, lambda i: i % 4 != 0)

    def test_load_labels_takes_several_files_in_order_and_sums_the_counters(self):
        a = write_jsonl(self.dir / "a.jsonl", self.rows[:20] + [{"case_id": "", "label": "pass"}, dict(self.rows[0], label="")])
        b = write_jsonl(self.dir / "b.jsonl", self.rows[20:30] + [{"case_id": "c999", "label": "maybe"}])
        loaded = calibration.load_labels([a, b])
        self.assertEqual([r["case_id"] for r in loaded], [r["case_id"] for r in self.rows[:30]])
        self.assertEqual((loaded.n_invalid, loaded.n_unlabeled), (2, 1))
        self.assertEqual(len(calibration.load_labels(str(a))), 20)  # one path: as before

    def test_a_csv_and_a_jsonl_can_be_mixed(self):
        labeling.export_labeling_csv(self.run, self.dir / "sheet.csv", sample=10, seed=1)
        text = (self.dir / "sheet.csv").read_text(encoding="utf-8-sig")
        rows = list(csv.DictReader(text.splitlines(True)))
        for r in rows:
            r["label"] = "pass"
        with open(self.dir / "sheet.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        b = write_jsonl(self.dir / "b.jsonl", [r for r in self.rows if r["case_id"] not in {x["case_id"] for x in rows}][:5])
        self.assertEqual(len(calibration.load_labels([self.dir / "sheet.csv", b])), 15)

    def test_as_label_list_accepts_paths_rows_lists_and_mixtures(self):
        a = write_jsonl(self.dir / "a.jsonl", self.rows[:5])
        self.assertEqual(len(calibration.as_label_list(a)), 5)
        self.assertEqual(len(calibration.as_label_list([a, self.rows[5:8], [self.rows[8]], self.rows[9]])), 10)  # a path, rows, a list in a list, one row
        loaded = calibration.load_labels(a)
        self.assertIs(calibration.as_label_list(loaded), loaded)
        with self.assertRaises(TypeError):
            calibration.as_label_list([a, 3])
        self.assertEqual(len(calibration.as_label_list(iter(self.rows[:4]))), 4)

    def test_calibrate_takes_the_files_directly_and_counts_every_row(self):
        a = write_jsonl(self.dir / "a.jsonl", self.rows[:30])
        b = write_jsonl(self.dir / "b.jsonl", self.rows[30:50])
        cal = calibration.calibrate(self.run, [a, b], **FAST)
        together = calibration.calibrate(self.run, self.rows, **FAST)
        self.assertEqual(cal["n_labels"], 50)
        for key in ("n", "kappa", "tpr", "tnr", "kappa_weighted", "kappa_weighted_ci", "confusion"):
            self.assertEqual(cal[key], together[key], key)
        self.assertEqual(calibration.calibrate(self.run, a, **FAST)["n"], 30)  # a single path works too

    def test_the_same_labeler_twice_the_later_file_wins(self):
        first = label_rows(self.run, lambda i: True, ["c001"])
        later = label_rows(self.run, lambda i: False, ["c001"])
        votes = calibration._collect_votes(calibration.as_label_list([first, later]), self.run.by_case(), "run-1").votes
        self.assertFalse(votes["c001"]["ann"].is_pass)


# ---------------------------------------------------------------------------------------------- carried-over labels
class CarriedOverEstimatorTests(unittest.TestCase):
    """Hand-computable designs: run 2 has 40 judge passes and 40 judge fails; some cases carry labels made on run 1."""

    def setUp(self):
        self.run2 = world_run("run-2", 80, verdict=lambda i: i < 40)  # c000-c039 judge pass, c040-c079 judge fail
        self.by = self.run2.by_case()

    def rows(self, cids, human, run_id):
        return [
            {"case_id": c, "label": "pass" if human(int(c[1:])) else "fail", "output_sha": self.by[c]["output"]["sha"], "labeler": "ann", "run_id": run_id,
             "stratified_on": "overall", "stratum": calibration.stratum_token(run_id, "overall", c, calibration.stratum_of(int(c[1:]) < 40))}
            for c in cids
        ]

    def scenario(self):
        # carried from run-1: 10 judge-pass cases (all human pass) and 8 judge-fail cases (all human fail)
        old = self.rows([f"c{i:03d}" for i in range(0, 10)], lambda i: True, "run-1") + self.rows([f"c{i:03d}" for i in range(40, 48)], lambda i: False, "run-1")
        # made on run-2 among the cases without a label: 10 of the 30 judge passes (1 human fail), 8 of the 32 judge fails (half human pass)
        new_pass = [f"c{i:03d}" for i in range(10, 20)]
        new_fail = [f"c{i:03d}" for i in range(48, 56)]
        new = self.rows(new_pass, lambda i: i != 19, "run-2") + self.rows(new_fail, lambda i: i % 2 == 0, "run-2")
        return old, new

    def test_carried_labels_are_a_census_and_the_new_labels_stand_for_the_rest(self):
        old, new = self.scenario()
        cal = calibration.calibrate(self.run2, old + new, **FAST)
        s = cal["strata"]
        self.assertEqual(s["estimator"], "carried_census")
        self.assertEqual(s["carried"], {"judge_pass": 10, "judge_fail": 8})
        self.assertEqual(s["uncovered"], {"judge_pass": 30, "judge_fail": 32})
        self.assertEqual(s["weights"], {"judge_pass": 3.0, "judge_fail": 4.0})
        self.assertEqual(s["labeled"], {"judge_pass": 20, "judge_fail": 16})
        self.assertEqual((cal["n"], cal["n_carried_cases"], cal["n_carried_labels"], cal["carried_run_ids"]), (36, 18, 18, ["run-1"]))
        # judge-pass stratum: 10 carried TP + 10 new (9 TP, 1 FP) x 3;  judge-fail stratum: 8 carried TN + 8 new (4 TN, 4 FN) x 4
        tp, fp, tn, fn = 10 + 9 * 3, 1 * 3, 8 + 4 * 4, 4 * 4
        self.assertAlmostEqual(cal["tpr"], tp / (tp + fn))
        self.assertAlmostEqual(cal["tnr"], tn / (tn + fp))
        self.assertAlmostEqual(cal["human_pass_rate_weighted"], (tp + fn) / (tp + fn + tn + fp))

    def test_pooling_every_label_under_one_set_of_weights_gives_another_answer(self):
        old, new = self.scenario()
        pooled = calibration.calibrate(self.run2, [dict(r, run_id="") for r in old] + new, **FAST)
        self.assertEqual(pooled["strata"]["estimator"], "horvitz_thompson")
        # judge-pass: 20 labels x 2 (19 TP, 1 FP);  judge-fail: 16 labels x 2.5 (12 TN, 4 FN) - the old sample was drawn on another verdict
        self.assertAlmostEqual(pooled["tpr"], 38 / (38 + 10))
        self.assertAlmostEqual(pooled["tnr"], 30 / (30 + 2))
        census = calibration.calibrate(self.run2, old + new, **FAST)
        self.assertNotAlmostEqual(census["tpr"], pooled["tpr"], places=2)
        self.assertNotAlmostEqual(census["tnr"], pooled["tnr"], places=2)

    def test_a_census_of_labels_carried_and_new_is_the_truth_exactly(self):
        human = lambda i: i % 5 != 0
        run1 = world_run("run-1", 80, verdict=lambda i: i < 40)
        old = label_rows(run1, human, run_id="run-1")  # every case labeled on run 1
        changed = {i for i in range(80) if i % 3 == 0}
        run2 = world_run("run-2", 80, changed=changed, verdict=lambda i: i < 50)  # 27 outputs changed; the judge changed too
        new_human = lambda i: (i % 5 != 0) if i not in changed else (i % 2 == 0)
        by = run2.by_case()
        new = label_rows(run2, new_human, [f"c{i:03d}" for i in sorted(changed)], run_id="run-2")
        plan = labeling.plan_relabel(run2, old)
        self.assertEqual(plan["uncovered"], len(changed))
        cal = calibration.calibrate(run2, old + new, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "carried_census")
        verdicts = {cid: r["passed"] for cid, r in by.items()}
        truth = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for cid, passed in verdicts.items():
            i = int(cid[1:])
            truth[{(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}[(new_human(i), passed)]] += 1
        self.assertEqual(cal["n"], 80)
        self.assertAlmostEqual(cal["tpr"], truth["tp"] / (truth["tp"] + truth["fn"]))
        self.assertAlmostEqual(cal["tnr"], truth["tn"] / (truth["tn"] + truth["fp"]))
        self.assertEqual(cal["confusion"], {"human_pass_judge_pass": truth["tp"], "human_pass_judge_fail": truth["fn"], "human_fail_judge_pass": truth["fp"], "human_fail_judge_fail": truth["tn"]})
        self.assertEqual(cal["n_carried_cases"], 80 - len(changed))

    def test_few_new_labels_are_said_to_widen_and_undercover_the_intervals(self):
        old, new = self.scenario()
        few = new[:4] + new[12:14]  # 6 labels made on this run stand for 62 cases without a carried label
        cal = calibration.calibrate(self.run2, old + few, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "carried_census")
        note = next(w for w in cal["warnings"] if "carried over" in w)
        self.assertIn("Only 6 labels made on this run", note)
        self.assertIn("cover less than their 95%", note)
        self.assertIn("label more of the cases that have no carried label", note)

    def test_no_few_labels_warning_with_enough_new_labels_or_when_every_other_case_is_labeled(self):
        old, new = self.scenario()
        self.assertEqual(len(new), 18)  # 18 new labels here: below MIN_LABELS (30) ...
        cal = calibration.calibrate(self.run2, old + new, **FAST)
        self.assertIn("Only 18 labels", next(w for w in cal["warnings"] if "carried over" in w))
        # ... but with every other case labeled the estimate is a census: nothing to warn about, however few they are
        human = lambda i: i % 4 != 0
        run1 = world_run("run-1", 40)
        run2 = world_run("run-2", 40, changed=frozenset(range(10)))
        rows = label_rows(run1, human, run_id="run-1") + label_rows(run2, human, [f"c{i:03d}" for i in range(10)], run_id="run-2")
        census = calibration.calibrate(run2, rows, **FAST)
        self.assertEqual(census["strata"]["estimator"], "carried_census")
        self.assertEqual(census["n_carried_cases"], 30)
        self.assertNotIn("Only", next(w for w in census["warnings"] if "carried over" in w))

    def test_the_carried_labels_are_reported_not_flagged_as_foreign_noise(self):
        old, new = self.scenario()
        cal = calibration.calibrate(self.run2, old + new, **FAST)
        self.assertEqual(cal["n_foreign_run_labels"], 18)  # every row of another run is still counted
        self.assertEqual(cal["foreign_run_ids"], ["run-1"])
        note = next(w for w in cal["warnings"] if "carried over" in w)
        self.assertIn("18 label rows (18 cases)", note)
        self.assertIn("'run-1'", note)
        self.assertIn("output is unchanged", note)
        self.assertFalse(any("matched by case id only" in w for w in cal["warnings"]))
        self.assertFalse(any("do not match this run" in w or "does not match this run" in w for w in cal["warnings"]))

    def test_old_labels_whose_output_changed_are_dropped_and_their_case_needs_a_new_label(self):
        old, new = self.scenario()
        changed_text = {r["case_id"] for r in old[:4]}
        records = []
        for rec in self.run2.results:
            if rec["case_id"] in changed_text:
                rec = {**rec, "output": {**rec["output"], "text": "rewritten", "sha": sha_of("rewritten")}}
            records.append(rec)
        run = make_run(records, run_id="run-2")
        cal = calibration.calibrate(run, old + new, **FAST)
        self.assertEqual(cal["n_excluded_sha_mismatch"], 4)
        self.assertEqual(cal["n_carried_cases"], 14)
        self.assertEqual(cal["strata"]["uncovered"]["judge_pass"], 30 + 4 - 0)  # the 4 cases lost their label: they are cases without one again

    def test_without_new_labels_for_a_stratum_the_labels_are_pooled_with_a_warning(self):
        old, new = self.scenario()
        only_pass = [r for r in new if int(r["case_id"][1:]) < 40]
        cal = calibration.calibrate(self.run2, old + only_pass, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "pooled")
        warning = next(w for w in cal["warnings"] if "no label made on this run" in w)
        self.assertIn("judge_fail", warning)
        self.assertIn("pooled", warning)
        self.assertIn("can bias", warning)
        self.assertEqual(cal["n_carried_cases"], 18)
        # pooled: judge-pass 20 x 2, judge-fail 8 (all TN) x 5
        self.assertAlmostEqual(cal["tnr"], 40 / (40 + 2))

    def test_only_carried_labels_and_unlabeled_cases_left_is_the_pooled_fallback_not_a_census(self):
        old, _ = self.scenario()
        cal = calibration.calibrate(self.run2, old, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "pooled")
        self.assertTrue(any("no label made on this run" in w for w in cal["warnings"]))

    def test_carried_cases_that_are_held_out_are_represented_by_the_remaining_carried_labels(self):
        old, new = self.scenario()
        held = {"c000", "c001", "c040"}
        cal = calibration.calibrate(self.run2, old + new, exclude_case_ids=held, **FAST)
        self.assertEqual(cal["n_excluded_holdout"], 3)
        self.assertEqual(cal["strata"]["estimator"], "carried_census")
        self.assertEqual(cal["strata"]["carried"], {"judge_pass": 8, "judge_fail": 7})
        self.assertEqual(cal["strata"]["carried_weights"], {"judge_pass": 10 / 8, "judge_fail": 8 / 7})
        self.assertEqual(cal["strata"]["uncovered"], {"judge_pass": 30, "judge_fail": 32})  # the held-out cases are not "unlabeled cases"
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 3.0, "judge_fail": 4.0})
        self.assertEqual(cal["n"], 33)
        # the carried cases here are all alike, so standing in for the held-out ones changes nothing
        full = calibration.calibrate(self.run2, old + new, **FAST)
        self.assertAlmostEqual(cal["tpr"], full["tpr"])
        self.assertAlmostEqual(cal["tnr"], full["tnr"])
        note = next(w for w in cal["warnings"] if "carried over" in w)
        self.assertIn("3 carried cases held out or tied are represented by the other carried labels", note)

    def test_the_weight_of_a_carried_label_is_the_share_of_carried_cases_it_stands_for(self):
        old, new = self.scenario()
        # judge-pass carried cases: c000-c005 human pass, c006-c009 human fail; c000 is held out
        old = [dict(r, label="pass" if int(r["case_id"][1:]) < 6 else "fail") if int(r["case_id"][1:]) < 10 else r for r in old]
        cal = calibration.calibrate(self.run2, old + new, exclude_case_ids={"c000"}, **FAST)
        w = 10 / 9
        tp, fp, tn, fn = 5 * w + 27, 4 * w + 3, 24, 16
        self.assertAlmostEqual(cal["tpr"], tp / (tp + fn))
        self.assertAlmostEqual(cal["tnr"], tn / (tn + fp))
        self.assertEqual(cal["strata"]["carried_weights"]["judge_pass"], w)

    def test_a_stratum_whose_carried_labels_are_all_held_out_hands_its_cases_to_the_new_labels(self):
        old, new = self.scenario()
        held = {f"c{i:03d}" for i in range(40, 48)}
        cal = calibration.calibrate(self.run2, old + new, exclude_case_ids=held, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "carried_census")
        self.assertEqual(cal["strata"]["carried"], {"judge_pass": 10, "judge_fail": 0})
        self.assertEqual(cal["strata"]["uncovered"], {"judge_pass": 30, "judge_fail": 40})
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 3.0, "judge_fail": 5.0})  # 8 new labels stand for 40 cases
        self.assertNotIn("judge_fail", cal["strata"]["carried_weights"])
        tp, fp, tn, fn = 10 + 27, 3, 4 * 5, 4 * 5
        self.assertAlmostEqual(cal["tpr"], tp / (tp + fn))
        self.assertAlmostEqual(cal["tnr"], tn / (tn + fp))

    def test_the_stratum_codes_of_carried_labels_are_not_checked_against_this_run(self):
        old, new = self.scenario()
        cal = calibration.calibrate(self.run2, old + new, **FAST)
        self.assertEqual(cal["strata"]["source"], "recorded")  # the new labels verify; the carried ones (another run's tokens) are not needed
        self.assertFalse(any("record a stratum" in w for w in cal["warnings"]))
        old_other_design = [dict(r, stratified_on="quality") for r in old]
        cal = calibration.calibrate(self.run2, old_other_design + new, **FAST)
        self.assertFalse(any("different designs" in w for w in cal["warnings"]))  # the design of an earlier sample is irrelevant to a census

    def test_a_label_of_another_run_without_a_sha_is_still_matched_and_still_warned_about(self):
        old, new = self.scenario()
        unverified = [dict(r, output_sha=None) for r in old]
        cal = calibration.calibrate(self.run2, unverified + new, **FAST)
        self.assertEqual(cal["n_carried_cases"], 0)
        self.assertEqual(cal["strata"]["estimator"], "horvitz_thompson")
        self.assertEqual(cal["n_foreign_run_labels"], 18)
        warning = next(w for w in cal["warnings"] if "no output sha to check" in w)
        self.assertIn("18 label rows", warning)
        self.assertIn("'run-1'", warning)
        self.assertIn("evalsprint label", warning)

    def test_without_carried_labels_nothing_changes(self):
        run, labels = confusion_scenario(tp=25, fn=2, fp=3, tn=15)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(cal["strata"]["estimator"], "horvitz_thompson")
        self.assertEqual((cal["n_carried_cases"], cal["n_carried_labels"], cal["carried_run_ids"]), (0, 0, []))
        self.assertIsNone(cal["strata"]["uncovered"])
        self.assertEqual(cal["strata"]["carried"], {"judge_pass": 0, "judge_fail": 0})

    def test_the_intervals_are_defined_and_contain_the_estimates(self):
        old, new = self.scenario()
        cal = calibration.calibrate(self.run2, old + new, iters=400)
        for key, point in (("tpr_ci", "tpr"), ("tnr_ci", "tnr"), ("kappa_weighted_ci", "kappa_weighted"), ("corrected_pass_rate_ci", "corrected_pass_rate")):
            ci = cal[key]
            self.assertIsNotNone(ci, key)
            self.assertLessEqual(ci[0], cal[point] + 1e-9, key)
            self.assertGreaterEqual(ci[1], cal[point] - 1e-9, key)
        again = calibration.calibrate(self.run2, old + new, iters=400)
        self.assertEqual(cal["kappa_weighted_ci"], again["kappa_weighted_ci"])  # seeded

    def test_a_judge_calibrated_against_the_overall_verdict_labels_uses_the_same_census(self):
        # two judges, sample stratified on the overall verdict, named judge calibrated
        def rec(i, quality, heur):
            judges = [
                {"judge": "quality", "passed": quality, "score": None, "reasoning": "", "details": {}, "cost_usd": 0, "usage": {}, "error": None},
                {"judge": "heur", "passed": heur, "score": None, "reasoning": "", "details": {}, "cost_usd": 0, "usage": {}, "error": None},
            ]
            return make_record(f"c{i:03d}", None, judges=judges, passed=quality and heur)

        records = [rec(i, i % 3 != 0, i % 7 != 0) for i in range(70)]
        run = make_run(records, run_id="run-2")
        by = run.by_case()
        human = lambda i: i % 4 != 0
        old = [{"case_id": f"c{i:03d}", "label": "pass" if human(i) else "fail", "output_sha": by[f"c{i:03d}"]["output"]["sha"], "labeler": "ann", "run_id": "run-1"} for i in range(0, 30)]
        new_ids = [c for c in by if int(c[1:]) >= 30]
        new = [{"case_id": c, "label": "pass" if human(int(c[1:])) else "fail", "output_sha": by[c]["output"]["sha"], "labeler": "ann", "run_id": "run-2",
                "stratified_on": "overall", "stratum": calibration.stratum_token("run-2", "overall", c, calibration.stratum_of(by[c]["passed"]))} for c in new_ids]
        cal = calibration.calibrate(run, old + new, judge="quality", **FAST)
        self.assertEqual(cal["strata"]["estimator"], "carried_census")
        self.assertEqual(cal["n"], 70)
        self.assertEqual(sum(cal["strata"]["uncovered"].values()), 40)  # the cases without a carried label were all labeled on this run


class FewshotLoopTests(TempDirCase):
    """The whole rubric-iteration loop: calibrate, export the few-shot examples and the holdout, change the judge, label only what is new,
    calibrate again with the holdout left out."""

    def test_the_loop_runs_end_to_end(self):
        human = lambda i: i % 4 != 0
        run1 = world_run("run-1", 80, verdict=lambda i: i % 3 != 0)
        labels1 = label_rows(run1, human)
        cal1 = calibration.calibrate(run1, labels1, **FAST)
        fewshot = calibration.export_fewshot(cal1, run1, self.dir / "fewshot.yaml", seed=4)
        holdout = set(calibration.load_holdout_ids(fewshot))
        self.assertTrue(holdout)
        # the rubric got the examples: the judge changes its mind on some cases, the outputs do not change
        run2 = world_run("run-2", 80, verdict=lambda i: (i % 3 != 0) or i % 5 == 0)
        self.assertEqual(labeling.plan_relabel(run2, labels1)["uncovered"], 0)  # nothing new to label: every output already has a label
        only_old = calibration.calibrate(run2, labels1, exclude_case_ids=holdout, **FAST)
        self.assertEqual(only_old["strata"]["estimator"], "carried_census")  # every case was labeled on run 1 and every output is unchanged
        self.assertEqual(only_old["strata"]["uncovered"], {"judge_pass": 0, "judge_fail": 0})
        self.assertEqual(sum(only_old["strata"]["carried"].values()), 80 - len(holdout))
        self.assertTrue(all(w > 1 for w in only_old["strata"]["carried_weights"].values()))  # the kept labels stand for the held-out cases
        self.assertEqual(only_old["n_excluded_holdout"], len(holdout))
        self.assertEqual(only_old["n_carried_cases"], 80 - len(holdout))
        self.assertTrue(set(only_old["labeled_case_ids"]).isdisjoint(holdout))

    def test_labels_of_a_sample_plus_a_new_sample_after_the_judge_changed(self):
        human = lambda i: i % 4 != 0
        run1 = world_run("run-1", 120, verdict=lambda i: i % 3 != 0)
        rows1 = label_rows(run1, human, [r["case_id"] for r in labeling.select_sample(run1, 60, seed=1)])
        cal1 = calibration.calibrate(run1, rows1, **FAST)
        fewshot = calibration.export_fewshot(cal1, run1, self.dir / "fewshot.yaml", seed=4)
        holdout = set(calibration.load_holdout_ids(fewshot))
        run2 = world_run("run-2", 120, verdict=lambda i: (i % 3 != 0) or i % 5 == 0)
        chosen = labeling.select_sample(run2, 30, seed=2, previous_labels=rows1)
        rows2 = label_rows(run2, human, [r["case_id"] for r in chosen], run_id="run-2")
        cal2 = calibration.calibrate(run2, rows1 + rows2, exclude_case_ids=holdout, **FAST)
        self.assertEqual(cal2["strata"]["estimator"], "carried_census")
        self.assertEqual(cal2["n_excluded_holdout"], len(holdout))
        self.assertEqual(cal2["n_carried_cases"], 60 - len(holdout))
        self.assertTrue(set(cal2["labeled_case_ids"]).isdisjoint(holdout))
        self.assertEqual(sum(cal2["strata"]["uncovered"].values()), 120 - 60)  # the cases no earlier label covers; the held-out ones are not among them


class ReportingTests(unittest.TestCase):
    def test_markdown_says_how_many_labels_were_carried_over(self):
        old = label_rows(world_run("run-1", 60), lambda i: i % 4 != 0, [f"c{i:03d}" for i in range(0, 30)])
        run2 = world_run("run-2", 60)
        new = label_rows(run2, lambda i: i % 4 != 0, [f"c{i:03d}" for i in range(30, 60)], run_id="run-2")
        cal = calibration.calibrate(run2, old + new, **FAST)
        text = calibration.to_markdown(cal)
        self.assertIn("30 carried over from an earlier run", text)
        self.assertNotIn("carried over", calibration.to_markdown(calibration.calibrate(run2, new, **FAST)))


# ------------------------------------------------------------------------------------------------- judges_sha, stale
class StaleCalibrationTests(unittest.TestCase):
    def run_with(self, sha, run_id="run-test"):
        run, labels = confusion_scenario(tp=25, fn=2, fp=3, tn=15)
        run.summary["judges_sha"] = sha
        return run, labels

    def test_the_calibration_records_the_judge_fingerprint_of_its_run(self):
        run, labels = self.run_with("0123456789ab")
        self.assertEqual(calibration.calibrate(run, labels, **FAST)["judges_sha"], "0123456789ab")
        run.summary.pop("judges_sha")
        self.assertIsNone(calibration.calibrate(run, labels, **FAST)["judges_sha"])

    def test_a_calibration_made_for_other_judges_is_stale(self):
        run, labels = self.run_with("aaaaaaaaaaaa")
        cal = calibration.calibrate(run, labels, **FAST)
        later, _ = self.run_with("bbbbbbbbbbbb")
        text = calibration.stale_calibration(cal, later)
        self.assertIn("calibration was made for a different judge configuration", text)
        self.assertIn("aaaaaaaaaaaa", text)
        self.assertIn("bbbbbbbbbbbb", text)
        self.assertEqual(text.count("calibration was made for a different judge configuration"), 1)
        self.assertIn(calibration.STALE_CALIBRATION_TEXT, text)

    def test_the_same_fingerprint_is_fine(self):
        run, labels = self.run_with("aaaaaaaaaaaa")
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertIsNone(calibration.stale_calibration(cal, run))

    def test_nothing_is_said_when_a_side_has_no_fingerprint(self):
        run, labels = self.run_with("aaaaaaaaaaaa")
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertIsNone(calibration.stale_calibration({k: v for k, v in cal.items() if k != "judges_sha"}, run))  # an older calibration.json
        self.assertIsNone(calibration.stale_calibration(cal, {"run_id": "x"}))
        self.assertIsNone(calibration.stale_calibration(cal, None))
        self.assertIsNone(calibration.stale_calibration({"judges_sha": None}, run))

    def test_a_run_a_summary_or_a_plain_sha_can_be_compared(self):
        cal = {"judges_sha": "aaaaaaaaaaaa"}
        run, _ = self.run_with("bbbbbbbbbbbb")
        for other in (run, run.summary, "bbbbbbbbbbbb"):
            self.assertIn("different judge configuration", calibration.stale_calibration(cal, other))
        self.assertIsNone(calibration.stale_calibration(cal, "aaaaaaaaaaaa"))

    def test_works_on_a_calibration_read_back_from_json(self):
        run, labels = self.run_with("aaaaaaaaaaaa")
        cal = json.loads(json.dumps(calibration.calibrate(run, labels, **FAST)))
        later, _ = self.run_with("cccccccccccc")
        self.assertIsNotNone(calibration.stale_calibration(cal, later))
        self.assertIsNone(calibration.stale_calibration(cal, run))


# --------------------------------------------------------------------------------------------------- CSV case ids
TRICKY_IDS = [
    "plain", "a=b", "x-1", "=cmd|' /C calc'!A0", "+1+1", "-2+3", "@SUM(1,1)", "\t=1", "\r=1", "'", "''", "'a", "'=x", "''=x", "'''-x", "'+", "=", "-", "@",
    "'\t=1", "é=ü", "", " =x", "=x ", "'abc'", "0=1",
]


class CsvCaseIdTests(TempDirCase):
    def test_neutralize_and_restore_are_exact_inverses(self):
        for text in TRICKY_IDS + ["".join(random.Random(i).choice("='+-@\t\rab ") for _ in range(random.Random(i).randint(0, 5))) for i in range(3000)]:
            self.assertEqual(calibration.restore_csv_id(calibration.neutralize_csv_id(text)), text, repr(text))

    def test_a_neutralised_id_never_starts_with_a_formula_character(self):
        for text in TRICKY_IDS:
            safe = calibration.neutralize_csv_id(text)
            self.assertNotIn(safe[:1], list("=+-@\t\r"), repr(text))

    def test_an_id_that_is_not_dangerous_is_untouched_and_a_dangerous_one_gets_one_apostrophe(self):
        for text in ("plain", "a=b", "x-1", "'abc'", "é=ü", ""):
            self.assertEqual(calibration.neutralize_csv_id(text), text)
        self.assertEqual(calibration.neutralize_csv_id("=1+1"), "'=1+1")
        self.assertEqual(calibration.neutralize_csv_id("'=1+1"), "''=1+1")  # an id that already starts with an apostrophe stays recoverable

    def test_an_id_an_older_version_wrote_untouched_reads_back_as_it_was(self):
        for text in ("=old", "-1", "@x", "plain", "'plain"):
            self.assertEqual(calibration.restore_csv_id(text) if text.startswith("'") and len(text) > 1 and text[1] in "=+-@\t\r" else text, text)
        self.assertEqual(calibration.restore_csv_id("=old"), "=old")
        self.assertEqual(calibration.restore_csv_id("'plain"), "'plain")

    def formula_run(self):
        ids = ["=cmd|' /C calc'!A0", "+SUM(A1)", "-5", "@evil", "normal-1", "'=quoted", "\t=tab", "a-b"]
        records = [make_record(cid, i % 2 == 0, text=f"out {i}") for i, cid in enumerate(ids)]
        return make_run(records, run_id="run-csv"), ids

    def test_the_exported_csv_holds_no_formula_in_the_case_id_column(self):
        run, ids = self.formula_run()
        out = labeling.export_labeling_csv(run, self.dir / "sheet.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            cells = [row["case_id"] for row in csv.DictReader(fh)]
        self.assertEqual(len(cells), len(ids))
        self.assertTrue(all(cell[:1] not in list("=+-@\t\r") for cell in cells), cells)
        self.assertEqual({calibration.restore_csv_id(c) for c in cells}, set(ids))

    def test_a_filled_in_csv_comes_back_with_the_original_ids_and_matches_the_run(self):
        run, ids = self.formula_run()
        out = labeling.export_labeling_csv(run, self.dir / "sheet.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            row["label"] = "pass"
        with open(out, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        loaded = calibration.load_labels(out)
        self.assertEqual({r["case_id"] for r in loaded}, set(ids))  # including the ones that start with a tab or an apostrophe
        cal = calibration.calibrate(run, loaded, **FAST)
        self.assertEqual((cal["n"], cal["n_unmatched"]), (len(ids), 0))
        # and the same ids are what covered_case_ids sees, so a formula-like id is not exported twice
        self.assertEqual(labeling.covered_case_ids(run, out), frozenset(ids))

    def test_semicolon_csv_saved_by_excel_keeps_the_neutralised_ids_working(self):
        run, ids = self.formula_run()
        out = labeling.export_labeling_csv(run, self.dir / "sheet.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            row["label"] = "fail"
        path = self.dir / "excel.csv"
        with open(path, "w", encoding="cp1252", errors="replace", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual({r["case_id"] for r in calibration.load_labels(path)}, set(ids))

    def test_jsonl_ids_are_never_rewritten(self):
        path = write_jsonl(self.dir / "l.jsonl", [{"case_id": "'=x", "label": "pass"}, {"case_id": "=x", "label": "fail"}])
        self.assertEqual([r["case_id"] for r in calibration.load_labels(path)], ["'=x", "=x"])

    def test_other_text_columns_are_still_neutralised(self):
        run, _ = self.formula_run()
        out = labeling.export_labeling_csv(run, self.dir / "sheet.csv")
        with open(out, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                for column in ("tags", "input", "reference", "expected_behavior", "output"):
                    self.assertNotIn(row[column][:1], list("=+-@\t\r"), column)


# -------------------------------------------------------------------------------------------------- labels budget
def _fails_with_probability(n: int, p_fail: float, probability: float) -> int:
    """The largest number F such that P(Binomial(n, p_fail) >= F) >= ``probability`` (0 when even one fail is rarer than that)."""
    from math import comb

    pmf = [comb(n, k) * p_fail**k * (1 - p_fail) ** (n - k) for k in range(n + 1)]
    best, tail = 0, 1.0
    for fails in range(n + 1):
        if tail >= probability:
            best = fails
        tail -= pmf[fails]
    return best


class LabelsBudgetTests(unittest.TestCase):
    def test_the_table_has_a_row_per_human_pass_rate_and_says_where_it_comes_from(self):
        table = calibration.labels_budget_table()
        self.assertEqual([row["human_pass_rate"] for row in table["rows"]], sorted(row["human_pass_rate"] for row in table["rows"]))
        self.assertGreaterEqual(len(table["rows"]), 4)
        for key in ("judge", "how_to_read", "source", "rows", "python", "reps"):
            self.assertIn(key, table)
        for row in table["rows"]:
            self.assertEqual(set(row["census"]), {"usable", "trustworthy", "tnr_reliable"})

    def test_the_census_figures_are_what_the_code_does(self):
        # Every census figure against the real calibrate on a perfect judge with every case labeled. The number of human fails is random
        # (Binomial over the run), and the outcome holds from a minimum number of fails on, so "at least 80% of the runs reach it" means
        # that the 20th percentile of the number of fails is enough: true for a run of the figure's size, false one case smaller.
        table = {row["human_pass_rate"]: row for row in calibration.labels_budget_table()["rows"]}
        flags = {"usable": lambda c: c["kappa_decision"] in ("usable", "trustworthy"), "trustworthy": lambda c: c["kappa_decision"] == "trustworthy",
                 "tnr_reliable": lambda c: c["tnr_reliable"]}
        for hp in sorted(table):
            for key, flag in flags.items():
                first = table[hp]["census"][key]
                for n, expected in ((first - 1, False), (first, True)):
                    fails = _fails_with_probability(n, 1 - hp, 0.8)  # P(number of human fails >= this) is at least 80%
                    records = [make_record(f"c{i:03d}", i >= fails) for i in range(n)]
                    labels = [label(r["case_id"], "pass" if r["passed"] else "fail") for r in records]
                    self.assertEqual(flag(calibration.calibrate(make_run(records), labels, seed=0)), expected, (hp, key, n, fails))

    def test_the_number_of_human_fails_matters_so_the_table_does_not_fix_it(self):
        # The 200-case run at 90% human pass: 20 fails is the average, but a run with 12 fails does not get there at the same label count.
        def trustworthy(fails, labels_made):
            records = [make_record(f"c{i:03d}", i >= fails) for i in range(200)]
            run = make_run(records)
            rows = []
            for r in labeling.select_sample(run, labels_made, seed=1):  # the labeling tool's sample, stratified on the verdict
                token = calibration.stratum_token(run.run_id, "overall", r["case_id"], calibration.stratum_of(r["passed"]))
                rows.append(dict(label(r["case_id"], "pass" if r["passed"] else "fail"), stratified_on="overall", stratum=token))
            return calibration.calibrate(run, rows, seed=0)["kappa_decision"] == "trustworthy"

        self.assertTrue(trustworthy(20, 100))
        self.assertFalse(trustworthy(12, 100))
        # and the published figure is the one for the 20th percentile of the number of fails, not for the average run
        row = {r["human_pass_rate"]: r for r in calibration.labels_budget_table()["rows"]}[0.9]
        self.assertGreaterEqual(row["subset"]["200"]["trustworthy"], 115)
        self.assertGreaterEqual(row["subset"]["400"]["trustworthy"], 125)

    def test_the_figures_never_get_easier_as_humans_pass_more(self):
        rows = calibration.labels_budget_table()["rows"]
        for key in ("usable", "trustworthy", "tnr_reliable"):
            needed = [row["census"][key] for row in rows]
            self.assertEqual(needed, sorted(needed), key)

    def test_every_figure_comes_with_its_spread_over_the_bootstrap_seeds(self):
        for row in calibration.labels_budget_table()["rows"]:
            for flag, figure in row["census"].items():
                low, high = row["spread"]["census"][flag]
                self.assertTrue(low <= figure <= high, (row["human_pass_rate"], flag))  # the figure is the one of bootstrap seed 0
            for size, figures in row["subset"].items():
                for flag, figure in figures.items():
                    low, high = row["spread"]["subset"][size][flag]
                    if figure is None:
                        self.assertEqual((low, high), (None, None))
                    else:  # rounded up to a multiple of 5 from the largest of the seeds: never below any seed, at most 4 labels above
                        self.assertEqual(figure % 5, 0)
                        self.assertTrue(high <= figure < high + 5, (row["human_pass_rate"], size, flag))

    def test_the_text_says_the_human_fails_are_random_and_that_it_is_a_floor(self):
        table = calibration.labels_budget_table()
        self.assertIn("Binomial", table["how_to_read"])
        self.assertIn("floor", table["how_to_read"])
        self.assertIn("80%", table["how_to_read"])
        self.assertIn("random", table["judge"])
        self.assertIn("Binomial", table["reps"]["census"])
        self.assertIn("Binomial", table["reps"]["subset_perfect_judge"])
        md = calibration.labels_budget_markdown()
        self.assertIn("more cases", md)  # a 200-case run at 95% human pass cannot reach trustworthy even if every case is labeled
        self.assertIn("not reached", md)

    def test_a_caller_cannot_change_the_table_for_the_next_one(self):
        first = calibration.labels_budget_table()
        first["rows"].clear()
        first["how_to_read"] = "x"
        self.assertTrue(calibration.labels_budget_table()["rows"])
        self.assertNotEqual(calibration.labels_budget_table()["how_to_read"], "x")

    def test_markdown_renders_the_table(self):
        text = calibration.labels_budget_markdown()
        self.assertIn("| Human pass rate |", text)
        self.assertGreaterEqual(text.count("\n|"), 6)
        self.assertIn("perfect", text.lower())

    def test_hint_for_a_human_pass_rate_picks_the_row_at_or_above_it(self):
        rows = calibration.labels_budget_table()["rows"]
        self.assertEqual(calibration.labels_budget_hint(0.5)["census"], rows[0]["census"])
        self.assertEqual(calibration.labels_budget_hint(0.96)["human_pass_rate"], rows[-1]["human_pass_rate"])
        self.assertIsNone(calibration.labels_budget_hint(None))


# ------------------------------------------------------------------------------------------------------ wording
class ScreeningFlagWordingTests(unittest.TestCase):
    def trustworthy(self):
        records = [make_record(f"c{i:03d}", i < 25) for i in range(50)]
        run = make_run(records)
        labels = [label(r["case_id"], "pass" if r["passed"] else "fail") for r in records]
        return calibration.calibrate(run, labels, **FAST)

    def test_a_trustworthy_judge_is_told_the_tnr_flag_is_a_screen(self):
        cal = self.trustworthy()
        self.assertEqual(cal["kappa_decision"], "trustworthy")
        self.assertIn("passed the screening check", cal["recommendation"])
        self.assertIn("not a guarantee", cal["recommendation"])
        self.assertNotIn("true-negative rate is reliable", cal["recommendation"])
        md = calibration.to_markdown(cal)
        self.assertIn("screening check passed", md)
        self.assertNotIn("TNR reliable", md)

    def test_the_rule_is_documented_as_a_screening_flag_in_the_output(self):
        rule = self.trustworthy()["tnr_reliable_rule"]
        self.assertEqual(rule["kind"], "screening flag")
        self.assertIn("not a guarantee", rule["meaning"])
        self.assertIn("selects lucky reads", rule["meaning"])
        self.assertEqual(rule["meaning"], calibration.TNR_SCREEN_MEANING)

    def test_when_the_screen_fails_the_message_says_so_and_never_calls_it_unreliable_as_a_fact(self):
        run, labels = confusion_scenario(tp=30, fn=1, fp=1, tn=6)  # 7 human fails: too few
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertFalse(cal["tnr_reliable"])
        texts = [cal["recommendation"]] + cal["warnings"]
        self.assertTrue(any("did not pass the screening check" in t for t in texts))
        self.assertFalse(any("is unreliable" in t for t in texts))
        for reason, text in calibration._TNR_UNRELIABLE_TEXT.items():
            self.assertNotIn("unreliable", text, reason)

    def test_the_held_back_judge_is_told_the_screen_not_passed(self):
        text = calibration._DECISION_TEXT["trustworthy"]
        self.assertIn("screening check", text)

    def test_the_few_fails_message_points_at_the_labels_budget(self):
        message = calibration._tnr_message("few_human_fails", 4)
        self.assertIn("labels budget", message)


if __name__ == "__main__":
    unittest.main()
