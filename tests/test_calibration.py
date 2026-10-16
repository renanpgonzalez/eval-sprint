import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evalsprint import calibration
from evalsprint.runner import RunResult, summarize
from evalsprint.stats import kappa_label


# ------------------------------------------------------------------ synthetic runs
def sha_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def make_record(
    case_id, judge_pass, *, text=None, input="What is the refund window?", reasoning=None, judges=None, passed=None, outcome=None
):
    """One result record per the ARCHITECTURE.md contract; a single judge named "llm" by default.

    ``outcome`` ("pass" | "fail" | "error") is written only when given: records of older runs have none."""
    text = f"answer for {case_id}" if text is None else text
    if judges is None:
        judges = [
            {
                "judge": "llm",
                "passed": judge_pass,
                "score": 1.0 if judge_pass else 0.0,
                "reasoning": reasoning or ("all criteria passed" if judge_pass else "grounded: invented a policy"),
                "details": {},
                "cost_usd": 0.0,
                "usage": {},
                "error": None,
            }
        ]
    record = {
        "case_id": case_id,
        "case": {
            "id": case_id,
            "input": input,
            "reference": None,
            "expected_behavior": "answer",
            "tags": [],
            "checks": [],
            "metadata": {},
        },
        "output": {
            "text": text,
            "sha": sha_of(text),
            "model": "mock",
            "usage": {},
            "cost_usd": None,
            "latency_s": None,
            "meta": {},
            "error": None,
        },
        "judges": judges,
        "passed": judge_pass if passed is None else passed,
    }
    if outcome is not None:
        record["outcome"] = outcome
        record["passed"] = outcome == "pass"
    return record


def make_run(records, run_id="run-test"):
    return RunResult(run_dir=None, summary={"run_id": run_id, **summarize(records)}, results=records)


def label(case_id, value, *, labeler=None, sha=None, note=""):
    return {"case_id": case_id, "label": value, "output_sha": sha, "note": note, "labeler": labeler, "run_id": None}


def confusion_scenario(tp, fn, fp, tn):
    """A run whose every case is labeled, with exactly this human-vs-judge confusion matrix."""
    records, labels = [], []
    for kind, n, judge_pass, human in (
        ("tp", tp, True, "pass"),
        ("fn", fn, False, "pass"),
        ("fp", fp, True, "fail"),
        ("tn", tn, False, "fail"),
    ):
        for i in range(n):
            case_id = f"{kind}-{i:03d}"
            records.append(make_record(case_id, judge_pass))
            labels.append(label(case_id, human))
    return make_run(records), labels


def with_unlabeled(run, n_pass, n_fail):
    """The same run plus cases nobody labeled (the judge passed n_pass of them and failed n_fail): the labels become a sample."""
    extra = [make_record(f"xp{i:03d}", True) for i in range(n_pass)] + [make_record(f"xf{i:03d}", False) for i in range(n_fail)]
    return make_run(list(run.results) + extra, run.summary["run_id"])


FAST = {"iters": 300}  # bootstrap draws; plenty for structural assertions and keeps tests quick


# --------------------------------------------------------------------- load_labels
class LoadLabelsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, content, mode="w", **kw):
        path = self.dir / name
        with open(path, mode, **kw) as fh:
            fh.write(content)
        return path

    def test_jsonl_normalises_labels_and_skips_bad_rows(self):
        lines = [
            '{"case_id": "a", "label": "PASS", "labeler": "ana", "output_sha": "abc", "note": "ok", "run_id": "r1"}',
            '{"case_id": "b", "label": 0}',
            "this is not json",
            "[1, 2]",
            '{"label": "pass"}',
            '{"case_id": "c", "label": "maybe"}',
            '{"case_id": "d", "label": ""}',
            "",
            '{"case_id": "e", "label": true}',
            '{"case_id": "f", "label": " No "}',
        ]
        path = self.write("labels.jsonl", "\n".join(lines))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            labels = calibration.load_labels(path)
        self.assertEqual(
            [(x["case_id"], x["label"]) for x in labels], [("a", "pass"), ("b", "fail"), ("e", "pass"), ("f", "fail")]
        )
        self.assertEqual(labels.n_invalid, 4)
        self.assertEqual(labels.n_unlabeled, 1)
        self.assertIn("skipped 4", stderr.getvalue())
        first = labels[0]
        self.assertEqual((first["labeler"], first["output_sha"], first["note"], first["run_id"]), ("ana", "abc", "ok", "r1"))
        self.assertIsNone(labels[1]["labeler"])
        self.assertIsNone(labels[1]["output_sha"])

    def test_every_accepted_spelling(self):
        for raw, expected in [
            ("pass", "pass"),
            ("Fail", "fail"),
            (1, "pass"),
            (0, "fail"),
            ("1", "pass"),
            ("0", "fail"),
            (True, "pass"),
            (False, "fail"),
            ("YES", "pass"),
            ("no", "fail"),
            ("true", "pass"),
            ("FALSE", "fail"),
        ]:
            self.assertEqual(calibration.normalize_label(raw), expected, raw)
        self.assertIsNone(calibration.normalize_label("unsure"))

    def test_csv_ignores_extra_columns_and_bom(self):
        content = (
            "case_id,tags,input,output,output_sha,run_id,label,note,labeler\n"
            "a,x; y,hi,out,abc,r1,PASS,fine,ana\n"
            "b,,q,out,def,r1,,,\n"
            "c,,q,out,ghi,r1,fail,bad answer,ana\n"
        )
        path = self.write("labels.csv", content, encoding="utf-8-sig")
        labels = calibration.load_labels(path)
        self.assertEqual(
            [(x["case_id"], x["label"], x["note"]) for x in labels], [("a", "pass", "fine"), ("c", "fail", "bad answer")]
        )
        self.assertEqual(labels.n_unlabeled, 1)
        self.assertEqual(labels.n_invalid, 0)

    def test_csv_saved_by_european_excel(self):
        # Semicolon-separated and cp1252-encoded, as Excel writes "CSV" in many locales.
        content = "case_id;label;note\nc-1;fail;resposta inválida\nc-2;pass;\n"
        path = self.write("labels.csv", content.encode("cp1252"), mode="wb")
        labels = calibration.load_labels(path)
        self.assertEqual([(x["case_id"], x["label"]) for x in labels], [("c-1", "fail"), ("c-2", "pass")])
        self.assertEqual(labels[0]["note"], "resposta inválida")

    def test_csv_without_label_column_counts_every_row_invalid(self):
        path = self.write("labels.csv", "case_id,note\na,x\nb,y\n")
        with contextlib.redirect_stderr(io.StringIO()):
            labels = calibration.load_labels(path)
        self.assertEqual((len(labels), labels.n_invalid), (0, 2))

    def test_unknown_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            calibration.load_labels(self.write("labels.txt", "x"))


# ------------------------------------------------------------------ core metrics
class MetricsTests(unittest.TestCase):
    def test_perfect_agreement(self):
        run, labels = confusion_scenario(tp=25, fn=0, fp=0, tn=15)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(cal["judge"], "overall")
        self.assertEqual(cal["run_id"], "run-test")
        self.assertEqual(cal["n"], 40)
        self.assertEqual(cal["agreement"], 1.0)
        self.assertEqual(cal["kappa"], 1.0)
        self.assertEqual(cal["kappa_label"], "almost perfect")
        self.assertEqual((cal["tpr"], cal["tnr"]), (1.0, 1.0))
        self.assertEqual(
            cal["confusion"],
            {"human_pass_judge_pass": 25, "human_pass_judge_fail": 0, "human_fail_judge_pass": 0, "human_fail_judge_fail": 15},
        )
        self.assertAlmostEqual(cal["corrected_pass_rate"], cal["run_observed_pass_rate"])
        self.assertEqual(cal["disagreements"], [])
        # 15 human fails, every one caught: the true-negative rate is measured (lower bound 0.76, half-width 0.12), so a perfect
        # judge with 40 labels is certified (v0.1 could never certify it: it demanded 5 missed bad outputs).
        self.assertTrue(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "ok")
        self.assertEqual(cal["kappa_decision"], "trustworthy")
        self.assertIn("trustworthy for gating", cal["recommendation"])
        self.assertNotIn("not called trustworthy", cal["recommendation"])
        self.assertNotIn("provisional", cal["recommendation"])

    def test_known_confusion_matrix_by_hand(self):
        # 40 labeled cases: TP=20 FN=5 FP=6 TN=9 (25 human pass, 15 human fail; judge passes 26).
        run, labels = confusion_scenario(tp=20, fn=5, fp=6, tn=9)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(
            cal["confusion"],
            {"human_pass_judge_pass": 20, "human_pass_judge_fail": 5, "human_fail_judge_pass": 6, "human_fail_judge_fail": 9},
        )
        self.assertAlmostEqual(cal["agreement"], 29 / 40)
        # po = 0.725, pe = (25/40)(26/40) + (15/40)(14/40) = 0.5375 -> kappa = 0.1875 / 0.4625
        self.assertAlmostEqual(cal["kappa"], 0.1875 / 0.4625)
        self.assertEqual(cal["kappa_label"], kappa_label(cal["kappa"]))
        self.assertAlmostEqual(cal["tpr"], 20 / 25)
        self.assertAlmostEqual(cal["tnr"], 9 / 15)
        self.assertAlmostEqual(cal["human_pass_rate"], 25 / 40)
        self.assertAlmostEqual(cal["judge_pass_rate_labeled"], 26 / 40)
        self.assertAlmostEqual(cal["run_observed_pass_rate"], 26 / 40)

    def test_rogan_gladen_correction(self):
        run, labels = confusion_scenario(tp=20, fn=5, fp=6, tn=9)
        cal = calibration.calibrate(run, labels, **FAST)
        # (p_obs + tnr - 1) / (tpr + tnr - 1) = (0.65 + 0.6 - 1) / (0.8 + 0.6 - 1) = 0.625
        self.assertAlmostEqual(cal["corrected_pass_rate"], 0.625)
        lo, hi = cal["corrected_pass_rate_ci"]
        self.assertLess(lo, 0.625)
        self.assertGreater(hi, 0.625)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        klo, khi = cal["kappa_ci"]
        self.assertLess(klo, cal["kappa"])
        self.assertGreater(khi, cal["kappa"])

    def test_stratified_sample_is_reweighted_to_the_runs_verdict_mix(self):
        # 100 cases: the judge passes 80 and fails 20. Humans label 20 of each (a stratified sample):
        # among judge-pass cases 18/20 are human pass, among judge-fail cases 5/20 are human pass.
        records = [make_record(f"c{i:03d}", i < 80) for i in range(100)]
        labels = [label(f"c{i:03d}", "pass" if i < 18 else "fail") for i in range(20)]
        labels += [label(f"c{i:03d}", "pass" if i < 85 else "fail") for i in range(80, 100)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertEqual(
            cal["confusion"],
            {"human_pass_judge_pass": 18, "human_pass_judge_fail": 5, "human_fail_judge_pass": 2, "human_fail_judge_fail": 15},
        )
        # Population joint: P(pass,pass) = .8*.9, P(pass,fail) = .2*.25, P(fail,pass) = .8*.1, P(fail,fail) = .2*.75
        self.assertAlmostEqual(cal["tpr"], 0.72 / 0.77)
        self.assertAlmostEqual(cal["tnr"], 0.15 / 0.23)
        self.assertAlmostEqual(cal["corrected_pass_rate"], 0.77)  # = P(human pass) = .72 + .05
        naive_tpr, naive_tnr = 18 / 23, 15 / 17
        self.assertGreater(abs(cal["tpr"] - naive_tpr), 0.1)
        self.assertGreater(abs(cal["tnr"] - naive_tnr), 0.2)
        self.assertEqual(cal["run_observed_pass_rate"], 0.8)
        self.assertEqual(cal["judge_pass_rate_labeled"], 0.5)

    def test_labels_on_judge_passes_only_leave_tnr_undefined(self):
        # The run has judge failures, but none was labeled: nothing tells us how the judge behaves there.
        records = [make_record(f"c{i}", i < 30) for i in range(40)]
        labels = [label(f"c{i}", "pass") for i in range(30)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertIsNone(cal["tpr"])
        self.assertIsNone(cal["tnr"])
        self.assertIsNone(cal["corrected_pass_rate"])
        self.assertIsNone(cal["corrected_pass_rate_ci"])
        self.assertIsNone(cal["kappa"])  # both raters said "pass" everywhere: kappa is undefined, not 1.0
        self.assertIsNone(cal["kappa_weighted"])  # and the run's judge-fail stratum has no label: the whole run cannot be estimated
        self.assertEqual(cal["kappa_decision"], "insufficient_data")
        self.assertEqual(cal["one_sided_labels"], "pass")
        self.assertIn("label cases the judge failed", cal["recommendation"])
        self.assertNotIn("revise the rubric", cal["recommendation"].lower())
        self.assertNotIn("trustworthy", cal["recommendation"])

    def test_no_human_failures_leaves_tnr_undefined_but_keeps_tpr(self):
        records = [make_record(f"c{i}", True) for i in range(20)] + [make_record("bad", False)]
        labels = [label(f"c{i}", "pass") for i in range(20)] + [label("bad", "pass")]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertIsNone(cal["tnr"])
        self.assertIsNone(cal["tnr_ci"])
        self.assertAlmostEqual(cal["tpr"], 20 / 21)

    def test_useless_judge_has_no_corrected_pass_rate(self):
        # Judge verdicts independent of the human label: Youden's J = 0, Rogan-Gladen is undefined.
        run, labels = confusion_scenario(tp=15, fn=15, fp=15, tn=15)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertAlmostEqual(cal["kappa"], 0.0)
        self.assertIsNone(cal["corrected_pass_rate"])
        self.assertIsNone(cal["corrected_pass_rate_ci"])

    def test_bootstrap_is_reproducible_for_a_seed(self):
        run, labels = confusion_scenario(tp=20, fn=5, fp=6, tn=9)
        a = calibration.calibrate(run, labels, seed=7, **FAST)
        b = calibration.calibrate(run, labels, seed=7, **FAST)
        c = calibration.calibrate(run, labels, seed=8, **FAST)
        self.assertEqual(a, b)
        self.assertNotEqual(a["kappa_ci"], c["kappa_ci"])

    def test_no_labels_at_all(self):
        run, _ = confusion_scenario(tp=3, fn=1, fp=1, tn=1)
        cal = calibration.calibrate(run, [], **FAST)
        self.assertEqual(cal["n"], 0)
        for key in (
            "agreement",
            "kappa",
            "kappa_ci",
            "tpr",
            "tnr",
            "corrected_pass_rate",
            "corrected_pass_rate_ci",
            "human_pass_rate",
        ):
            self.assertIsNone(cal[key], key)
        self.assertIn("No usable labels", cal["recommendation"])
        self.assertIn("No usable labels", calibration.to_markdown(cal))

    def test_empty_run(self):
        cal = calibration.calibrate(make_run([]), [label("x", "pass")], **FAST)
        self.assertEqual((cal["n"], cal["n_unmatched"]), (0, 1))
        self.assertIsNone(cal["run_observed_pass_rate"])


# ------------------------------------------------------------ matching and consensus
class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.records = [make_record(f"c{i:02d}", i % 2 == 0) for i in range(12)]
        self.run = make_run(self.records)

    def test_sha_mismatch_is_excluded_and_counted(self):
        good = self.records[0]["output"]["sha"]
        labels = [
            label("c00", "pass", sha=good),
            label("c01", "fail", sha="deadbeef0000"),  # labeled a different output
            label("c02", "pass", sha=self.records[2]["output"]["sha"].upper()),  # case-insensitive
            label("c04", "pass", sha=sha_of("answer for c04") + "f" * 52),  # full-length sha256 prefix-matches
            label("c06", "pass"),  # no sha: cannot be verified, kept
        ]
        cal = calibration.calibrate(self.run, labels, **FAST)
        self.assertEqual(cal["n_excluded_sha_mismatch"], 1)
        self.assertEqual(cal["n"], 4)
        self.assertEqual(cal["n_labels"], 5)
        self.assertIn("excluded because the AI output changed", cal["recommendation"])

    def test_unmatched_case_ids_are_counted(self):
        labels = [label("c00", "pass"), label("nope", "pass"), label("also-nope", "fail")]
        cal = calibration.calibrate(self.run, labels, **FAST)
        self.assertEqual(cal["n_unmatched"], 2)
        self.assertEqual(cal["n"], 1)

    def test_invalid_row_count_from_load_labels_is_surfaced(self):
        labels = calibration.LabelList([label("c00", "pass")], n_invalid=3)
        self.assertEqual(calibration.calibrate(self.run, labels, **FAST)["n_invalid_labels"], 3)
        self.assertEqual(calibration.calibrate(self.run, [label("c00", "pass")], **FAST)["n_invalid_labels"], 0)

    def test_hand_built_label_dicts_are_normalised_and_garbage_is_counted(self):
        labels = [
            {"case_id": "c00", "label": "PASS"},
            {"case_id": "c01", "label": False},
            {"case_id": "c02", "label": "meh"},
            {"label": "pass"},
        ]
        cal = calibration.calibrate(self.run, labels, **FAST)
        self.assertEqual(cal["n"], 2)
        self.assertEqual(cal["n_invalid_labels"], 2)
        self.assertEqual((cal["confusion"]["human_pass_judge_pass"], cal["confusion"]["human_fail_judge_fail"]), (1, 1))

    def test_majority_vote_and_ties(self):
        labels = [
            label("c00", "pass", labeler="ana"),
            label("c00", "pass", labeler="bo"),
            label("c00", "fail", labeler="cy"),  # majority pass
            label("c01", "fail", labeler="ana"),
            label("c01", "pass", labeler="bo"),  # tie -> excluded
            label("c02", "fail", labeler="ana"),
            label("c02", "fail", labeler="bo"),
            label("c02", "pass", labeler="cy"),  # majority fail
        ]
        cal = calibration.calibrate(self.run, labels, **FAST)
        self.assertEqual(cal["n"], 2)
        self.assertEqual(cal["n_tied_cases"], 1)
        # c00: human pass / judge pass; c02: human fail / judge pass (records with even index pass)
        self.assertEqual(cal["confusion"]["human_pass_judge_pass"], 1)
        self.assertEqual(cal["confusion"]["human_fail_judge_pass"], 1)
        self.assertEqual([d["case_id"] for d in cal["disagreements"]], ["c02"])

    def test_a_labelers_relabel_replaces_their_earlier_label(self):
        labels = [label("c00", "fail", labeler="Ana"), label("c00", "pass", labeler="ana"), label("c00", "pass", labeler="bo")]
        cal = calibration.calibrate(self.run, labels, **FAST)
        self.assertEqual(cal["n"], 1)
        self.assertEqual(cal["confusion"]["human_pass_judge_pass"], 1)
        self.assertEqual(cal["n_tied_cases"], 0)

    def test_inter_annotator_kappa_needs_ten_shared_cases(self):
        # ana and bo both label all 12 cases (6 pass / 6 fail each) and disagree on 2 of them.
        ana = [label(f"c{i:02d}", "pass" if i < 6 else "fail", labeler="ana") for i in range(12)]
        bo_values = ["pass" if i < 6 else "fail" for i in range(12)]
        bo_values[0], bo_values[11] = "fail", "pass"
        bo = [label(f"c{i:02d}", v, labeler="bo") for i, v in enumerate(bo_values)]
        cal = calibration.calibrate(self.run, ana + bo, **FAST)
        ia = cal["inter_annotator"]
        self.assertEqual(ia["labelers"], ["ana", "bo"])
        self.assertEqual(ia["n"], 12)
        self.assertAlmostEqual(ia["kappa"], (10 / 12 - 0.5) / 0.5)  # po = 10/12, pe = 0.5
        self.assertEqual(cal["n_tied_cases"], 2)  # the two disagreements are 1-1 ties for the judge comparison
        self.assertEqual(cal["n"], 10)
        self.assertIn("Human-human kappa", calibration.to_markdown(cal))

        fewer = calibration.calibrate(self.run, ana[:9] + bo[:9], **FAST)
        self.assertIsNone(fewer["inter_annotator"])
        self.assertIsNone(calibration.calibrate(self.run, ana, **FAST)["inter_annotator"])

    def test_low_inter_annotator_kappa_is_called_out(self):
        ana = [label(f"c{i:02d}", "pass" if i < 6 else "fail", labeler="ana") for i in range(12)]
        bo = [label(f"c{i:02d}", "pass" if i % 2 else "fail", labeler="bo") for i in range(12)]
        cal = calibration.calibrate(self.run, ana + bo, **FAST)
        self.assertLess(cal["inter_annotator"]["kappa"], 0.6)
        self.assertIn("pass/fail definition is ambiguous", cal["recommendation"])


# ------------------------------------------------------------- named judge / overall
class JudgeSelectionTests(unittest.TestCase):
    def setUp(self):
        def judges(llm_pass, regex_pass, llm_error=None):
            return [
                {
                    "judge": "llm",
                    "passed": llm_pass,
                    "score": None,
                    "reasoning": "" if llm_error else "llm says " + ("ok" if llm_pass else "bad"),
                    "details": {},
                    "cost_usd": 0,
                    "usage": {},
                    "error": llm_error,
                },
                {
                    "judge": "regex",
                    "passed": regex_pass,
                    "score": None,
                    "reasoning": "regex says " + ("ok" if regex_pass else "no match"),
                    "details": {},
                    "cost_usd": 0,
                    "usage": {},
                    "error": None,
                },
            ]

        self.records = [
            make_record("a", None, judges=judges(True, True), passed=True),
            make_record("b", None, judges=judges(True, False), passed=False),
            make_record("c", None, judges=judges(False, True), passed=False),
            make_record("d", None, judges=judges(False, True, llm_error="ValueError: unparsable"), passed=False),
        ]
        # a judge that only applies to some cases
        self.records.append(make_record("e", None, judges=[judges(True, True)[1]], passed=True))
        self.run = make_run(self.records)
        self.labels = [label(cid, "pass") for cid in "abcde"]

    def test_overall_uses_the_case_verdict_and_skips_errored_cases(self):
        cal = calibration.calibrate(self.run, self.labels, **FAST)
        self.assertEqual(cal["judge"], "overall")
        self.assertEqual(cal["n"], 4)  # "d" had a judge error: no verdict, not a judge fail (records without `outcome`)
        self.assertEqual(cal["n_no_judge_verdict"], 1)
        self.assertEqual(cal["confusion"]["human_pass_judge_pass"], 2)
        self.assertEqual(cal["confusion"]["human_pass_judge_fail"], 2)
        self.assertEqual(cal["n_run_errors"], 1)
        self.assertAlmostEqual(cal["run_observed_pass_rate"], 2 / 4)

    def test_named_judge_skips_cases_where_it_did_not_run_or_errored(self):
        cal = calibration.calibrate(self.run, self.labels, judge="llm", **FAST)
        self.assertEqual(cal["judge"], "llm")
        self.assertEqual(cal["n"], 3)  # a, b, c; d errored, e never ran
        self.assertEqual(cal["n_no_judge_verdict"], 2)
        self.assertEqual(cal["confusion"]["human_pass_judge_pass"], 2)
        self.assertEqual(cal["confusion"]["human_pass_judge_fail"], 1)
        self.assertAlmostEqual(cal["run_observed_pass_rate"], 2 / 3)  # errored/absent verdicts are not fails

    def test_unknown_judge_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            calibration.calibrate(self.run, self.labels, judge="ghost")
        self.assertIn("llm", str(ctx.exception))

    def test_overall_disagreement_shows_the_failing_judges_reasoning(self):
        cal = calibration.calibrate(self.run, self.labels, **FAST)
        reasoning = {d["case_id"]: d["judge_reasoning"] for d in cal["disagreements"]}
        self.assertEqual(reasoning["b"], "regex says no match")
        self.assertEqual(reasoning["c"], "llm says bad")
        self.assertNotIn("d", reasoning)  # errored: no verdict, so no disagreement


# ---------------------------------------------------------- disagreements / report
class DisagreementTests(unittest.TestCase):
    def test_fields_truncation_and_order(self):
        long_text = "x" * 2000
        records = [
            make_record("lenient-1", True, text=long_text, input={"question": "q" * 900}, reasoning="looked fine"),
            make_record("strict-1", False, reasoning="missing the citation"),
            make_record("lenient-2", True),
            make_record("agree", True),
        ]
        labels = [
            label("strict-1", "pass", labeler="ana", note="citation is not required here"),
            label("lenient-1", "fail", labeler="ana", note="invents a discount"),
            label("lenient-2", "fail"),
            label("agree", "pass"),
        ]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        ids = [d["case_id"] for d in cal["disagreements"]]
        self.assertEqual(ids, ["lenient-1", "lenient-2", "strict-1"])  # human fail / judge pass first
        first = cal["disagreements"][0]
        self.assertEqual((first["human"], first["judge"]), ("fail", "pass"))
        self.assertEqual(first["judge_reasoning"], "looked fine")
        self.assertEqual(first["note"], "invents a discount")
        self.assertLessEqual(len(first["output"]), 500)
        self.assertLessEqual(len(first["input"]), 500)
        self.assertTrue(first["output"].endswith("…"))
        self.assertTrue(first["input"].startswith('{"question"'))  # dict inputs are serialised
        last = cal["disagreements"][2]
        self.assertEqual((last["human"], last["judge"]), ("pass", "fail"))
        self.assertEqual(last["judge_reasoning"], "missing the citation")

    def test_notes_of_several_labelers_are_joined(self):
        records = [make_record("c", True)]
        labels = [
            label("c", "fail", labeler="ana", note="wrong"),
            label("c", "fail", labeler="bo", note="also wrong"),
            label("c", "pass", labeler="cy"),
        ]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertEqual(cal["disagreements"][0]["note"], "ana: wrong | bo: also wrong")


class RecommendationTests(unittest.TestCase):
    def cal(self, tp, fn, fp, tn):
        run, labels = confusion_scenario(tp, fn, fp, tn)
        return calibration.calibrate(run, labels, **FAST)

    def rec(self, tp, fn, fp, tn):
        return self.cal(tp, fn, fp, tn)["recommendation"]

    def test_small_sample_asks_for_more_labels_and_never_says_trustworthy(self):
        cal = self.cal(tp=8, fn=0, fp=0, tn=6)  # perfect agreement, but only 14 labels
        self.assertEqual(cal["kappa_decision"], "insufficient_data")
        text = cal["recommendation"]
        self.assertIn("Only 14 cases labeled", text)
        self.assertIn("50-100", text)
        self.assertIn("include failures", text)
        self.assertNotIn("trustworthy for gating", text)

    def test_few_human_failures_make_tnr_unreliable(self):
        cal = self.cal(tp=35, fn=0, fp=1, tn=4)
        text = cal["recommendation"]
        self.assertNotIn("Only 40", text)
        self.assertFalse(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "few_human_fails")
        self.assertIn("only 5 cases labeled fail by humans", text)
        self.assertIn("true-negative rate", text)
        self.assertIn("the ones the judge failed", text)
        self.assertIn("the ones it passed", text)

    def test_a_weak_true_negative_rate_is_called_low_not_imprecise(self):
        cal = self.cal(tp=40, fn=2, fp=5, tn=5)  # 10 human fails, the judge caught half of them
        self.assertFalse(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "low_tnr")
        self.assertIn("appears to miss more than 25% of the bad outputs", cal["recommendation"])
        self.assertNotIn("labeled fail by humans", cal["recommendation"])

    def test_an_imprecise_but_good_true_negative_rate_asks_for_more_labels(self):
        cal = self.cal(tp=25, fn=0, fp=5, tn=15)  # TNR 0.75 but the interval is +/-0.177 wide
        self.assertEqual(cal["tnr_reliable_reason"], "wide_ci")
        self.assertIn("too imprecise", cal["recommendation"])
        self.assertIn("within +/-18 points", cal["recommendation"])

    def test_high_kappa_with_a_tight_ci_is_trustworthy(self):
        cal = self.cal(tp=95, fn=3, fp=6, tn=90)  # kappa ~0.90, CI lower bound above 0.8
        self.assertEqual(cal["kappa_decision"], "trustworthy")
        self.assertGreaterEqual(cal["kappa_weighted_ci"][0], 0.8)
        self.assertIn("trustworthy for gating", cal["recommendation"])
        self.assertIn("true-negative rate passed the screening check", cal["recommendation"])  # a screen, not a guarantee (C20)
        self.assertIn("not a guarantee", cal["recommendation"])
        self.assertTrue(cal["tnr_reliable"])
        self.assertNotIn("Only", cal["recommendation"])

    def test_high_point_estimate_with_a_wide_ci_is_not_trustworthy(self):
        # 60 labels, kappa 0.87 (point estimate would have said "trustworthy"), but the CI reaches below 0.8
        cal = self.cal(tp=28, fn=2, fp=2, tn=28)
        self.assertGreaterEqual(round(cal["kappa"], 2), 0.8)
        self.assertLess(cal["kappa_weighted_ci"][0], 0.8)
        self.assertEqual(cal["kappa_decision"], "usable")
        self.assertNotIn("trustworthy for gating", cal["recommendation"])

    def test_middle_kappa_is_usable_with_fewshot_advice(self):
        cal = self.cal(tp=90, fn=10, fp=10, tn=90)  # kappa 0.8, CI 0.68-0.91: lower bound >= 0.6, < 0.8
        self.assertEqual(cal["kappa_decision"], "usable")
        text = cal["recommendation"]
        self.assertIn("usable", text)
        self.assertIn("few-shot `examples:`", text)
        self.assertNotIn("trustworthy", text)

    def test_ci_straddling_the_bar_is_inconclusive_and_says_how_many_labels_would_help(self):
        run, labels = confusion_scenario(tp=25, fn=5, fp=5, tn=25)  # point estimate 0.67 would have said "usable"
        cal = calibration.calibrate(with_unlabeled(run, 200, 200), labels, **FAST)  # the 60 labels are a sample of 460 cases
        self.assertEqual(cal["kappa_decision"], "inconclusive")
        self.assertFalse(cal["labels_needed_capped"])
        text = cal["recommendation"]
        self.assertIn("inconclusive", text)
        self.assertIn("Do not gate on it yet", text)
        self.assertIn(f"About {cal['labels_needed_estimate']} labels", text)
        self.assertGreater(cal["labels_needed_estimate"], 60)

    def test_when_every_labelable_case_is_labeled_it_says_to_add_cases_not_labels(self):
        cal = self.cal(tp=25, fn=5, fp=5, tn=25)  # 60 cases, all labeled: nothing left to label
        self.assertEqual(cal["kappa_decision"], "inconclusive")
        self.assertEqual((cal["n"], cal["n_labelable"], cal["labels_needed_estimate"]), (60, 60, 60))
        self.assertTrue(cal["labels_needed_capped"])
        text = cal["recommendation"]
        self.assertIn("All 60 labelable cases are already labeled: add cases or repeats for more precision", text)
        self.assertNotIn("About", text)

    def test_a_run_smaller_than_the_labels_needed_is_told_to_label_all_of_it(self):
        run, labels = confusion_scenario(tp=25, fn=5, fp=5, tn=25)
        cal = calibration.calibrate(with_unlabeled(run, 35, 35), labels, **FAST)  # 130 cases, 60 labeled
        self.assertEqual((cal["n"], cal["n_labelable"]), (60, 130))
        self.assertGreater(calibration._labels_needed(cal["n"], cal["kappa_weighted_ci"]), 130)  # what it would take without the cap
        self.assertEqual(cal["labels_needed_estimate"], 130)
        self.assertTrue(cal["labels_needed_capped"])
        self.assertIn("This run has only 130 labelable cases", cal["recommendation"])
        self.assertIn("label all 130 cases; add cases or repeats for more precision", cal["recommendation"])
        self.assertNotIn("About", cal["recommendation"])

    def test_low_kappa_says_do_not_gate(self):
        cal = self.cal(tp=45, fn=45, fp=45, tn=45)  # kappa 0.0
        self.assertEqual(cal["kappa_decision"], "not_trustworthy")
        text = cal["recommendation"]
        self.assertIn("do not gate on this judge", text)
        self.assertIn("rubric criteria", text)

    def test_messages_combine(self):
        text = self.rec(tp=3, fn=3, fp=3, tn=3)
        self.assertIn("Only 12 cases labeled:", text)
        self.assertIn("only 6 cases labeled fail by humans", text)
        self.assertNotIn("do not gate", text)  # too little data to say either way


class MarkdownTests(unittest.TestCase):
    def test_report_contains_metrics_matrix_recommendation_and_capped_disagreements(self):
        records = [make_record(f"bad-{i:02d}", True, reasoning=f"looks fine {i}") for i in range(14)]
        records += [make_record(f"ok-{i:02d}", True) for i in range(10)] + [make_record(f"low-{i:02d}", False) for i in range(6)]
        labels = [label(f"bad-{i:02d}", "fail", note=f"wrong {i}\nsecond line | pipe") for i in range(14)]
        labels += [label(f"ok-{i:02d}", "pass") for i in range(10)] + [label(f"low-{i:02d}", "fail") for i in range(6)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        md = calibration.to_markdown(cal)
        self.assertIn("## Judge calibration: overall (run `run-test`)", md)
        self.assertIn("**30** labeled cases used.", md)
        for text in (
            "| Agreement |",
            "| Cohen's kappa (whole run, weighted; the decision uses it) |",
            "| Cohen's kappa (labeled sample, unweighted; secondary) |",
            "| TPR",
            "| TNR",
            "corrected for judge errors",
            "| **Human fail** | 14 | 6 |",
            "**Recommendation:**",
        ):
            self.assertIn(text, md)
        self.assertIn("showing 10 of 14", md)
        self.assertEqual(md.count("human **fail**, judge **pass**"), 10)
        self.assertIn("wrong 0 second line | pipe", md)  # newlines are collapsed so the list stays readable
        self.assertNotIn("nan", md.lower())

    def test_report_survives_undefined_metrics_and_notes_exclusions(self):
        records = [make_record(f"c{i}", True) for i in range(5)]
        labels = [label(f"c{i}", "pass") for i in range(5)] + [label("ghost", "pass"), label("c0", "fail", sha="0" * 12)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        md = calibration.to_markdown(cal)
        self.assertIn("n/a", md)
        self.assertIn("1 labels on an output that has since changed", md)
        self.assertIn("1 labels with an unknown case id", md)

    def test_reweighting_is_disclosed(self):
        records = [make_record(f"c{i:03d}", i < 80) for i in range(100)]
        labels = [label(f"c{i:03d}", "pass") for i in range(20)] + [label(f"c{i:03d}", "fail") for i in range(80, 100)]
        md = calibration.to_markdown(calibration.calibrate(make_run(records), labels, **FAST))
        self.assertIn("re-weighted", md)
        run, full_labels = confusion_scenario(20, 5, 6, 9)
        self.assertNotIn("re-weighted", calibration.to_markdown(calibration.calibrate(run, full_labels, **FAST)))

    def test_output_is_json_serialisable(self):
        run, labels = confusion_scenario(20, 5, 6, 9)
        json.dumps(calibration.calibrate(run, labels, **FAST))


# ------------------------------------------------------ errors are not verdicts (C1 / C5)
class ErrorOutcomeTests(unittest.TestCase):
    def test_errored_cases_have_no_verdict_for_overall_and_named_judge(self):
        records = [make_record(f"ok{i}", True, outcome="pass") for i in range(6)]
        records += [make_record(f"bad{i}", False, outcome="fail") for i in range(4)]
        # A target error: the judge row looks fine, but the case has no verdict at all.
        records.append(make_record("err-target", True, outcome="error"))
        records.append(make_record("err-judge", False, outcome="error"))
        labels = [label(r["case_id"], "pass") for r in records]
        for judge in (None, "llm"):
            cal = calibration.calibrate(make_run(records), labels, judge=judge, **FAST)
            self.assertEqual(cal["n"], 10, judge)
            self.assertEqual(cal["n_no_judge_verdict"], 2, judge)
            self.assertEqual(cal["n_run_errors"], 2, judge)
            self.assertAlmostEqual(cal["run_observed_pass_rate"], 0.6, msg=judge)  # errors are not judge fails
            self.assertTrue(any("2 run cases ended in an error" in w for w in cal["warnings"]), judge)

    def test_legacy_records_without_outcome_treat_target_and_judge_errors_as_errors(self):
        target_err = make_record("t", False)
        target_err["output"]["error"] = "TimeoutError: upstream"
        judge_err = make_record("j", False)
        judge_err["judges"][0]["error"] = "ValueError: unparsable"
        records = [make_record(f"c{i}", i % 2 == 0) for i in range(6)] + [target_err, judge_err]
        self.assertEqual([calibration.record_outcome(r) for r in (target_err, judge_err, records[0], records[1])], ["error", "error", "pass", "fail"])
        cal = calibration.calibrate(make_run(records), [label(r["case_id"], "pass") for r in records], **FAST)
        self.assertEqual((cal["n"], cal["n_run_errors"]), (6, 2))

    def test_explicit_outcome_wins_over_passed(self):
        record = make_record("x", True)
        record["outcome"] = "fail"
        self.assertEqual(calibration.record_outcome(record), "fail")
        record["outcome"] = "error"
        self.assertEqual(calibration.record_outcome(record), "error")


# ------------------------------------------------- Horvitz-Thompson weights on recorded strata
def _two_judge_record(case_id, quality, heuristic):
    def result(name, passed):
        return {"judge": name, "passed": passed, "score": None, "reasoning": "", "details": {}, "cost_usd": 0, "usage": {}, "error": None}

    return make_record(case_id, None, judges=[result("quality", quality), result("heur", heuristic)], passed=quality and heuristic)


class HorvitzThompsonTests(unittest.TestCase):
    """100 cases, two judges, overall verdict = quality AND heuristic:
    60 overall-pass (p), 10 quality-pass but heuristic-fail (m), 30 quality-fail (q).
    The labeling sample is stratified on the OVERALL verdict (10 labels from each stratum), the workflow
    `evalsprint label` + `evalsprint calibrate --judge quality`."""

    def setUp(self):
        records = [_two_judge_record(f"p{i:02d}", True, True) for i in range(60)]
        records += [_two_judge_record(f"m{i:02d}", True, False) for i in range(10)]
        records += [_two_judge_record(f"q{i:02d}", False, True) for i in range(30)]
        self.run = make_run(records)
        human = {f"p{i:02d}": i != 9 for i in range(10)}  # p09 human fail
        human.update({"m00": True, "m01": False, "m02": False})
        human.update({f"q{i:02d}": i < 2 for i in range(7)})  # q00 q01 human pass; q02..q06 human fail
        self.human = human

    def labels(self, **extra):
        return [dict(label(cid, "pass" if ok else "fail"), **extra) for cid, ok in self.human.items()]

    def tool_labels(self, design="overall", judge=None):
        from evalsprint import labeling

        by_case = self.run.by_case()
        return [
            dict(
                label(cid, "pass" if ok else "fail"),
                stratified_on=design,
                stratum=labeling._stratum_code(self.run, by_case[cid], True, judge),
            )
            for cid, ok in self.human.items()
        ]

    def check_overall_strata_numbers(self, cal):
        # HT weights: overall-pass stratum 60/10 = 6 per label, overall-fail stratum 40/10 = 4 per label.
        # tp = 9*6 + 1*4, fp = 1*6 + 2*4, fn = 2*4, tn = 5*4
        self.assertAlmostEqual(cal["tpr"], 58 / 66)
        self.assertAlmostEqual(cal["tnr"], 20 / 34)
        self.assertAlmostEqual(cal["agreement_weighted"], 78 / 100)
        self.assertAlmostEqual(cal["human_pass_rate_weighted"], 66 / 100)
        self.assertAlmostEqual(cal["run_observed_pass_rate"], 70 / 100)  # the named judge's own pass rate
        self.assertAlmostEqual(cal["corrected_pass_rate"], (0.70 + 20 / 34 - 1) / (58 / 66 + 20 / 34 - 1))
        self.assertEqual(cal["strata"]["run_cases"], {"judge_pass": 60, "judge_fail": 40})
        self.assertEqual(cal["strata"]["labeled"], {"judge_pass": 10, "judge_fail": 10})
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 6.0, "judge_fail": 4.0})
        # the unweighted sample numbers are still reported as such
        self.assertEqual(cal["confusion"], {"human_pass_judge_pass": 10, "human_pass_judge_fail": 2, "human_fail_judge_pass": 3, "human_fail_judge_fail": 5})
        self.assertAlmostEqual(cal["agreement"], 15 / 20)

    def test_named_judge_uses_the_sampling_strata_recorded_by_the_tool(self):
        cal = calibration.calibrate(self.run, self.tool_labels(), judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertEqual(cal["strata"]["stratified_on"], "overall")
        self.assertFalse(any("do not record how they were sampled" in w for w in cal["warnings"]))

    def test_the_strata_recorded_are_opaque_and_the_plain_values_also_work(self):
        tool = self.tool_labels()
        self.assertTrue(all(row["stratum"].startswith("s-") for row in tool))
        self.assertTrue(all(row["stratum"] not in calibration.STRATA for row in tool))
        plain = self.labels(stratified_on="overall")
        for row in plain:
            row["stratum"] = calibration.stratum_of(self.run.by_case()[row["case_id"]]["passed"])
        cal = calibration.calibrate(self.run, plain, judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertEqual(cal["strata"]["source"], "recorded")

    def test_labels_without_a_recorded_design_fall_back_to_the_overall_verdict_with_a_warning(self):
        cal = calibration.calibrate(self.run, self.labels(), judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertEqual(cal["strata"]["source"], "recomputed")
        self.assertTrue(any("do not record how they were sampled" in w for w in cal["warnings"]))

    def test_no_warning_when_the_fallback_is_exact(self):
        # calibrating the OVERALL verdict from labels that predate `stratified_on`: the fallback is exactly right
        cal = calibration.calibrate(self.run, self.labels(), **FAST)
        self.assertFalse(any("do not record how they were sampled" in w for w in cal["warnings"]))

    def test_naive_weights_on_the_judges_own_verdict_would_be_different(self):
        # Stratifying on the named judge's own verdict (the pre-cycle-2 behaviour) gives another answer.
        cal = calibration.calibrate(self.run, self.labels(stratified_on="quality"), judge="quality", **FAST)
        w_pass, w_fail = 70 / 13, 30 / 7  # quality-pass: 13 labeled of 70, quality-fail: 7 labeled of 30
        tn, fp = 5 * w_fail, 3 * w_pass
        self.assertAlmostEqual(cal["tnr"], tn / (tn + fp))
        self.assertGreater(abs(cal["tnr"] - 20 / 34), 0.01)
        self.assertEqual(cal["strata"]["stratified_on"], "quality")

    def test_random_sample_is_post_stratified_on_the_calibrated_verdict(self):
        by_design = calibration.calibrate(self.run, self.labels(stratified_on="none"), judge="quality", **FAST)
        as_quality = calibration.calibrate(self.run, self.labels(stratified_on="quality"), judge="quality", **FAST)
        self.assertAlmostEqual(by_design["tnr"], as_quality["tnr"])
        self.assertEqual(by_design["strata"]["stratified_on"], "quality")
        self.assertEqual(by_design["strata"]["design"], "none")
        overall = calibration.calibrate(self.run, self.labels(stratified_on="none"), **FAST)
        self.assertEqual(overall["strata"]["stratified_on"], "overall")

    def test_a_stratum_without_labels_leaves_the_rates_undefined_with_a_warning(self):
        only_pass_stratum = [row for row in self.tool_labels() if row["case_id"].startswith("p")]
        cal = calibration.calibrate(self.run, only_pass_stratum, judge="quality", **FAST)
        self.assertIsNone(cal["tpr"])
        self.assertIsNone(cal["tnr"])
        self.assertIsNone(cal["agreement_weighted"])
        self.assertIsNone(cal["corrected_pass_rate"])
        self.assertTrue(any("No labeled case in the judge_fail stratum" in w for w in cal["warnings"]))

    def test_labels_from_another_run_are_flagged_and_weights_use_this_runs_verdicts(self):
        stale = self.tool_labels()
        for row in stale:
            row["stratum"] = calibration.stratum_token("some-other-run", "overall", row["case_id"], "judge_fail")
        cal = calibration.calibrate(self.run, stale, judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertEqual(cal["strata"]["source"], "recomputed")
        self.assertTrue(any("does not match this run" in w for w in cal["warnings"]))

    def test_a_recorded_plain_stratum_that_contradicts_the_run_is_flagged(self):
        rows = self.labels(stratified_on="overall")
        for row in rows:
            row["stratum"] = "judge_pass"  # wrong for the overall-fail cases
        cal = calibration.calibrate(self.run, rows, judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertTrue(any("does not match this run" in w for w in cal["warnings"]))

    def test_mixed_designs_warn_and_follow_the_most_common_one(self):
        rows = self.labels(stratified_on="overall")
        rows[0]["stratified_on"] = "quality"
        cal = calibration.calibrate(self.run, rows, judge="quality", **FAST)
        self.assertEqual(cal["strata"]["design"], "overall")
        self.assertTrue(any("different designs" in w for w in cal["warnings"]))

    def test_a_design_naming_an_unknown_judge_falls_back_with_a_warning(self):
        cal = calibration.calibrate(self.run, self.labels(stratified_on="ghost"), judge="quality", **FAST)
        self.check_overall_strata_numbers(cal)
        self.assertTrue(any("'ghost', which is not in this run" in w for w in cal["warnings"]))

    def test_census_weights_are_one(self):
        records = [make_record(f"c{i}", i % 3 != 0) for i in range(30)]
        labels = [label(f"c{i}", "pass" if i % 5 else "fail") for i in range(30)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertEqual(cal["strata"]["weights"], {"judge_pass": 1.0, "judge_fail": 1.0})
        self.assertAlmostEqual(cal["agreement_weighted"], cal["agreement"])
        self.assertAlmostEqual(cal["kappa_weighted"], cal["kappa"])
        self.assertAlmostEqual(cal["human_pass_rate_weighted"], cal["human_pass_rate"])

    def test_bootstrap_intervals_cover_the_point_estimates_and_are_reproducible(self):
        cal = calibration.calibrate(self.run, self.tool_labels(), judge="quality", seed=3, **FAST)
        again = calibration.calibrate(self.run, self.tool_labels(), judge="quality", seed=3, **FAST)
        self.assertEqual(cal, again)
        for key, point in (("tpr_ci", cal["tpr"]), ("tnr_ci", cal["tnr"]), ("agreement_weighted_ci", cal["agreement_weighted"])):
            lo, hi = cal[key]
            self.assertLessEqual(lo, point, key)
            self.assertGreaterEqual(hi, point, key)


class StratumTokenTests(unittest.TestCase):
    def test_token_round_trip_and_run_binding(self):
        token = calibration.stratum_token("run-1", "overall", "c1", "judge_fail")
        self.assertEqual(calibration.decode_stratum(token, "run-1", "overall", "c1"), "judge_fail")
        self.assertIsNone(calibration.decode_stratum(token, "run-2", "overall", "c1"))  # another run
        self.assertIsNone(calibration.decode_stratum(token, "run-1", "quality", "c1"))  # another design
        self.assertIsNone(calibration.decode_stratum(token, "run-1", "overall", "c2"))  # another case
        self.assertEqual(calibration.decode_stratum("judge_pass", "run-1", "overall", "c1"), "judge_pass")
        self.assertIsNone(calibration.decode_stratum("", "run-1", "overall", "c1"))
        self.assertIsNone(calibration.decode_stratum("garbage", "run-1", "overall", "c1"))

    def test_the_token_does_not_reveal_the_verdict_or_group_cases(self):
        tokens = {
            (case_id, stratum): calibration.stratum_token("r", "overall", case_id, stratum)
            for case_id in ("a", "b", "c")
            for stratum in calibration.STRATA
        }
        self.assertEqual(len(set(tokens.values())), 6)  # even two cases of one stratum never share a token
        for token in tokens.values():
            self.assertNotIn("pass", token)
            self.assertNotIn("fail", token)
            self.assertNotIn("judge", token)


# ---------------------------------------------------------------- kappa_decision
class KappaDecisionTests(unittest.TestCase):
    kd = staticmethod(calibration.kappa_decision)

    def test_thresholds_on_the_ci(self):
        self.assertEqual(self.kd(60, 0.9, [0.80, 0.95]), "trustworthy")
        self.assertEqual(self.kd(60, 0.9, [0.79, 0.95]), "usable")
        self.assertEqual(self.kd(60, 0.7, [0.60, 0.80]), "usable")
        self.assertEqual(self.kd(60, 0.7, [0.59, 0.80]), "inconclusive")
        self.assertEqual(self.kd(60, 0.3, [0.10, 0.59]), "not_trustworthy")
        self.assertEqual(self.kd(60, 0.3, [0.10, 0.60]), "inconclusive")

    def test_the_point_estimate_alone_decides_nothing(self):
        # same point estimate (0.70), three different CIs, three different decisions
        self.assertEqual(self.kd(400, 0.70, [0.62, 0.78]), "usable")
        self.assertEqual(self.kd(40, 0.70, [0.40, 0.90]), "inconclusive")
        self.assertEqual(self.kd(400, 0.30, [0.22, 0.38]), "not_trustworthy")

    def test_too_little_data_wins_over_a_lucky_ci(self):
        self.assertEqual(self.kd(29, 1.0, [1.0, 1.0]), "insufficient_data")
        self.assertEqual(self.kd(30, 1.0, [1.0, 1.0]), "inconclusive")  # 30 labels: enough to measure, not to release
        self.assertEqual(self.kd(40, 1.0, [1.0, 1.0]), "trustworthy")
        self.assertEqual(self.kd(0, None, None), "insufficient_data")

    def test_a_judge_is_released_only_from_the_minimum_number_of_labels(self):
        self.assertEqual(calibration.MIN_LABELS_RELEASE, 40)
        self.assertEqual(self.kd(39, 0.9, [0.85, 0.95]), "inconclusive")  # would be trustworthy with one more label
        self.assertEqual(self.kd(40, 0.9, [0.85, 0.95]), "trustworthy")
        self.assertEqual(self.kd(39, 0.7, [0.62, 0.78]), "inconclusive")
        self.assertEqual(self.kd(40, 0.7, [0.62, 0.78]), "usable")
        # the rule can be tightened or relaxed per call (the measurement scripts switch it off with 0)
        self.assertEqual(self.kd(60, 0.7, [0.62, 0.78], min_labels_release=100), "inconclusive")
        self.assertEqual(self.kd(35, 0.7, [0.62, 0.78], min_labels_release=0), "usable")

    def test_the_minimum_only_holds_back_a_release_never_a_rejection(self):
        # "do not gate" needs no minimum: the whole CI is already below the bar, and more labels would be needed only to be sure
        self.assertEqual(self.kd(30, 0.2, [0.05, 0.45]), "not_trustworthy")
        self.assertEqual(self.kd(35, 0.3, [0.10, 0.59]), "not_trustworthy")
        self.assertEqual(self.kd(35, 0.7, [0.40, 0.90]), "inconclusive")

    def test_trustworthy_also_needs_a_reliable_true_negative_rate(self):
        self.assertEqual(self.kd(200, 0.95, [0.90, 0.98], tnr_reliable=True), "trustworthy")
        self.assertEqual(self.kd(200, 0.95, [0.90, 0.98], tnr_reliable=False), "usable")
        # it does not weaken the other bars: an unreliable TNR does not turn a straddling CI into anything
        self.assertEqual(self.kd(200, 0.7, [0.55, 0.85], tnr_reliable=False), "inconclusive")
        self.assertEqual(self.kd(200, 0.3, [0.10, 0.50], tnr_reliable=False), "not_trustworthy")
        self.assertEqual(self.kd(200, 0.7, [0.62, 0.80], tnr_reliable=False), "usable")

    def test_undefined_kappa_and_missing_ci(self):
        self.assertEqual(self.kd(60, None, None), "insufficient_data")
        self.assertEqual(self.kd(60, 0.7, None), "inconclusive")

    def test_perfect_agreement_on_a_handful_of_cases_is_not_trustworthy(self):
        run, labels = confusion_scenario(tp=8, fn=0, fp=0, tn=6)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(cal["kappa"], 1.0)
        self.assertEqual(cal["kappa_decision"], "insufficient_data")

    def test_the_decision_rests_on_the_weighted_kappa_ci_not_the_sample_one(self):
        # 200 cases, the judge passes 180. The 40 labels are stratified 20/20 on the judge verdict, so the judge-fail stratum
        # (where humans agree) is over-represented: the sample kappa is 0.90 with a lower bound above 0.60, but the whole run's
        # kappa is 0.64 with a wide interval. The decision must follow the second.
        records = [make_record(f"p{i:03d}", True) for i in range(180)] + [make_record(f"f{i:03d}", False) for i in range(20)]
        labels = [label(f"p{i:03d}", "pass" if i < 18 else "fail") for i in range(20)] + [label(f"f{i:03d}", "fail") for i in range(20)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        self.assertGreater(cal["kappa"], 0.85)
        self.assertGreaterEqual(cal["kappa_ci"][0], calibration.KAPPA_USABLE)  # the unweighted CI alone would have released it
        self.assertAlmostEqual(cal["kappa_weighted"], 0.6429, places=3)
        self.assertLess(cal["kappa_weighted_ci"][0], calibration.KAPPA_USABLE)
        self.assertEqual(cal["kappa_decision"], "inconclusive")
        self.assertEqual(cal["kappa_decision_basis"], "weighted")
        self.assertEqual(cal["kappa_decision"], calibration.kappa_decision(cal["n"], cal["kappa_weighted"], cal["kappa_weighted_ci"], cal["tnr_reliable"]))
        self.assertIn("Weighted kappa 0.64", cal["recommendation"])

    def test_the_unweighted_sample_kappa_is_still_reported_as_secondary(self):
        run, labels = confusion_scenario(tp=95, fn=3, fp=6, tn=90)  # a census: both estimands coincide
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertAlmostEqual(cal["kappa"], cal["kappa_weighted"])
        self.assertEqual(cal["kappa_ci"], calibration.calibrate(run, labels, **FAST)["kappa_ci"])
        for key in ("kappa", "kappa_ci", "kappa_ci_bootstrap", "kappa_weighted", "kappa_weighted_ci", "kappa_weighted_ci_bootstrap", "kappa_ci_smoothing"):
            self.assertIn(key, cal)
        md = calibration.to_markdown(cal)
        self.assertIn("Cohen's kappa (whole run, weighted; the decision uses it)", md)
        self.assertIn("Cohen's kappa (labeled sample, unweighted; secondary)", md)

    def test_a_good_judge_held_back_only_by_the_minimum_says_so(self):
        run, labels = confusion_scenario(tp=15, fn=1, fp=1, tn=18)  # 35 labels, kappa 0.89, lower bound of the CI above 0.6
        cal = calibration.calibrate(with_unlabeled(run, 20, 20), labels, **FAST)  # 75 labelable cases
        self.assertEqual(cal["n"], 35)
        self.assertGreaterEqual(cal["kappa_weighted_ci"][0], calibration.KAPPA_USABLE)
        self.assertEqual(cal["kappa_decision"], "inconclusive")
        self.assertIn("only 35 cases are labeled and at least 40 are needed", cal["recommendation"])
        self.assertTrue(any("qualifies as usable, but only 35 cases are labeled" in w for w in cal["warnings"]))
        self.assertEqual(cal["min_labels_release"], 40)
        # the same evidence with 40 labels is released
        run40, labels40 = confusion_scenario(tp=17, fn=1, fp=1, tn=21)
        self.assertIn(calibration.calibrate(run40, labels40, **FAST)["kappa_decision"], {"usable", "trustworthy"})

    def test_a_perfect_judge_is_usable_and_says_why_until_it_has_enough_human_fails(self):
        # kappa interval above 0.8 (n = 2009), but only 9 human fails: the true-negative rate is not measured yet
        cal = calibration.calibrate(*confusion_scenario(tp=2000, fn=0, fp=0, tn=9), **FAST)
        self.assertGreaterEqual(cal["kappa_weighted_ci"][0], calibration.KAPPA_TRUSTWORTHY)
        self.assertFalse(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "few_human_fails")
        self.assertEqual(cal["kappa_decision"], "usable")
        self.assertIn("not called trustworthy", cal["recommendation"])
        self.assertTrue(any("stays 'usable' and is not called 'trustworthy'" in w for w in cal["warnings"]))
        # one more human fail, caught: the same judge is certified
        ten = calibration.calibrate(*confusion_scenario(tp=2000, fn=0, fp=0, tn=10), **FAST)
        self.assertTrue(ten["tnr_reliable"])
        self.assertEqual(ten["kappa_decision"], "trustworthy")
        self.assertFalse(any("not called 'trustworthy'" in w for w in ten["warnings"]))
        reliable = calibration.calibrate(*confusion_scenario(tp=95, fn=3, fp=6, tn=90), **FAST)
        self.assertEqual(reliable["kappa_decision"], "trustworthy")
        self.assertFalse(any("not called 'trustworthy'" in w for w in reliable["warnings"]))

    def test_a_very_good_judge_that_misses_a_few_bad_outputs_can_be_trustworthy(self):
        # v0.1 demanded 5 missed bad outputs and called 2 "too few": what counts now is how well the TNR is measured
        cal = calibration.calibrate(*confusion_scenario(tp=95, fn=1, fp=2, tn=90), **FAST)
        self.assertGreaterEqual(cal["kappa_weighted_ci"][0], calibration.KAPPA_TRUSTWORTHY)
        self.assertTrue(cal["tnr_reliable"])
        self.assertEqual(cal["kappa_decision"], "trustworthy")

    def test_constant_verdicts_are_insufficient_data(self):
        records = [make_record(f"c{i}", True) for i in range(40)]
        cal = calibration.calibrate(make_run(records), [label(f"c{i}", "pass") for i in range(40)], **FAST)
        self.assertIsNone(cal["kappa"])
        self.assertEqual(cal["kappa_decision"], "insufficient_data")
        self.assertEqual(cal["one_sided_labels"], "pass")
        self.assertIn("label cases the judge failed", cal["recommendation"])
        self.assertNotIn("revise the rubric", cal["recommendation"].lower())
        # the mirror image: a judge that fails everything is asked for cases it passed, never for a rubric revision
        records = [make_record(f"c{i}", False) for i in range(40)]
        cal = calibration.calibrate(make_run(records), [label(f"c{i}", "fail") for i in range(40)], **FAST)
        self.assertEqual(cal["one_sided_labels"], "fail")
        self.assertIn("label cases the judge passed", cal["recommendation"])
        self.assertNotIn("revise the rubric", cal["recommendation"].lower())

    def test_a_judge_that_passes_everything_against_mixed_human_labels_has_kappa_zero(self):
        # not "insufficient data": the judge has no skill, and that is the answer (kappa 0, the whole CI below the bar)
        cal = calibration.calibrate(*confusion_scenario(tp=28, fn=0, fp=12, tn=0), **FAST)
        self.assertEqual(cal["kappa"], 0.0)
        self.assertEqual(cal["kappa_weighted"], 0.0)
        self.assertEqual(cal["kappa_weighted_ci"], [0.0, 0.0])
        self.assertEqual(cal["kappa_decision"], "not_trustworthy")
        self.assertIn("do not gate on this judge", cal["recommendation"])
        # the same for a judge that fails everything against mixed human labels
        cal = calibration.calibrate(*confusion_scenario(tp=0, fn=12, fp=0, tn=28), **FAST)
        self.assertEqual(cal["kappa_weighted"], 0.0)
        self.assertEqual(cal["kappa_decision"], "not_trustworthy")

    def test_the_decision_is_in_the_output_and_the_markdown(self):
        run, labels = confusion_scenario(tp=95, fn=3, fp=6, tn=90)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertIn(cal["kappa_decision"], {"trustworthy", "usable", "inconclusive", "not_trustworthy", "insufficient_data"})
        self.assertIn("**Decision: trustworthy", calibration.to_markdown(cal))


class TnrReliableTests(unittest.TestCase):
    """``tnr_reliable`` = at least 10 human fails, TNR >= 0.75 and a Wilson interval (at the delta-method effective n)
    within +/-0.175. The dedicated monotonicity / coverage / determinism tests are in tests/test_calibration_v02.py."""

    def cal(self, tp, fn, fp, tn):
        run, labels = confusion_scenario(tp, fn, fp, tn)
        return calibration.calibrate(run, labels, **FAST)

    def reliable(self, tp, fn, fp, tn):
        return self.cal(tp, fn, fp, tn)["tnr_reliable"]

    def test_the_rule_constants(self):
        self.assertEqual(calibration.MIN_HUMAN_FAILS, 10)
        self.assertEqual(calibration.TNR_MAX_HALF_WIDTH, 0.175)
        self.assertEqual(calibration.TNR_MIN_RELIABLE, 0.75)
        self.assertFalse(hasattr(calibration, "MIN_MISSED_BAD"))  # the v0.1 count of missed bad outputs is gone

    def test_needs_ten_human_failures(self):
        self.assertTrue(self.reliable(tp=30, fn=0, fp=0, tn=10))
        self.assertFalse(self.reliable(tp=30, fn=0, fp=0, tn=9))
        self.assertEqual(self.cal(30, 0, 0, 9)["tnr_reliable_reason"], "few_human_fails")

    def test_a_perfect_judge_is_reliable_once_the_interval_is_narrow_enough(self):
        cal = self.cal(tp=25, fn=0, fp=0, tn=15)
        self.assertTrue(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "ok")
        lo, hi = cal["tnr_ci"]
        self.assertAlmostEqual(lo, 0.759, places=2)
        self.assertEqual(hi, 1.0)

    def test_a_judge_below_the_floor_is_low_however_precise_the_interval(self):
        for counts in ((40, 2, 4, 6), (40, 2, 8, 12), (200, 5, 100, 100)):
            cal = self.cal(*counts)
            self.assertFalse(cal["tnr_reliable"], counts)
            self.assertEqual(cal["tnr_reliable_reason"], "low_tnr", counts)
        self.assertLess(self.cal(200, 5, 100, 100)["tnr"], calibration.TNR_MIN_RELIABLE)

    def test_a_good_but_imprecise_tnr_is_flagged_wide(self):
        cal = self.cal(tp=25, fn=0, fp=5, tn=15)  # TNR 0.75, half-width 0.177 > 0.175
        self.assertEqual(cal["tnr"], 0.75)
        self.assertFalse(cal["tnr_reliable"])
        self.assertEqual(cal["tnr_reliable_reason"], "wide_ci")

    def test_a_strong_judge_with_a_handful_of_misses_is_reliable(self):
        # 1 miss among 30 human fails: the number of misses is no longer the criterion, the precision is
        cal = self.cal(tp=70, fn=0, fp=1, tn=29)
        self.assertTrue(cal["tnr_reliable"])
        self.assertFalse(any("true-negative rate" in w for w in cal["warnings"]))

    def test_reliable_scenario_has_no_tnr_warning(self):
        cal = self.cal(tp=95, fn=3, fp=6, tn=90)
        self.assertTrue(cal["tnr_reliable"])
        self.assertFalse(any("true-negative rate" in w for w in cal["warnings"]))
        self.assertNotIn("true-negative rate (how often", cal["recommendation"])

    def test_the_rule_and_its_reason_are_in_the_output(self):
        cal = self.cal(tp=95, fn=3, fp=6, tn=90)
        self.assertEqual(cal["tnr_reliable_rule"]["min_human_fails"], 10)
        self.assertEqual(cal["tnr_reliable_rule"]["max_half_width"], 0.175)
        self.assertEqual(cal["tnr_reliable_rule"]["min_tnr"], 0.75)
        self.assertIn("Wilson", cal["tnr_reliable_rule"]["interval"])
        self.assertEqual(cal["tnr_reliable_rule"]["kind"], "screening flag")
        self.assertIn("tnr_ci_bootstrap", cal)


class LabelsNeededTests(unittest.TestCase):
    def test_scales_like_one_over_sqrt_n(self):
        self.assertEqual(calibration._labels_needed(60, [0.4, 0.8]), 240)  # half width 0.2 -> 4x the labels
        self.assertEqual(calibration._labels_needed(400, [0.65, 0.75]), 400)  # already within +/-0.1
        self.assertEqual(calibration._labels_needed(100, [0.3, 0.9]), 900)

    def test_bounds_and_missing_ci(self):
        self.assertIsNone(calibration._labels_needed(50, None))
        self.assertEqual(calibration._labels_needed(8, [1.0, 1.0]), calibration.MIN_LABELS_RELEASE)  # never below the release minimum
        self.assertEqual(calibration._labels_needed(100, [-1.0, 1.0]), calibration.MAX_LABELS_NEEDED)

    def test_the_estimate_is_capped_at_the_labelable_cases_of_the_run(self):
        self.assertEqual(calibration._labels_needed(60, [0.4, 0.8], cap=150), 150)  # 240 needed, only 150 cases exist
        self.assertEqual(calibration._labels_needed(60, [0.4, 0.8], cap=1000), 240)  # a cap above the need changes nothing
        self.assertEqual(calibration._labels_needed(60, [0.4, 0.8], cap=None), 240)
        self.assertEqual(calibration._labels_needed(60, [0.4, 0.8], cap=0), 240)  # a missing frame size is not a cap of zero
        self.assertIsNone(calibration._labels_needed(60, None, cap=150))

    def test_a_run_with_fewer_cases_than_the_release_minimum_says_so(self):
        run, labels = confusion_scenario(tp=15, fn=1, fp=1, tn=13)  # 30 cases, all labeled: can never reach 40
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual((cal["n"], cal["n_labelable"]), (30, 30))
        self.assertIn(cal["kappa_decision"], {"inconclusive", "not_trustworthy"})
        self.assertNotIn(cal["kappa_decision"], {"usable", "trustworthy"})
        self.assertTrue(any("Only 30 cases of this run can be labeled" in w and "fewer than the 40" in w for w in cal["warnings"]))

    def test_fewer_labels_need_more_and_more_labels_need_fewer(self):
        # labels are a sample of a big run here, so the estimate is not capped by the run's size
        small_run, small_labels = confusion_scenario(25, 5, 5, 25)
        large_run, large_labels = confusion_scenario(250, 50, 50, 250)
        small = calibration.calibrate(with_unlabeled(small_run, 1000, 1000), small_labels, **FAST)["labels_needed_estimate"]
        large = calibration.calibrate(with_unlabeled(large_run, 1000, 1000), large_labels, **FAST)["labels_needed_estimate"]
        self.assertIsInstance(small, int)
        self.assertGreater(small, 60)  # 60 labels are not enough
        self.assertLessEqual(large, 600)  # 600 labels already are (about)
        self.assertLess(large, small * 5)

    def test_undefined_kappa_has_no_estimate(self):
        records = [make_record(f"c{i}", True) for i in range(40)]
        cal = calibration.calibrate(make_run(records), [label(f"c{i}", "pass") for i in range(40)], **FAST)
        self.assertIsNone(cal["labels_needed_estimate"])


class WarningsTests(unittest.TestCase):
    def test_clean_calibration_has_no_warnings(self):
        run, labels = confusion_scenario(tp=95, fn=3, fp=6, tn=90)
        self.assertEqual(calibration.calibrate(run, labels, **FAST)["warnings"], [])

    def test_skipped_rows_and_small_samples_are_listed(self):
        records = [make_record(f"c{i:02d}", i % 2 == 0) for i in range(12)]
        labels = calibration.LabelList(
            [
                label("c00", "pass", sha="deadbeef0000"),
                label("ghost", "pass"),
                label("c01", "fail", labeler="ana"),
                label("c01", "pass", labeler="bo"),
                label("c02", "pass"),
                label("c03", "fail"),
            ],
            n_invalid=2,
        )
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        text = " | ".join(cal["warnings"])
        for needle in (
            "1 label(s) excluded because the AI output changed",
            "1 label(s) have an unknown case id",
            "2 label(s) were unreadable",
            "1 labeled case(s) skipped because the labelers tied",
            "Only 2 cases labeled",
        ):
            self.assertIn(needle, text)

    def test_zero_disagreements_warn_that_the_intervals_are_degenerate(self):
        run, labels = confusion_scenario(tp=25, fn=0, fp=0, tn=15)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertTrue(any("No disagreement in 40 labels" in w and "rule of three" in w for w in cal["warnings"]))

    def test_judge_no_better_than_chance_warns(self):
        run, labels = confusion_scenario(tp=15, fn=15, fp=15, tn=15)
        cal = calibration.calibrate(run, labels, **FAST)
        self.assertTrue(any("no better than chance" in w for w in cal["warnings"]))

    def test_warnings_are_json_serialisable_strings(self):
        cal = calibration.calibrate(*confusion_scenario(3, 3, 3, 3), **FAST)
        self.assertTrue(cal["warnings"])
        self.assertTrue(all(isinstance(w, str) for w in cal["warnings"]))
        json.dumps(cal)


def skewed_run(disagreements=0, where="pass", n_pass=177, n_fail=23, l_pass=20, l_fail=20):
    """A run judged 88.5% pass, labeled the way a stratified sample is (l_pass + l_fail cases).

    ``disagreements`` cases are labeled against the judge: human "fail" among the judge-pass labels (``where="pass"``) or human
    "pass" among the judge-fail labels (``where="fail"``)."""
    records = [make_record(f"p{i:03d}", True) for i in range(n_pass)] + [make_record(f"f{i:03d}", False) for i in range(n_fail)]
    labels = [label(f"p{i:03d}", "fail" if where == "pass" and i < disagreements else "pass") for i in range(l_pass)]
    labels += [label(f"f{i:03d}", "pass" if where == "fail" and i < disagreements else "fail") for i in range(l_fail)]
    return make_run(records), labels


RELEASED = ("usable", "trustworthy")


class SmoothedKappaCITests(unittest.TestCase):
    """Few or no disagreements: the plain bootstrap sees none in most resamples and collapses (0 disagreements: [1, 1]).

    The decision interval is a bootstrap that adds pseudo-counts (KAPPA_CI_SMOOTHING) to every confusion cell the run allows,
    so its lower bound falls monotonically with the number of disagreements and 1 disagreement is never easier than 0."""

    def test_the_smoothing_constant(self):
        self.assertEqual(calibration.KAPPA_CI_SMOOTHING, 0.5)

    def test_zero_disagreements_do_not_collapse_the_decision_interval(self):
        cal = calibration.calibrate(*skewed_run(0), **FAST)
        self.assertEqual(cal["kappa"], 1.0)
        self.assertEqual(cal["kappa_ci_bootstrap"], [1.0, 1.0])  # the plain bootstrap alone reads as certainty; kept for transparency
        self.assertEqual(cal["kappa_weighted_ci_bootstrap"], [1.0, 1.0])
        lo, hi = cal["kappa_weighted_ci"]
        self.assertLess(lo, 0.8)
        self.assertGreater(lo, 0.0)
        self.assertEqual(hi, 1.0)
        self.assertLess(cal["kappa_ci"][0], 1.0)  # the secondary (unweighted) interval is smoothed too
        self.assertEqual(cal["kappa_ci_smoothing"], calibration.KAPPA_CI_SMOOTHING)
        self.assertNotEqual(cal["kappa_decision"], "trustworthy")

    def test_the_lower_bound_never_rises_with_more_disagreements(self):
        for where in ("pass", "fail"):
            lows = []
            for d in range(6):
                cal = calibration.calibrate(*skewed_run(d, where), iters=2000, seed=3)
                lows.append(cal["kappa_weighted_ci"][0])
            for d in range(5):
                self.assertLessEqual(lows[d + 1], lows[d] + 0.03, (where, d, lows))  # 0.03: bootstrap noise at 2000 draws
            self.assertLess(lows[1], lows[0], (where, lows))  # the point of the rule: 1 disagreement is harder than 0
            self.assertLess(lows[5], lows[0] - 0.1, (where, lows))

    def test_one_disagreement_is_never_easier_than_none_for_the_decision(self):
        for where in ("pass", "fail"):
            for l_pass, l_fail in ((20, 20), (40, 20), (50, 50)):
                zero = calibration.calibrate(*skewed_run(0, where, l_pass=l_pass, l_fail=l_fail), **FAST)
                one = calibration.calibrate(*skewed_run(1, where, l_pass=l_pass, l_fail=l_fail), **FAST)
                two = calibration.calibrate(*skewed_run(2, where, l_pass=l_pass, l_fail=l_fail), **FAST)
                for harder, easier in ((one, zero), (two, one)):
                    if harder["kappa_decision"] in RELEASED:
                        self.assertIn(easier["kappa_decision"], RELEASED, (where, l_pass, l_fail))
                    if harder["kappa_decision"] == "trustworthy":
                        self.assertEqual(easier["kappa_decision"], "trustworthy", (where, l_pass, l_fail))

    def test_more_perfect_labels_raise_the_bound(self):
        few = calibration.calibrate(*confusion_scenario(tp=25, fn=0, fp=0, tn=15), **FAST)
        many = calibration.calibrate(*confusion_scenario(tp=60, fn=0, fp=0, tn=40), **FAST)
        self.assertGreater(many["kappa_weighted_ci"][0], few["kappa_weighted_ci"][0] + 0.05)

    def test_perfect_labels_with_enough_human_fails_make_a_judge_trustworthy(self):
        # the smoothing keeps 40 perfect labels from being certified on a lucky sample only when they hold too few human fails
        for tp, tn in ((25, 15), (60, 40), (300, 200)):
            cal = calibration.calibrate(*confusion_scenario(tp=tp, fn=0, fp=0, tn=tn), **FAST)
            self.assertEqual(cal["kappa_decision"], "trustworthy", (tp, tn))

    def test_fewer_than_ten_human_fails_are_never_trustworthy(self):
        for tn in range(10):
            cal = calibration.calibrate(*confusion_scenario(tp=400, fn=0, fp=0, tn=tn), **FAST)
            self.assertNotEqual(cal["kappa_decision"], "trustworthy", tn)
        for fp in range(10):  # the judge missed some of the few bad outputs: still fewer than 10 human fails
            cal = calibration.calibrate(*confusion_scenario(tp=400, fn=0, fp=fp, tn=9 - fp), **FAST)
            self.assertNotEqual(cal["kappa_decision"], "trustworthy", fp)
            self.assertEqual(cal["tnr_reliable_reason"] in ("few_human_fails", "undefined"), True, fp)

    def test_a_constant_judge_keeps_its_zero_kappa_interval(self):
        # pseudo-counts only go to cells the run allows: a judge that never fails anything cannot get a human-pass/judge-fail cell
        cal = calibration.calibrate(*confusion_scenario(tp=28, fn=0, fp=12, tn=0), **FAST)
        self.assertEqual(cal["kappa_ci"], [0.0, 0.0])
        self.assertEqual(cal["kappa_weighted_ci"], [0.0, 0.0])
        self.assertEqual(cal["kappa_decision"], "not_trustworthy")

    def test_smoothing_zero_is_the_plain_bootstrap(self):
        run, labels = skewed_run(0)
        with mock.patch.object(calibration, "KAPPA_CI_SMOOTHING", 0.0):
            cal = calibration.calibrate(run, labels, **FAST)
        self.assertEqual(cal["kappa_weighted_ci"], cal["kappa_weighted_ci_bootstrap"])
        self.assertEqual(cal["kappa_ci"], cal["kappa_ci_bootstrap"])
        self.assertEqual(cal["kappa_weighted_ci"], [1.0, 1.0])
        self.assertEqual(cal["kappa_ci_smoothing"], 0.0)

    def test_smoothed_and_plain_agree_when_there_are_many_disagreements(self):
        cal = calibration.calibrate(*confusion_scenario(tp=180, fn=20, fp=20, tn=180), iters=1000, seed=1)
        for smoothed, plain in ((cal["kappa_weighted_ci"], cal["kappa_weighted_ci_bootstrap"]), (cal["kappa_ci"], cal["kappa_ci_bootstrap"])):
            self.assertAlmostEqual(smoothed[0], plain[0], delta=0.03)
            self.assertAlmostEqual(smoothed[1], plain[1], delta=0.03)

    def test_the_interval_is_reproducible_for_a_seed(self):
        run, labels = skewed_run(1)
        a = calibration.calibrate(run, labels, seed=5, **FAST)
        b = calibration.calibrate(run, labels, seed=5, **FAST)
        c = calibration.calibrate(run, labels, seed=6, **FAST)
        self.assertEqual(a["kappa_weighted_ci"], b["kappa_weighted_ci"])
        self.assertNotEqual(a["kappa_weighted_ci"], c["kappa_weighted_ci"])

    def test_the_warning_explains_what_was_done_and_appears_once(self):
        cal = calibration.calibrate(*skewed_run(0), **FAST)
        (text,) = [w for w in cal["warnings"] if "No disagreement" in w]
        for needle in ("40 labels", "degenerate", "rule of three", "kappa interval is smoothed", "stays wide"):
            self.assertIn(needle, text)

    def test_it_reaches_the_markdown_and_the_recommendation(self):
        cal = calibration.calibrate(*skewed_run(0), **FAST)
        md = calibration.to_markdown(cal)
        self.assertIn(f"95% CI {cal['kappa_weighted_ci'][0]:.2f} to 1.00", md)
        self.assertIn("No disagreement in 40 labels", md)
        self.assertIn(f"Weighted kappa 1.00 (almost perfect, 95% CI {cal['kappa_weighted_ci'][0]:.2f}-1.00)", cal["recommendation"])
        json.dumps(cal)

    def test_a_sample_with_a_disagreement_gets_no_degenerate_warning(self):
        cal = calibration.calibrate(*confusion_scenario(tp=95, fn=3, fp=6, tn=90), **FAST)
        self.assertFalse(any("No disagreement" in w for w in cal["warnings"]))
        one = calibration.calibrate(*confusion_scenario(tp=24, fn=1, fp=0, tn=15), **FAST)
        self.assertFalse(any("No disagreement" in w for w in one["warnings"]))
        self.assertLess(one["kappa_weighted_ci"][0], calibration.calibrate(*confusion_scenario(tp=25, fn=0, fp=0, tn=15), **FAST)["kappa_weighted_ci"][0])

    def test_undefined_kappa_and_tiny_samples_are_unchanged(self):
        records = [make_record(f"c{i}", True) for i in range(40)]
        cal = calibration.calibrate(make_run(records), [label(f"c{i}", "pass") for i in range(40)], **FAST)
        self.assertIsNone(cal["kappa_ci"])
        self.assertIsNone(cal["kappa_weighted_ci"])
        self.assertEqual(cal["kappa_decision"], "insufficient_data")
        small = calibration.calibrate(*confusion_scenario(tp=8, fn=0, fp=0, tn=6), **FAST)
        self.assertEqual(small["kappa_decision"], "insufficient_data")  # n < 30 still wins over any interval


class NewMarkdownTests(unittest.TestCase):
    def test_decision_warnings_and_corrected_ci_scope_are_rendered(self):
        cal = calibration.calibrate(*confusion_scenario(tp=40, fn=2, fp=4, tn=6), **FAST)
        md = calibration.to_markdown(cal)
        self.assertIn("**Decision:", md)
        self.assertIn("### Warnings", md)
        self.assertIn("low: the judge misses more than 25% of the bad outputs", md)  # next to the TNR figure, with the reason
        self.assertIn("describes the population the test set was sampled from", md)
        self.assertNotIn("nan", md.lower())

    def test_weighted_agreement_is_shown_next_to_the_sample_value(self):
        records = [make_record(f"c{i:03d}", i < 80) for i in range(100)]
        labels = [label(f"c{i:03d}", "pass" if i < 18 else "fail") for i in range(20)]
        labels += [label(f"c{i:03d}", "pass" if i < 85 else "fail") for i in range(80, 100)]
        cal = calibration.calibrate(make_run(records), labels, **FAST)
        md = calibration.to_markdown(cal)
        self.assertNotAlmostEqual(cal["agreement_weighted"], cal["agreement"], places=2)
        self.assertIn("labeled sample:", md)
        self.assertIn("re-weighted", md)

    def test_markdown_survives_an_old_calibration_dict_without_the_new_keys(self):
        cal = calibration.calibrate(*confusion_scenario(20, 5, 6, 9), **FAST)
        for key in ("kappa_decision", "warnings", "agreement_weighted", "tnr_reliable", "labels_needed_estimate", "strata", "corrected_pass_rate_ci_note"):
            cal.pop(key)
        md = calibration.to_markdown(cal)
        self.assertIn("## Judge calibration", md)
        self.assertNotIn("Decision:", md)


class LoadLabelsStrataTests(unittest.TestCase):
    def test_stratum_and_stratified_on_survive_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp, "labels.jsonl")
            jsonl.write_text(
                '{"case_id": "a", "label": "pass", "stratum": "s-0123456789", "stratified_on": "overall"}\n'
                '{"case_id": "b", "label": "fail"}\n',
                encoding="utf-8",
            )
            rows = calibration.load_labels(jsonl)
            self.assertEqual((rows[0]["stratum"], rows[0]["stratified_on"]), ("s-0123456789", "overall"))
            self.assertEqual((rows[1]["stratum"], rows[1]["stratified_on"]), (None, None))
            csv_path = Path(tmp, "labels.csv")
            csv_path.write_text("case_id,stratum,stratified_on,label\na,s-0123456789,quality,pass\n", encoding="utf-8")
            (row,) = calibration.load_labels(csv_path)
            self.assertEqual((row["stratum"], row["stratified_on"]), ("s-0123456789", "quality"))


if __name__ == "__main__":
    unittest.main()
