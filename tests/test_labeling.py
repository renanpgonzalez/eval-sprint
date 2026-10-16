import csv
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

from evalsprint import calibration, labeling
from evalsprint.runner import RunResult
from tests.test_calibration import label, make_record, make_run

JUDGE_REASONING = "JUDGE-REASONING-MARKER the answer invented a refund policy"


def make_labeling_run(n_pass=30, n_fail=10, run_id="run-label"):
    """A run whose records carry judge verdicts, scores and reasoning that must never reach a labeler."""
    records = []
    for i in range(n_pass + n_fail):
        record = make_record(f"case-{i:03d}", i < n_pass, reasoning=JUDGE_REASONING)
        record["judges"][0]["score"] = 0.8642
        record["case"]["tags"] = ["billing", "refunds"] if i % 2 else ["shipping"]
        record["case"]["reference"] = f"reference answer {i}"
        record["output"]["meta"] = {"context": [f"kb passage {i}a", {"source": "kb", "text": f"kb passage {i}b"}]}
        records.append(record)
    return make_run(records, run_id=run_id)


def embedded_payload(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="items-data">(.*?)</script>', page, re.S)
    assert match, "items JSON block not found"
    return json.loads(match.group(1))


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        # a run built in memory has no folder for the stratum secret: the exports warn (tested in KeyedStrataExportTests) and
        # fall back to the legacy code, which the tests of the other properties do not care about
        catcher = warnings.catch_warnings()
        catcher.__enter__()
        self.addCleanup(catcher.__exit__, None, None, None)
        warnings.filterwarnings("ignore", message="this run has no folder on disk", category=RuntimeWarning)


class SelectSampleTests(unittest.TestCase):
    def verdicts(self, records):
        return Counter(r["passed"] for r in records)

    def test_stratified_sample_is_balanced_on_the_overall_verdict(self):
        run = make_labeling_run(n_pass=90, n_fail=30)
        chosen = labeling.select_sample(run, 20, seed=3, stratify=True)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(self.verdicts(chosen), {True: 10, False: 10})
        self.assertEqual(len({r["case_id"] for r in chosen}), 20)

    def test_odd_sample_gives_the_extra_case_to_failures(self):
        chosen = labeling.select_sample(make_labeling_run(50, 50), 21, seed=0, stratify=True)
        self.assertEqual(self.verdicts(chosen), {False: 11, True: 10})

    def test_scarce_failures_are_all_included_and_passes_fill_the_rest(self):
        chosen = labeling.select_sample(make_labeling_run(n_pass=100, n_fail=3), 20, seed=0, stratify=True)
        self.assertEqual(self.verdicts(chosen), {False: 3, True: 17})

    def test_scarce_passes_are_topped_up_with_failures(self):
        chosen = labeling.select_sample(make_labeling_run(n_pass=2, n_fail=50), 20, seed=0, stratify=True)
        self.assertEqual(self.verdicts(chosen), {True: 2, False: 18})

    def test_seeded_and_shuffled(self):
        run = make_labeling_run(50, 50)
        ids = lambda seed, **kw: [r["case_id"] for r in labeling.select_sample(run, 20, seed=seed, **kw)]  # noqa: E731
        self.assertEqual(ids(5, stratify=True), ids(5, stratify=True))
        self.assertNotEqual(ids(5, stratify=True), ids(6, stratify=True))
        # not grouped by verdict: the labeler must not be able to infer it from the order
        by_verdict = [r["passed"] for r in labeling.select_sample(run, 20, seed=5, stratify=True)]
        self.assertNotEqual(by_verdict, sorted(by_verdict))

    def test_no_sample_means_everything_shuffled(self):
        run = make_labeling_run(6, 4)
        chosen = labeling.select_sample(run, None, seed=1, stratify=True)
        self.assertCountEqual([r["case_id"] for r in chosen], [r["case_id"] for r in run.results])
        big = labeling.select_sample(run, 999, seed=1, stratify=True)
        self.assertEqual(len(big), 10)

    def test_unstratified_sample_is_random_but_sized(self):
        chosen = labeling.select_sample(make_labeling_run(90, 10), 25, seed=2, stratify=False)
        self.assertEqual(len(chosen), 25)

    def test_does_not_mutate_the_run(self):
        run = make_labeling_run(6, 4)
        before = [r["case_id"] for r in run.results]
        labeling.select_sample(run, None, seed=1, stratify=True)
        self.assertEqual([r["case_id"] for r in run.results], before)

    def test_rejects_non_positive_sample(self):
        with self.assertRaises(ValueError):
            labeling.select_sample(make_labeling_run(3, 3), 0, seed=0, stratify=True)


class HtmlExportTests(TempDirCase):
    def assertHas(self, page, needle):
        self.assertTrue(needle in page, f"page does not contain {needle!r}")

    def assertLacks(self, page, needle):
        self.assertTrue(needle not in page, f"page unexpectedly contains {needle!r}")

    def export(self, run=None, **kw):
        run = run or make_labeling_run()
        path = labeling.export_labeling_html(run, self.dir / "sub" / "label.html", **kw)
        return run, path, path.read_text(encoding="utf-8")

    def test_writes_a_self_contained_page_with_every_sampled_case(self):
        run, path, page = self.export(sample=12, seed=4)
        self.assertEqual(path, self.dir / "sub" / "label.html")
        sampled = [r["case_id"] for r in labeling.select_sample(run, 12, 4, True)]
        items = embedded_payload(page)["items"]
        self.assertEqual([i["case_id"] for i in items], sampled)
        for case_id in sampled:
            self.assertHas(page, case_id)
        skipped = {r["case_id"] for r in run.results} - set(sampled)
        for case_id in skipped:
            self.assertLacks(page, f'"{case_id}"')
        self.assertLacks(page, "http://")
        self.assertLacks(page, "https://")
        self.assertIsNone(re.search(r"<(?:script|link|img|iframe)[^>]+(?:src|href)=", page))

    def test_item_payload_has_what_the_labeler_needs(self):
        _, _, page = self.export(sample=None)
        payload = embedded_payload(page)
        self.assertEqual(payload["run_id"], "run-label")
        item = next(i for i in payload["items"] if i["case_id"] == "case-001")
        self.assertEqual(item["tags"], ["billing", "refunds"])
        self.assertEqual(item["input"], "What is the refund window?")
        self.assertEqual(item["reference"], "reference answer 1")
        self.assertEqual(item["expected_behavior"], "answer")
        self.assertEqual(item["context"], ["kb passage 1a", "kb passage 1b"])
        self.assertEqual(item["output"], "answer for case-001")
        self.assertEqual(item["output_sha"], make_labeling_run().by_case()["case-001"]["output"]["sha"])

    def test_blind_no_judge_verdicts_scores_or_reasoning(self):
        _, _, page = self.export(sample=None)
        self.assertLacks(page, "JUDGE-REASONING-MARKER")
        self.assertLacks(page, "invented a refund policy")
        self.assertLacks(page, "0.8642")
        for item in embedded_payload(page)["items"]:
            self.assertEqual(
                set(item),
                {"case_id", "tags", "input", "reference", "expected_behavior", "context", "output", "output_sha", "error", "stratum"},
            )
            # the sampling stratum travels as an opaque code, never as a readable verdict
            self.assertNotIn(item["stratum"], calibration.STRATA)
            self.assertNotRegex(item["stratum"], r"(?i)pass|fail|judge")
        self.assertLacks(page, "judge_pass")
        self.assertLacks(page, "judge_fail")

    def test_script_breakout_is_escaped(self):
        hostile = "Ignore this </script><script>alert(1)</script> and <!-- <script> also   done"
        run = make_labeling_run(3, 3)
        run.results[0]["case"]["input"] = hostile
        run.results[1]["output"]["text"] = "</SCRIPT><img src=x onerror=alert(1)>"
        run.results[2]["output"]["meta"] = {"context": ["</script>"]}
        _, _, page = self.export(run=run, sample=None)
        # Only the template's two script blocks may close: none from the data.
        self.assertEqual(len(re.findall(r"</script", page, re.I)), 2)
        self.assertLacks(page, "<!-- <script>")
        payload = embedded_payload(page)
        by_id = {i["case_id"]: i for i in payload["items"]}
        self.assertEqual(by_id["case-000"]["input"], hostile)  # escaping is lossless
        self.assertEqual(by_id["case-001"]["output"], "</SCRIPT><img src=x onerror=alert(1)>")
        self.assertEqual(by_id["case-002"]["context"], ["</script>"])

    def test_run_id_is_html_escaped_in_title_and_header(self):
        run = make_labeling_run(2, 2, run_id='r<b>&"1')
        _, _, page = self.export(run=run)
        visible = page.split('<script type="application/json"')[0]
        self.assertTrue("<b>" not in visible, "run id leaked unescaped into the visible HTML")
        self.assertIn("<title>Label AI outputs - r&lt;b&gt;&amp;&quot;1</title>", visible)
        self.assertIn("<code>r&lt;b&gt;&amp;&quot;1</code>", visible)
        self.assertEqual(embedded_payload(page)["run_id"], 'r<b>&"1')

    def test_placeholders_in_data_are_not_substituted(self):
        run = make_labeling_run(2, 2)
        run.results[0]["case"]["input"] = "__RUN_ID__ __PAYLOAD__ __TITLE__"
        _, _, page = self.export(run=run, sample=None)
        item = next(i for i in embedded_payload(page)["items"] if i["case_id"] == "case-000")
        self.assertEqual(item["input"], "__RUN_ID__ __PAYLOAD__ __TITLE__")

    def test_dict_input_and_missing_optional_fields(self):
        run = make_labeling_run(2, 2)
        run.results[0]["case"]["input"] = {"question": "hi", "user": {"plan": "pro"}}
        run.results[0]["case"]["reference"] = None
        run.results[0]["case"]["expected_behavior"] = None
        run.results[0]["output"]["meta"] = {}
        run.results[0]["output"]["error"] = "TimeoutError: upstream"
        run.results[0]["outcome"] = "fail"  # errored cases are never sampled; force this one in to test the field
        _, _, page = self.export(run=run, sample=None)
        item = next(i for i in embedded_payload(page)["items"] if i["case_id"] == "case-000")
        self.assertEqual(json.loads(item["input"]), {"question": "hi", "user": {"plan": "pro"}})
        self.assertEqual((item["reference"], item["expected_behavior"], item["context"]), ("", "", []))
        self.assertEqual(item["error"], "TimeoutError: upstream")

    def test_empty_run_still_produces_a_page(self):
        _, _, page = self.export(run=make_run([], run_id="empty"))
        self.assertEqual(embedded_payload(page)["items"], [])

    def test_page_features_named_in_the_contract(self):
        _, _, page = self.export()
        for needle in (
            '<meta name="viewport"',
            "prefers-color-scheme: dark",
            "Download labels.jsonl",
            "Copy JSONL",
            "new Blob(",
            '"labels-"',
            "localStorage",
            'key === "p"',
            "ArrowLeft",
            "ArrowRight",
            "labeled",
            'for="labeler"',
            'for="note"',
            ":focus-visible",
        ):
            self.assertHas(page, needle)
        # every localStorage access sits inside a try block
        for match in re.finditer(r"localStorage\.(?:getItem|setItem)", page):
            self.assertIn("try {", page[max(0, match.start() - 120) : match.start()])

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_inline_script_is_valid_javascript(self):
        _, _, page = self.export()
        scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
        self.assertEqual(len(scripts), 1)
        js = self.dir / "page.js"
        js.write_text(scripts[0], encoding="utf-8")
        result = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class CsvExportTests(TempDirCase):
    def read_rows(self, path):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def test_columns_bom_and_rows(self):
        run = make_labeling_run(6, 4)
        path = labeling.export_labeling_csv(run, self.dir / "out" / "labeling.csv", sample=None, seed=1)
        self.assertEqual(path.read_bytes()[:3], b"\xef\xbb\xbf")
        rows = self.read_rows(path)
        self.assertEqual(
            list(rows[0]),
            [
                "case_id", "tags", "input", "reference", "expected_behavior", "output", "output_sha", "run_id",
                "stratum", "stratified_on", "label", "note",
            ],
        )
        self.assertEqual(len(rows), 10)
        self.assertCountEqual([r["case_id"] for r in rows], [r["case_id"] for r in run.results])
        first = rows[0]
        record = run.by_case()[first["case_id"]]
        self.assertEqual(first["output_sha"], record["output"]["sha"])
        self.assertEqual(first["run_id"], "run-label")
        self.assertEqual((first["label"], first["note"]), ("", ""))
        self.assertEqual(first["input"], "What is the refund window?")

    def test_blind_and_sampled(self):
        run = make_labeling_run(30, 10)
        path = labeling.export_labeling_csv(run, self.dir / "l.csv", sample=8, seed=2)
        text = path.read_text(encoding="utf-8-sig")
        self.assertNotIn("JUDGE-REASONING-MARKER", text)
        self.assertNotIn("0.8642", text)
        rows = self.read_rows(path)
        self.assertEqual([r["case_id"] for r in rows], [r["case_id"] for r in labeling.select_sample(run, 8, 2, True)])

    def test_multiline_unicode_and_formula_cells(self):
        run = make_labeling_run(2, 2)
        run.results[0]["output"]["text"] = 'línea uno\nlínea "dos", con coma\n日本語 ✓'
        run.results[1]["output"]["text"] = '=HYPERLINK("http://evil.example","click")'
        run.results[2]["case"]["input"] = "+1 555 0100 is my number"
        path = labeling.export_labeling_csv(run, self.dir / "l.csv", sample=None, seed=0)
        by_id = {r["case_id"]: r for r in self.read_rows(path)}
        self.assertEqual(by_id["case-000"]["output"], 'línea uno\nlínea "dos", con coma\n日本語 ✓')
        self.assertTrue(by_id["case-001"]["output"].startswith("'=HYPERLINK"))
        self.assertTrue(by_id["case-002"]["input"].startswith("'+1 555"))
        self.assertEqual(by_id["case-002"]["case_id"], "case-002")  # identifiers are never altered

    def test_roundtrip_through_a_spreadsheet_into_calibration(self):
        run = make_labeling_run(n_pass=30, n_fail=30)
        path = labeling.export_labeling_csv(run, self.dir / "l.csv", sample=40, seed=9)
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, row in enumerate(rows):
            if i == 3:
                continue  # left unlabeled by the human
            row["label"] = "PASS" if run.by_case()[row["case_id"]]["passed"] else "fail"
            row["note"] = "looks wrong, really" if row["label"] == "fail" else ""
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # Excel re-saves with a BOM
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        labels = calibration.load_labels(path)
        self.assertEqual(len(labels), 39)
        self.assertEqual((labels.n_invalid, labels.n_unlabeled), (0, 1))
        self.assertTrue(all(row["output_sha"] and row["run_id"] == "run-label" for row in labels))
        cal = calibration.calibrate(run, labels, iters=200)
        self.assertEqual(cal["n"], 39)
        self.assertEqual(cal["n_excluded_sha_mismatch"], 0)
        self.assertEqual(cal["n_unmatched"], 0)
        self.assertEqual(cal["agreement"], 1.0)
        self.assertEqual(cal["kappa"], 1.0)
        # the stratified sample had half judge failures; the reweighted rates still describe the run
        self.assertAlmostEqual(cal["corrected_pass_rate"], 0.5)


class LabelsContractTests(TempDirCase):
    def test_jsonl_lines_in_the_page_format_load_and_calibrate(self):
        """The JSONL the page's download button produces (see toJsonl in the template)."""
        run = make_labeling_run(n_pass=20, n_fail=20)
        page_labels = [
            {
                "case_id": r["case_id"],
                "label": "pass" if r["passed"] else "fail",
                "output_sha": r["output"]["sha"],
                "note": "",
                "labeler": "renan",
                "run_id": "run-label",
            }
            for r in run.results
        ]
        path = self.dir / "labels-run-label.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in page_labels), encoding="utf-8")
        labels = calibration.load_labels(path)
        cal = calibration.calibrate(run, labels, iters=200)
        self.assertEqual((cal["n"], cal["n_excluded_sha_mismatch"]), (40, 0))
        self.assertEqual(cal["kappa"], 1.0)


def two_judge_run(n_both=30, n_quality_only=10, n_neither=20, run_id="run-two"):
    """Overall verdict = quality AND heuristic. `quality_only` passes quality but fails the heuristic."""

    def result(name, passed):
        return {"judge": name, "passed": passed, "score": None, "reasoning": "SECRET", "details": {}, "cost_usd": 0, "usage": {}, "error": None}

    records = []
    for i in range(n_both):
        records.append(make_record(f"both-{i:03d}", None, judges=[result("quality", True), result("heur", True)], passed=True))
    for i in range(n_quality_only):
        records.append(make_record(f"qonly-{i:03d}", None, judges=[result("quality", True), result("heur", False)], passed=False))
    for i in range(n_neither):
        records.append(make_record(f"neither-{i:03d}", None, judges=[result("quality", False), result("heur", True)], passed=False))
    return make_run(records, run_id=run_id)


class SelectSampleErrorsAndJudgeTests(unittest.TestCase):
    def test_errored_cases_are_never_sampled(self):
        records = [make_record(f"ok-{i}", i % 2 == 0, outcome="pass" if i % 2 == 0 else "fail") for i in range(20)]
        records += [make_record(f"err-{i}", False, outcome="error") for i in range(5)]
        run = make_run(records)
        for sample in (None, 12, 999):
            for stratify in (True, False):
                chosen = labeling.select_sample(run, sample, seed=1, stratify=stratify)
                self.assertFalse([r for r in chosen if r["case_id"].startswith("err")], (sample, stratify))
        self.assertEqual(len(labeling.select_sample(run, None, seed=1)), 20)

    def test_records_without_outcome_treat_target_and_judge_errors_as_errors(self):
        legacy_target = make_record("legacy-target", False)
        legacy_target["output"]["error"] = "TimeoutError: upstream"
        legacy_judge = make_record("legacy-judge", False)
        legacy_judge["judges"][0]["error"] = "ValueError: unparsable"
        run = make_run([make_record("fine", True), make_record("bad", False), legacy_target, legacy_judge])
        ids = {r["case_id"] for r in labeling.select_sample(run, None, seed=0)}
        self.assertEqual(ids, {"fine", "bad"})

    def test_an_explicit_outcome_overrides_the_legacy_error_inference(self):
        record = make_record("x", False, outcome="fail")
        record["output"]["error"] = "kept: the runner says this case was scored"
        self.assertEqual([r["case_id"] for r in labeling.select_sample(make_run([record]), None)], ["x"])

    def test_stratifies_on_the_named_judges_verdict(self):
        run = two_judge_run(30, 10, 20)  # quality passes 40, fails 20; the overall verdict passes only 30
        by_overall = labeling.select_sample(run, 20, seed=4, stratify=True)
        self.assertEqual(Counter(r["passed"] for r in by_overall), {True: 10, False: 10})
        by_quality = labeling.select_sample(run, 20, seed=4, stratify=True, judge="quality")
        quality = lambda r: next(j["passed"] for j in r["judges"] if j["judge"] == "quality")  # noqa: E731
        self.assertEqual(Counter(quality(r) for r in by_quality), {True: 10, False: 10})

    def test_named_judge_sampling_skips_cases_the_judge_did_not_score(self):
        run = two_judge_run(10, 5, 5)
        run.results[0]["judges"][0]["error"] = "ValueError: unparsable"  # quality errored here
        run.results[1]["judges"] = [j for j in run.results[1]["judges"] if j["judge"] != "quality"]  # never ran here
        ids = {r["case_id"] for r in labeling.select_sample(run, None, judge="quality")}
        self.assertNotIn(run.results[0]["case_id"], ids)
        self.assertNotIn(run.results[1]["case_id"], ids)
        self.assertEqual(len(ids), 18)

    def test_unknown_judge_is_an_error_naming_the_choices(self):
        with self.assertRaises(ValueError) as ctx:
            labeling.select_sample(two_judge_run(), 10, judge="ghost")
        self.assertIn("quality", str(ctx.exception))

    def test_default_sampling_is_unchanged_for_runs_without_errors(self):
        run = make_labeling_run(50, 50)
        again = [r["case_id"] for r in labeling.select_sample(run, 30, seed=7)]
        self.assertEqual(again, [r["case_id"] for r in labeling.select_sample(run, 30, seed=7, stratify=True, judge=None)])


class StrataInExportsTests(TempDirCase):
    def strata_of(self, rows, run, judge=None):
        return {
            row["case_id"]: calibration.decode_stratum(row["stratum"], run.run_id, row["stratified_on"], row["case_id"])
            for row in rows
        }

    def test_html_payload_records_the_design_and_an_opaque_stratum_per_item(self):
        run = two_judge_run()
        path = labeling.export_labeling_html(run, self.dir / "a.html", sample=30, seed=2)
        payload = embedded_payload(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stratified_on"], "overall")
        decoded = self.strata_of(
            [dict(it, stratified_on=payload["stratified_on"]) for it in payload["items"]], run
        )
        for case_id, stratum in decoded.items():
            self.assertEqual(stratum, "judge_pass" if run.by_case()[case_id]["passed"] else "judge_fail")

    def test_html_export_for_a_named_judge_stratifies_and_records_it(self):
        run = two_judge_run()
        path = labeling.export_labeling_html(run, self.dir / "b.html", sample=24, seed=2, judge="quality")
        payload = embedded_payload(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stratified_on"], "quality")
        items = [dict(it, stratified_on="quality") for it in payload["items"]]
        decoded = self.strata_of(items, run)
        quality = lambda cid: next(j["passed"] for j in run.by_case()[cid]["judges"] if j["judge"] == "quality")  # noqa: E731
        self.assertEqual(Counter(decoded.values()), {"judge_pass": 12, "judge_fail": 12})
        for case_id, stratum in decoded.items():
            self.assertEqual(stratum, "judge_pass" if quality(case_id) else "judge_fail")

    def test_csv_rows_carry_stratified_on_and_stratum(self):
        run = two_judge_run()
        path = labeling.export_labeling_csv(run, self.dir / "c.csv", sample=30, seed=5, judge="quality")
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["stratified_on"] for r in rows}, {"quality"})
        self.assertTrue(all(r["stratum"] for r in rows))
        text = path.read_text(encoding="utf-8-sig")
        for needle in ("judge_pass", "judge_fail", "SECRET"):
            self.assertNotIn(needle, text)  # the sheet is the labeler's screen: no readable verdict
        self.assertEqual(len({r["stratum"] for r in rows}), len(rows))  # the codes do not even group cases

    def test_random_sample_records_no_stratum(self):
        run = two_judge_run()
        path = labeling.export_labeling_csv(run, self.dir / "r.csv", sample=30, seed=5, stratify=False)
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({(r["stratified_on"], r["stratum"]) for r in rows}, {("none", "")})
        payload = embedded_payload(labeling.export_labeling_html(run, self.dir / "r.html", sample=30, stratify=False).read_text(encoding="utf-8"))
        self.assertEqual(payload["stratified_on"], "none")
        self.assertEqual({it["stratum"] for it in payload["items"]}, {""})

    def test_plain_strata_can_be_switched_on(self):
        run = two_judge_run()
        old = labeling.BLIND_STRATA
        labeling.BLIND_STRATA = False
        try:
            path = labeling.export_labeling_csv(run, self.dir / "p.csv", sample=30, seed=5)
        finally:
            labeling.BLIND_STRATA = old
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["stratum"] for r in rows}, {"judge_pass", "judge_fail"})

    def test_export_judge_must_exist(self):
        with self.assertRaises(ValueError):
            labeling.export_labeling_csv(two_judge_run(), self.dir / "x.csv", judge="ghost")
        with self.assertRaises(ValueError):
            labeling.export_labeling_html(two_judge_run(), self.dir / "x.html", judge="ghost")

    def test_csv_roundtrip_into_a_named_judge_calibration_uses_the_recorded_strata(self):
        run = two_judge_run(60, 20, 40)
        path = labeling.export_labeling_csv(run, self.dir / "l.csv", sample=40, seed=3)  # stratified on the OVERALL verdict
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:  # a perfect labeler who agrees with the quality judge
            quality = next(j["passed"] for j in run.by_case()[row["case_id"]]["judges"] if j["judge"] == "quality")
            row["label"] = "pass" if quality else "fail"
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cal = calibration.calibrate(run, calibration.load_labels(path), judge="quality", iters=100)
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertEqual(cal["strata"]["stratified_on"], "overall")
        self.assertEqual(cal["strata"]["run_cases"], {"judge_pass": 60, "judge_fail": 60})
        self.assertFalse(any("do not record how they were sampled" in w for w in cal["warnings"]))
        self.assertEqual((cal["tpr"], cal["tnr"], cal["kappa"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(cal["human_pass_rate_weighted"], 80 / 120, delta=0.06)  # quality passes 80 of 120 (sampling noise)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_page_download_writes_stratified_on_and_stratum(self):
        """Run the page's own script against a stub DOM and read what the Copy button hands to the clipboard."""
        run = two_judge_run(6, 2, 4)
        page = labeling.export_labeling_html(run, self.dir / "js.html", sample=None, seed=1).read_text(encoding="utf-8")
        script = re.findall(r"<script>(.*?)</script>", page, re.S)[0]
        payload = embedded_payload(page)
        harness = self.dir / "harness.js"
        harness.write_text(
            """
const vm = require("vm"), fs = require("fs");
const script = fs.readFileSync(process.argv[2], "utf8");
const payload = fs.readFileSync(process.argv[3], "utf8");
const handlers = {}, elements = {}, captured = [];
function element(id) {
  if (elements[id]) return elements[id];
  const base = { textContent: id === "items-data" ? payload : "", value: "", checked: false };
  elements[id] = new Proxy(base, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === "addEventListener") return (evt, fn) => { handlers[id + ":" + evt] = fn; };
      if (k === "classList") return { toggle() {} };
      return () => undefined;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return elements[id];
}
const labels = {};
for (const it of JSON.parse(payload).items) labels[it.case_id] = { label: it.case_id.startsWith("both") ? "pass" : "fail", note: "n" };
const store = { ["evalsprint:labels:" + JSON.parse(payload).run_id]: JSON.stringify({ labeler: "ana", autoAdvance: true, index: 0, labels }) };
const sandbox = {
  JSON, Math, console, Promise, Blob: function () {}, URL: { createObjectURL() { return "u"; }, revokeObjectURL() {} },
  setTimeout() {},
  document: { getElementById: element, createElement: () => element("_created_" + Math.random()), addEventListener() {}, body: { appendChild() {} } },
  window: { localStorage: { getItem: (k) => store[k] || null, setItem() {} }, scrollTo() {}, addEventListener() {} },
  navigator: { clipboard: { writeText: (text) => { captured.push(text); return Promise.resolve(); } } },
};
vm.runInNewContext(script, sandbox);
handlers["copy:click"]();
console.log(JSON.stringify(captured));
""",
            encoding="utf-8",
        )
        script_path, payload_path = self.dir / "page.js", self.dir / "payload.json"
        script_path.write_text(script, encoding="utf-8")
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(["node", str(harness), str(script_path), str(payload_path)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        (text,) = json.loads(result.stdout)
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row["stratified_on"], "overall")
            self.assertEqual(row["run_id"], "run-two")
            self.assertEqual(row["labeler"], "ana")
            decoded = calibration.decode_stratum(row["stratum"], "run-two", "overall", row["case_id"])
            self.assertEqual(decoded, "judge_pass" if run.by_case()[row["case_id"]]["passed"] else "judge_fail")


class NamedJudgeBiasTests(unittest.TestCase):
    """The workflow the README documents: label (stratified on the overall verdict), then calibrate --judge NAME.

    Simulated with the real sampler and the real calibration on runs where a second (heuristic) judge catches half of
    the bad outputs independently. Seeded, so deterministic. Cycle-1 finding: TNR bias -12.7 pp with 40 labels."""

    N, LABELS, REPS = 200, 40, 200

    def one(self, seed, design):
        rng = random.Random(9000 + seed)

        def judge(name, passed):
            return {"judge": name, "passed": passed, "score": None, "reasoning": "", "details": {}, "cost_usd": 0, "usage": {}, "error": None}

        records, humans = [], {}
        for i in range(self.N):
            human = rng.random() < 0.85
            quality = rng.random() < 0.95 if human else rng.random() >= 0.80
            heuristic = rng.random() < 0.99 if human else rng.random() >= 0.50
            records.append(make_record(f"c{i:04d}", None, judges=[judge("quality", quality), judge("heur", heuristic)], passed=quality and heuristic))
            humans[f"c{i:04d}"] = human
        run = make_run(records)
        rows = []
        for record in labeling.select_sample(run, self.LABELS, seed=seed, stratify=True):
            row = label(record["case_id"], "pass" if humans[record["case_id"]] else "fail")
            row["stratified_on"] = design
            row["stratum"] = calibration.stratum_token(run.run_id, design, record["case_id"], calibration.stratum_of(record["passed"]))
            rows.append(row)
        cal = calibration.calibrate(run, rows, judge="quality", seed=seed, iters=10)
        quality = {r["case_id"]: r["judges"][0]["passed"] for r in records}
        tn = sum(1 for c, h in humans.items() if not h and not quality[c])
        fail = sum(1 for h in humans.values() if not h)
        return cal["tnr"] - tn / fail

    def test_ht_weights_on_the_recorded_strata_remove_the_bias(self):
        biases = [self.one(s, "overall") for s in range(self.REPS)]
        self.assertLess(abs(sum(biases) / len(biases)), 0.03)

    def test_weighting_on_the_judges_own_verdict_was_biased(self):
        # what calibrate did before: strata = the calibrated judge's verdict, although the sample was stratified on the overall one
        biases = [self.one(s, "quality") for s in range(self.REPS)]
        self.assertLess(sum(biases) / len(biases), -0.08)


class KeyedStrataExportTests(TempDirCase):
    """The stratum code of a label row is an HMAC keyed with `<run_dir>/.label_secret` (created at the first export)."""

    TOKEN = re.compile(r"^s-[0-9a-f]{16}$")

    def saved(self, run=None, name="run"):
        run = run or two_judge_run()
        return RunResult(run_dir=self.dir / name, summary=run.summary, results=run.results)

    def rows_of(self, path):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def secret_path(self, run):
        return Path(run.run_dir) / ".label_secret"

    def test_the_first_export_creates_a_private_secret_and_never_writes_it_out(self):
        run = self.saved()
        html = labeling.export_labeling_html(run, self.dir / "out" / "l.html", sample=30, seed=2)
        path = self.secret_path(run)
        self.assertTrue(path.is_file())
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        secret_hex = path.read_text(encoding="ascii").strip()
        self.assertRegex(secret_hex, r"^[0-9a-f]{64}$")
        csv_path = labeling.export_labeling_csv(run, self.dir / "out" / "l.csv", sample=30, seed=2)
        for exported in (html, csv_path):
            text = exported.read_text(encoding="utf-8-sig")
            self.assertNotIn(secret_hex, text)
            self.assertNotIn(".label_secret", text)
            self.assertNotIn(secret_hex[:20], text)
        self.assertEqual(sorted(p.name for p in (self.dir / "out").iterdir()), ["l.csv", "l.html"])  # nothing else is written next to them

    def test_tokens_are_hmac_codes_that_only_the_secret_decodes(self):
        run = self.saved()
        rows = self.rows_of(labeling.export_labeling_csv(run, self.dir / "l.csv", sample=40, seed=4))
        secret = calibration.read_label_secret(run.run_dir)
        self.assertIsNotNone(secret)
        self.assertEqual(len({r["stratum"] for r in rows}), len(rows))  # they do not group cases
        for row in rows:
            self.assertRegex(row["stratum"], self.TOKEN)
            truth = "judge_pass" if run.by_case()[row["case_id"]]["passed"] else "judge_fail"
            self.assertEqual(calibration.stratum_status(row["stratum"], run.run_id, row["stratified_on"], row["case_id"], secret), (truth, "hmac"))
            # what a reader with the source code and the exported file can compute matches nothing
            self.assertEqual(calibration.decode_stratum(row["stratum"], run.run_id, row["stratified_on"], row["case_id"]), None)
            for stratum in calibration.STRATA:
                self.assertNotEqual(row["stratum"], calibration.stratum_token(run.run_id, row["stratified_on"], row["case_id"], stratum))

    def test_html_tokens_are_keyed_too(self):
        run = self.saved()
        payload = embedded_payload(labeling.export_labeling_html(run, self.dir / "l.html", sample=30, seed=2).read_text(encoding="utf-8"))
        self.assertTrue(payload["items"])
        for item in payload["items"]:
            self.assertRegex(item["stratum"], self.TOKEN)

    def test_a_second_export_reuses_the_secret_so_earlier_labels_stay_verifiable(self):
        run = self.saved()
        first = self.rows_of(labeling.export_labeling_csv(run, self.dir / "a.csv", sample=60, seed=1))
        secret_text = self.secret_path(run).read_text(encoding="ascii")
        second = self.rows_of(labeling.export_labeling_csv(run, self.dir / "b.csv", sample=60, seed=9))
        self.assertEqual(self.secret_path(run).read_text(encoding="ascii"), secret_text)
        tokens_a = {r["case_id"]: r["stratum"] for r in first}
        tokens_b = {r["case_id"]: r["stratum"] for r in second}
        common = set(tokens_a) & set(tokens_b)
        self.assertTrue(common)
        self.assertTrue(all(tokens_a[c] == tokens_b[c] for c in common))  # the same case in the same run has the same code
        other = self.saved(name="other-run-folder")
        labeling.export_labeling_csv(other, self.dir / "c.csv", sample=10, seed=1)
        self.assertNotEqual(self.secret_path(other).read_text(encoding="ascii"), secret_text)  # one secret per run folder

    def test_a_garbled_secret_is_replaced_by_a_valid_one(self):
        run = self.saved()
        run.run_dir.mkdir(parents=True)
        self.secret_path(run).write_text("garbage\n", encoding="ascii")
        rows = self.rows_of(labeling.export_labeling_csv(run, self.dir / "g.csv", sample=20, seed=1))
        secret = calibration.read_label_secret(run.run_dir)
        self.assertIsNotNone(secret)
        self.assertTrue(all(self.TOKEN.match(r["stratum"]) for r in rows))

    def test_the_tokens_verify_in_calibrate_and_a_lost_secret_is_warned_about(self):
        run = self.saved(two_judge_run(60, 20, 40))
        path = labeling.export_labeling_csv(run, self.dir / "l.csv", sample=40, seed=3)
        rows = self.rows_of(path)
        for row in rows:  # a perfect labeler who agrees with the overall verdict
            row["label"] = "pass" if run.by_case()[row["case_id"]]["passed"] else "fail"
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cal = calibration.calibrate(run, calibration.load_labels(path), iters=100)
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertEqual(cal["strata"]["n_unverifiable_tokens"], 0)
        self.assertFalse(any(".label_secret" in w or "does not match" in w for w in cal["warnings"]))
        self.assertEqual((cal["tpr"], cal["tnr"]), (1.0, 1.0))
        self.secret_path(run).unlink()  # the run folder was copied without its hidden file
        lost = calibration.calibrate(run, calibration.load_labels(path), iters=100)
        self.assertEqual(lost["strata"]["source"], "recomputed")
        self.assertEqual(lost["strata"]["n_unverifiable_tokens"], len(rows))
        self.assertTrue(any(".label_secret" in w and "recomputed" in w for w in lost["warnings"]))
        self.assertEqual(lost["strata"]["weights"], cal["strata"]["weights"])  # the weights never came from the token
        self.assertEqual((lost["tpr"], lost["tnr"], lost["kappa_weighted"]), (cal["tpr"], cal["tnr"], cal["kappa_weighted"]))

    def test_legacy_tokens_of_earlier_exports_still_calibrate(self):
        run = self.saved(two_judge_run(60, 20, 40))
        rows = []
        for record in labeling.select_sample(run, 40, seed=3, stratify=True):
            row = label(record["case_id"], "pass" if record["passed"] else "fail")
            row["stratified_on"] = "overall"
            row["stratum"] = calibration.stratum_token(run.run_id, "overall", record["case_id"], calibration.stratum_of(record["passed"]))
            self.assertRegex(row["stratum"], r"^s-[0-9a-f]{10}$")
            rows.append(row)
        labeling.export_labeling_csv(run, self.dir / "new.csv", sample=10, seed=1)  # a secret now exists next to the run
        cal = calibration.calibrate(run, rows, iters=50)
        self.assertEqual(cal["strata"]["source"], "recorded")
        self.assertFalse(any("does not match" in w or ".label_secret" in w for w in cal["warnings"]))

    def test_a_run_without_a_folder_falls_back_to_the_legacy_code_with_a_warning(self):
        run = two_judge_run()
        with self.assertWarnsRegex(RuntimeWarning, "no folder on disk"):
            rows = self.rows_of(labeling.export_labeling_csv(run, self.dir / "n.csv", sample=20, seed=1))
        for row in rows:
            self.assertRegex(row["stratum"], r"^s-[0-9a-f]{10}$")
            self.assertIsNotNone(calibration.decode_stratum(row["stratum"], run.run_id, row["stratified_on"], row["case_id"]))
        with self.assertWarnsRegex(RuntimeWarning, "no folder on disk"):
            labeling.export_labeling_html(run, self.dir / "n.html", sample=20, seed=1)

    def test_a_secret_that_cannot_be_written_falls_back_with_a_warning_and_the_export_still_works(self):
        blocker = self.dir / "a-file"
        blocker.write_text("x", encoding="utf-8")
        base = two_judge_run()
        run = RunResult(run_dir=blocker / "run", summary=base.summary, results=base.results)  # its parent is a file
        with self.assertWarnsRegex(RuntimeWarning, "could not create"):
            rows = self.rows_of(labeling.export_labeling_csv(run, self.dir / "u.csv", sample=20, seed=1))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(re.match(r"^s-[0-9a-f]{10}$", r["stratum"]) for r in rows))

    def test_no_secret_is_needed_for_a_random_sample_or_for_plain_strata(self):
        run = self.saved()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning either
            labeling.export_labeling_csv(run, self.dir / "r.csv", sample=20, seed=1, stratify=False)
            labeling.export_labeling_html(run, self.dir / "r.html", sample=20, seed=1, stratify=False)
        self.assertFalse(self.secret_path(run).exists())
        old = labeling.BLIND_STRATA
        labeling.BLIND_STRATA = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                rows = self.rows_of(labeling.export_labeling_csv(run, self.dir / "p.csv", sample=20, seed=1))
        finally:
            labeling.BLIND_STRATA = old
        self.assertFalse(self.secret_path(run).exists())
        self.assertTrue({r["stratum"] for r in rows} <= {"judge_pass", "judge_fail"})


if __name__ == "__main__":
    unittest.main()
