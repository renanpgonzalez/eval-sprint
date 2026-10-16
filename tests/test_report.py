"""Tests for the HTML report (evalsprint/report.py).

Runs, comparisons and calibrations are built by hand from the contracts in
docs/ARCHITECTURE.md and orchestration/.../cycle2/contracts.md (C1 outcome and
error accounting, C2 repeats, C3 gate, C5 calibration, C6 report), so nothing
here depends on the runner, compare or calibration modules producing them.
"""

import html as html_lib
import math
import re
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from evalsprint.report import (
    DASH,
    MAX_FAILURES,
    fmt_ci,
    fmt_latency,
    fmt_money,
    fmt_pct,
    render_report,
    write_report,
)
from evalsprint.runner import RunResult

XSS = "<script>alert(1)</script>"
XSS_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"


# ---------------------------------------------------------------------------- builders
def judge(name="quality", passed=True, criteria=None, reasoning="", cost=0.002, error=None):
    return {
        "judge": name,
        "passed": passed,
        "score": 1.0 if passed else 0.5,
        "reasoning": reasoning,
        "details": {"criteria": criteria} if criteria else {},
        "cost_usd": cost,
        "usage": {},
        "error": error,
    }


def record(
    case_id,
    passed=True,
    tags=("billing",),
    input_="How do I get a refund?",
    text="Within 30 days.",
    cost=0.001,
    latency=0.5,
    error=None,
    judges=None,
    outcome=None,
    error_source=None,
    attempts=None,
    stability=None,
    meta=None,
    **case_extra,
):
    """One results.jsonl record with the second-contract fields (`outcome`, `error_source`, `attempts`)."""
    if judges is None:
        judges = [judge(passed=passed, criteria=None if passed else {"grounded": {"pass": False, "reasoning": "claims 60 days"}, "polite": {"pass": True, "reasoning": "fine"}})]
    if outcome is None:
        if error:
            outcome, error_source = "error", "target"
        elif any(j.get("error") for j in judges):
            outcome, error_source = "error", "judge"
        else:
            outcome = "pass" if passed else "fail"
    rec = {
        "case_id": case_id,
        "case": {"id": case_id, "input": input_, "reference": None, "expected_behavior": None, "tags": list(tags), "checks": [], "metadata": {}, **case_extra},
        "case_sha": "0123456789ab",
        "output": {
            "text": text,
            "sha": "abc123",
            "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0, "cache_write_tokens": 0},
            "cost_usd": cost,
            "latency_s": latency,
            "meta": dict(meta or {}),
            "error": error,
        },
        "judges": judges,
        "outcome": outcome,
        "error_source": error_source if outcome == "error" else None,
        "passed": outcome == "pass",
    }
    if attempts is not None:
        rec["attempts"] = attempts
        rec["stability"] = stability
    return rec


def legacy(rec):
    """The same record as the first version of the runner wrote it."""
    rec = dict(rec)
    for key in ("outcome", "error_source", "case_sha", "attempts", "stability"):
        rec.pop(key, None)
    return rec


def _wilson(k, n, z=1.96):
    if not n:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (centre - margin) / den), min(1.0, (centre + margin) / den)]


def _rate(k, n):
    return {"n": n, "n_passed": k, "pass_rate": (k / n) if n else None, "ci": _wilson(k, n)}


def build_summary(records, tag_groups=None, stability=None, legacy_keys=False):
    """summary.json as defined by contracts C1, C2 and C8 (errors are excluded from every rate)."""
    n = len(records)
    scored = [r for r in records if r["outcome"] != "error"]
    n_pass = sum(1 for r in records if r["outcome"] == "pass")
    n_fail = sum(1 for r in records if r["outcome"] == "fail")
    errors = [r for r in records if r["outcome"] == "error"]
    by_tag_counts = {}
    for r in scored:
        for tag in r["case"]["tags"] or ["(untagged)"]:
            k = by_tag_counts.setdefault(tag, [0, 0])
            k[0] += 1
            k[1] += int(r["outcome"] == "pass")
    by_judge = {}
    for r in records:
        if (r["output"].get("error")):
            continue
        for j in r["judges"]:
            by_judge.setdefault(j["judge"], []).append(j)
    costs = [r["output"]["cost_usd"] for r in records if r["output"].get("cost_usd") is not None]
    latencies = sorted(r["output"]["latency_s"] for r in records if r["output"].get("latency_s") is not None)
    summary = {
        "n_cases": n,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_scored": n_pass + n_fail,
        "pass_rate": (n_pass / (n_pass + n_fail)) if (n_pass + n_fail) else None,
        "pass_rate_ci": _wilson(n_pass, n_pass + n_fail),
        "n_errors": len(errors),
        "n_target_errors": sum(1 for r in errors if r["error_source"] == "target"),
        "n_judge_errors": sum(1 for r in errors if r["error_source"] == "judge"),
        "error_rate": (len(errors) / n) if n else 0.0,
        "by_tag": {t: _rate(p, k) for t, (k, p) in sorted(by_tag_counts.items())},
        "by_judge": {
            name: {
                **_rate(sum(1 for j in rows if j["passed"] and not j.get("error")), sum(1 for j in rows if not j.get("error"))),
                "mean_score": 0.9,
                "n_errors": sum(1 for j in rows if j.get("error")),
                "cost_usd": sum(j.get("cost_usd") or 0 for j in rows),
            }
            for name, rows in by_judge.items()
        },
        "cost": {
            "system_total_usd": sum(costs),
            "system_mean_usd": (sum(costs) / len(costs)) if costs else None,
            "system_per_1k_usd": (sum(costs) / len(costs) * 1000) if costs else None,
            "judge_total_usd": sum(j.get("cost_usd") or 0 for r in records for j in r["judges"]),
            "n_priced": len(costs),
        },
        "latency": {
            "mean_s": (sum(latencies) / len(latencies)) if latencies else None,
            "p50_s": latencies[len(latencies) // 2] if latencies else None,
            "p95_s": latencies[-1] if latencies else None,
            "max_s": latencies[-1] if latencies else None,
        },
        "tokens": {"input": 100 * n, "output": 20 * n, "cache_read": 0, "cache_write": 0},
        "models": dict(Counter(r["output"].get("model") or "unknown" for r in records)),
    }
    if tag_groups is not None:
        summary["tag_groups"] = tag_groups
    if stability is not None:
        summary["stability"] = stability
    if legacy_keys:  # the first-version summary: errors counted inside the pass rate, no scored/failed split
        for key in ("n_failed", "n_scored", "n_target_errors", "n_judge_errors"):
            summary.pop(key)
    return summary


def make_run(records, tag_groups=None, stability=None, **summary_extra):
    summary = {
        "run_id": "20260929-100000-v2",
        "suite": "support-bot",
        "label": "v2",
        "created_at": "2026-09-29T10:00:00+00:00",
        "evalsprint_version": "0.1.0",
        "target": {"callable": "bot.py:answer", "params": {"model": "claude-haiku-4-5", "prompt": "v2"}},
        "judges": ["quality"],
        "pass_rule": "all",
        "gate": {},
        "dataset": "/work/evals/cases.jsonl",
        **build_summary(records, tag_groups, stability),
        "duration_s": 1.5,
    }
    summary.update(summary_extra)
    return RunResult(run_dir=None, summary=summary, results=records)


def sample_run():
    return make_run(
        [
            record("a-1", True),
            record("b-2", False, tags=("shipping",), input_="Where is my order?", text="I don't know."),
            record("c-3", True, tags=("billing", "core")),
        ]
    )


def regression_comparison():
    return {
        "verdict": "regression",
        "reasons": ["pass rate dropped 33.3 p.p. (limit 3.0)", f"critical tag: {XSS}"],
        "warnings": ["critical tag 'legal' matches no paired case"],
        "baseline_run_id": "20260901-090000-v1",
        "candidate_run_id": "20260929-100000-v2",
        "n_paired": 3,
        "unpaired": {"only_baseline": ["gone-1"], "only_candidate": []},
        "pass_rate": {"baseline": 1.0, "candidate": 0.6667, "delta": -0.3333, "baseline_ci": [0.4, 1.0], "candidate_ci": [0.2, 0.9], "b_pass_to_fail": 1, "c_fail_to_pass": 0, "p_value": 1.0},
        "score": None,
        "by_tag": {"billing": {"n": 2, "baseline": 1.0, "candidate": 1.0, "delta": 0.0}, "shipping": {"n": 1, "baseline": 1.0, "candidate": 0.0, "delta": -1.0}},
        "newly_failing": ["b-2"],
        "newly_passing": ["z-9"],
        "critical_regressions": ["b-2"],
        "cost": {"baseline_mean_usd": 0.001, "candidate_mean_usd": 0.00125, "delta_pct": 25.0},
        "latency_p95": {"baseline_s": 1.0, "candidate_s": 0.9, "delta_pct": -10.0},
        "error_rate": {"baseline": 0.0, "candidate": 0.0},
        "gate": {"max_pass_rate_drop": 0.03, "alpha": 0.05, "require_significance": False, "min_pass_rate": None, "critical_tags": ["shipping"], "max_cost_increase_pct": None, "max_p95_latency_increase_pct": None, "max_error_rate": 0.05},
    }


def improvement_comparison():
    cmp = regression_comparison()
    cmp.update(verdict="pass", reasons=[], warnings=[], newly_failing=[], newly_passing=["b-2"], critical_regressions=[], unpaired={"only_baseline": [], "only_candidate": []})
    cmp["pass_rate"] = {**cmp["pass_rate"], "baseline": 0.6, "candidate": 0.642, "delta": 0.042, "p_value": 0.5}
    return cmp


def cycle2_comparison(**overrides):
    """A comparison carrying the keys the third contract (C3) adds."""
    cmp = regression_comparison()
    cmp["pass_rate"] = {**cmp["pass_rate"], "p_value": 0.016, "p_value_two_sided": 0.031}
    cmp.update(
        removed_cases=["old-1", "old-2"],
        changed_cases=["b-2"],
        errors={
            "baseline": {"n_errors": 0, "n_judge_errors": 0, "n_target_errors": 0, "error_rate": 0.0},
            "candidate": {"n_errors": 3, "n_judge_errors": 2, "n_target_errors": 1, "error_rate": 0.0667},
        },
        effective_max_pass_rate_drop=0.05,
        gate_mode="auto-small",
        calibration_status={"judge": "quality", "kappa": 0.72, "kappa_ci": [0.5, 0.9], "ok": False},
    )
    cmp["gate"] = {**cmp["gate"], "max_pass_rate_drop": "auto", "max_removed_cases": 0, "max_changed_cases": 0, "min_judge_kappa": 0.6, "require_calibration": False}
    return cmp


def calibration_dict(**overrides):
    cal = {
        "judge": "overall",
        "run_id": "20260929-100000-v2",
        "n": 40,
        "n_excluded_sha_mismatch": 2,
        "n_unmatched": 1,
        "agreement": 0.875,
        "kappa": 0.72,
        "kappa_label": "substantial",
        "kappa_ci": [0.5, 0.9],
        "tpr": 0.9,
        "tnr": 0.8,
        "confusion": {"human_pass_judge_pass": 18, "human_pass_judge_fail": 2, "human_fail_judge_pass": 3, "human_fail_judge_fail": 17},
        "human_pass_rate": 0.5,
        "judge_pass_rate_labeled": 0.525,
        "run_observed_pass_rate": 0.7,
        "corrected_pass_rate": 0.714,
        "corrected_pass_rate_ci": [0.6, 0.82],
        "disagreements": [
            {"case_id": "d-1", "human": "fail", "judge": "pass", "judge_reasoning": "looked grounded", "note": f"invented a fee {XSS}", "input": "Fee?", "output": "It is $5."},
        ],
        "recommendation": "Judge is usable; add the disagreements as few-shot examples.",
    }
    cal.update(overrides)
    return cal


def cycle2_calibration(**overrides):
    """A calibration carrying the keys the fifth contract (C5) adds."""
    extra = {
        "agreement_weighted": 0.91,
        "tnr_reliable": False,
        "kappa_decision": "inconclusive",
        "labels_needed_estimate": 120,
        "warnings": ["only 3 human-fail cases got past the judge, the true-negative rate is unreliable"],
    }
    extra.update(overrides)
    return calibration_dict(**extra)


class BalancedHTML(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag not in self.VOID:
            self.errors.append(tag)


class TestFormatting(unittest.TestCase):
    def test_money(self):
        self.assertEqual(fmt_money(0.0123), "$0.0123")
        self.assertEqual(fmt_money(0.5), "$0.5000")
        self.assertEqual(fmt_money(12.3412), "$12.34")
        self.assertEqual(fmt_money(1234.5), "$1,234.50")
        self.assertEqual(fmt_money(0.00001), "<$0.0001")

    def test_missing_values_are_dashes(self):
        self.assertEqual(fmt_money(None), DASH)
        self.assertEqual(fmt_money(0.0), DASH)
        self.assertEqual(fmt_pct(None), DASH)
        self.assertEqual(fmt_ci(None), DASH)
        self.assertEqual(fmt_ci([None, 0.5]), DASH)
        self.assertEqual(fmt_latency(None), DASH)

    def test_percentages_and_latency(self):
        self.assertEqual(fmt_pct(0.8333), "83.3%")
        self.assertEqual(fmt_pct(1.0), "100.0%")
        self.assertEqual(fmt_ci([0.721, 0.914]), "72.1–91.4%")
        self.assertEqual(fmt_ci([0.721, 0.914], sep="-"), "72.1-91.4%")
        self.assertEqual(fmt_latency(0.0004), "<1 ms")
        self.assertEqual(fmt_latency(0.25), "250 ms")
        self.assertEqual(fmt_latency(2.345), "2.35 s")


class ReportTestCase(unittest.TestCase):
    """assertIn/assertNotIn that name the snippet instead of dumping the whole document."""

    def assertHas(self, snippet, doc):
        if snippet not in doc:
            self.fail(f"report does not contain {snippet!r}")

    def assertLacks(self, snippet, doc):
        if snippet in doc:
            self.fail(f"report unexpectedly contains {snippet!r}")

    def assertBalanced(self, doc):
        parser = BalancedHTML()
        parser.feed(doc)
        self.assertEqual((parser.stack, parser.errors), ([], []))

    def assertSelfContained(self, doc):
        for external in ("<script", "<link", "@import", "url(", "src=", "http://", "https://"):
            self.assertLacks(external, doc)


def plain(doc):
    """The document with tags removed, for asserting on what a reader sees."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", doc))


def seen(doc):
    """`plain` with HTML entities decoded (apostrophes, quotes): the text as the reader sees it."""
    return html_lib.unescape(plain(doc))


class TestRenderReport(ReportTestCase):
    def test_self_contained_responsive_document(self):
        doc = render_report(sample_run(), regression_comparison(), calibration_dict())
        self.assertTrue(doc.startswith("<!doctype html>"))
        self.assertHas('name="viewport"', doc)
        self.assertHas("prefers-color-scheme:dark", doc.replace(" ", ""))
        self.assertHas("@media print", doc)
        self.assertHas(":root{", doc)
        self.assertSelfContained(doc)
        self.assertBalanced(doc)

    def test_output_is_deterministic(self):
        run = sample_run()
        self.assertEqual(render_report(run, regression_comparison()), render_report(run, regression_comparison()))

    def test_header_and_kpis(self):
        doc = render_report(sample_run())
        for expected in ("support-bot", "v2", "20260929-100000-v2", "2026-09-29 10:00 UTC", "bot.py:answer", "claude-haiku-4-5", "cases.jsonl", "3 cases", "every judge must pass"):
            self.assertHas(expected, doc)
        self.assertHas("66.7%", doc)  # pass rate: 2 of 3 cases pass
        self.assertHas("95% CI", doc)
        self.assertHas("$1.00", doc)  # cost per 1k requests
        self.assertHas("500 ms", doc)  # latency p50
        self.assertLacks("Regression detected", doc)
        self.assertLacks("vs baseline", doc)

    def test_failures_are_listed_with_criteria_and_anchors(self):
        doc = render_report(sample_run())
        self.assertHas('<details class="case" id="case-b-2">', doc)
        self.assertLacks('id="case-a-1"', doc)  # passing cases are not listed
        self.assertHas("Where is my order?", doc)
        self.assertHas("I don&#x27;t know.", doc)
        self.assertHas("✘ fail", doc)
        self.assertHas("✔ pass", doc)
        self.assertHas("claims 60 days", doc)
        self.assertHas("1 of 3 scored cases failed", doc.replace("<strong>", "").replace("</strong>", ""))

    def test_untrusted_strings_are_escaped_everywhere(self):
        bad = record(
            f"id-{XSS}",
            False,
            tags=(XSS,),
            input_=XSS,
            text=f'"><img src=x onerror=alert(2)> {XSS}',
            judges=[judge(f"j-{XSS}", False, criteria={f"c-{XSS}": {"pass": False, "reasoning": XSS}}, error=None)],
            reference=XSS,
            expected_behavior=XSS,
        )
        bad["output"]["meta"]["context"] = [XSS]
        err_target = record(f"t-{XSS}", False, text="", error=XSS)
        err_judge = record(f"j-{XSS}", False, judges=[judge(f"jj-{XSS}", False, error=XSS)])
        run = make_run(
            [bad, err_target, err_judge],
            tag_groups={XSS: [XSS], f"g-{XSS}": [f"missing-{XSS}"]},
            stability={"n_repeated": 1, "flip_rate": 1.0},
            suite=XSS,
            label=XSS,
            target={"callable": XSS, "params": {"model": XSS, XSS: XSS}},
            judges=[XSS],
        )
        bad["attempts"] = [{"outcome": XSS, "passed": False, "output_sha": "1"}, {"outcome": "fail", "passed": False, "output_sha": "2"}]
        bad["stability"] = 0.5
        cal = cycle2_calibration(judge=XSS, recommendation=XSS, warnings=[XSS], kappa_decision=XSS, disagreements=[{"case_id": XSS, "human": XSS, "judge": XSS, "judge_reasoning": XSS, "note": XSS, "input": XSS, "output": XSS}])
        cmp = cycle2_comparison()
        cmp.update(
            newly_failing=[bad["case_id"]],
            critical_regressions=[bad["case_id"]],
            reasons=[XSS],
            baseline_run_id=XSS,
            candidate_run_id=XSS,
            warnings=[XSS],
            removed_cases=[XSS],
            changed_cases=[XSS],
            gate_mode=XSS,
            calibration_status={"judge": XSS, "kappa": 0.5, "kappa_ci": [0.3, 0.7], "ok": False},
        )
        base_bad = record(bad["case_id"], True, text=f"<img src=x onerror=alert(3)> {XSS}")
        baseline = make_run([base_bad])
        doc = render_report(run, cmp, cal, baseline=baseline, monthly_requests=1000)
        self.assertLacks(XSS, doc)
        self.assertLacks("<img", doc)
        self.assertHas(XSS_ESCAPED, doc)
        self.assertBalanced(doc)

    def test_failure_cap_is_announced(self):
        run = make_run([record(f"f-{i:03d}", False) for i in range(MAX_FAILURES + 5)])
        doc = render_report(run)
        self.assertEqual(doc.count('<details class="case" id='), MAX_FAILURES)
        self.assertHas(f"Showing {MAX_FAILURES} of {MAX_FAILURES + 5} failing cases", doc)
        self.assertHas("5 hidden", doc)

    def test_no_failures(self):
        doc = render_report(make_run([record("a-1"), record("a-2")]))
        self.assertHas("No failing cases", doc)
        self.assertHas("All <strong>2</strong> scored cases passed", doc)
        self.assertLacks('<details class="case"', doc)

    def test_missing_costs_latencies_and_usage_render_dashes(self):
        records = [record(f"a-{i}", True, cost=None, latency=None, judges=[judge(cost=0.0)]) for i in range(3)]
        for r in records:
            r["output"].update(model=None, usage={})
        doc = render_report(make_run(records))
        for label in ("Cost / 1k requests", "Mean cost / request", "Latency p50", "Latency p95", "Eval run cost"):
            self.assertRegex(doc, rf'{re.escape(label)}</div><div class="tile-value">{DASH}</div>')
        self.assertHas("no cost data reported", doc)
        self.assertHas("No cost data", doc)
        for junk in ("None", "nan", "inf"):
            self.assertLacks(junk, doc)
        self.assertBalanced(doc)

    def test_partially_priced_run_is_flagged(self):
        run = make_run([record("a-1"), record("a-2", cost=None), record("a-3", cost=None)])
        doc = render_report(run)
        self.assertHas("priced 1 of 3 requests", doc)
        self.assertHas("Only 1 of 3 requests could be priced", doc)

    def test_zero_cases(self):
        doc = render_report(make_run([]))
        self.assertHas("No failing cases", doc)
        self.assertLacks("0.0–100.0%", doc)  # no interval for an empty run
        self.assertLacks("None", doc)
        self.assertBalanced(doc)

    def test_case_ids_with_spaces_and_quotes_get_valid_anchors(self):
        odd = 'a b"c'
        cmp = regression_comparison()
        cmp.update(newly_failing=[odd], critical_regressions=[])
        doc = render_report(make_run([record(odd, False)]), cmp)
        self.assertHas('id="case-a_b&quot;c"', doc)
        self.assertHas('href="#case-a_b%22c"', doc)

    def test_tags_table_is_sorted_weakest_first_and_flags_small_samples(self):
        records = [record(f"g-{i}", True, tags=("good",)) for i in range(12)] + [record(f"w-{i}", i < 2, tags=("weak",)) for i in range(12)]
        doc = render_report(make_run(records))
        self.assertLess(doc.index("<strong>weak</strong>"), doc.index("<strong>good</strong>"))
        self.assertLacks("small sample", doc)
        self.assertHas("Weakest segment: <strong>weak</strong> at 16.7% (n = 12)", doc)
        small = render_report(sample_run())
        self.assertHas("small sample", small)
        self.assertLacks("Weakest segment", small)  # no tag has enough cases to single out

    def test_key_findings(self):
        records = [record(f"a-{i}", False, tags=("x",)) for i in range(3)] + [record("a-9", True, tags=("y",))]
        records.append(record("boom", False, text="", error="Timeout"))
        doc = render_report(make_run(records), None, calibration_dict(n=12, kappa=0.4, kappa_label="fair"))
        self.assertHas("Most frequent failure: <code>quality / grounded</code> in <strong>3</strong> of 3 failing cases", doc)
        self.assertHas("<strong>1</strong> of 5 request(s) errored (1 in the target, 0 in a judge)", doc)
        self.assertHas("κ 0.40, 12 labeled cases): treat the automated pass rate with caution; the labeled sample is small.", doc)

    def test_untagged_only_suite_explains_how_to_get_a_breakdown(self):
        doc = render_report(make_run([record("a-1", tags=()), record("a-2", False, tags=())]))
        self.assertHas("No cases carry tags", doc)
        self.assertLacks("small sample", doc)

    def test_judges_section(self):
        doc = render_report(sample_run())
        self.assertHas('<h2>Judges</h2>', doc)
        self.assertHas("<strong>quality</strong>", doc)
        self.assertHas("$0.0060", doc)  # 3 x $0.002

    def test_regression_banner_and_signed_deltas(self):
        doc = render_report(sample_run(), regression_comparison())
        self.assertHas("Regression detected", doc)
        self.assertHas("pass rate dropped 33.3 p.p.", doc)
        self.assertHas("critical tag:", doc)
        self.assertHas("critical tag &#x27;legal&#x27; matches no paired case", doc)  # gate warnings are shown
        self.assertHas("only one of the two runs", doc)
        # pass rate down: sign, arrow and words, coloured as bad
        self.assertHas('<span class="delta bad"><span aria-hidden="true">▼</span> −33.3 p.p. <span class="delta-note">vs baseline · worse</span></span>', doc)
        # cost up 25%: increase is bad
        self.assertHas('<span class="delta bad"><span aria-hidden="true">▲</span> +25.0% <span class="delta-note">vs baseline · worse</span>', doc)
        # p95 latency down 10%: decrease is good
        self.assertHas('<span class="delta good"><span aria-hidden="true">▼</span> −10.0% <span class="delta-note">vs baseline · better</span>', doc)
        self.assertHas("no change", doc)  # error rate unchanged
        self.assertHas("Gate thresholds", doc)
        self.assertHas("McNemar p = 1.000", doc)  # a first-version comparison: the two-sided test

    def test_improvement_is_worded_and_green(self):
        doc = render_report(sample_run(), improvement_comparison())
        self.assertHas("No regression", doc)
        self.assertLacks("Regression detected", doc)
        self.assertHas('<span class="delta good"><span aria-hidden="true">▲</span> +4.2 p.p. <span class="delta-note">vs baseline · better</span></span>', doc)

    def test_regressions_section_links_to_failure_entries_and_lists_them_first(self):
        run = make_run([record("a-1", False), record("b-2", False, tags=("shipping",)), record("c-3", False)])
        doc = render_report(run, regression_comparison())
        self.assertHas('<a href="#case-b-2">b-2</a> <span class="badge bad">critical</span>', doc)
        self.assertHas("1 case(s) fixed", doc)
        first_failure = doc.index('<details class="case" id=')
        self.assertTrue(doc[first_failure:].startswith('<details class="case" id="case-b-2"'))
        self.assertHas("critical regression", doc)
        # the tag table shows the per-tag delta
        self.assertHas("Change vs baseline", doc)
        self.assertHas("−100.0 p.p.", doc)

    def test_no_regressions_section_without_comparison(self):
        self.assertLacks('id="regressions"', render_report(sample_run()))

    def test_calibration_section_and_tile(self):
        doc = render_report(sample_run(), None, calibration_dict())
        self.assertHas('<h2>Calibration</h2>', doc)
        self.assertHas("Calibrated pass rate", doc)
        self.assertHas("71.4%", doc)
        self.assertHas("κ 0.72 (substantial)", doc)
        self.assertHas("judge too lenient", doc)
        self.assertHas(">17<", doc)  # confusion cell
        self.assertHas("Judge is usable", doc)
        self.assertHas("d-1", doc)
        self.assertHas("human: fail", doc)
        self.assertHas("2 label(s) were excluded", doc)
        self.assertHas("Judge agreement with human labels is <strong>substantial</strong>", doc)
        self.assertLacks("Judge verdict:", doc)  # no kappa_decision in a first-version calibration

    def test_calibration_edge_cases(self):
        empty = calibration_dict(n=0, kappa=None, kappa_label="n/a", kappa_ci=None, tpr=None, tnr=None, agreement=None, corrected_pass_rate=None, corrected_pass_rate_ci=None, disagreements=[], confusion={}, recommendation="")
        doc = render_report(sample_run(), None, empty)
        self.assertHas("No labeled cases could be matched", doc)
        self.assertBalanced(doc)

        unestimable = calibration_dict(corrected_pass_rate=None, corrected_pass_rate_ci=None, disagreements=[], judge="quality")
        doc = render_report(sample_run(), None, unestimable)
        self.assertHas("not estimable", doc)
        self.assertHas("agree on every labeled case", doc)
        self.assertHas("judge quality", doc)
        self.assertLacks("The interval on the calibrated pass rate", doc)  # nothing to explain without an estimate
        self.assertBalanced(doc)

    def test_cost_section(self):
        doc = render_report(sample_run(), regression_comparison())
        self.assertHas("Input (uncached)", doc)
        self.assertHas("<code>claude-haiku-4-5</code>", doc)
        self.assertHas("100.0%", doc)  # model share
        self.assertHas("System</strong> (feature under test)", doc)
        self.assertHas("in the baseline", doc)


class TestErrorsAreNotFailures(ReportTestCase):
    """C1: a case that errored is neither a pass nor a fail, and never shows up as a quality failure."""

    def run_with_errors(self):
        records = [
            record("ok-1"),
            record("ok-2"),
            record("bad-1", False),
            record("boom", False, text="", error="RuntimeError: upstream exploded"),
            record("judge-1", False, judges=[judge("quality", False, error="parse failure: no JSON"), judge("length", True)]),
        ]
        return make_run(records, gate={"max_error_rate": 0.05})

    def test_errors_get_their_own_section_split_by_source(self):
        doc = render_report(self.run_with_errors())
        self.assertHas('<h2>Errors</h2>', doc)
        self.assertHas("RuntimeError: upstream exploded", doc)
        self.assertHas("quality: parse failure: no JSON", doc)
        self.assertRegex(doc, r'<span class="case-id">boom</span></td><td><span class="badge warn">target</span>')
        self.assertRegex(doc, r'<span class="case-id">judge-1</span></td><td><span class="badge warn">judge</span>')
        text = plain(doc)
        self.assertIn("2 of 5 case(s) errored: 1 in the target", text)
        self.assertIn("1 in a judge", text)
        self.assertIn("not a quality failure", text)

    def test_errors_are_not_listed_as_failures_and_do_not_lower_the_pass_rate(self):
        doc = render_report(self.run_with_errors())
        self.assertHas('id="case-bad-1"', doc)
        self.assertLacks('id="case-boom"', doc)
        self.assertLacks('id="case-judge-1"', doc)
        text = plain(doc)
        self.assertIn("1 of 3 scored cases failed", text)
        self.assertIn("Errored cases are not listed here", text)
        # 2 of 3 scored cases pass: 66.7%, not 2 of 5
        self.assertHas("66.7%", doc)
        self.assertHas("2 of 3 scored cases", doc)
        self.assertHas("2 errored, not counted", doc)
        self.assertHas("5 cases (3 scored)", doc)

    def test_error_tile_splits_target_and_judge(self):
        doc = render_report(self.run_with_errors())
        self.assertHas("target 1 · judge 1", doc)
        self.assertHas("40.0% of requests, not quality failures", doc)

    def test_unhealthy_run_is_called_out_before_any_number(self):
        doc = render_report(self.run_with_errors())
        self.assertHas('class="health bad" role="alert"', doc)
        self.assertHas("Unhealthy run.", doc)
        self.assertHas("1 judge error(s) and 1 target error(s)", doc)
        self.assertLess(doc.index("Unhealthy run."), doc.index("Pass rate"))
        self.assertHas("above the 5.0% limit", doc)

    def test_a_few_errors_under_the_limit_are_a_note_not_an_alarm(self):
        records = [record(f"ok-{i}") for i in range(30)] + [record("boom", False, text="", error="Timeout")]
        doc = render_report(make_run(records, gate={"max_error_rate": 0.05}))
        self.assertLacks("Unhealthy run", doc)
        self.assertHas("1 case(s) errored", doc)
        self.assertHas('class="health" role="note"', doc)

    def test_the_limit_comes_from_the_comparison_gate_first(self):
        cmp = regression_comparison()
        cmp["gate"] = {**cmp["gate"], "max_error_rate": 0.5}
        doc = render_report(self.run_with_errors(), cmp)
        self.assertLacks("Unhealthy run", doc)  # 40% is under the comparison's 50% limit

    def test_comparison_shows_errors_separately(self):
        doc = render_report(self.run_with_errors(), cycle2_comparison())
        self.assertHas("Errors (not counted as failures): baseline 0, candidate 3 (1 target, 2 judge)", doc)

    def test_a_run_without_errors_has_no_errors_section_or_alarm(self):
        doc = render_report(sample_run())
        self.assertLacks('id="errors"', doc)
        self.assertLacks("Unhealthy", doc)
        self.assertLacks("errored", doc.replace("errored, not counted", ""))

    def test_long_error_messages_are_clipped(self):
        doc = render_report(make_run([record("boom", False, text="", error="x" * 5000)]))
        self.assertNotIn("x" * 1000, doc)
        self.assertHas("…", doc)

    def test_runs_written_before_the_outcome_field_still_render(self):
        records = [legacy(r) for r in self.run_with_errors().results]
        summary = {k: v for k, v in self.run_with_errors().summary.items() if k not in {"n_failed", "n_scored", "n_target_errors", "n_judge_errors"}}
        summary["n_errors"], summary["error_rate"] = 1, 0.2  # the old runner counted target errors only
        doc = render_report(RunResult(None, summary, records))
        self.assertHas("RuntimeError: upstream exploded", doc)
        self.assertLacks('id="case-boom"', doc)
        self.assertBalanced(doc)


class TestStability(ReportTestCase):
    def repeated_run(self):
        stable = record("s-1", True, attempts=[{"outcome": "pass", "passed": True, "output_sha": "a"}] * 3, stability=1.0)
        flips = record(
            "f-1",
            False,
            attempts=[
                {"outcome": "pass", "passed": True, "output_sha": "a"},
                {"outcome": "fail", "passed": False, "output_sha": "b"},
                {"outcome": "fail", "passed": False, "output_sha": "c"},
            ],
            stability=2 / 3,
        )
        plain_case = record("p-1", True)  # not repeated
        return make_run([stable, flips, plain_case], stability={"n_repeated": 2, "flip_rate": 0.5})

    def test_stability_section_tile_and_header(self):
        doc = render_report(self.repeated_run())
        self.assertHas('<h2>Stability</h2>', doc)
        text = plain(doc)
        self.assertIn("2 case(s) ran 3 times each", text)
        self.assertIn("Flip rate 50.0%", text)
        self.assertIn("1 case(s) flipped", text)
        self.assertIn("The output text changed between attempts in 1 of 2 repeated cases", text)
        self.assertRegex(doc, r'Flip rate</div><div class="tile-value">50\.0%</div>')
        self.assertHas("2 repeated cases × 3 attempts, majority outcome", doc)
        self.assertHas('<a href="#case-f-1">f-1</a>', doc)
        self.assertHas("1 / 2 / 0", doc)  # pass / fail / error
        self.assertHas("67%", doc)  # agreement with the majority
        self.assertHas("Noise floor: <strong>50.0%</strong> of the 2 repeated cases", doc)

    def test_repeats_header_uses_the_repeat_setting_of_the_summary(self):
        run = self.repeated_run()
        run.summary["repeat"] = {"n": 5, "tags": ["safety", "billing"]}
        doc = render_report(run)
        self.assertHas("2 repeated cases (billing 5 attempts, safety 5 attempts, every other case once), majority outcome", doc)
        self.assertIn("2 case(s) ran 3 times each", plain(doc))  # the records say what really ran

    def test_repeat_setting_is_escaped_and_garbage_falls_back_to_the_attempts(self):
        run = self.repeated_run()
        run.summary["repeat"] = {"n": 5, "tags": [XSS]}
        doc = render_report(run)
        self.assertLacks(XSS, doc)
        self.assertBalanced(doc)
        for bad in (None, [], "x", {"n": None, "tags": "safety"}, {"n": -3, "tags": None}, {"n": "many", "tags": [None, 3]}):
            run = self.repeated_run()
            run.summary["repeat"] = bad
            doc = render_report(run)
            self.assertBalanced(doc)
            self.assertHas("2 repeated cases × 3 attempts", doc)  # what the records show

    def test_attempts_are_shown_in_the_failure_entry(self):
        doc = render_report(self.repeated_run())
        self.assertIn("Attempts: 3 (1 pass, 2 fail) (agreement 67%)", plain(doc))

    def test_all_repeated_cases_stable(self):
        stable = record("s-1", True, attempts=[{"outcome": "pass", "passed": True, "output_sha": "a"}] * 3, stability=1.0)
        doc = render_report(make_run([stable], stability={"n_repeated": 1, "flip_rate": 0.0}))
        self.assertHas("Every repeated case gave the same verdict on every attempt", doc)
        self.assertHas("Flip rate 0.0%", plain(doc))

    def test_nothing_repeated_says_so_when_the_run_has_the_stability_block(self):
        doc = render_report(make_run([record("a-1")], stability={"n_repeated": 0, "flip_rate": None}))
        self.assertHas("No case was repeated", doc)
        self.assertLacks('<h2>Stability</h2>', doc)
        self.assertLacks("Flip rate", doc)

    def test_runs_without_the_stability_block_are_silent_about_it(self):
        doc = render_report(sample_run())
        self.assertLacks("repeated", doc)
        self.assertLacks("Flip rate", doc)
        self.assertLacks('<h2>Stability</h2>', doc)

    def test_garbage_stability_values_do_not_crash(self):
        for bad in (None, [], "x", {"n_repeated": None, "flip_rate": "nan"}, {"n_repeated": 2, "flip_rate": None}):
            run = make_run([record("a-1")])
            run.summary["stability"] = bad
            self.assertBalanced(render_report(run))


class TestTagGroups(ReportTestCase):
    def grouped_run(self):
        records = [
            record("a-1", True, tags=("answerable", "plans")),
            record("a-2", False, tags=("answerable", "billing")),
            record("o-1", True, tags=("out_of_scope", "mobile")),
            record("o-2", False, tags=("out_of_scope", "sla")),
            record("s-1", True, tags=("safety", "fraud")),
        ]
        return make_run(records, tag_groups={"type": ["answerable", "out_of_scope", "safety"], "topic": ["plans", "billing", "mobile"], "unused": ["nothing-here"]})

    def test_one_table_per_group_and_other_for_the_rest(self):
        doc = render_report(self.grouped_run())
        self.assertHas("<h3>type</h3>", doc)
        self.assertHas("<h3>topic</h3>", doc)
        self.assertHas("<h3>other</h3>", doc)  # sla and fraud belong to no group
        self.assertLacks("<h3>unused</h3>", doc)  # a group with no tag in the run is not drawn
        self.assertHas('aria-label="Quality by tag: type"', doc)
        self.assertEqual(doc.count("<h3>type</h3>"), 1)
        type_block = doc[doc.index("<h3>type</h3>"):doc.index("<h3>topic</h3>")]
        self.assertIn("<strong>answerable</strong>", type_block)
        self.assertNotIn("<strong>plans</strong>", type_block)
        other_block = doc[doc.index("<h3>other</h3>"):]
        other_block = other_block[: other_block.index("</table>")]
        self.assertIn("<strong>sla</strong>", other_block)
        self.assertIn("<strong>fraud</strong>", other_block)
        self.assertHas("a case is counted once per table", plain(doc))
        self.assertBalanced(doc)

    def test_each_table_is_sorted_weakest_first(self):
        doc = render_report(self.grouped_run())
        type_block = doc[doc.index("<h3>type</h3>"):doc.index("<h3>topic</h3>")]
        self.assertLess(type_block.index("<strong>answerable</strong>"), type_block.index("<strong>safety</strong>"))  # 50% before 100%

    def test_group_tables_get_the_comparison_deltas(self):
        cmp = regression_comparison()
        cmp["by_tag"] = {"answerable": {"n": 2, "baseline": 1.0, "candidate": 0.5, "delta": -0.5}}
        doc = render_report(self.grouped_run(), cmp)
        self.assertHas("−50.0 p.p.", doc)
        self.assertEqual(doc.count("Change vs baseline"), 3)

    def test_without_groups_the_single_table_is_kept(self):
        doc = render_report(sample_run())
        self.assertLacks("<h3>other</h3>", doc)
        self.assertHas('aria-label="Quality by tag"', doc)

    def test_empty_or_malformed_groups_fall_back_to_one_table(self):
        for bad in ({}, [], None, "type", {"type": "answerable"}, {"type": None}):
            run = self.grouped_run()
            run.summary["tag_groups"] = bad
            doc = render_report(run)
            self.assertHas('aria-label="Quality by tag"', doc)
            self.assertLacks("<h3>type</h3>", doc)

    def test_a_tag_in_two_groups_appears_in_both(self):
        run = self.grouped_run()
        run.summary["tag_groups"] = {"one": ["safety"], "two": ["safety", "fraud"]}
        doc = render_report(run)
        self.assertEqual(doc.count("<strong>safety</strong>"), 2)


class TestComparisonKeys(ReportTestCase):
    def test_removed_and_changed_cases(self):
        run = make_run([record("b-2", False, tags=("shipping",)), record("c-3")])
        doc = render_report(run, cycle2_comparison())
        text = plain(doc)
        self.assertIn("Removed cases (2)", text)
        self.assertIn("Changed cases (1)", text)
        self.assertIn("A shrinking test set can hide a regression", text)
        self.assertIn("update the baseline in a reviewed change", text)
        self.assertHas("old-1", doc)
        self.assertHas("2 baseline case(s) are missing from this run", doc)
        self.assertHas("1 case(s) were edited since the baseline", doc)
        self.assertBalanced(doc)

    def test_gate_mode_pvalues_errors_and_calibration_status(self):
        doc = render_report(sample_run(), cycle2_comparison())
        self.assertHas("McNemar one-sided p = 0.016", doc)
        self.assertHas("Gate mode: auto-small", doc)
        self.assertHas("Judge quality calibration: κ 0.72 (95% CI 0.50 to 0.90) is below the gate minimum", doc)
        self.assertHas("Mode for this comparison", doc)
        self.assertIn("a drop above 5.0 p.p. fails, no significance test", plain(doc))
        self.assertHas("automatic, by test-set size", doc)  # max_pass_rate_drop: auto does not crash the thresholds table
        for label in ("Max removed cases", "Max changed cases", "Min judge kappa", "Calibration required"):
            self.assertHas(label, doc)

    def test_gate_modes_are_worded(self):
        for mode, phrase in (("auto-large", "one-sided McNemar test also says"), ("fixed", "fixed limit")):
            cmp = cycle2_comparison()
            cmp["gate_mode"] = mode
            self.assertIn(phrase, plain(render_report(sample_run(), cmp)))

    def test_judges_changed_is_shown_in_the_verdict_banner(self):
        cmp = cycle2_comparison(); cmp["judges_changed"] = True
        cmp["reasons"] = ["judge configuration or rubric changed: update the baseline in a reviewed PR (judges_sha aaa → bbb)"]
        doc = render_report(sample_run(), cmp)
        self.assertHas("<strong>Judge configuration or rubric changed</strong> since the baseline", doc)
        self.assertIn("judges_sha aaa → bbb", plain(doc))  # the comparison's own reason still shows
        cmp = cycle2_comparison(); cmp["judges_changed"] = False
        doc = render_report(sample_run(), cmp)
        self.assertHas("Judges unchanged since the baseline", doc)
        self.assertLacks("Judge configuration or rubric changed", doc)

    def test_judges_changed_unknown_or_garbage_says_nothing_and_does_not_crash(self):
        for value in (None, "yes", 1, [], {"a": XSS}):
            cmp = cycle2_comparison(); cmp["judges_changed"] = value
            doc = render_report(sample_run(), cmp)
            self.assertLacks("Judges unchanged", doc)
            self.assertLacks("Judge configuration or rubric changed</strong>", doc)
            self.assertLacks(XSS, doc)
            self.assertBalanced(doc)
        cmp = cycle2_comparison()
        self.assertLacks("Judges unchanged", render_report(sample_run(), cmp))  # a comparison that predates the key

    def test_allow_judge_changes_gate_setting_has_a_label(self):
        cmp = cycle2_comparison(); cmp["gate"]["allow_judge_changes"] = True
        self.assertHas("Judge changes allowed", render_report(sample_run(), cmp))

    def test_calibration_status_ok(self):
        cmp = cycle2_comparison()
        cmp["calibration_status"] = {"judge": "quality", "kappa": 0.85, "kappa_ci": [0.8, 0.9], "ok": True}
        self.assertHas("meets the gate minimum", render_report(sample_run(), cmp))

    def test_first_version_comparison_renders_without_the_new_keys(self):
        doc = render_report(sample_run(), regression_comparison())
        for absent in ("Removed cases", "Changed cases", "Gate mode", "calibration:", "McNemar one-sided"):
            self.assertLacks(absent, doc)

    def test_garbage_in_new_comparison_keys_does_not_crash(self):
        cmp = cycle2_comparison()
        cmp.update(removed_cases=None, changed_cases=None, errors=None, effective_max_pass_rate_drop=None, gate_mode="", calibration_status=None)
        self.assertBalanced(render_report(sample_run(), cmp))
        cmp.update(errors={"baseline": None, "candidate": {"n_errors": None}}, effective_max_pass_rate_drop="x", gate_mode="auto-small", calibration_status={"kappa": None, "kappa_ci": None})
        self.assertBalanced(render_report(sample_run(), cmp))


class TestBaselineSideBySide(ReportTestCase):
    def runs(self):
        base = make_run([record("a-1", True), record("b-2", True, text="Baseline says 14 days.", tags=("shipping",)), record("c-3", True)])
        cand = make_run([record("a-1", True), record("b-2", False, text="Candidate says 60 days.", tags=("shipping",)), record("c-3", False, text="I do not know.")])
        return base, cand

    def test_newly_failing_cases_show_both_outputs(self):
        base, cand = self.runs()
        cmp = regression_comparison()
        cmp.update(newly_failing=["b-2", "c-3"], critical_regressions=[])
        doc = render_report(cand, cmp, baseline=base)
        self.assertHas('<div class="sbs">', doc)
        self.assertEqual(doc.count('<div class="sbs">'), 2)
        entry = doc[doc.index('id="case-b-2"'):doc.index('id="case-c-3"')]
        self.assertIn("Baseline says 14 days.", entry)
        self.assertIn("Candidate says 60 days.", entry)
        self.assertLess(entry.index("Baseline says"), entry.rindex("Candidate says"))  # the panes: baseline first
        self.assertIn('Baseline <span class="badge pass">pass</span>', entry)
        self.assertIn("Baseline vs candidate output", entry)
        self.assertBalanced(doc)

    def test_only_newly_failing_cases_get_the_pane(self):
        base, cand = self.runs()
        cmp = regression_comparison()
        cmp.update(newly_failing=["b-2"], critical_regressions=[])
        doc = render_report(cand, cmp, baseline=base)
        self.assertEqual(doc.count('<div class="sbs">'), 1)  # c-3 also fails but the comparison did not flag it

    def test_without_a_baseline_there_is_no_pane(self):
        _, cand = self.runs()
        doc = render_report(cand, regression_comparison())
        self.assertLacks('class="sbs"', doc)

    def test_a_case_missing_from_the_baseline_says_so(self):
        base, cand = self.runs()
        base.results = [r for r in base.results if r["case_id"] != "b-2"]
        cmp = regression_comparison()
        cmp.update(newly_failing=["b-2"], critical_regressions=[])
        doc = render_report(cand, cmp, baseline=base)
        self.assertHas("not in the baseline run", doc)

    def test_newly_failing_is_derived_from_the_baseline_when_there_is_no_comparison(self):
        base, cand = self.runs()
        doc = render_report(cand, None, baseline=base)
        self.assertEqual(doc.count('<div class="sbs">'), 2)
        self.assertLacks("Regression detected", doc)

    def test_an_errored_baseline_output_is_labelled(self):
        base, cand = self.runs()
        base.results[1] = record("b-2", False, text="", error="Timeout")
        cmp = regression_comparison()
        cmp.update(newly_failing=["b-2"], critical_regressions=[])
        doc = render_report(cand, cmp, baseline=base)
        self.assertHas('Baseline <span class="badge warn">error</span>', doc)
        self.assertHas("(empty output)", doc)


class TestMonthlyProjection(ReportTestCase):
    def test_projection_table_and_tile(self):
        doc = render_report(sample_run(), monthly_requests=100_000)
        self.assertHas("Monthly projection", doc)
        self.assertHas("Per month (100,000 requests)", doc)
        self.assertRegex(doc, r"Est\. cost / month</div><div class=\"tile-value\">\$100\.00</div>")  # $0.001 x 100k
        self.assertHas("at 100,000 requests per month", doc)
        self.assertIn("mean system cost per request measured on this test set", plain(doc))
        self.assertBalanced(doc)

    def test_projection_against_the_baseline_run(self):
        base = make_run([record(f"a-{i}", True, cost=0.002) for i in range(3)])
        doc = render_report(sample_run(), regression_comparison(), baseline=base, monthly_requests=10_000)
        table = doc[doc.index("Monthly cost projection"):]
        table = table[: table.index("</table>")]
        self.assertIn("$20.00", table)  # baseline: $0.002 x 10k
        self.assertIn("$10.00", table)  # this run
        self.assertIn(f"{chr(0x2212)}$10.00", table)  # change: a saving is shown with a minus sign

    def test_projection_falls_back_to_the_comparison_baseline_cost(self):
        doc = render_report(sample_run(), regression_comparison(), monthly_requests=1_000)
        table = doc[doc.index("Monthly cost projection"):]
        self.assertIn("$1.00", table)  # baseline_mean_usd 0.001 x 1000
        self.assertIn("Change per month", table)

    def test_invalid_monthly_requests_are_ignored(self):
        for bad in (None, 0, -5, "many", True, float("nan"), float("inf")):
            doc = render_report(sample_run(), monthly_requests=bad)
            self.assertLacks("Monthly projection", doc)
            self.assertLacks("Est. cost / month", doc)

    def test_no_cost_data_says_so(self):
        records = [record(f"a-{i}", True, cost=None) for i in range(2)]
        doc = render_report(make_run(records), monthly_requests=1000)
        self.assertHas("no projection at 1,000 requests per month", doc)

    def test_partially_priced_projection_is_flagged(self):
        run = make_run([record("a-1"), record("a-2", cost=None)])
        self.assertHas("the projection is understated", render_report(run, monthly_requests=1000))

    def test_write_report_passes_the_new_arguments_through(self):
        base, cand = TestBaselineSideBySide().runs()
        cmp = regression_comparison()
        cmp.update(newly_failing=["b-2"], critical_regressions=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(Path(tmp) / "r" / "report.html", cand, cmp, None, baseline=base, monthly_requests=5000)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text, render_report(cand, cmp, None, base, 5000))
            self.assertIn("Monthly projection", text)
            self.assertIn('class="sbs"', text)


class TestCalibrationDecision(ReportTestCase):
    def test_every_decision_is_worded(self):
        expected = {
            "trustworthy": "at or above 0.8",
            "usable": "at or above 0.6",
            "inconclusive": "too wide to say",
            "not_trustworthy": "Do not gate on this judge yet",
            "insufficient_data": "fewer than 30 labeled cases",
        }
        for decision, phrase in expected.items():
            doc = render_report(sample_run(), None, cycle2_calibration(kappa_decision=decision))
            self.assertIn(phrase, plain(doc), decision)
            self.assertHas("Judge verdict:", doc)
            self.assertHas(decision.replace("_", " "), doc)
            self.assertBalanced(doc)

    def test_decision_replaces_the_point_estimate_label_in_tile_and_findings(self):
        doc = render_report(sample_run(), None, cycle2_calibration(kappa_decision="inconclusive"))
        self.assertHas("κ 0.72 (inconclusive)", doc)
        self.assertHas("Judge agreement with human labels is <strong>inconclusive</strong> (κ 0.72, 95% CI 0.50 to 0.90, 40 labeled cases)", doc)
        self.assertLacks("substantial", doc)  # the point-estimate label is not shown when a decision exists

    def test_warnings_and_reliability_flags(self):
        doc = render_report(sample_run(), None, cycle2_calibration())
        self.assertHas('<ul class="cal-warnings">', doc)
        self.assertHas("the true-negative rate is unreliable", doc)
        self.assertHas("screening check not passed", doc)  # no reason recorded: the generic wording, never "unreliable"
        self.assertLacks("unreliable: too few bad outputs", doc)
        self.assertHas("About 120 labeled cases would narrow the kappa interval to ±0.1", doc)
        self.assertHas("Agreement, weighted to the run", doc)
        self.assertHas("91.0%", doc)

    def test_reliable_tnr_has_no_caveat_and_trustworthy_needs_no_more_labels(self):
        doc = render_report(sample_run(), None, cycle2_calibration(kappa_decision="trustworthy", tnr_reliable=True, warnings=[], labels_needed_estimate=0))
        self.assertLacks("unreliable", doc)
        self.assertLacks('<ul class="cal-warnings">', doc)
        self.assertLacks("would narrow", doc)

    def test_the_calibrated_rate_interval_is_worded_for_the_population(self):
        doc = render_report(sample_run(), None, cycle2_calibration())
        self.assertIn("describes the population this test set was sampled from, not only these exact cases", plain(doc))

    def test_missing_or_garbage_new_calibration_keys_do_not_crash(self):
        for extra in ({"kappa_decision": None}, {"labels_needed_estimate": None, "kappa_decision": "usable"}, {"warnings": None}, {"warnings": [None, ""]}, {"agreement_weighted": None}, {"tnr_reliable": None}):
            self.assertBalanced(render_report(sample_run(), None, calibration_dict(**extra)))

    def test_warnings_show_even_when_nothing_matched(self):
        empty = cycle2_calibration(n=0, kappa=None, kappa_ci=None, tpr=None, tnr=None, corrected_pass_rate=None, corrected_pass_rate_ci=None, disagreements=[], confusion={}, recommendation="", warnings=["no labels matched"])
        doc = render_report(sample_run(), None, empty)
        self.assertHas("No labeled cases could be matched", doc)
        self.assertHas("no labels matched", doc)

    # ---- loop-back 1: the verdict is decided on the weighted kappa; the unweighted one is secondary
    def weighted_calibration(self, **overrides):
        extra = {
            "kappa": 0.80,
            "kappa_ci": [0.36, 1.0],
            "kappa_weighted": 0.75,
            "kappa_weighted_ci": [0.33, 1.0],
            "kappa_decision": "inconclusive",
            "kappa_decision_basis": "weighted",
            "min_labels_release": 40,
            "n_labelable": 45,
            "labels_needed_estimate": 45,
            "labels_needed_capped": True,
            "kappa_ci_smoothing": 0.5,
        }
        extra.update(overrides)
        return cycle2_calibration(**extra)

    def test_weighted_kappa_is_the_primary_number_everywhere(self):
        doc = render_report(sample_run(), None, self.weighted_calibration())
        text = seen(doc)
        self.assertIn("weighted κ 0.75 (inconclusive)", text)  # the tile
        self.assertIn("Judge agreement with human labels is inconclusive (weighted κ 0.75, 95% CI 0.33 to 1.00, 40 labeled cases)", text)
        self.assertIn("Cohen's kappa (whole run, weighted) 0.75 (inconclusive), 95% CI 0.33 to 1.00", text)
        self.assertIn("Cohen's kappa (labeled sample, unweighted) 0.80, 95% CI 0.36 to 1.00", text)
        self.assertIn("the verdict is decided on this one", text)
        self.assertIn("secondary: describes the labels that were drawn, not the run", text)
        self.assertIn("The verdict rests on the weighted kappa", text)
        self.assertNotIn("κ 0.72", text)  # the legacy point estimate of calibration_dict is not what is shown
        self.assertNotIn("0.80 (inconclusive)", text)  # the unweighted value never carries the decision
        self.assertBalanced(doc)

    def test_legacy_calibration_without_weighted_keys_falls_back_to_the_unweighted_pair(self):
        doc = render_report(sample_run(), None, cycle2_calibration(kappa_decision="inconclusive"))
        text = seen(doc)
        self.assertIn("κ 0.72 (inconclusive)", text)
        self.assertIn("Cohen's kappa 0.72 (inconclusive), 95% CI 0.50 to 0.90", text)
        self.assertIn("(κ 0.72, 95% CI 0.50 to 0.90, 40 labeled cases)", text)
        for absent in ("weighted κ", "unweighted", "whole run, weighted", "The verdict rests on the weighted kappa", "labelable in this run"):
            self.assertNotIn(absent, text)

    def test_capped_labels_needed_says_to_label_everything_and_add_cases(self):
        doc = render_report(sample_run(), None, self.weighted_calibration())
        text = seen(doc)
        self.assertIn("This run has only 45 labelable cases, fewer than the labels needed for a ±0.1 interval: label all 45; add cases or repeats for more precision.", text)
        self.assertNotIn("would narrow the kappa interval", text)
        self.assertIn("of 45 labelable in this run", text)
        done = seen(render_report(sample_run(), None, self.weighted_calibration(n=45)))
        self.assertIn("All 45 labelable cases are already labeled: add cases or repeats for more precision.", done)
        self.assertNotIn("This run has only", done)
        uncapped = seen(render_report(sample_run(), None, self.weighted_calibration(labels_needed_capped=False, labels_needed_estimate=278)))
        self.assertIn("About 278 labeled cases would narrow the kappa interval to ±0.1.", uncapped)
        self.assertNotIn("labelable cases", uncapped)

    def test_the_minimum_number_of_labels_for_a_release_is_stated_when_it_matters(self):
        few = seen(render_report(sample_run(), None, self.weighted_calibration(n=32)))
        self.assertIn("At least 40 labeled cases are needed before a judge is released (usable or trustworthy).", few)
        enough = seen(render_report(sample_run(), None, self.weighted_calibration(n=40)))
        self.assertNotIn("At least 40 labeled cases", enough)
        for decision in ("usable", "trustworthy"):  # a released judge never carries the minimum
            released = seen(render_report(sample_run(), None, self.weighted_calibration(n=60, kappa_decision=decision, labels_needed_capped=False)))
            self.assertNotIn("At least 40 labeled cases", released)
        absent = seen(render_report(sample_run(), None, self.weighted_calibration(n=32, min_labels_release=None)))
        self.assertNotIn("At least", absent)

    def test_weighted_keys_present_but_undefined_or_garbage_do_not_crash_or_borrow_the_unweighted(self):
        undefined = render_report(sample_run(), None, self.weighted_calibration(kappa_weighted=None, kappa_weighted_ci=None, kappa_decision="insufficient_data"))
        self.assertIn("weighted κ n/a", seen(undefined))
        self.assertIn("Cohen's kappa (labeled sample, unweighted) 0.80", seen(undefined))
        self.assertNotIn("weighted κ 0.80", seen(undefined))
        self.assertBalanced(undefined)
        for extra in (
            {"kappa_weighted": "x", "kappa_weighted_ci": "abc"},
            {"kappa_weighted_ci": [None, 1.0]},
            {"kappa_weighted_ci": [0.1]},
            {"n_labelable": "many", "labels_needed_capped": "yes", "min_labels_release": "forty"},
            {"n_labelable": -3},
            {"kappa_ci": "oops"},
        ):
            with self.subTest(extra=extra):
                self.assertBalanced(render_report(sample_run(), None, self.weighted_calibration(**extra)))

    def test_escapes_the_new_calibration_keys(self):
        doc = render_report(sample_run(), None, self.weighted_calibration(kappa_decision_basis=XSS, n_labelable=XSS, min_labels_release=XSS, kappa_weighted_ci=[XSS, XSS]))
        self.assertLacks("<script>alert", doc)
        self.assertBalanced(doc)

    def test_gate_status_names_the_kappa_it_shows(self):
        for basis, phrase in (("weighted", "calibration: weighted κ 0.72 (95% CI 0.50 to 0.90)"), ("legacy", "calibration: unweighted κ 0.72 (95% CI 0.50 to 0.90)"), (None, "calibration: κ 0.72 (95% CI 0.50 to 0.90)")):
            cmp = regression_comparison()
            cmp["calibration_status"] = {"judge": "quality", "kappa": 0.72, "kappa_ci": [0.5, 0.9], "ok": False, "basis": basis}
            self.assertIn(phrase, seen(render_report(sample_run(), cmp)), str(basis))


class TestCycle3Keys(ReportTestCase):
    """Keys added in cycle 3 (C9 auto gate, C10 attempt errors, C11 protocol, C12 resume), read defensively."""

    def auto_comparison(self, **overrides):
        cmp = cycle2_comparison()
        cmp.update(gate_mode="auto", effective_alpha=0.2405, effective_min_net_loss=3, effective_max_pass_rate_drop=2 / 45)
        cmp.update(overrides)
        return cmp

    def test_the_single_auto_gate_rule_is_worded_with_its_alpha_and_floor(self):
        doc = render_report(sample_run(), self.auto_comparison())
        text = seen(doc)
        self.assertIn("Gate mode: auto (alpha 0.24)", text)
        self.assertIn("automatic, one rule for every test-set size: fails on a net loss of at least 3 cases (a drop above 4.4 p.p.) "
                      "and a one-sided McNemar test at alpha 0.24 says it is unlikely to be noise", text)
        self.assertLacks("auto-small", doc)
        self.assertBalanced(doc)

    def test_auto_without_a_test_or_with_garbage_does_not_crash(self):
        text = seen(render_report(sample_run(), self.auto_comparison(effective_alpha=None, effective_min_net_loss=1)))
        self.assertIn("fails on a net loss of at least 1 case (a drop above 4.4 p.p.)", text)
        self.assertNotIn("McNemar test at alpha", text)
        for extra in ({"effective_alpha": "x", "effective_min_net_loss": "y"}, {"effective_alpha": [1], "effective_min_net_loss": None}, {"effective_max_pass_rate_drop": None}, {"effective_min_net_loss": 0}):
            with self.subTest(extra=extra):
                self.assertBalanced(render_report(sample_run(), self.auto_comparison(**extra)))

    def test_both_generations_of_gate_mode_still_render(self):
        for mode in ("auto-small", "auto-large", "fixed", "off", "auto"):
            with self.subTest(mode=mode):
                doc = render_report(sample_run(), self.auto_comparison(gate_mode=mode))
                self.assertHas(f"Gate mode: {mode}", doc)
        self.assertIn("the pass-rate drop is not checked", seen(render_report(sample_run(), self.auto_comparison(gate_mode="off"))))

    def test_new_gate_keys_are_labelled_in_the_thresholds(self):
        cmp = self.auto_comparison()
        cmp["gate"] = {**cmp["gate"], "allow_repeat_change": False, "require_same_kit_version": True, "tag_drop_warning_pp": 20, "tag_drop_min_cases": 10}
        text = seen(render_report(sample_run(), cmp))
        for label, value in (("Repeat-protocol change allowed", "no"), ("Same evalsprint version required", "yes"), ("Warn when a tag drops by (p.p.)", "20"), ("Cases a tag needs for that warning", "10")):
            self.assertIn(f"{label} {value}", text)

    # ---- attempts
    def repeated_run(self, error_attempts=1, singles=2, **summary_extra):
        attempts = [{"outcome": "pass", "output_sha": "a"}, {"outcome": "pass", "output_sha": "b"}, {"outcome": "pass", "output_sha": "c"}]
        for k in range(error_attempts):
            attempts[k] = {"outcome": "error", "error_source": "target", "output_sha": None}
        records = [record("s-1", True, tags=("safety",), attempts=attempts, stability=1.0)]
        records += [record(f"a-{k}", k != 1) for k in range(1, singles + 1)]
        return make_run(records, stability={"n_repeated": 1, "flip_rate": 1.0}, **summary_extra)

    def test_an_errored_attempt_inside_a_case_with_a_verdict_is_shown(self):
        run = self.repeated_run(singles=37)  # 3 attempts of s-1 and 37 single cases: 1 of 40 attempts errored
        doc = render_report(run)
        text = seen(doc)
        self.assertIn("1 attempt(s) errored inside repeated cases that still got a verdict", text)
        self.assertIn("Errors inside repeated cases (1)", text)
        self.assertIn("1 (target 1 · judge 0)", text)
        self.assertIn("attempts: 1 of 40 errored (2.5%)", text)
        self.assertHas('id="errors"', doc)
        self.assertNotIn("Unhealthy run.", text)  # 2.5% is under the 5% limit: a calm note, not an alarm
        self.assertBalanced(doc)

    def test_an_attempt_error_rate_above_the_limit_makes_the_run_unhealthy_even_with_no_errored_case(self):
        run = self.repeated_run(error_attempts=2)  # 2 of 5 attempts: 40%
        text = seen(render_report(run))
        self.assertIn("Unhealthy run.", text)
        self.assertIn("2 of 5 attempts errored (40.0% of attempts, limit 5.0%)", text)
        self.assertIn("Attempt error rate 40.0% (2 of 5 attempts) is above the 5.0% limit", text)

    def test_the_summary_counters_win_over_the_records(self):
        run = self.repeated_run(singles=37, n_attempts=100, n_attempt_errors=2, n_attempt_target_errors=2, n_attempt_judge_errors=0, attempt_error_rate=0.02)
        text = seen(render_report(run))
        self.assertIn("attempts: 2 of 100 errored (2.0%)", text)
        self.assertNotIn("Unhealthy run.", text)
        self.assertIn("2 attempt(s) errored inside repeated cases that still got a verdict", text)  # the calm tone

    def test_a_run_without_repeats_or_errors_shows_no_attempt_noise(self):
        text = seen(render_report(sample_run()))
        for absent in ("attempts:", "Errors inside repeated cases", "attempt(s) errored"):
            self.assertNotIn(absent, text)

    def test_attempt_errors_in_a_comparison(self):
        cmp = self.auto_comparison()
        cmp["errors"] = {
            "baseline": {"n_errors": 0, "n_attempt_errors": 0, "n_attempts": 100, "attempt_error_rate": 0.0},
            "candidate": {"n_errors": 0, "n_attempt_errors": 4, "n_attempts": 100, "attempt_error_rate": 0.04},
        }
        text = seen(render_report(sample_run(), cmp))
        self.assertIn("Attempt errors (not counted as failures): baseline 0 of 100 attempts (0.0%), candidate 4 of 100 attempts (4.0%)", text)
        quiet = self.auto_comparison()
        quiet["errors"] = {"baseline": {"n_attempt_errors": 0}, "candidate": {"n_attempt_errors": 0}}
        self.assertNotIn("Attempt errors", seen(render_report(sample_run(), quiet)))
        for junk in ({"baseline": None, "candidate": {"n_attempt_errors": "x"}}, {"baseline": {"n_attempt_errors": 1}, "candidate": {}}, "text", None):
            with self.subTest(junk=junk):
                cmp["errors"] = junk
                self.assertBalanced(render_report(sample_run(), cmp))

    def test_flaky_crashes_are_listed_and_counted_in_the_findings(self):
        cmp = self.auto_comparison(flaky_crashes=["a-1", XSS])
        doc = render_report(sample_run(), cmp)
        text = seen(doc)
        self.assertIn("Flaky crashes (2)", text)
        self.assertIn("crashed in some attempts of this run although the majority outcome is pass", text)
        self.assertIn("2 case(s) that passed in the baseline crashed", text)
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)
        for junk in ("a-1", None, [None, 3, {"x": 1}], {}):
            with self.subTest(junk=junk):
                self.assertBalanced(render_report(sample_run(), self.auto_comparison(flaky_crashes=junk)))
        self.assertNotIn("Flaky crashes", seen(render_report(sample_run(), self.auto_comparison(flaky_crashes=[]))))

    # ---- protocol
    def test_repeat_changed(self):
        protocol = {"baseline": {"n": 1, "tags": None}, "candidate": {"n": 5, "tags": ["safety"]}}
        text = seen(render_report(sample_run(), self.auto_comparison(repeat_changed=True, repeat_protocol=protocol)))
        self.assertIn("Evaluation protocol changed (baseline: every case once; candidate: safety 5 attempts, every other case once)", text)
        self.assertIn("only comparable under the same repeat setting", text)
        same = seen(render_report(sample_run(), self.auto_comparison(repeat_changed=False, repeat_protocol={"baseline": protocol["candidate"], "candidate": protocol["candidate"]})))
        self.assertIn("Same repeat protocol as the baseline (safety 5 attempts, every other case once)", same)
        self.assertNotIn("Evaluation protocol changed", same)
        unknown = seen(render_report(sample_run(), self.auto_comparison(repeat_changed=None)))
        self.assertNotIn("repeat protocol", unknown)
        self.assertNotIn("Evaluation protocol changed", unknown)

    def test_repeat_protocol_junk_and_escaping(self):
        for junk in ({"baseline": "x", "candidate": [1]}, {"baseline": {"n": "many", "tags": "safety"}, "candidate": {"n": None}}, None, "protocol"):
            with self.subTest(junk=junk):
                self.assertBalanced(render_report(sample_run(), self.auto_comparison(repeat_changed=True, repeat_protocol=junk)))
        doc = render_report(sample_run(), self.auto_comparison(repeat_changed=True, repeat_protocol={"baseline": {"n": 1}, "candidate": {"n": 3, "tags": [XSS]}}))
        self.assertLacks(XSS, doc)

    def test_kit_version(self):
        text = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "0.1.0", "candidate": "0.2.0", "changed": True})))
        self.assertIn("evalsprint version differs (baseline 0.1.0, candidate 0.2.0)", text)
        same = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "0.2.0", "candidate": "0.2.0", "changed": False})))
        self.assertIn("evalsprint 0.2.0 on both runs", same)
        spelled = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "v0.2.0", "candidate": "0.2", "changed": False})))
        self.assertIn("evalsprint v0.2.0 and 0.2 (the same release)", spelled)  # two spellings of one release
        for quiet in ({"baseline": None, "candidate": "0.2.0", "changed": None}, None, "0.2.0", {"changed": False}):
            with self.subTest(quiet=quiet):
                doc = render_report(sample_run(), self.auto_comparison(kit_version=quiet))
                for absent in ("version differs", "on both runs", "the same release"):
                    self.assertNotIn(absent, seen(doc))
                self.assertBalanced(doc)
        doc = render_report(sample_run(), self.auto_comparison(kit_version={"baseline": XSS, "candidate": XSS, "changed": True}))
        self.assertLacks(XSS, doc)

    # ---- tag warnings
    def tag_warning(self, **overrides):
        return {"tag": "billing", "n": 14, "baseline": 0.9286, "candidate": 0.6429, "delta": -0.2857, **overrides}

    def test_tag_warnings_are_a_table_a_badge_and_a_finding_never_a_reason(self):
        run = sample_run()
        cmp = self.auto_comparison(tag_warnings=[self.tag_warning(), self.tag_warning(tag="shipping", delta=-0.21, n=11)])
        doc = render_report(run, cmp)
        text = seen(doc)
        self.assertIn("Big drop in a tag (a warning, not a gate reason)", text)
        self.assertIn("Tag billing and 1 more dropped a lot against the baseline (92.9% → 64.3%, 14 paired cases): a warning, not a gate reason.", text)
        self.assertHas("big drop", doc)
        self.assertIn("billing 14 92.9% 64.3%", text)
        self.assertBalanced(doc)
        self.assertNotIn("Big drop in a tag", seen(render_report(run, self.auto_comparison(tag_warnings=[]))))

    def test_tag_warnings_junk_and_escaping(self):
        for junk in (None, "x", [None, 3, {"tag": None}, {"tag": "t"}, {"tag": "t", "n": "x", "baseline": 1, "candidate": 0}], {"tag": "billing"}):
            with self.subTest(junk=junk):
                self.assertBalanced(render_report(sample_run(), self.auto_comparison(tag_warnings=junk)))
        doc = render_report(sample_run(), self.auto_comparison(tag_warnings=[self.tag_warning(tag=XSS)]))
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)
        untagged = self.tag_warning()
        untagged.pop("delta")  # a missing delta is recomputed from the two rates
        self.assertIn("−28.6 p.p.", seen(render_report(sample_run(), self.auto_comparison(tag_warnings=[untagged]))))

    # ---- resume
    def test_a_resumed_run_says_what_it_reused_and_what_it_cost(self):
        run = make_run(sample_run().results, resumed_from={"run_id": "20260929-090000-old", "reused_records": 30, "reused_cost_usd": 1.25, "retried_errors": 2})
        doc = render_report(run)
        text = seen(doc)
        self.assertIn("Resumed from 20260929-090000-old : 30 record(s) reused without calling the target or the judges again ($1.25 of their cost counted as already spent); 2 errored record(s) were run again", text)
        self.assertIn("$1.25 of the run's cost was spent before it was resumed", text)
        self.assertBalanced(doc)

    def test_a_resume_with_a_fingerprint_mismatch_reused_nothing(self):
        run = make_run(sample_run().results, resumed_from={"run_id": "old", "reused_records": 0, "fingerprint_mismatch": ["judges_sha", "repeat"]})
        text = seen(render_report(run))
        self.assertIn("nothing was reused, because its judges (judges_sha), repeat protocol (repeat) differ from this run's, so every case ran again", text)
        self.assertNotIn("was spent before it was resumed", text)
        self.assertNotIn("source file of the target callable was edited", text)

    def test_a_resume_that_found_the_target_source_edited_says_so(self):
        run = make_run(sample_run().results, resumed_from={"run_id": "old", "reused_records": 0, "fingerprint_mismatch": ["target_source"]})
        text = seen(render_report(run))
        self.assertIn("nothing was reused, because its target source file (target_source) differs from this run's (the source file of the target callable was edited), so every case ran again", text)

    def test_every_part_of_the_fingerprint_has_its_words_and_an_unknown_part_is_shown_as_it_is(self):
        run = make_run(sample_run().results, resumed_from={"run_id": "old", "fingerprint_mismatch": ["judges_sha", "target", "target_source", "repeat", "brand_new_part"]})
        text = seen(render_report(run))
        self.assertIn("judges (judges_sha), target callable or parameters (target), target source file (target_source), repeat protocol (repeat), brand_new_part differ", text)

    def test_the_header_names_the_source_file_of_the_target(self):
        sha = "0123456789abcdef0123456789abcdef"
        target = {"callable": "bot.py:answer", "params": {"prompt": "v2"}, "source_sha": sha}
        doc = render_report(make_run(sample_run().results, target=target))
        text = seen(doc)
        self.assertIn("Target source 0123456789ab (sha256 of the file behind the callable)", text)  # 12 characters, like the judges fingerprint
        self.assertNotIn(sha, text)
        self.assertBalanced(doc)
        self.assertSelfContained(doc)

    def test_the_target_source_row_is_left_out_when_the_run_has_no_readable_one(self):
        for junk in (None, "", "   ", 12, ["abc"], {"x": 1}):
            with self.subTest(source_sha=junk):
                doc = render_report(make_run(sample_run().results, target={"callable": "bot.py:answer", "params": {}, "source_sha": junk}))
                self.assertNotIn("Target source", seen(doc))
                self.assertBalanced(doc)
        for target in (None, "bot.py:answer", [], {"callable": "bot.py:answer"}):
            with self.subTest(target=target):
                self.assertNotIn("Target source", seen(render_report(make_run(sample_run().results, target=target))))
        self.assertNotIn("Target source", seen(render_report(sample_run())))

    def test_target_parameters_of_another_shape_do_not_break_the_header(self):
        for params in ("x", [1, 2], 3, None):
            with self.subTest(params=params):
                doc = render_report(make_run(sample_run().results, target={"callable": "bot.py:answer", "params": params}))
                self.assertIn("bot.py:answer", seen(doc))
                self.assertBalanced(doc)

    def test_a_hostile_source_sha_and_fingerprint_part_are_escaped(self):
        doc = render_report(make_run(sample_run().results, target={"callable": "bot.py:answer", "params": {}, "source_sha": XSS}, resumed_from={"run_id": "old", "fingerprint_mismatch": [XSS]}))
        self.assertLacks(XSS, doc)
        self.assertBalanced(doc)

    def test_resumed_from_in_other_shapes(self):
        self.assertIn("Resumed from 20260929-090000-old", seen(render_report(make_run(sample_run().results, resumed_from="20260929-090000-old"))))
        for junk in (None, 3, [], {}, {"run_id": XSS, "reused_records": "x", "reused_cost_usd": "y"}, {"fingerprint_mismatch": XSS}):
            with self.subTest(junk=junk):
                doc = render_report(make_run(sample_run().results, resumed_from=junk))
                self.assertLacks(XSS, doc)
                self.assertBalanced(doc)
        self.assertNotIn("Resumed from", seen(render_report(sample_run())))

    def test_kit_version_and_judges_fingerprint_in_the_header(self):
        run = make_run(sample_run().results, evalsprint_version="0.1.0", kit_version="0.2.0", judges_sha="1fe04ba73d3e99aabbcc")
        text = seen(render_report(run))
        self.assertIn("Kit version evalsprint 0.2.0", text)  # kit_version wins over evalsprint_version
        self.assertIn("Judges fingerprint 1fe04ba73d3e", text)
        self.assertNotIn("1fe04ba73d3e99", text)
        self.assertIn("Generated by the Eval Sprint kit 0.2.0", text)
        bare = make_run(sample_run().results)
        del bare.summary["evalsprint_version"]
        self.assertNotIn("Judges fingerprint", seen(render_report(bare)))

    def test_dark_mode_follows_the_os_and_an_explicit_theme(self):
        doc = render_report(sample_run())
        self.assertHas('@media screen and (prefers-color-scheme: dark){', doc)
        self.assertHas(':root:not([data-theme="light"]){', doc)
        self.assertHas(':root[data-theme="dark"]{color-scheme:dark;', doc)
        self.assertHas(':root[data-theme="light"]{color-scheme:light}', doc)
        self.assertNotIn("@@", doc)


class TestCycle4Keys(ReportTestCase):
    """Keys added in cycle 4 (C17 per-tag protocol, C18 placeholder / retries / oversize, C20 stale calibration, patch versions, screening flag)."""

    def auto_comparison(self, **overrides):
        cmp = cycle2_comparison()
        cmp.update(gate_mode="auto", effective_alpha=0.2405, effective_min_net_loss=3, effective_max_pass_rate_drop=2 / 45)
        cmp.update(overrides)
        return cmp

    # ---- placeholder (C18)
    def test_a_placeholder_run_says_so_before_any_number(self):
        doc = render_report(make_run([record("a-1", True)], placeholder=True))
        text = seen(doc)
        self.assertIn("Placeholder run.", text)
        self.assertIn("its verdicts say nothing about the real product", text)
        self.assertIn("Do not record it as a baseline", text)
        self.assertHas('class="sim-banner placeholder-banner" role="note"', doc)
        self.assertLess(doc.index("Placeholder run."), doc.index('class="toc"'))  # above the table of contents, so above every number
        self.assertIn("placeholder run: not a measure of the real product", text)  # also on the tile that gets screenshotted alone
        self.assertIn("This is a placeholder run", text)  # and the first key finding
        self.assertBalanced(doc)
        self.assertSelfContained(doc)

    def test_only_a_literal_true_marks_a_placeholder(self):
        for value in (False, None, "true", 1, "yes", [True], {}, 0):
            with self.subTest(value=value):
                doc = render_report(make_run([record("a-1", True)], placeholder=value))
                self.assertLacks("Placeholder run", doc)
                self.assertLacks("placeholder-banner", doc.replace(".placeholder-banner", ""))
        self.assertLacks("Placeholder run", render_report(sample_run()))

    def test_a_simulated_placeholder_shows_both_notes(self):
        doc = render_report(make_run([record("a-1", True, meta={"simulated": True})], placeholder=True))
        self.assertIn("Placeholder run.", seen(doc))
        self.assertIn("Simulated run.", seen(doc))
        self.assertBalanced(doc)

    def test_a_placeholder_baseline_is_a_banner_a_fact_and_the_warning_is_not_repeated(self):
        warning = "the baseline run is a placeholder (simulated target or mock judge: its verdicts say nothing about the real product), so this comparison is meaningless; record a baseline from a real run and commit that one"
        doc = render_report(sample_run(), self.auto_comparison(warnings=["critical tag 'legal' matches no paired case", warning]))
        text = seen(doc)
        self.assertHas('class="health bad placeholder-banner" role="alert"', doc)
        self.assertIn("Placeholder baseline. The baseline run is a placeholder", text)
        self.assertEqual(text.count("the baseline run is a placeholder"), 0)  # lower-case original: only the banner's capitalised copy
        self.assertEqual(text.count("he baseline run is a placeholder"), 1)
        self.assertHas("<strong>Baseline is a placeholder run</strong>: this verdict says nothing about the real product", doc)
        self.assertIn("critical tag 'legal' matches no paired case", text)  # the other warnings stay
        self.assertHas("<strong>The baseline is a placeholder run</strong>: the comparison against it is meaningless", doc)  # a key finding
        self.assertLess(doc.index("Placeholder baseline."), doc.index('class="verdict'))
        self.assertBalanced(doc)

    def test_a_placeholder_baseline_run_is_enough_without_compares_warning(self):
        baseline = make_run([record("a-1", True)], placeholder=True)
        doc = render_report(sample_run(), self.auto_comparison(), None, baseline)
        self.assertIn("Placeholder baseline. The baseline run is a placeholder (simulated target or mock judge", seen(doc))
        self.assertLacks("Placeholder baseline", render_report(sample_run(), self.auto_comparison(), None, make_run([record("a-1", True)])))
        self.assertLacks("Placeholder baseline", render_report(sample_run(), self.auto_comparison()))
        self.assertLacks("Placeholder baseline", render_report(sample_run(), None, None, baseline))  # nothing to compare

    def test_the_verdict_banner_alone_keeps_every_warning(self):
        from evalsprint.report import _verdict_banner

        warning = "the baseline run is a placeholder (simulated target or mock judge), so this comparison is meaningless"
        cmp = self.auto_comparison(warnings=[warning])
        self.assertIn(warning, seen(_verdict_banner(cmp)))  # called without the banner above it, nothing is dropped
        self.assertNotIn("the baseline run is a placeholder", seen(_verdict_banner(cmp, True)))

    def test_junk_warnings_and_baselines_do_not_break_the_placeholder_banner(self):
        for warnings in (None, "placeholder", [None, 3, {"x": 1}], ["placeholder but no b-word"]):
            with self.subTest(warnings=warnings):
                doc = render_report(sample_run(), self.auto_comparison(warnings=warnings))
                self.assertBalanced(doc)
                self.assertLacks("Placeholder baseline", doc)
        doc = render_report(sample_run(), self.auto_comparison(warnings=[f"the baseline is a placeholder {XSS}"]))
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)

    # ---- target retries (C18)
    def retried_run(self, **extra):
        crashed = {**record("boom", False, text="", error="TimeoutError: upstream"), "target_retries_used": 1}
        return make_run([record("a-1", True), crashed], **extra)

    def test_target_retries_show_in_the_header_the_tile_the_findings_and_the_errors_section(self):
        doc = render_report(self.retried_run(n_target_retries=3))
        text = seen(doc)
        self.assertIn("Target retries 3 retry call(s) after a target crash, made within the run (paid, and counted in the cost)", text)
        self.assertIn("3 target retry call(s) after a crash, not counted as errors", text)
        self.assertIn("3 target call(s) crashed and were retried within the run", text)
        self.assertIn("3 target retry call(s) were made within the run after a crash", text)
        self.assertIn("this count is the only trace of a flaky target", text)
        self.assertIn("still failing after 1 retry call(s)", text)  # the errored case says how many retries it used
        self.assertBalanced(doc)

    def test_retries_are_counted_from_the_records_when_the_summary_has_no_counter(self):
        run = make_run([{**record("a-1", True), "target_retries_used": 2}, {**record("b-1", True), "target_retries_used": 1}, record("c-1", True)])
        self.assertIn("Target retries 3 retry call(s)", seen(render_report(run)))
        run = make_run([{**record("a-1", True), "target_retries_used": 2}], n_target_retries=7)
        self.assertIn("Target retries 7 retry call(s)", seen(render_report(run)))  # the summary wins

    def test_no_retries_no_trace(self):
        for run in (sample_run(), make_run([record("a-1", True)], n_target_retries=0), make_run([{**record("a-1", True), "target_retries_used": 0}])):
            doc = render_report(run)
            self.assertLacks("Target retries", doc)
            self.assertNotIn("retry call", seen(doc))

    def test_junk_retry_counts_never_break_the_report(self):
        for junk in (None, "3", -2, [], {}, float("nan"), float("inf"), True, XSS):
            with self.subTest(junk=junk):
                doc = render_report(make_run([{**record("a-1", True), "target_retries_used": junk}], n_target_retries=junk))
                self.assertBalanced(doc)
                self.assertLacks(XSS, doc)
                self.assertNotIn("retry call", seen(doc))

    # ---- the configured target_retries next to the retry calls made (C22)
    def test_the_header_gives_the_configured_retries_next_to_the_calls_that_were_made(self):
        doc = render_report(self.retried_run(target_retries=2, n_target_retries=3))
        self.assertIn("Target retries configured target_retries: 2 · 3 retry call(s) made after a target crash (paid, and counted in the cost)", seen(doc))
        self.assertBalanced(doc)
        self.assertSelfContained(doc)

    def test_a_run_that_configured_retries_and_needed_none_still_says_what_it_configured(self):
        """Two runs with different `target_retries` are not comparable (the gate blocks it), so the report states the value even when no call was retried."""
        for configured in (0, 1, 5):
            with self.subTest(configured=configured):
                text = seen(render_report(make_run([record("a-1", True)], target_retries=configured, n_target_retries=0)))
                self.assertIn(f"Target retries configured target_retries: {configured} · no retry call was made", text)
                self.assertNotIn("retry call(s) made", text)
        text = seen(render_report(make_run([{**record("a-1", True), "target_retries_used": 0}], target_retries=1)))
        self.assertIn("configured target_retries: 1 · no retry call was made", text, "the records say none was retried")

    def test_a_run_that_does_not_say_how_many_calls_were_retried_says_so(self):
        text = seen(render_report(make_run([record("a-1", True)], target_retries=1)))
        self.assertIn("configured target_retries: 1 · retry calls made: not recorded", text)

    def test_a_summary_without_the_configured_value_keeps_the_old_row(self):
        """An older run has no `target_retries`: nothing is invented, and the row is the one of before when calls were retried."""
        doc = render_report(self.retried_run(n_target_retries=3))
        self.assertIn("Target retries 3 retry call(s) after a target crash, made within the run (paid, and counted in the cost)", seen(doc))
        self.assertNotIn("configured target_retries", seen(doc))

    def test_a_configured_value_that_is_not_a_count_is_not_shown(self):
        for junk in (None, "2", -1, 1.5, [], {}, float("nan"), float("inf"), True, XSS):
            with self.subTest(junk=junk):
                doc = render_report(make_run([record("a-1", True)], target_retries=junk, n_target_retries=0))
                self.assertBalanced(doc)
                self.assertLacks(XSS, doc)
                self.assertNotIn("configured target_retries", seen(doc))
                self.assertLacks("Target retries", doc)

    # ---- a run over part of the suite (C26)
    def limited(self, **selection):
        return {"limited": True, "n_selected": 12, "n_config": 45, "limit": None, "tags": None, "ids": None, **selection}

    def test_a_limited_run_says_it_covers_part_of_the_suite_and_is_no_baseline(self):
        doc = render_report(make_run([record("a-1", True)], selection=self.limited(limit=12)))
        text = seen(doc)
        self.assertIn("Selection limited run covers 12 of the 45 cases of the suite (limit 12): it is not a baseline for the whole suite", text)
        self.assertHas('<span class="badge warn">limited run</span>', doc)
        self.assertLess(text.index("Test set"), text.index("Selection"), "the row sits right under the test set it qualifies")
        self.assertBalanced(doc)
        self.assertSelfContained(doc)

    def test_the_limited_run_note_names_the_tags_and_the_case_ids_that_were_chosen(self):
        text = seen(render_report(make_run([record("a-1", True)], selection=self.limited(n_selected=8, tags=["safety", "billing"], ids=["a-1", 7]))))
        self.assertIn("covers 8 of the 45 cases of the suite (tags: safety, billing; case ids: a-1, 7)", text)

    def test_a_long_list_of_case_ids_is_cut_and_counted(self):
        ids = [f"case-{i:02d}" for i in range(9)]
        text = seen(render_report(make_run([record("a-1", True)], selection=self.limited(ids=ids))))
        self.assertIn("case ids: case-00, case-01, case-02, case-03, case-04 and 4 more)", text)
        self.assertNotIn("case-05", text)

    def test_a_run_that_is_not_limited_has_no_selection_row(self):
        for selection in (None, {}, {"limited": False, "n_selected": 45, "n_config": 45}, {"limited": "true"}, {"limited": 1}, {"n_selected": 3, "n_config": 45}, "limited", ["limited"]):
            with self.subTest(selection=selection):
                doc = render_report(make_run([record("a-1", True)], selection=selection))
                self.assertLacks("limited run", doc)
                self.assertLacks("Selection", doc)
                self.assertBalanced(doc)
        self.assertLacks("Selection", render_report(sample_run()))  # an older summary has no `selection`

    def test_a_limited_run_with_unreadable_counts_still_says_it_is_partial(self):
        for selection in ({"limited": True}, {"limited": True, "n_selected": "12", "n_config": None, "tags": "safety", "ids": {"a": 1}, "limit": "x"}):
            with self.subTest(selection=selection):
                doc = render_report(make_run([record("a-1", True)], selection=selection))
                self.assertIn("limited run covers only part of the cases of the suite: it is not a baseline for the whole suite", seen(doc))
                self.assertBalanced(doc)

    def test_a_flag_in_the_lists_is_not_shown_as_a_tag_or_a_case_id(self):
        """`isinstance(True, int)` is true in Python: a forged `ids: [true]` must not print "True" as if it were a case id."""
        text = seen(render_report(make_run([record("a-1", True)], selection=self.limited(tags=[True, "safety", False], ids=[True, False, "a-1"]))))
        self.assertIn("(tags: safety; case ids: a-1)", text)
        self.assertNotIn("True", text)
        self.assertNotIn("False", text)

    def test_the_limit_of_a_limited_run_is_shown_only_when_it_is_a_number_above_zero(self):
        for limit, shown in ((12, "(limit 12)"), (0, None), (None, None), (True, None), ("12", None), (-3, None), (0.5, None), (float("nan"), None), (float("inf"), None)):
            with self.subTest(limit=limit):
                text = seen(render_report(make_run([record("a-1", True)], selection=self.limited(limit=limit))))
                if shown:
                    self.assertIn(shown, text)
                else:
                    self.assertNotIn("limit ", text)
                    self.assertIn("limited run covers 12 of the 45 cases of the suite: it is not a baseline", text)

    def test_the_selection_note_escapes_what_comes_from_the_run(self):
        doc = render_report(make_run([record("a-1", True)], selection=self.limited(tags=[XSS], ids=[XSS, 3, None, [XSS]])))
        self.assertLacks(XSS, doc)
        self.assertHas(f"tags: {XSS_ESCAPED}", doc)
        self.assertHas(f"case ids: {XSS_ESCAPED}, 3)", doc)
        self.assertBalanced(doc)

    # ---- oversize (C18)
    def test_oversize_outputs_are_an_honest_note_and_not_a_judge_reading(self):
        doc = render_report(make_run([record("a-1", True), record("big", False)], n_oversize=2))
        text = seen(doc)
        self.assertIn("2 judge verdict(s) are automatic fails because the output was longer than the judge's max_output_chars", text)
        self.assertIn("the judge was not called (cost $0) and never read it", text)
        self.assertIn("2 judge verdict(s) are automatic fails (output over max_output_chars)", text)  # the pass-rate tile
        self.assertHas('class="health oversize-note" role="note"', doc)
        self.assertBalanced(doc)

    def test_oversize_is_counted_from_the_records_when_the_summary_has_no_counter(self):
        big = judge("quality", False, reasoning="output exceeds max_output_chars (30000 > 20000)")
        big["details"] = {"criteria": {}, "oversize": True, "output_chars": 30000}
        run = make_run([record("a-1", True), record("big", False, judges=[big]), record("big-2", False, judges=[dict(big, judge="second")])])
        self.assertIn("2 judge verdict(s) are automatic fails", seen(render_report(run)))
        self.assertIn("5 judge verdict(s) are automatic fails", seen(render_report(make_run([record("a-1", True)], n_oversize=5))))
        self.assertIn("1 judge verdict(s) are automatic fails", seen(render_report(make_run([record("big", False, judges=[big])], n_oversize=1))))

    def test_no_oversize_no_note_and_junk_is_ignored(self):
        self.assertNotIn("automatic fails", seen(render_report(sample_run())))
        for junk in (0, None, "2", -4, [], float("nan"), True):
            with self.subTest(junk=junk):
                doc = render_report(make_run([record("a-1", True)], n_oversize=junk))
                self.assertBalanced(doc)
                self.assertNotIn("automatic fails", seen(doc))
        weird = make_run([{**record("a-1", True), "judges": [{"judge": "q", "passed": True, "details": None}, {"judge": "r", "passed": True, "details": "oversize"}, {"judge": "s", "passed": False, "details": {"oversize": True}}]}])
        self.assertIn("1 judge verdict(s) are automatic fails", seen(render_report(weird)))

    # ---- stale calibration (C20)
    SHA_RUN, SHA_CAL = "a" * 64, "b" * 64

    def stale_pair(self, run_sha="a" * 64, cal_sha="b" * 64, **cal_extra):
        run = make_run([record("a-1", True), record("b-1", False)], judges_sha=run_sha) if run_sha else make_run([record("a-1", True)])
        cal = cycle2_calibration(**cal_extra)
        if cal_sha:
            cal["judges_sha"] = cal_sha
        return run, cal

    def test_a_calibration_made_for_other_judges_is_flagged_in_the_section_the_tile_and_the_findings(self):
        from evalsprint.calibration import STALE_CALIBRATION_TEXT

        run, cal = self.stale_pair()
        doc = render_report(run, None, cal)
        text = seen(doc)
        self.assertIn(STALE_CALIBRATION_TEXT, text)
        self.assertIn(f"judges_sha {self.SHA_CAL[:12]} in the calibration, {self.SHA_RUN[:12]} in this run", text)  # short, like the header's fingerprint
        self.assertNotIn(self.SHA_CAL, text)
        self.assertIn(f"stale: {STALE_CALIBRATION_TEXT}", text)  # tile
        self.assertIn("Stale calibration: this calibration was made for a different judge configuration", text)  # finding
        self.assertHas('<ul class="cal-warnings"><li>This calibration was made for a different judge configuration', doc)  # first warning of the section
        self.assertBalanced(doc)

    def test_the_report_uses_the_wording_of_the_calibration_module(self):
        from evalsprint import report
        from evalsprint.calibration import STALE_CALIBRATION_TEXT, stale_calibration

        self.assertEqual(report.STALE_CALIBRATION_TEXT, STALE_CALIBRATION_TEXT)
        run, cal = self.stale_pair()
        shortened = stale_calibration(cal, run).replace(self.SHA_CAL, self.SHA_CAL[:12]).replace(self.SHA_RUN, self.SHA_RUN[:12])
        self.assertEqual(report._stale_calibration_text(cal, run), shortened)  # the same sentence, with 12-character fingerprints

    def test_matching_or_missing_fingerprints_say_nothing(self):
        for run_sha, cal_sha in (("a" * 64, "a" * 64), (None, "b" * 64), ("a" * 64, None), (None, None), ("a" * 64, ""), ("", "b" * 64)):
            with self.subTest(run=bool(run_sha), cal=bool(cal_sha)):
                run, cal = self.stale_pair(run_sha=run_sha, cal_sha=cal_sha)
                if run_sha == "":
                    run.summary["judges_sha"] = ""
                doc = render_report(run, None, cal)
                self.assertNotIn("different judge configuration", seen(doc))
                self.assertNotIn("Stale calibration", seen(doc))

    def test_a_warning_the_calibration_already_carries_is_not_repeated(self):
        run, cal = self.stale_pair(warnings=["This calibration was made for a different judge configuration (kept from calibrate)"])
        text = seen(render_report(run, None, cal))
        self.assertEqual(text.count("calibration was made for a different judge configuration"), 3)  # the warning, the tile and the key finding
        self.assertNotIn("in the calibration", text)  # our own longer copy was not added

    def test_garbage_fingerprints_do_not_break_the_calibration_section(self):
        for junk in (None, 3, ["x"], {"a": 1}, XSS):
            with self.subTest(junk=junk):
                run, cal = self.stale_pair()
                cal["judges_sha"] = junk
                doc = render_report(run, None, cal)
                self.assertBalanced(doc)
                self.assertLacks(XSS, doc)
        run, cal = self.stale_pair()
        run.summary["judges_sha"] = XSS
        self.assertLacks(XSS, render_report(run, None, cal))

    # ---- per-tag repeat protocol (C17)
    def per_tag_run(self, repeat):
        safety = record("s-1", True, tags=("safety",), attempts=[{"outcome": "pass", "passed": True, "output_sha": "a"}] * 5, stability=1.0)
        other = record(
            "o-1", False, tags=("billing",),
            attempts=[{"outcome": "pass", "passed": True, "output_sha": "a"}, {"outcome": "fail", "passed": False, "output_sha": "b"}, {"outcome": "fail", "passed": False, "output_sha": "c"}],
            stability=2 / 3,
        )
        return make_run([safety, other, record("p-1", True)], stability={"n_repeated": 2, "flip_rate": 0.5}, repeat=repeat)

    def test_the_header_names_the_per_tag_protocol_and_the_stability_section_the_attempt_counts(self):
        doc = render_report(self.per_tag_run({"n": 3, "per_tag": {"safety": 5}}))
        text = seen(doc)
        self.assertIn("2 repeated cases (safety 5 attempts, every other case 3 attempts), majority outcome", text)
        self.assertIn("2 case(s) ran 5 times (1), 3 times (1)", text)
        self.assertIn("the protocol gives each tag its own number of attempts", text)
        self.assertBalanced(doc)

    def test_legacy_tags_and_per_tag_render_alike(self):
        legacy_doc = seen(render_report(self.per_tag_run({"n": 5, "tags": ["safety"]})))
        new_doc = seen(render_report(self.per_tag_run({"n": 1, "per_tag": {"safety": 5}})))
        self.assertIn("(safety 5 attempts, every other case once)", legacy_doc)
        self.assertIn("(safety 5 attempts, every other case once)", new_doc)

    def uniform_run(self, repeat):
        cases = [record(f"u-{i}", True, attempts=[{"outcome": "pass", "passed": True, "output_sha": "a"}] * 3, stability=1.0) for i in range(2)]
        return make_run(cases + [record("p-1", True)], stability={"n_repeated": 2, "flip_rate": 0.0}, repeat=repeat)

    def test_a_uniform_protocol_keeps_the_short_header(self):
        self.assertIn("2 repeated cases × 3 attempts, majority outcome", seen(render_report(self.uniform_run({"n": 3, "per_tag": {}}))))
        self.assertIn("2 repeated cases × 3 attempts, majority outcome", seen(render_report(self.uniform_run({"n": 3, "per_tag": {"billing": 3}}))))
        self.assertIn("2 repeated cases × 3 attempts, majority outcome", seen(render_report(self.uniform_run(None))))  # no protocol: what the records show

    def test_a_protocol_with_nothing_repeated_is_still_worded(self):
        run = make_run([record("a-1", True)], stability={"n_repeated": 0, "flip_rate": None}, repeat={"n": 3, "per_tag": {"safety": 5}})
        text = seen(render_report(run))
        self.assertIn("Repeats no case ran more than once · protocol: safety 5 attempts, every other case 3 attempts", text)
        self.assertIn("No case was repeated", text)
        self.assertIn("{n: 3, per_tag: {safety: 5}}", text)  # the suggestion is the per-tag protocol, not the old tags spelling
        self.assertNotIn("tags: [safety]", text)

    def test_the_comparison_protocol_is_worded_per_tag(self):
        protocol = {"baseline": {"n": 3, "per_tag": {}}, "candidate": {"n": 3, "per_tag": {"safety": 5}}}
        text = seen(render_report(sample_run(), self.auto_comparison(repeat_changed=True, repeat_protocol=protocol)))
        self.assertIn("Evaluation protocol changed (baseline: every case 3 attempts; candidate: safety 5 attempts, every other case 3 attempts)", text)
        same = seen(render_report(sample_run(), self.auto_comparison(repeat_changed=False, repeat_protocol={"baseline": protocol["candidate"], "candidate": protocol["candidate"]})))
        self.assertIn("Same repeat protocol as the baseline (safety 5 attempts, every other case 3 attempts)", same)

    def test_protocol_junk_and_escaping(self):
        junk = [None, [], "x", {"n": "many", "per_tag": "safety"}, {"n": 3, "per_tag": {"safety": "five"}}, {"n": 3, "per_tag": {"safety": 0}}, {"n": 0, "per_tag": {}}, {"per_tag": [1]}, True, 7]
        for bad in junk:
            with self.subTest(repeat=bad):
                doc = render_report(self.per_tag_run(bad))
                self.assertBalanced(doc)
                self.assertIn("2 case(s) ran 5 times (1), 3 times (1)", seen(doc))  # the records say what ran when the protocol is unreadable
        doc = render_report(self.per_tag_run({"n": 3, "per_tag": {XSS: 5}}))
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)

    def test_a_bare_integer_repeat_is_every_case_n_times(self):
        self.assertIn("2 repeated cases × 4 attempts", seen(render_report(self.uniform_run(4))))  # the setting of the summary wins over the records

    # ---- evalsprint patch versions (C20)
    def test_a_patch_difference_is_worded_as_a_patch_and_a_minor_one_as_a_version_change(self):
        patch = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "0.2.0", "candidate": "0.2.1", "changed": True})))
        self.assertIn("evalsprint patch version differs (baseline 0.2.0, candidate 0.2.1)", patch)
        self.assertIn("a patch release can fix how a case is scored", patch)  # no compare warning in this comparison: the fact explains itself
        warned = seen(render_report(sample_run(), self.auto_comparison(
            kit_version={"baseline": "0.2.0", "candidate": "0.2.1", "changed": True},
            warnings=["evalsprint patch version differs between baseline and candidate (baseline 0.2.0, candidate 0.2.1): scoring is expected to be the same, but a patch release can fix how a case is scored"])))
        self.assertEqual(warned.count("a patch release can fix how a case is scored"), 1)  # explained once, by compare's warning
        self.assertEqual(warned.count("evalsprint patch version differs"), 2)  # the short fact and the warning
        minor = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "0.2.1", "candidate": "0.3.0", "changed": True})))
        self.assertIn("evalsprint version differs (baseline 0.2.1, candidate 0.3.0)", minor)
        self.assertNotIn("patch version", minor)

    def test_a_comparison_written_before_c20_still_shows_a_patch_difference(self):
        old = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "0.2.0", "candidate": "0.2.1", "changed": False})))
        self.assertIn("evalsprint patch version differs (baseline 0.2.0, candidate 0.2.1)", old)
        self.assertNotIn("same release series", old)

    def test_the_same_release_spelled_twice_is_not_a_difference(self):
        text = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "v0.2", "candidate": "0.2.0", "changed": False})))
        self.assertIn("evalsprint v0.2 and 0.2.0 (the same release)", text)
        self.assertNotIn("differs", text)

    def test_non_numeric_versions_are_a_version_difference_not_a_patch(self):
        text = seen(render_report(sample_run(), self.auto_comparison(kit_version={"baseline": "dev", "candidate": "main", "changed": True})))
        self.assertIn("evalsprint version differs (baseline dev, candidate main)", text)
        self.assertNotIn("patch version", text)

    # ---- tnr_reliable as a screening flag (C20)
    def tnr_text(self, **extra):
        return seen(render_report(sample_run(), None, cycle2_calibration(**extra)))

    def test_a_failed_screening_check_says_which_check_with_the_recorded_thresholds(self):
        rule = {"min_human_fails": 10, "max_half_width": 0.15, "min_tnr": 0.75, "kind": "screening flag"}
        cases = {
            "few_human_fails": "screening check not passed: fewer than 10 bad outputs labeled",
            "wide_ci": "screening check not passed: interval wider than ±15 points",
            "low_tnr": "low: the judge misses more than 25% of the bad outputs (below 75%)",
            "undefined": "screening check not passed: interval undefined",
            "something_new": "screening check not passed",
            None: "screening check not passed",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                text = self.tnr_text(tnr_reliable=False, tnr_reliable_reason=reason, tnr_reliable_rule=rule)
                self.assertIn(f"human-rejected outputs the judge also fails; {expected}", text)
                self.assertNotIn("unreliable: too few", text)

    def test_the_thresholds_are_optional(self):
        self.assertIn("screening check not passed: too few bad outputs labeled", self.tnr_text(tnr_reliable=False, tnr_reliable_reason="few_human_fails"))
        self.assertIn("screening check not passed: interval too wide", self.tnr_text(tnr_reliable=False, tnr_reliable_reason="wide_ci", tnr_reliable_rule="junk"))
        self.assertIn("low: the judge misses too many of the bad outputs", self.tnr_text(tnr_reliable=False, tnr_reliable_reason="low_tnr", tnr_reliable_rule={}))

    def test_a_passed_screen_is_a_screen_not_a_guarantee(self):
        text = self.tnr_text(tnr_reliable=True, warnings=[])
        self.assertIn("screening check passed (a screen, not a guarantee)", text)
        self.assertNotIn("unreliable", text)

    def test_the_calibrations_own_explanation_is_shown_and_there_is_a_fallback(self):
        meaning = "tnr_reliable is a screening flag, not a guarantee: a judge whose true rate is 70% passed it in 2% to 28% of the runs"
        self.assertIn(meaning, self.tnr_text(tnr_reliable=True, tnr_reliable_rule={"kind": "screening flag", "meaning": meaning}))
        fallback = self.tnr_text(tnr_reliable=False)
        self.assertIn("The true-negative screening flag is a screen, not a guarantee", fallback)
        for absent in (None,):
            self.assertNotIn("screening flag is a screen", self.tnr_text(tnr_reliable=absent))

    def test_the_trustworthy_verdict_mentions_the_screening_check(self):
        text = self.tnr_text(kappa_decision="trustworthy", tnr_reliable=True, warnings=[], labels_needed_estimate=0)
        self.assertIn("true-negative screening check passed (a screen, not a guarantee)", text)

    def test_screening_text_is_escaped(self):
        doc = render_report(sample_run(), None, cycle2_calibration(tnr_reliable=False, tnr_reliable_reason="wide_ci", tnr_reliable_rule={"meaning": XSS, "max_half_width": 0.1}))
        self.assertLacks(XSS, doc)
        self.assertHas(XSS_ESCAPED, doc)
        self.assertBalanced(doc)


class TestSimulatedBanner(ReportTestCase):
    def test_banner_when_the_target_says_it_is_a_simulator(self):
        run = make_run([record("a-1", meta={"simulated": True}), record("b-1", meta={"simulated": True})])
        doc = render_report(run)
        self.assertHas('class="sim-banner"', doc)
        self.assertIn("Simulated run.", plain(doc))
        self.assertIn("not from a language model", plain(doc))

    def test_no_banner_for_real_runs(self):
        run = make_run([record("a-1", meta={"simulated": False}), record("b-1")])
        self.assertLacks("sim-banner", render_report(run).replace(".sim-banner", ""))
        self.assertLacks("Simulated run", render_report(run))


class TestWriteReport(unittest.TestCase):
    def test_creates_parent_directories_and_returns_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "dir" / "report.html"
            written = write_report(target, sample_run(), regression_comparison(), calibration_dict())
            self.assertEqual(written, target)
            self.assertEqual(target.read_text(encoding="utf-8"), render_report(sample_run(), regression_comparison(), calibration_dict()))


if __name__ == "__main__":
    unittest.main()
