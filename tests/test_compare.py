"""Tests for evalsprint.compare: pairing, the `auto` drop rule, every gate rule, calibration link, markdown.

Runs are built from hand-made result records (see `build`) that follow the record
contract of docs/ARCHITECTURE.md / cycle-2 C1-C2 (`outcome`, `error_source`, `case_sha`,
`attempts`).  The summary is minimal on purpose: compare reads only the run id and
the gate block, so these tests do not depend on how the runner aggregates.
(The CI workflow templates are tested in tests/test_templates.py.)
"""

from __future__ import annotations

import html
import json
import re
import unittest

from evalsprint.compare import (
    MAX_LISTED_IDS,
    SMALL_SET,
    _code,
    auto_alpha,
    auto_min_net_loss,
    auto_trips,
    compare_runs,
    normalize_gate_mode,
    resolve_gate,
    to_markdown,
)
from evalsprint.gate_defaults import DEFAULT_GATE, GATE_KEYS
from evalsprint.runner import RunResult

# ------------------------------------------------------------------------ builders


def _pick(spec, i):
    """Per-case value: a callable of the index, a list indexed by it, or a constant."""
    if callable(spec):
        return spec(i)
    if isinstance(spec, list):
        return spec[i]
    return spec


def flags(n: int, fail=(), error=()) -> list:
    """`n` passing cases except the indexes in `fail` (fail) and `error` (target error)."""
    failing, erroring = set(fail), set(error)
    return ["error" if i in erroring else (i not in failing) for i in range(n)]


def build(
    pass_flags,
    *,
    tags=None,
    scores=None,
    cost=None,
    latency=None,
    error=None,
    prefix="c",
    sha=None,
    attempts=None,
    legacy=False,
    llm=False,
) -> list[dict]:
    """Result records shaped like runner output.

    Flag items: True / False (pass / fail), "error" (target error), "judge_error" (the judge
    failed).  `scores` items: None, a float, or a tuple (one per judge).  `attempts` items: a list
    of outcome strings, e.g. ["pass", "fail", "pass"].  `legacy=True` drops outcome / error_source /
    case_sha (records from before cycle 2).  `llm=True` gives the judges LLM-style details.
    """
    records = []
    for i, flag in enumerate(pass_flags):
        cid = f"{prefix}{i:03d}"
        score = _pick(scores, i)
        score = () if score is None else (score if isinstance(score, tuple) else (score,))
        err = _pick(error, i) or ("timeout" if flag == "error" else None)
        judge_error = flag == "judge_error"
        passed = flag is True and not err
        outcome = "error" if (err or judge_error) else ("pass" if passed else "fail")
        judge_scores = score or ((None,) if (judge_error or llm) else ())
        judges = [
            {
                "judge": f"j{k}",
                "passed": passed,
                "score": s,
                "details": {"criteria": {}} if llm else {},
                "usage": {"input_tokens": 100} if llm else {},
                "error": "judge outage" if judge_error and k == 0 else None,
            }
            for k, s in enumerate(judge_scores)
        ]
        rec = {
            "case_id": cid,
            "case": {"id": cid, "input": "q", "tags": list(_pick(tags, i) or [])},
            "output": {
                "text": "answer",
                "sha": "abc",
                "model": "m",
                "usage": {},
                "cost_usd": _pick(cost, i),
                "latency_s": _pick(latency, i),
                "meta": {},
                "error": err,
            },
            "judges": judges,
            "passed": passed,
        }
        if not legacy:
            rec["outcome"] = outcome
            rec["error_source"] = "target" if err else ("judge" if judge_error else None)
            rec["case_sha"] = _pick(sha, i) if sha is not None else f"{cid}-v1"
        runs = _pick(attempts, i)
        if runs:
            # attempt items: "pass" | "fail" | "error" (target error) | "judge_error"
            rec["attempts"] = [
                {
                    "outcome": "error" if o in ("error", "judge_error") else o,
                    "passed": o == "pass",
                    "output_sha": f"s{k}",
                    "error_source": {"error": "target", "judge_error": "judge"}.get(o),
                }
                for k, o in enumerate(runs)
            ]
            rec["stability"] = max(runs.count("pass"), runs.count("fail")) / len(runs)
            rec["n_attempt_errors"] = sum(o in ("error", "judge_error") for o in runs)  # C10 record key
        records.append(rec)
    return records


JUDGES_SHA = "0123456789ab"


def make_run(run_id: str, records: list[dict], gate: dict | None = None, judges_sha: str | None = JUDGES_SHA, **summary_extra) -> RunResult:
    summary = {"run_id": run_id, "target_retries": 1}  # the config value a v0.2 runner always records (C22)
    if judges_sha is not None:
        summary["judges_sha"] = judges_sha
    if gate is not None:
        summary["gate"] = gate
    summary.update(summary_extra)  # e.g. repeat={"n": 3, "tags": None}, evalsprint_version="0.2.0", target_retries=None (a v0.1 summary)
    return RunResult(run_dir=None, summary=summary, results=records)


def compare(
    base_flags, cand_flags, *, gate=None, summary_gate=None, base_kw=None, cand_kw=None, calibration=None,
    base_sha=JUDGES_SHA, cand_sha=JUDGES_SHA, base_summary=None, cand_summary=None,
) -> dict:  # fmt: skip
    """Compare two same-id runs built from pass flags (`base_summary` / `cand_summary`: extra summary keys)."""
    base = make_run("base", build(base_flags, **(base_kw or {})), judges_sha=base_sha, **(base_summary or {}))
    cand = make_run("cand", build(cand_flags, **(cand_kw or {})), gate=summary_gate, judges_sha=cand_sha, **(cand_summary or {}))
    return compare_runs(base, cand, gate=gate, calibration=calibration)


def warning_with(c: dict, fragment: str) -> list[str]:
    return [w for w in c["warnings"] if fragment in w]


# 100 cases, 90 passing in the baseline (c090..c099 fail): the workhorse scenario.
BASE_100 = flags(100, fail=range(90, 100))
FIXED_3PP = {"max_pass_rate_drop": 0.03}  # the pre-cycle-2 default, for tests of the fixed mode


def drop_by(k: int, n: int = 100, base_fail=range(90, 100)) -> list:
    """Candidate flags with the first `k` baseline-passing cases flipped to fail."""
    return flags(n, fail=[*base_fail, *range(k)])


# ------------------------------------------------------------- auto pass-rate rule


class AutoMode(unittest.TestCase):
    """C9: ONE rule for every n. Trips when the net loss is >= max(3, ceil(n/50)) cases AND the exact one-sided McNemar
    p-value is below alpha(n) = 0.0575 + 0.1925 / (1 + (n/85)^4.5)."""

    def test_four_clean_regressions_trip_at_100_cases(self):
        c = compare(BASE_100, drop_by(4))  # b=4, c=0: p = 1/16 = 0.0625 < alpha(100) = 0.120
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["gate_mode"], "auto")
        self.assertAlmostEqual(c["effective_max_pass_rate_drop"], 0.02)  # a net loss strictly above it trips: exactly 3 cases
        self.assertEqual(c["effective_min_net_loss"], 3)
        self.assertAlmostEqual(c["effective_alpha"], auto_alpha(100))
        self.assertEqual(
            c["reasons"],
            ["pass rate dropped 4.0 p.p. (auto rule for 100 paired cases: net loss 4 cases vs 3 needed, one-sided McNemar p=0.062 vs α=0.120)"],
        )
        pr = c["pass_rate"]
        self.assertAlmostEqual(pr["baseline"], 0.90)
        self.assertAlmostEqual(pr["candidate"], 0.86)
        self.assertEqual((pr["b_pass_to_fail"], pr["c_fail_to_pass"]), (4, 0))

    def test_three_clean_regressions_are_one_step_of_alpha_away(self):
        # p = 1/8 = 0.125 needs alpha(n) above it: true up to n = 97 (alpha = 0.126), false from n = 98 (alpha = 0.124)
        self.assertEqual(compare(flags(97), flags(97, fail=range(3)))["verdict"], "regression")
        c = compare(flags(98), flags(98, fail=range(3)))
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "not counted")
        self.assertIn("net loss 3 cases vs 3 needed, one-sided McNemar p=0.125 vs α=0.124", w)

    def test_a_net_loss_below_the_floor_never_fires_however_clean(self):
        c = compare(BASE_100, drop_by(2))  # p = 0.25 anyway
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual((c["reasons"], c["warnings"]), ([], []))
        c = compare(flags(500, fail=range(400, 500)), flags(500, fail=[*range(5), *range(400, 500)]))
        self.assertEqual(c["verdict"], "pass")  # 5 clean regressions in 500: p = 0.031 < alpha, but 1 p.p. is under the 10-case floor
        [w] = warning_with(c, "significant")
        self.assertIn("it is significant (one-sided McNemar p=0.031 < α=0.058)", w)
        self.assertIn("the net loss is only 5 cases and the auto rule needs at least 10", w)
        self.assertIn("not counted", w)

    def test_a_noisy_drop_with_a_big_enough_net_loss_passes_with_a_warning(self):
        # 8 pass->fail, 4 fail->pass: net loss 4 >= 3, but one-sided p = 0.194 > alpha(100)
        c = compare(BASE_100, flags(100, fail={*range(8), *range(94, 100)}))
        self.assertEqual((c["pass_rate"]["b_pass_to_fail"], c["pass_rate"]["c_fail_to_pass"]), (8, 4))
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "not counted")
        self.assertIn("pass rate dropped 4.0 p.p. (auto rule for 100 paired cases: net loss 4 cases vs 3 needed, one-sided McNemar p=0.194 vs α=0.120)", w)
        self.assertIn("not significant at this suite size", w)
        self.assertNotIn("significant at α", w)

    def test_no_switch_and_no_cliff_around_150(self):
        # 6 clean regressions (p = 1/64) trip at every n around the old switch, with the same wording and mode
        for n in (147, 148, 149, 150, 151, 152, 153):
            with self.subTest(n=n):
                c = compare(flags(n), flags(n, fail=range(6)))
                self.assertEqual((c["gate_mode"], c["verdict"]), ("auto", "regression"))
                self.assertEqual(c["effective_min_net_loss"], auto_min_net_loss(n))
                self.assertAlmostEqual(c["effective_alpha"], auto_alpha(n))
                self.assertIn(f"auto rule for {n} paired cases", c["reasons"][0])
        # 3 clean regressions (p = 1/8) trip up to n = 97 and not beyond: alpha(n) falls below 1/8 there, one case, one step
        verdicts = {n: compare(flags(n), flags(n, fail=range(3)))["verdict"] for n in (60, 90, 97, 98, 100, 149, 150, 200)}
        self.assertEqual([verdicts[n] for n in (60, 90, 97)], ["regression"] * 3)
        self.assertEqual([verdicts[n] for n in (98, 100, 149, 150, 200)], ["pass"] * 5)

    def test_the_smallest_clean_regression_that_trips_grows_by_at_most_one_case_per_step(self):
        """Adjacent suite sizes never disagree by more than one case about how many clean regressions are enough."""
        smallest = {n: next(b for b in range(1, n + 1) if auto_trips(b, 0, n)) for n in range(30, 501)}
        steps = [smallest[n + 1] - smallest[n] for n in range(30, 500)]
        self.assertTrue(all(0 <= s <= 1 for s in steps), [n for n, s in zip(range(30, 500), steps) if not 0 <= s <= 1])
        self.assertEqual((smallest[30], smallest[45], smallest[150], smallest[200], smallest[500]), (3, 3, 4, 5, 10))

    def test_tiny_test_set_needs_three_cases(self):
        # n = 45: a net loss of 2 cases is 4.4 p.p. (passes), 3 cases is 6.7 p.p. (fails: p = 0.125 < alpha(45) = 0.239)
        self.assertEqual(compare(flags(45), flags(45, fail=[0, 1]))["verdict"], "pass")
        c = compare(flags(45), flags(45, fail=[0, 1, 2]))
        self.assertEqual(c["verdict"], "regression")
        self.assertAlmostEqual(c["effective_alpha"], 0.2396, places=3)
        self.assertAlmostEqual(c["effective_max_pass_rate_drop"], 2 / 45)

    def test_a_larger_suite_needs_a_significant_drop(self):
        base = flags(200, fail=range(180, 200))
        cand = lambda b, c=0: flags(200, fail=[*range(b), *range(180 + c, 200)])  # noqa: E731
        c = compare(base, cand(5))  # 2.5 p.p. clean: p = 1/32 < alpha(200) = 0.0615 and net loss 5 >= 4
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["pass rate dropped 2.5 p.p. (auto rule for 200 paired cases: net loss 5 cases vs 4 needed, one-sided McNemar p=0.031 vs α=0.062)"])
        self.assertEqual(compare(base, cand(3))["verdict"], "pass")  # net loss 3 < 4
        c = compare(base, cand(4))  # net loss 4 >= 4 but p = 1/16 = 0.0625 is not below alpha(200) = 0.0615
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(len(warning_with(c, "not counted")), 1)
        c = compare(base, cand(12, 6))  # net loss 6 but p = 0.119
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "not counted")
        self.assertIn("net loss 6 cases vs 4 needed, one-sided McNemar p=0.119 vs α=0.062", w)

    def test_significance_uses_the_one_sided_test(self):
        # 5 pass->fail, 0 fail->pass on 150 cases: one-sided p = 0.031 (two-sided 0.0625), alpha(150) = 0.089
        base = flags(150, fail=range(140, 150))
        c = compare(base, flags(150, fail=[*range(5), *range(140, 150)]))
        self.assertAlmostEqual(c["pass_rate"]["p_value"], 1 / 32)
        self.assertAlmostEqual(c["pass_rate"]["p_value_two_sided"], 1 / 16)
        self.assertEqual(c["verdict"], "regression")

    def test_the_gate_alpha_and_require_significance_do_not_apply_and_say_so(self):
        base = flags(150, fail=range(140, 150))
        cand = flags(150, fail=[*range(5), *range(140, 150)])
        c = compare(base, cand, gate={"alpha": 0.01, "require_significance": True})
        self.assertEqual(c["verdict"], "regression")  # auto's own alpha(150) = 0.089 decides
        self.assertEqual(len(warning_with(c, "require_significance is ignored")), 1)
        self.assertEqual(len(warning_with(c, "gate.alpha (0.01) is ignored")), 1)
        self.assertEqual(warning_with(compare(base, cand), "ignored"), [])  # the default alpha is silent

    def test_improvement_passes_and_lists_newly_passing(self):
        c = compare(BASE_100, flags(100))
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["newly_passing"], [f"c{i:03d}" for i in range(90, 100)])
        self.assertEqual(c["newly_failing"], [])

    def test_wilson_intervals_are_reported_for_both_runs(self):
        pr = compare(BASE_100, drop_by(6))["pass_rate"]
        for key, rate in (("baseline_ci", 0.90), ("candidate_ci", 0.84)):
            lo, hi = pr[key]
            self.assertLess(lo, rate)
            self.assertGreater(hi, rate)

    def test_cases_left_out_as_errors_do_not_count_toward_n(self):
        # 155 shared cases; 6 are left out of the pairing (a judge error), so 149 are paired: floor ceil(149/50) = 3, not 4
        c = compare(flags(155), ["judge_error" if i >= 149 else True for i in range(155)], gate={"max_error_rate": None})
        self.assertEqual((c["n_paired"], c["effective_min_net_loss"]), (149, 3))
        # ... and so are 6 cases that failed in the baseline and then crashed in the candidate
        c = compare(flags(155, fail=range(149, 155)), flags(155, error=range(149, 155)), gate={"max_error_rate": None})
        self.assertEqual((c["n_paired"], c["effective_min_net_loss"]), (149, 3))

    def test_crashes_on_passing_cases_do_count_toward_n(self):
        # 155 shared cases, 6 crash in the candidate on cases that passed: all 155 are paired (floor ceil(155/50) = 4)
        c = compare(flags(155), flags(155, error=range(149, 155)), gate={"max_error_rate": None})
        self.assertEqual((c["n_paired"], c["effective_min_net_loss"]), (155, 4))
        self.assertEqual(c["gate_mode"], "auto")

    def test_no_paired_cases_has_no_effective_numbers(self):
        c = compare_runs(make_run("b", []), make_run("c", []))
        self.assertEqual(c["verdict"], "regression")  # nothing was compared
        self.assertEqual((c["gate_mode"], c["effective_max_pass_rate_drop"], c["effective_alpha"], c["effective_min_net_loss"]), ("auto", None, None, None))


class AutoRuleMatchesTheContract(unittest.TestCase):
    """The shipped rule against a from-scratch reading of C9 on many (n, b, c), through compare_runs."""

    @staticmethod
    def contract(b, c, n):
        import math

        floor = max(3, math.ceil(n / 50))
        alpha = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)
        m = b + c
        p_one_sided = sum(math.comb(m, i) for i in range(b, m + 1)) / 2**m
        return b - c >= floor and p_one_sided < alpha

    def test_verdict_matches_on_a_grid(self):
        for n in (10, 30, 45, 99, 149, 150, 151, 200, 320):
            k = int(n * 0.8)
            for b in range(0, min(16, k + 1)):
                for c in range(0, min(9, n - k + 1)):
                    with self.subTest(n=n, b=b, c=c):
                        base = flags(n, fail=range(k, n))
                        cand_fail = [*range(b), *range(k + c, n)]
                        res = compare(base, flags(n, fail=cand_fail))
                        self.assertEqual((res["pass_rate"]["b_pass_to_fail"], res["pass_rate"]["c_fail_to_pass"]), (b, c))
                        self.assertEqual(res["verdict"] == "regression", self.contract(b, c, n))
                        self.assertEqual(auto_trips(b, c, n), self.contract(b, c, n))


class FixedMode(unittest.TestCase):
    def test_numeric_limit_is_a_fixed_absolute_rule_without_significance(self):
        self.assertEqual(compare(BASE_100, drop_by(3), gate=FIXED_3PP)["verdict"], "pass")  # 0.87 - 0.90 float noise
        c = compare(BASE_100, drop_by(4), gate=FIXED_3PP)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual((c["gate_mode"], c["effective_max_pass_rate_drop"]), ("fixed", 0.03))
        self.assertEqual(c["reasons"], ["pass rate dropped 4.0 p.p. (limit 3.0 p.p.; one-sided McNemar p=0.062)"])

    def test_custom_limit_and_disabled_rule(self):
        self.assertEqual(compare(BASE_100, drop_by(6), gate={"max_pass_rate_drop": 0.10})["verdict"], "pass")
        c = compare(BASE_100, drop_by(6), gate={"max_pass_rate_drop": None})
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual((c["gate_mode"], c["effective_max_pass_rate_drop"]), ("off", None))

    def test_require_significance_suppresses_a_noisy_drop_but_warns(self):
        gate = {**FIXED_3PP, "require_significance": True}
        c = compare(BASE_100, drop_by(4), gate=gate)  # 4 pass->fail: one-sided p = 0.0625 >= 0.05
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["reasons"], [])
        [w] = warning_with(c, "not counted")
        self.assertIn("require_significance is on", w)

    def test_require_significance_still_flags_a_significant_drop(self):
        c = compare(BASE_100, drop_by(10), gate={**FIXED_3PP, "require_significance": True})
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(warning_with(c, "not counted"), [])
        self.assertIn("significant at α=0.05", c["reasons"][0])

    def test_one_sided_p_and_alpha_are_used(self):
        # 5 pass->fail: one-sided p = 0.031 < 0.05, but two-sided 0.0625 would not be
        gate = {"max_pass_rate_drop": 0.02, "require_significance": True}
        self.assertEqual(compare(BASE_100, drop_by(5), gate=gate)["verdict"], "regression")
        self.assertEqual(compare(BASE_100, drop_by(5), gate={**gate, "alpha": 0.01})["verdict"], "pass")

    def test_significance_uses_both_directions_of_discordance(self):
        # 8 pass->fail and 4 fail->pass: -4 p.p., one-sided p = 0.19: not significant
        cand_fail = {*range(8), *range(94, 100)}  # c000..c007 regress; c090..c093 recover
        c = compare(BASE_100, flags(100, fail=cand_fail), gate={**FIXED_3PP, "require_significance": True})
        self.assertEqual((c["pass_rate"]["b_pass_to_fail"], c["pass_rate"]["c_fail_to_pass"]), (8, 4))
        self.assertEqual(c["verdict"], "pass")

    def test_require_significance_is_ignored_in_auto_mode_and_says_so(self):
        c = compare(BASE_100, drop_by(6), gate={"require_significance": True})
        self.assertEqual(c["verdict"], "regression")  # auto applies its own test (p = 1/64 < alpha(100)) and gate.require_significance does not matter
        self.assertEqual(len(warning_with(c, "require_significance is ignored")), 1)


class PValues(unittest.TestCase):
    def test_both_p_values_are_reported(self):
        pr = compare(BASE_100, drop_by(6))["pass_rate"]
        self.assertAlmostEqual(pr["p_value"], 1 / 64)  # one-sided: the candidate is worse
        self.assertAlmostEqual(pr["p_value_two_sided"], 2 / 64)

    def test_one_sided_p_is_high_when_the_candidate_improved(self):
        pr = compare(BASE_100, flags(100))["pass_rate"]
        self.assertAlmostEqual(pr["p_value"], 1.0)  # 0 worse vs 10 better: no evidence of a drop
        self.assertAlmostEqual(pr["p_value_two_sided"], 2 / 1024)

    def test_no_discordant_pairs(self):
        pr = compare(BASE_100, BASE_100)["pass_rate"]
        self.assertEqual((pr["p_value"], pr["p_value_two_sided"]), (1.0, 1.0))


class MinPassRateRule(unittest.TestCase):
    def test_below_floor_triggers(self):
        c = compare(flags(100, fail=range(24)), flags(100, fail=range(26)), gate={"min_pass_rate": 0.80})
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["candidate pass rate 74.0% is below the minimum 80.0%"])

    def test_at_floor_and_above_passes(self):
        at_floor = flags(100, fail=range(20))
        self.assertEqual(compare(at_floor, at_floor, gate={"min_pass_rate": 0.80})["verdict"], "pass")
        self.assertEqual(compare(BASE_100, BASE_100, gate={"min_pass_rate": 0.85})["verdict"], "pass")

    def test_unset_floor_never_triggers(self):
        base, cand = flags(100, fail=range(50)), flags(100, fail=range(52))
        self.assertEqual(compare(base, cand)["verdict"], "pass")

    def test_floor_looks_at_whole_candidate_run_not_just_paired_cases(self):
        base = make_run("base", build(flags(50)))
        cand = make_run("cand", build(flags(50)) + build(flags(50, fail=range(50)), prefix="new"))
        c = compare_runs(base, cand, gate={"min_pass_rate": 0.8})
        self.assertEqual(c["pass_rate"]["candidate"], 1.0)  # paired view is perfect
        self.assertAlmostEqual(c["pass_rate"]["candidate_overall"], 0.5)
        self.assertEqual(c["verdict"], "regression")

    def test_floor_leaves_judge_errors_out_of_the_pass_rate(self):
        # 90 pass, 5 fail, 5 judge errors: 90 / 95 = 94.7%, not 90%, on the whole run and on the baseline's cases
        cand = ["judge_error" if 5 <= i < 10 else (i >= 5) for i in range(100)]
        c = compare(flags(100), cand, gate={"min_pass_rate": 0.92, "max_error_rate": None})
        self.assertAlmostEqual(c["pass_rate"]["candidate_overall"], 90 / 95)
        self.assertAlmostEqual(c["pass_rate"]["candidate_paired_pessimistic"], 90 / 95)
        self.assertNotIn("below the minimum", " ".join(c["reasons"]))

    def test_floor_counts_target_errors_on_the_baselines_cases_as_failures(self):
        # 90 pass, 5 fail, 5 target errors on cases that passed: the whole run says 94.7%, the baseline's cases 90%
        cand = flags(100, fail=range(5), error=range(5, 10))
        c = compare(flags(100), cand, gate={"min_pass_rate": 0.92, "max_error_rate": None, "max_pass_rate_drop": None})
        self.assertAlmostEqual(c["pass_rate"]["candidate_overall"], 90 / 95)
        self.assertAlmostEqual(c["pass_rate"]["candidate_paired_pessimistic"], 0.90)
        self.assertEqual(c["pass_rate"]["candidate_paired_pessimistic_n"], 100)
        self.assertEqual(
            c["reasons"],
            [
                "candidate pass rate on the 100 cases that also exist in the baseline is 90.0%, below the minimum 92.0% "
                "(target errors count as failures; added cases do not count towards it)"
            ],
        )

    def test_appended_easy_cases_cannot_lift_the_candidate_over_the_floor(self):
        # critic-engineering 7b: 2 of 45 broken (82.2% on the baseline's cases), 100 easy cases appended (86.2% overall)
        base = make_run("base", build(flags(45, fail=range(6))))
        cand = make_run("cand", build(flags(45, fail=[*range(6), 6, 7])) + build(flags(100), prefix="easy"))
        c = compare_runs(base, cand, gate={"min_pass_rate": 0.85})
        self.assertGreater(c["pass_rate"]["candidate_overall"], 0.85)
        self.assertAlmostEqual(c["pass_rate"]["candidate_paired_pessimistic"], 37 / 45)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(len(c["reasons"]), 1)
        self.assertIn("candidate pass rate on the 45 cases that also exist in the baseline is 82.2%, below the minimum 85.0%", c["reasons"][0])

    def test_both_floors_below_the_minimum_with_the_same_value_give_one_reason(self):
        c = compare(flags(100, fail=range(24)), flags(100, fail=range(26)), gate={"min_pass_rate": 0.80})
        self.assertEqual(c["pass_rate"]["candidate_paired_pessimistic"], c["pass_rate"]["candidate_overall"])
        self.assertEqual(len(c["reasons"]), 1)

    def test_both_floors_report_their_own_reason_when_they_differ(self):
        base = make_run("base", build(flags(50)))
        cand = make_run("cand", build(flags(50, fail=range(5))) + build(flags(50, fail=range(20)), prefix="new"))
        c = compare_runs(base, cand, gate={"min_pass_rate": 0.95, "max_pass_rate_drop": None})
        self.assertEqual(len(c["reasons"]), 2)
        self.assertIn("candidate pass rate 75.0% is below the minimum 95.0%", c["reasons"][0])
        self.assertIn("on the 50 cases that also exist in the baseline is 90.0%", c["reasons"][1])

    def test_edited_cases_are_not_part_of_the_baselines_cases_for_the_floor(self):
        base = make_run("base", build(flags(50)))
        cand = make_run("cand", build(flags(50), sha=lambda i: "x" if i < 5 else f"c{i:03d}-v1"))
        c = compare_runs(base, cand, gate={"min_pass_rate": 0.9, "max_changed_cases": None})
        self.assertEqual(c["pass_rate"]["candidate_paired_pessimistic_n"], 45)


class CriticalTagsRule(unittest.TestCase):
    GATE = {"critical_tags": ["safety"]}

    @staticmethod
    def tags(i):
        return ["safety"] if i < 10 else ["general"]

    def run_case(self, cand_fail, base_fail=(), gate=None, tags=None, base_extra=None, cand_extra=None):
        kw = {"tags": tags or self.tags}
        return compare(
            flags(100, fail=base_fail),
            flags(100, fail=[*base_fail, *cand_fail]),
            gate=gate or self.GATE,
            base_kw={**kw, **(base_extra or {})},
            cand_kw={**kw, **(cand_extra or {})},
        )

    def test_critical_flip_triggers_even_when_overall_drop_is_small(self):
        c = self.run_case(cand_fail=[0, 1])  # -2 p.p. overall: the pass-rate rule stays quiet
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["critical tag 'safety': 2 cases went pass→fail (c000, c001)"])
        self.assertEqual(c["critical_regressions"], ["c000", "c001"])

    def test_single_case_uses_singular(self):
        self.assertIn("1 case went pass→fail (c000)", self.run_case(cand_fail=[0])["reasons"][0])

    def test_non_critical_flip_does_not_trigger(self):
        c = self.run_case(cand_fail=[50, 51])
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["critical_regressions"], [])

    def test_critical_case_failing_in_both_runs_is_not_a_regression(self):
        c = self.run_case(cand_fail=[], base_fail=[0, 1])
        self.assertEqual(c["verdict"], "pass")

    def test_no_critical_tags_configured(self):
        self.assertEqual(self.run_case(cand_fail=[0, 1], gate={"critical_tags": []})["verdict"], "pass")

    def test_long_case_lists_are_truncated_in_the_reason(self):
        c = self.run_case(cand_fail=list(range(8)), gate={**self.GATE, "max_pass_rate_drop": None})
        self.assertIn("(c000, c001, c002, c003, c004 and 3 more)", c["reasons"][0])
        self.assertEqual(len(c["critical_regressions"]), 8)

    def test_case_with_two_critical_tags_is_reported_under_each(self):
        c = self.run_case(
            cand_fail=[0],
            gate={"critical_tags": ["safety", "legal"]},
            tags=lambda i: ["safety", "legal"] if i == 0 else ["general"],
        )
        self.assertEqual(len(c["reasons"]), 2)
        self.assertEqual(c["critical_regressions"], ["c000"])

    def test_string_critical_tag_is_treated_as_one_tag(self):
        c = self.run_case(cand_fail=[0], gate={"critical_tags": "safety"})
        self.assertEqual(c["gate"]["critical_tags"], ["safety"])
        self.assertEqual(c["verdict"], "regression")

    def test_misspelled_critical_tag_warns_instead_of_passing_silently(self):
        c = self.run_case(cand_fail=[0], gate={"critical_tags": ["safty"]})
        self.assertEqual(c["verdict"], "pass")
        self.assertIn("critical tag 'safty' matches no paired case", c["warnings"][0])

    def test_retagging_a_case_in_the_candidate_does_not_hide_it(self):
        # the PR drops the tag from c000 in the candidate's dataset: tags are unioned across both runs
        c = compare(
            flags(100),
            flags(100, fail=[0]),
            gate=self.GATE,
            base_kw={"tags": self.tags},
            cand_kw={"tags": lambda i: ["general"]},
        )
        self.assertEqual(c["critical_regressions"], ["c000"])
        self.assertEqual(c["verdict"], "regression")

    # -- repeated critical cases (C2) ------------------------------------------------

    def test_unrepeated_critical_cases_get_a_repeat_suggestion(self):
        c = self.run_case(cand_fail=[])
        [w] = warning_with(c, "fewer than 5 times")
        self.assertIn("10 of 10 paired", w)
        self.assertIn("add per_tag: {safety: 5} under repeat in eval.yaml", w)  # C17: the critical tag only, never a whole-protocol block
        self.assertIn("keep your n otherwise, the other cases must not change", w)
        self.assertNotIn("tags: [", w)

    def test_repeat_warning_quantifies_the_false_alarms_it_is_about(self):
        [w] = warning_with(self.run_case(cand_fail=[]), "fewer than 5 times")  # 10 critical cases
        self.assertIn("false alarms with 10 critical cases at ~5% noise: 39% with n: 1, 7.0% with n: 3, 1.2% with n: 5", w)

    def test_critical_false_alarm_matches_the_closed_form(self):
        from evalsprint.compare import _critical_false_alarm

        # 8 healthy critical cases, 5% per-attempt noise: 1 - (1 - q(1-q))^8 with q = P(majority of k attempts fail)
        self.assertAlmostEqual(_critical_false_alarm(8, 1), 1 - (1 - 0.05 * 0.95) ** 8)
        q3 = 3 * 0.05**2 - 2 * 0.05**3
        self.assertAlmostEqual(_critical_false_alarm(8, 3), 1 - (1 - q3 * (1 - q3)) ** 8)
        self.assertAlmostEqual(_critical_false_alarm(8, 3), 0.0561, places=4)
        self.assertAlmostEqual(_critical_false_alarm(8, 5), 0.0092, places=4)
        self.assertLess(_critical_false_alarm(8, 5), 0.01)  # the <= 1% the kit promises for 8 critical cases
        self.assertAlmostEqual(_critical_false_alarm(8, 4), _critical_false_alarm(8, 4, 0.05))  # ties -> fail: even n is worse than n-1
        self.assertGreater(_critical_false_alarm(8, 4), _critical_false_alarm(8, 3))
        self.assertEqual(_critical_false_alarm(0, 3), 0.0)

    def test_critical_cases_repeated_5_times_get_no_repeat_warning(self):
        rep = {"attempts": lambda i: ["pass"] * 5 if i < 10 else None}
        c = self.run_case(cand_fail=[], base_extra=rep, cand_extra=rep)
        self.assertEqual(warning_with(c, "fewer than 5 times"), [])

    def test_critical_cases_repeated_only_3_times_still_get_the_warning(self):
        rep = {"attempts": lambda i: ["pass"] * 3 if i < 10 else None}
        c = self.run_case(cand_fail=[], base_extra=rep, cand_extra=rep)
        [w] = warning_with(c, "fewer than 5 times")
        self.assertIn("10 of 10 paired", w)
        self.assertIn("per_tag: {safety: 5}", w)

    def test_repeat_warning_when_only_one_side_was_repeated_enough(self):
        rep = {"attempts": lambda i: ["pass"] * 5 if i < 10 else None}
        c = self.run_case(cand_fail=[], cand_extra=rep)  # the baseline predates `repeat`
        self.assertEqual(len(warning_with(c, "fewer than 5 times")), 1)

    def test_repeat_warning_counts_the_cases_that_fall_short(self):
        rep5 = {"attempts": lambda i: ["pass"] * 5 if i < 6 else (["pass"] * 3 if i < 10 else None)}
        c = self.run_case(cand_fail=[], base_extra=rep5, cand_extra=rep5)
        [w] = warning_with(c, "fewer than 5 times")
        self.assertIn("4 of 10 paired", w)

    def test_repeat_warning_names_only_the_tags_that_need_it(self):
        rep = {"attempts": lambda i: ["pass"] * 5 if i < 10 else None}
        tags = lambda i: ["safety"] if i < 10 else ["legal"]  # noqa: E731
        c = self.run_case(cand_fail=[], gate={"critical_tags": ["safety", "legal"]}, tags=tags, base_extra=rep, cand_extra=rep)
        [w] = warning_with(c, "fewer than 5 times")
        self.assertIn("per_tag: {legal: 5}", w)
        self.assertNotIn("safety", w)

    def test_a_case_with_two_critical_tags_is_counted_once(self):
        tags = lambda i: ["safety", "legal"] if i < 4 else ["general"]  # noqa: E731
        c = self.run_case(cand_fail=[], gate={"critical_tags": ["safety", "legal"]}, tags=tags)
        [w] = warning_with(c, "fewer than 5 times")
        self.assertIn("4 of 4 paired", w)
        self.assertIn("per_tag: {safety: 5, legal: 5}", w)

    def test_flip_of_a_repeated_case_shows_how_many_attempts_failed(self):
        base_rep = {"attempts": lambda i: ["pass"] * 3 if i < 10 else None}
        cand_rep = {"attempts": lambda i: ["fail", "fail", "pass"] if i == 0 else (["pass"] * 3 if i < 10 else None)}
        c = self.run_case(cand_fail=[0], base_extra=base_rep, cand_extra=cand_rep)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["critical tag 'safety': 1 case went pass→fail (c000 (2/3 attempts failed))"])

    def test_critical_case_that_crashed_in_the_candidate_is_a_critical_regression(self):
        # a target error on a critical case that passed is a pass -> fail, not "unverified"
        c = compare(
            flags(100),
            flags(100, error=[0]),
            gate={**self.GATE, "max_error_rate": 0.05},
            base_kw={"tags": self.tags},
            cand_kw={"tags": self.tags},
        )
        self.assertEqual(c["critical_regressions"], ["c000"])
        self.assertEqual(c["critical_unverified"], [])
        self.assertEqual(c["target_error_regressions"], ["c000"])
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["critical tag 'safety': 1 case went pass→fail (c000 (crashed in the candidate))"])

    def test_critical_case_whose_judge_errored_in_the_candidate_is_not_silently_dropped(self):
        # a judge error leaves no verdict (infrastructure), but a critical case must not go unverified
        c = compare(
            flags(100),
            ["judge_error"] + [True] * 99,
            gate={**self.GATE, "max_error_rate": 0.05},
            base_kw={"tags": self.tags},
            cand_kw={"tags": self.tags},
        )
        self.assertEqual(c["critical_regressions"], [])
        self.assertEqual(c["critical_unverified"], ["c000"])
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("critical tag 'safety': 1 case passed in the baseline but the judge errored in the candidate", c["reasons"][0])
        self.assertIn("could not be verified", c["reasons"][0])

    def test_critical_case_that_already_failed_and_crashes_is_not_flagged(self):
        c = compare(
            flags(100, fail=[0]),
            flags(100, error=[0]),
            gate=self.GATE,
            base_kw={"tags": self.tags},
            cand_kw={"tags": self.tags},
        )
        self.assertEqual(c["critical_unverified"], [])
        self.assertEqual(c["verdict"], "pass")

    def test_tag_only_carried_by_an_errored_case_is_not_reported_as_a_typo(self):
        c = compare(
            flags(100),
            flags(100, error=[0]),
            gate={"critical_tags": ["rare"]},
            base_kw={"tags": lambda i: ["rare"] if i == 0 else []},
            cand_kw={"tags": lambda i: ["rare"] if i == 0 else []},
        )
        self.assertEqual(warning_with(c, "matches no paired case"), [])


class CostRule(unittest.TestCase):
    GATE = {"max_cost_increase_pct": 25}

    def costs(self, base, cand, gate=None):
        return compare(
            flags(20),
            flags(20),
            gate=self.GATE if gate is None else gate,
            base_kw={"cost": base},
            cand_kw={"cost": cand},
        )

    def test_increase_beyond_limit_triggers(self):
        c = self.costs(0.010, 0.0138)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["mean cost per request +38% (limit +25%)"])
        self.assertAlmostEqual(c["cost"]["delta_pct"], 38.0)
        self.assertAlmostEqual(c["cost"]["baseline_mean_usd"], 0.010)
        self.assertAlmostEqual(c["cost"]["candidate_mean_usd"], 0.0138)

    def test_increase_within_limit_passes(self):
        c = self.costs(0.010, 0.011)
        self.assertEqual(c["verdict"], "pass")
        self.assertAlmostEqual(c["cost"]["delta_pct"], 10.0)

    def test_cost_decrease_passes(self):
        self.assertEqual(self.costs(0.010, 0.005)["verdict"], "pass")

    def test_no_limit_means_no_rule_and_no_cost_warning(self):
        c = self.costs(0.010, 0.100, gate={})
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(warning_with(c, "cost limit"), [])

    def test_mean_ignores_cases_without_cost(self):
        c = self.costs(lambda i: 0.010 if i % 2 else None, lambda i: 0.012 if i % 2 else None, gate={})
        self.assertAlmostEqual(c["cost"]["baseline_mean_usd"], 0.010)
        self.assertAlmostEqual(c["cost"]["candidate_mean_usd"], 0.012)

    def test_missing_baseline_cost_gives_no_percentage_and_a_warning(self):
        c = self.costs(None, 0.010)
        self.assertIsNone(c["cost"]["baseline_mean_usd"])
        self.assertIsNone(c["cost"]["delta_pct"])
        self.assertEqual(c["verdict"], "pass")
        self.assertIn("baseline run has no cost data", c["warnings"][0])

    def test_missing_candidate_cost_warns(self):
        c = self.costs(0.010, None)
        self.assertIsNone(c["cost"]["delta_pct"])
        self.assertIn("candidate run has no cost data", c["warnings"][0])

    def test_zero_baseline_cost_avoids_division_by_zero(self):
        c = self.costs(0.0, 0.010)
        self.assertIsNone(c["cost"]["delta_pct"])
        self.assertIn("baseline cost is zero", c["warnings"][0])

    def test_no_costs_at_all_and_no_limit_is_fine(self):
        c = self.costs(None, None, gate={})
        self.assertEqual(c["cost"], {"baseline_mean_usd": None, "candidate_mean_usd": None, "delta_pct": None})
        self.assertEqual(c["verdict"], "pass")


class LatencyRule(unittest.TestCase):
    GATE = {"max_p95_latency_increase_pct": 30}

    def latencies(self, base, cand, gate=None):
        return compare(
            flags(20),
            flags(20),
            gate=self.GATE if gate is None else gate,
            base_kw={"latency": base},
            cand_kw={"latency": cand},
        )

    def test_increase_beyond_limit_triggers(self):
        c = self.latencies(1.0, 1.5)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"], ["p95 latency +50% (limit +30%)"])
        self.assertAlmostEqual(c["latency_p95"]["delta_pct"], 50.0)

    def test_p95_not_mean_drives_the_rule(self):
        # 18 fast cases and 2 slow ones: the mean rises only 20%, but p95 lands on the slow tail.
        c = self.latencies(1.0, lambda i: 3.0 if i >= 18 else 1.0)
        self.assertAlmostEqual(c["latency_p95"]["candidate_s"], 3.0)
        self.assertAlmostEqual(c["latency_p95"]["delta_pct"], 200.0)
        self.assertEqual(c["verdict"], "regression")

    def test_increase_within_limit_passes(self):
        self.assertEqual(self.latencies(1.0, 1.2)["verdict"], "pass")

    def test_no_limit_no_rule(self):
        self.assertEqual(self.latencies(1.0, 9.0, gate={})["verdict"], "pass")

    def test_missing_latency_warns_and_does_not_crash(self):
        c = self.latencies(None, 1.0)
        self.assertIsNone(c["latency_p95"]["baseline_s"])
        self.assertIsNone(c["latency_p95"]["delta_pct"])
        self.assertEqual(c["verdict"], "pass")
        self.assertIn("baseline run has no latency data", c["warnings"][0])


class ErrorRateRule(unittest.TestCase):
    def run_with_errors(self, k, source="target", gate=None):
        """100 cases; the first k error in the candidate.  Target errors hit cases that already failed in the
        baseline (a crash on a passing case is a regression, see TargetErrorsAreRegressions), judge errors hit
        cases that passed."""
        base = flags(100, fail=range(k) if source == "target" else ())
        cand = ["judge_error" if (source == "judge" and i < k) else ("error" if i < k else True) for i in range(100)]
        return compare(base, cand, gate=gate)

    def test_error_rate_above_limit_triggers_with_not_a_quality_signal_wording(self):
        c = self.run_with_errors(6)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(
            c["reasons"],
            ["run unhealthy: 0 judge errors, 6 target errors (6.0% of cases, limit 5.0%) — not a quality signal"],
        )
        self.assertAlmostEqual(c["error_rate"]["candidate"], 0.06)
        self.assertEqual(c["error_rate"]["baseline"], 0.0)

    def test_judge_errors_are_counted_separately(self):
        c = self.run_with_errors(7, source="judge")
        self.assertIn("run unhealthy: 7 judge errors, 0 target errors", c["reasons"][0])
        self.assertEqual(
            {k: c["errors"]["candidate"][k] for k in ("n_errors", "n_judge_errors", "n_target_errors", "error_rate")},
            {"n_errors": 7, "n_judge_errors": 7, "n_target_errors": 0, "error_rate": 0.07},
        )

    def test_error_rate_at_limit_passes(self):
        self.assertEqual(self.run_with_errors(5)["verdict"], "pass")

    def test_limit_can_be_raised_or_disabled(self):
        for gate in ({"max_error_rate": 0.10}, {"max_error_rate": None}):
            with self.subTest(gate=gate):
                self.assertEqual(self.run_with_errors(6, gate=gate)["verdict"], "pass")

    def test_judge_and_baseline_failing_cases_never_reach_the_pass_rate(self):
        # 6 target errors on cases that already failed: the paired pass rate is over the 94 other cases only
        c = self.run_with_errors(6)
        self.assertEqual(c["n_paired"], 94)
        self.assertEqual(c["pass_rate"]["candidate"], 1.0)
        self.assertEqual(c["pass_rate"]["delta"], 0.0)
        self.assertEqual(c["newly_failing"], [])
        self.assertEqual(c["target_error_regressions"], [])
        self.assertEqual(len(c["excluded_errors"]), 6)
        self.assertEqual(len(warning_with(c, "already failed in the baseline crashed")), 1)
        # 6 judge errors on cases that passed: the same
        c = self.run_with_errors(6, source="judge")
        self.assertEqual(c["n_paired"], 94)
        self.assertEqual(c["newly_failing"], [])
        self.assertEqual(len(warning_with(c, "the judge errored in the candidate")), 1)

    def test_each_kind_of_error_has_its_own_warning(self):
        base = ["error"] * 2 + [False] * 2 + [True] * 96
        cand = [True] * 2 + ["error"] * 2 + ["judge_error"] * 2 + [True] * 94
        c = compare(base, cand, gate={"max_error_rate": None})
        [w_base] = warning_with(c, "errored in the baseline")
        [w_crash] = warning_with(c, "already failed in the baseline crashed in the candidate")
        [w_judge] = warning_with(c, "the judge errored in the candidate")
        self.assertTrue(w_base.startswith("2 cases left out of the comparison because they errored in the baseline (c000, c001)"))
        self.assertIn("a baseline error says nothing about the candidate", w_base)
        self.assertTrue(w_crash.startswith("2 cases that already failed in the baseline crashed in the candidate (target error; c002, c003)"))
        self.assertIn("nothing left to regress", w_crash)
        self.assertIn("they count towards the error rate", w_crash)
        self.assertTrue(w_judge.startswith("2 cases left out of the comparison because the judge errored in the candidate run (c004, c005)"))
        self.assertIn("infrastructure failures, not quality failures", w_judge)
        self.assertEqual(c["excluded_errors"], ["c000", "c001", "c002", "c003", "c004", "c005"])
        self.assertEqual(c["n_paired"], 94)

    def test_excluded_errors_warnings_read_naturally_for_one_case(self):
        [w] = warning_with(compare(flags(100), ["judge_error"] + [True] * 99), "the judge errored in the candidate")
        self.assertTrue(w.startswith("1 case left out of the comparison because the judge errored"))
        self.assertIn("so it is not counted as a failure", w)
        [w] = warning_with(compare(flags(100, fail=[0]), flags(100, error=[0])), "already failed in the baseline crashed")
        self.assertTrue(w.startswith("1 case that already failed in the baseline crashed in the candidate"))
        self.assertIn("but it counts towards the error rate", w)
        [w] = warning_with(compare(flags(100, error=[0]), flags(100)), "errored in the baseline")
        self.assertTrue(w.startswith("1 case left out of the comparison because it errored in the baseline"))

    def test_errors_block_has_both_runs(self):
        c = self.run_with_errors(6)
        self.assertEqual(
            c["errors"]["baseline"],
            {
                "n_errors": 0, "n_judge_errors": 0, "n_target_errors": 0, "error_rate": 0.0,
                "n_attempts": 100, "n_attempt_errors": 0, "n_attempt_target_errors": 0, "n_attempt_judge_errors": 0, "attempt_error_rate": 0.0,
            },
        )  # fmt: skip
        self.assertEqual(c["errors"]["candidate"]["n_target_errors"], 6)
        self.assertEqual((c["errors"]["candidate"]["n_attempts"], c["errors"]["candidate"]["n_attempt_target_errors"]), (100, 6))

    def test_error_rate_uses_whole_candidate_run(self):
        base = make_run("base", build(flags(10)))
        cand = make_run("cand", build(flags(10)) + build(flags(10, error=range(10)), prefix="new"))
        c = compare_runs(base, cand)
        self.assertAlmostEqual(c["error_rate"]["candidate"], 0.5)
        self.assertEqual(c["verdict"], "regression")

    def test_unhealthy_baseline_warns(self):
        c = compare(flags(100, error=range(10)), flags(100))
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "baseline run is unhealthy")
        self.assertIn("10.0% errors", w)
        self.assertIn("refresh the baseline", w)


class TargetErrorsAreRegressions(unittest.TestCase):
    """Loop-back 1: a crash the change introduces on a case that passed is a failure, not "no signal".

    Judge errors (infrastructure) and baseline errors stay out of the pass rates; the error-rate ceiling and the
    pass-rate floor are also evaluated on the cases present in the baseline, so added cases cannot dilute them.
    """

    # critic-methodology W8/W10: 60 cases, c058 and c059 already fail in the baseline
    BASE_60 = flags(60, fail=[58, 59])

    def w8(self, crash, wrong, gate=None, n=60):
        """The PR breaks `crash + wrong` previously passing cases: the first `crash` of them by raising."""
        broken = [40 + i for i in range(crash + wrong)]
        cand = flags(n, fail=[*broken[crash:], 58, 59], error=broken[:crash])
        return compare(self.BASE_60, cand, gate=gate)

    # ---- the delta, McNemar, the verdict

    def test_crashes_count_like_wrong_answers_in_the_delta_and_mcnemar(self):
        hidden, honest = self.w8(2, 2), self.w8(0, 4)  # 4 regressions: 2 as target crashes vs 4 wrong answers
        self.assertEqual(hidden["verdict"], "regression")
        self.assertEqual(honest["verdict"], "regression")
        for key in ("baseline", "candidate", "delta", "b_pass_to_fail", "c_fail_to_pass", "p_value", "p_value_two_sided"):
            self.assertEqual(hidden["pass_rate"][key], honest["pass_rate"][key], key)
        self.assertEqual(hidden["n_paired"], honest["n_paired"])
        self.assertAlmostEqual(hidden["pass_rate"]["delta"], -4 / 60)
        self.assertEqual(hidden["newly_failing"], honest["newly_failing"])
        self.assertEqual(hidden["target_error_regressions"], ["c040", "c041"])
        self.assertEqual(honest["target_error_regressions"], [])
        self.assertEqual(hidden["excluded_errors"], [])

    def test_the_drop_reason_says_which_failures_were_crashes(self):
        hidden = self.w8(2, 2)
        self.assertEqual(
            hidden["reasons"][0],
            "pass rate dropped 6.7 p.p. (auto rule for 60 paired cases: net loss 4 cases vs 3 needed, one-sided McNemar p=0.062 vs α=0.217); "
            "of these, 2 cases crashed in the candidate (target error) and count as failures: c040, c041",
        )
        self.assertEqual(
            self.w8(0, 4)["reasons"],
            ["pass rate dropped 6.7 p.p. (auto rule for 60 paired cases: net loss 4 cases vs 3 needed, one-sided McNemar p=0.062 vs α=0.217)"],
        )

    def test_crashes_alone_within_the_limit_pass_but_are_never_silent(self):
        # critic-engineering scenario 11: 2 of 45 crash (-4.4 p.p., error rate 4.4%): the same verdict as 2 wrong answers
        base = flags(45, fail=range(6))
        crash = compare(base, flags(45, fail=range(6), error=[6, 7]))
        wrong = compare(base, flags(45, fail=[*range(6), 6, 7]))
        self.assertEqual(crash["verdict"], wrong["verdict"])
        self.assertEqual(crash["verdict"], "pass")
        self.assertEqual(crash["pass_rate"]["delta"], wrong["pass_rate"]["delta"])
        self.assertAlmostEqual(crash["pass_rate"]["delta"], -2 / 45)
        self.assertEqual(crash["newly_failing"], ["c006", "c007"])
        [w] = warning_with(crash, "crashed in the candidate")
        self.assertIn("2 cases that passed in the baseline crashed in the candidate", w)
        self.assertIn("the drop (4.4 p.p.) is within the limit, but check them", w)
        md = to_markdown(crash)
        self.assertIn("**Newly failing (2):** `c006`, `c007` — 2 of them crashed in the candidate", md)
        self.assertEqual(warning_with(wrong, "crashed"), [])

    def test_three_crashes_of_45_block_like_three_wrong_answers(self):
        base = flags(45, fail=range(6))
        crash = compare(base, flags(45, fail=range(6), error=[6, 7, 8]))
        self.assertEqual(crash["verdict"], "regression")
        self.assertIn("pass rate dropped 6.7 p.p.", crash["reasons"][0])

    def test_the_auto_rule_applies_its_significance_test_to_crashes_too(self):
        base = flags(200, fail=range(180, 200))
        crash = compare(base, flags(200, fail=range(180, 200), error=range(7)))  # -3.5 p.p., p = 1/128
        wrong = compare(base, flags(200, fail=[*range(7), *range(180, 200)]))
        self.assertEqual(crash["gate_mode"], "auto")
        self.assertEqual(crash["verdict"], "regression")
        self.assertEqual(crash["pass_rate"]["p_value"], wrong["pass_rate"]["p_value"])
        self.assertEqual(warning_with(crash, "significant"), [])

    def test_a_crash_and_a_recovery_offset_like_any_other_flip(self):
        base = flags(100, fail=[50])
        c = compare(base, flags(100, error=[0]), gate={"max_error_rate": None})  # c000 crashes, c050 recovers
        self.assertEqual((c["pass_rate"]["b_pass_to_fail"], c["pass_rate"]["c_fail_to_pass"]), (1, 1))
        self.assertEqual(c["pass_rate"]["delta"], 0.0)
        self.assertEqual(c["newly_passing"], ["c050"])

    def test_an_outright_target_outage_is_a_full_drop_and_an_unhealthy_run(self):
        c = compare(flags(100, fail=range(10)), ["error"] * 100)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["pass_rate"]["b_pass_to_fail"], 90)
        self.assertEqual(c["pass_rate"]["candidate"], 0.0)
        self.assertEqual(len(c["target_error_regressions"]), 90)
        self.assertTrue(any(r.startswith("run unhealthy: 0 judge errors, 100 target errors") for r in c["reasons"]))

    # ---- what stays out

    def test_judge_errors_on_passing_cases_are_not_failures(self):
        c = compare(flags(100), ["judge_error"] * 6 + [True] * 94, gate={"max_error_rate": None})
        self.assertEqual(c["newly_failing"], [])
        self.assertEqual(c["pass_rate"]["delta"], 0.0)
        self.assertEqual(c["n_paired"], 94)
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["target_error_regressions"], [])
        self.assertEqual(len(c["excluded_errors"]), 6)

    def test_baseline_errors_stay_out_even_when_the_candidate_also_errors(self):
        base = ["error"] * 3 + [True] * 97
        c = compare(base, ["error", "judge_error", True] + [True] * 97, gate={"max_error_rate": None})
        self.assertEqual(c["n_paired"], 97)
        self.assertEqual(c["target_error_regressions"], [])
        self.assertEqual(c["excluded_errors"], ["c000", "c001", "c002"])

    def test_a_crash_on_a_case_that_failed_in_the_baseline_is_not_a_regression(self):
        c = compare(flags(100, fail=[0]), flags(100, error=[0]), gate={"max_error_rate": None})
        self.assertEqual(c["target_error_regressions"], [])
        self.assertEqual(c["newly_failing"], [])
        self.assertEqual(c["excluded_errors"], ["c000"])
        self.assertEqual(c["n_paired"], 99)

    def test_a_repeated_case_crashes_only_when_at_least_half_of_its_attempts_error(self):
        # C2: errors >= half of the attempts -> the case is an error; one crash among three attempts is not
        base_kw = {"attempts": lambda i: ["pass"] * 3 if i < 2 else None}
        outcomes = {0: ["pass", "error", "pass"], 1: ["error", "error", "pass"]}
        records = build(flags(10), attempts=lambda i: outcomes.get(i))
        records[0]["outcome"], records[0]["error_source"] = "pass", None
        records[1]["outcome"], records[1]["error_source"] = "error", "target"
        records[1]["output"]["error"] = "timeout"
        c = compare_runs(make_run("b", build(flags(10), **base_kw)), make_run("c", records), gate={"max_error_rate": None})
        self.assertEqual(c["target_error_regressions"], ["c001"])
        self.assertEqual(c["newly_failing"], ["c001"])

    def test_crashed_cases_do_not_feed_cost_latency_or_score(self):
        cost = lambda i: 0.5 if i == 0 else 0.01  # noqa: E731  (the crashed case has a bogus candidate cost)
        base_kw = {"cost": 0.01, "latency": 1.0, "scores": 0.8}
        cand_kw = {"cost": cost, "latency": lambda i: 60.0 if i == 0 else 1.0, "scores": lambda i: None if i == 0 else 0.8}
        c = compare(flags(100), flags(100, error=[0]), base_kw=base_kw, cand_kw=cand_kw, gate={"max_error_rate": None})
        self.assertEqual(c["n_paired"], 100)
        self.assertAlmostEqual(c["cost"]["candidate_mean_usd"], 0.01)
        self.assertAlmostEqual(c["latency_p95"]["candidate_s"], 1.0)
        self.assertEqual(c["score"]["n"], 99)

    # ---- the error-rate ceiling cannot be diluted

    def test_added_cases_cannot_dilute_the_error_rate(self):
        # critic-methodology W10: 4 crashes on 60 cases + 100 easy cases appended -> 2.5% of the run, 6.7% of the baseline's cases
        base = make_run("base", build(self.BASE_60))
        cand = make_run("cand", build(flags(60, fail=[58, 59], error=[40, 41, 42, 43])) + build(flags(100), prefix="easy"))
        c = compare_runs(base, cand, gate={"max_pass_rate_drop": None})
        self.assertAlmostEqual(c["error_rate"]["candidate"], 4 / 160)
        self.assertAlmostEqual(c["errors"]["candidate"]["paired_error_rate"], 4 / 60)
        self.assertEqual((c["errors"]["candidate"]["paired_n_errors"], c["errors"]["candidate"]["paired_n_cases"]), (4, 60))
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(
            c["reasons"],
            [
                "run unhealthy on the cases that also exist in the baseline: 0 judge errors, 4 target errors "
                "(6.7% of 60 cases, limit 5.0%; 2.5% over the whole run, where added cases dilute it) — not a quality signal"
            ],
        )

    def test_w10_is_also_a_pass_rate_drop_with_the_default_gate(self):
        base = make_run("base", build(self.BASE_60))
        cand = make_run("cand", build(flags(60, fail=[58, 59], error=[40, 41, 42, 43])) + build(flags(100), prefix="easy"))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 60)
        self.assertAlmostEqual(c["pass_rate"]["delta"], -4 / 60)
        self.assertEqual(len(c["reasons"]), 2)
        self.assertTrue(c["reasons"][0].startswith("pass rate dropped 6.7 p.p."))
        self.assertTrue(c["reasons"][1].startswith("run unhealthy on the cases that also exist in the baseline"))

    def test_the_whole_run_reason_wins_when_both_rates_are_over_the_limit(self):
        c = self.run_errors(8)
        self.assertEqual(len([r for r in c["reasons"] if "run unhealthy" in r]), 1)
        self.assertTrue([r for r in c["reasons"] if "run unhealthy" in r][0].startswith("run unhealthy: 0 judge errors, 8 target errors"))

    def run_errors(self, k):
        return compare(flags(100), flags(100, error=range(k)))

    def test_errors_on_added_cases_count_only_in_the_whole_run(self):
        base = make_run("base", build(flags(100)))
        cand = make_run("cand", build(flags(100)) + build(flags(20, error=range(20)), prefix="new"))
        c = compare_runs(base, cand)
        self.assertAlmostEqual(c["error_rate"]["candidate"], 20 / 120)
        self.assertEqual(c["errors"]["candidate"]["paired_error_rate"], 0.0)
        self.assertEqual(c["errors"]["candidate"]["paired_n_cases"], 100)
        self.assertEqual(c["verdict"], "regression")  # still unhealthy as a whole

    def test_the_paired_error_rate_counts_judge_errors_and_edited_cases_too(self):
        base = make_run("base", build(flags(50)))
        cand = make_run(
            "cand",
            build(["judge_error"] * 2 + ["error"] * 2 + [True] * 46, sha=lambda i: "x" if i in (2, 3) else f"c{i:03d}-v1")
            + build(flags(100), prefix="easy"),
        )
        c = compare_runs(base, cand, gate={"max_changed_cases": None, "max_pass_rate_drop": None})
        self.assertEqual(c["errors"]["candidate"]["paired_n_errors"], 4)
        self.assertEqual(c["errors"]["candidate"]["paired_n_cases"], 50)
        self.assertAlmostEqual(c["errors"]["candidate"]["paired_error_rate"], 0.08)
        self.assertIn("2 judge errors, 2 target errors (8.0% of 50 cases", c["reasons"][0])

    def test_the_paired_error_rate_is_zero_without_cases_in_common(self):
        c = compare_runs(make_run("b", build(flags(5), prefix="a")), make_run("c", build(flags(5), prefix="z")))
        self.assertEqual(c["errors"]["candidate"]["paired_error_rate"], 0.0)
        self.assertEqual(c["errors"]["candidate"]["paired_n_cases"], 0)
        self.assertIsNone(c["pass_rate"]["candidate_paired_pessimistic"])

    def test_the_error_rate_limit_can_still_be_switched_off(self):
        base = make_run("base", build(self.BASE_60))
        cand = make_run("cand", build(flags(60, fail=[58, 59], error=[40, 41, 42, 43])) + build(flags(100), prefix="easy"))
        c = compare_runs(base, cand, gate={"max_error_rate": None, "max_pass_rate_drop": None})
        self.assertEqual(c["verdict"], "pass")

    # ---- unchanged when nothing crashed

    def test_without_target_errors_the_new_numbers_equal_the_old_ones(self):
        c = compare(BASE_100, drop_by(4))
        self.assertEqual(c["pass_rate"]["candidate_paired_pessimistic"], c["pass_rate"]["candidate"])
        self.assertEqual(c["pass_rate"]["candidate_paired_pessimistic"], c["pass_rate"]["candidate_overall"])
        self.assertEqual(c["errors"]["candidate"]["paired_error_rate"], c["error_rate"]["candidate"])
        self.assertEqual(c["target_error_regressions"], [])

    def test_random_runs_match_a_from_scratch_reading_of_the_rules(self):
        import random

        rng = random.Random(20260929)
        kinds = ["pass", "pass", "pass", "fail", "target", "judge"]
        for trial in range(300):
            n = rng.randint(1, 60)
            base_kinds = [rng.choice(["pass", "pass", "pass", "fail", "target", "judge"]) for _ in range(n)]
            cand_kinds = [rng.choice(kinds) for _ in range(n)]
            to_flag = {"pass": True, "fail": False, "target": "error", "judge": "judge_error"}
            base = make_run("b", build([to_flag[k] for k in base_kinds]))
            cand = make_run("c", build([to_flag[k] for k in cand_kinds]))
            c = compare_runs(base, cand, gate={"max_error_rate": None})
            b = cc = k_base = k_cand = paired = 0
            crashed = []
            for i, (kb, kc) in enumerate(zip(base_kinds, cand_kinds)):
                if kb in ("target", "judge"):
                    continue
                if kc == "judge" or (kc == "target" and kb == "fail"):
                    continue
                if kc == "target":
                    kc = "fail"
                    crashed.append(f"c{i:03d}")
                paired += 1
                k_base += kb == "pass"
                k_cand += kc == "pass"
                b += kb == "pass" and kc == "fail"
                cc += kb == "fail" and kc == "pass"
            with self.subTest(trial=trial):
                self.assertEqual(c["n_paired"], paired)
                self.assertEqual((c["pass_rate"]["b_pass_to_fail"], c["pass_rate"]["c_fail_to_pass"]), (b, cc))
                self.assertEqual(c["target_error_regressions"], crashed)
                if paired:
                    self.assertAlmostEqual(c["pass_rate"]["delta"], (k_cand - k_base) / paired)
                errs = sum(k in ("target", "judge") for k in cand_kinds)
                self.assertEqual(c["errors"]["candidate"]["paired_n_errors"], errs)
                pess = [k for k in cand_kinds if k != "judge"]
                if pess:
                    self.assertAlmostEqual(c["pass_rate"]["candidate_paired_pessimistic"], pess.count("pass") / len(pess))


class MultipleRules(unittest.TestCase):
    def test_every_triggered_rule_is_listed(self):
        c = compare(
            BASE_100,
            drop_by(6),
            gate={"min_pass_rate": 0.9, "max_cost_increase_pct": 10},
            base_kw={"cost": 0.01},
            cand_kw={"cost": 0.02},
        )
        self.assertEqual(len(c["reasons"]), 3)
        self.assertTrue(c["reasons"][0].startswith("pass rate dropped"))
        self.assertTrue(c["reasons"][1].startswith("candidate pass rate"))
        self.assertTrue(c["reasons"][2].startswith("mean cost per request"))


# ---------------------------------------------------------------- test-set integrity


class GateWeakening(unittest.TestCase):
    """Cycle-1 critic scenarios (24_gate_weakening.sh): a PR must not pass by shrinking or rewriting the test set."""

    # 45 cases; "main" fails 4, the PR reverts the prompt and fails 17 (13 of them newly).
    BASE = flags(45, fail=[41, 42, 43, 44])
    PR_FLAGS = flags(45, fail=[*range(13), 41, 42, 43, 44])

    def test_honest_pr_with_the_full_test_set_is_a_regression(self):
        c = compare(self.BASE, self.PR_FLAGS)
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("pass rate dropped 28.9 p.p.", c["reasons"][0])

    def test_deleting_the_17_failing_cases_is_a_regression_with_an_explicit_reason(self):
        failing = {i for i, ok in enumerate(self.PR_FLAGS) if not ok}
        keep = [i for i in range(45) if i not in failing]
        base = make_run("main", build(self.BASE))
        pr = make_run("pr-trim", [r for r in build(self.PR_FLAGS) if int(r["case_id"][1:]) in keep])
        c = compare_runs(base, pr)
        self.assertEqual(c["n_paired"], 28)
        self.assertEqual(c["pass_rate"]["delta"], 0.0)  # the trimmed set looks perfect...
        self.assertEqual(len(c["removed_cases"]), 17)
        self.assertEqual(c["unpaired"]["only_baseline"], c["removed_cases"])
        self.assertEqual(c["verdict"], "regression")  # ...and still fails
        [reason] = c["reasons"]
        self.assertIn("17 baseline cases missing from the candidate", reason)
        self.assertIn("update the baseline in a reviewed PR", reason)

    def test_the_removal_limit_is_configurable_and_can_be_switched_off(self):
        base = make_run("main", build(flags(20)))
        pr = make_run("pr", build(flags(17)))
        self.assertEqual(compare_runs(base, pr)["verdict"], "regression")
        self.assertEqual(compare_runs(base, pr, gate={"max_removed_cases": 3})["verdict"], "pass")
        self.assertEqual(compare_runs(base, pr, gate={"max_removed_cases": 2})["verdict"], "regression")
        self.assertEqual(compare_runs(base, pr, gate={"max_removed_cases": None})["verdict"], "pass")

    def test_removed_case_reason_quotes_ids_and_truncates(self):
        base = make_run("main", build(flags(20)))
        c = compare_runs(base, make_run("pr", build(flags(10))))
        self.assertIn("(c010, c011, c012, c013, c014 and 5 more)", c["reasons"][0])
        self.assertIn("10 baseline cases missing", c["reasons"][0])
        one = compare_runs(base, make_run("pr", build(flags(19))))
        self.assertIn("1 baseline case missing", one["reasons"][0])

    def test_editing_a_failing_case_is_a_regression_with_an_explicit_reason(self):
        # the PR rewrites the input of the cases it fails so they now pass: same ids, new content
        edited = {0, 1, 2}
        base = make_run("main", build(self.BASE[:45]))
        cand_records = build(flags(45), sha=lambda i: f"c{i:03d}-EDITED" if i in edited else f"c{i:03d}-v1")
        c = compare_runs(base, make_run("pr", cand_records))
        self.assertEqual(c["changed_cases"], ["c000", "c001", "c002"])
        self.assertEqual(c["n_paired"], 42)  # edited cases are not paired, so they cannot flatter the delta
        self.assertEqual(c["verdict"], "regression")
        [reason] = c["reasons"]
        self.assertIn("test cases were edited", reason)
        self.assertIn("3 cases differ from the baseline (c000, c001, c002)", reason)
        self.assertIn("update the baseline in a reviewed PR", reason)

    def test_one_edited_case_is_enough_by_default_and_the_limit_is_configurable(self):
        base = make_run("main", build(flags(30)))
        cand = make_run("pr", build(flags(30), sha=lambda i: "changed" if i == 5 else f"c{i:03d}-v1"))
        self.assertEqual(compare_runs(base, cand)["verdict"], "regression")
        self.assertEqual(compare_runs(base, cand, gate={"max_changed_cases": 1})["verdict"], "pass")
        self.assertEqual(compare_runs(base, cand, gate={"max_changed_cases": None})["verdict"], "pass")

    def test_deleting_and_editing_together_report_both_reasons(self):
        base = make_run("main", build(flags(30)))
        cand = make_run("pr", build(flags(25), sha=lambda i: "changed" if i < 2 else f"c{i:03d}-v1"))
        c = compare_runs(base, cand)
        self.assertEqual(len(c["reasons"]), 2)
        self.assertEqual((len(c["removed_cases"]), len(c["changed_cases"])), (5, 2))

    def test_adding_cases_is_not_a_regression_and_not_paired(self):
        base = make_run("main", build(flags(30)))
        cand = make_run("pr", build(flags(30)) + build(flags(5, fail=range(5)), prefix="new"))
        c = compare_runs(base, cand)
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["unpaired"]["only_candidate"], [f"new{i:03d}" for i in range(5)])
        self.assertEqual(c["n_paired"], 30)

    def test_all_error_baseline_cannot_be_used_to_pass_a_bad_candidate(self):
        # critic scenario (c): the baseline was recorded while every target call failed
        base = make_run("broken", build(["error"] * 45))
        c = compare_runs(base, make_run("pr_v1", build(flags(45, fail=range(17)))))
        self.assertEqual(c["n_paired"], 0)
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("no comparable cases: 45 cases in common, but 0 were edited and 45 errored", c["reasons"][0])
        self.assertIn("the baseline run is unhealthy (100.0% errors)", " ".join(c["warnings"]))

    def test_judge_outage_in_the_candidate_is_an_unhealthy_run_not_a_quality_drop(self):
        base = make_run("main", build(flags(45)))
        c = compare_runs(base, make_run("pr", build(["judge_error"] * 45)))
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("run unhealthy: 45 judge errors, 0 target errors", " ".join(c["reasons"]))
        self.assertIn("not a quality signal", " ".join(c["reasons"]))
        self.assertEqual(c["pass_rate"]["delta"], None)  # nothing comparable: no fake -100 p.p.


class Pairing(unittest.TestCase):
    def setUp(self):
        # Baseline: c000..c049 (all pass) + 5 baseline-only cases. Candidate: c000..c049 with c000 failing + 3 new cases.
        self.base = make_run("base", build(flags(50)) + build(flags(5), prefix="old"))
        self.cand = make_run("cand", build(flags(50, fail=[0])) + build(flags(3, fail=[0, 1, 2]), prefix="new"))
        self.c = compare_runs(self.base, self.cand, gate={"max_error_rate": None, "max_removed_cases": None})

    def test_unpaired_ids_are_reported_and_excluded_from_the_delta(self):
        c = self.c
        self.assertEqual(c["n_paired"], 50)
        self.assertEqual(c["unpaired"]["only_baseline"], [f"old{i:03d}" for i in range(5)])
        self.assertEqual(c["unpaired"]["only_candidate"], ["new000", "new001", "new002"])
        self.assertEqual(c["removed_cases"], c["unpaired"]["only_baseline"])
        self.assertAlmostEqual(c["pass_rate"]["baseline"], 1.0)
        self.assertAlmostEqual(c["pass_rate"]["candidate"], 0.98)
        self.assertEqual(c["newly_failing"], ["c000"])  # new failing cases are not "newly failing"

    def test_markdown_states_that_only_paired_cases_are_compared(self):
        md = to_markdown(self.c)
        self.assertIn("Only cases present in both runs are compared", md)
        self.assertIn("those 50 paired cases", md)
        self.assertIn("Baseline only (5)", md)
        self.assertIn("`old000`", md)
        self.assertIn("Candidate only (3)", md)
        self.assertIn("`new002`", md)

    def test_no_unpaired_note_when_runs_match(self):
        self.assertNotIn("Only cases present in both runs", to_markdown(compare(BASE_100, BASE_100)))

    def test_newly_failing_follows_candidate_order(self):
        base = make_run("b", build(flags(5)))
        shuffled = build(flags(5, fail=[1, 3]))[::-1]
        c = compare_runs(base, make_run("c", shuffled), gate={"max_pass_rate_drop": None})
        self.assertEqual(c["newly_failing"], ["c003", "c001"])

    def test_disjoint_runs_fail_closed_without_crashing(self):
        base = make_run("b", build(flags(4), prefix="a"))
        cand = make_run("c", build(flags(6), prefix="z"))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 0)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["reasons"][0], "no cases in common between baseline (4 cases) and candidate (6 cases); nothing to compare")
        self.assertIsNone(c["pass_rate"]["delta"])
        self.assertIsNone(c["pass_rate"]["p_value"])
        self.assertIsNone(c["pass_rate"]["p_value_two_sided"])
        self.assertIsNone(c["pass_rate"]["baseline_ci"])
        self.assertIsNone(c["score"])
        self.assertEqual(c["by_tag"], {})
        md = to_markdown(c)
        self.assertIn("❌ Regression detected", md)
        self.assertIn("nothing to compare", md)
        self.assertNotIn("small test set", md)

    def test_two_empty_runs_do_not_crash(self):
        c = compare_runs(make_run("b", []), make_run("c", []))
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(c["error_rate"], {"baseline": 0.0, "candidate": 0.0})
        to_markdown(c)

    def test_a_case_that_errored_in_the_baseline_or_the_judge_is_left_out_of_the_pairing(self):
        base = make_run("b", build(flags(100, error=[0], fail=[1])))
        cand = make_run("c", build(flags(100, error=[1])))  # c001 failed in the baseline and crashes now: nothing to regress
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 98)
        self.assertEqual(c["excluded_errors"], ["c000", "c001"])
        judge = compare_runs(make_run("b", build(flags(100))), make_run("c", build(["judge_error"] + [True] * 99)))
        self.assertEqual(judge["n_paired"], 99)
        self.assertEqual(judge["excluded_errors"], ["c000"])

    def test_a_target_error_on_a_case_that_passed_in_the_baseline_is_paired_as_a_failure(self):
        base = make_run("b", build(flags(100)))
        cand = make_run("c", build(flags(100, error=[1])))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 100)
        self.assertEqual(c["excluded_errors"], [])
        self.assertEqual(c["newly_failing"], ["c001"])
        self.assertEqual(c["target_error_regressions"], ["c001"])

    def test_edited_and_errored_cases_are_left_out_of_by_tag_too(self):
        tags = lambda i: ["a"] if i < 5 else ["b"]  # noqa: E731
        base = make_run("b", build(flags(10), tags=tags))
        cand = make_run(
            "c",
            build(
                ["judge_error" if i == 1 else (i not in (0, 5)) for i in range(10)],
                tags=tags,
                sha=lambda i: "x" if i == 2 else f"c{i:03d}-v1",
            ),
        )
        c = compare_runs(base, cand, gate={"max_error_rate": None, "max_changed_cases": None})
        self.assertEqual(c["by_tag"]["a"]["n"], 3)  # c000, c003, c004 (c001's judge errored, c002 edited)
        self.assertEqual(c["by_tag"]["b"]["n"], 5)

    def test_a_crash_on_a_passing_case_counts_in_the_by_tag_table(self):
        tags = lambda i: ["a"] if i < 5 else ["b"]  # noqa: E731
        base = make_run("b", build(flags(10), tags=tags))
        cand = make_run("c", build(flags(10, error=[1]), tags=tags))
        c = compare_runs(base, cand, gate={"max_error_rate": None})
        self.assertEqual((c["by_tag"]["a"]["n"], c["by_tag"]["a"]["baseline"], c["by_tag"]["a"]["candidate"]), (5, 1.0, 0.8))
        self.assertAlmostEqual(c["by_tag"]["a"]["delta"], -0.2)
        self.assertEqual(c["by_tag"]["b"]["delta"], 0.0)

    def test_unpaired_lists_are_json_friendly(self):
        json.dumps(self.c)


class LegacyRecords(unittest.TestCase):
    """Records written before C1/C2 (no outcome / case_sha): derive the outcome, pair by id, warn."""

    def test_old_records_are_paired_by_id_with_a_warning(self):
        base = make_run("old", build(BASE_100, legacy=True))
        cand = make_run("old2", build(drop_by(6), legacy=True))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 100)
        self.assertEqual(c["verdict"], "regression")
        [w] = warning_with(c, "without case_sha")
        self.assertIn("baseline and candidate run has records without case_sha", w)
        self.assertIn("paired by id only", w)

    def test_only_the_side_that_lacks_case_sha_is_named(self):
        c = compare_runs(make_run("old", build(flags(100), legacy=True)), make_run("new", build(flags(100))))
        self.assertIn("the baseline run has records without case_sha", warning_with(c, "without case_sha")[0])
        c = compare_runs(make_run("new", build(flags(100))), make_run("old", build(flags(100), legacy=True)))
        self.assertIn("the candidate run has records without case_sha", warning_with(c, "without case_sha")[0])

    def test_new_records_never_trigger_the_legacy_warning(self):
        self.assertEqual(warning_with(compare(BASE_100, BASE_100), "case_sha"), [])

    def test_outcome_is_derived_from_passed_and_the_target_error(self):
        base = make_run("b", build(flags(100), legacy=True))
        cand = make_run("c", build(flags(100, error=range(6)), legacy=True))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 100)  # the crashes hit cases that passed: paired, as failures
        self.assertEqual(c["target_error_regressions"], [f"c00{i}" for i in range(6)])
        self.assertEqual(c["errors"]["candidate"]["n_target_errors"], 6)
        self.assertTrue(any("run unhealthy" in r for r in c["reasons"]))

    def test_a_legacy_target_error_on_an_already_failing_case_is_left_out(self):
        base = make_run("b", build(flags(100, fail=range(6)), legacy=True))
        cand = make_run("c", build(flags(100, error=range(6)), legacy=True))
        c = compare_runs(base, cand)
        self.assertEqual(c["n_paired"], 94)
        self.assertEqual(c["target_error_regressions"], [])
        self.assertEqual(c["errors"]["candidate"]["n_target_errors"], 6)
        self.assertIn("run unhealthy", c["reasons"][0])

    def test_a_failing_legacy_case_with_a_judge_error_is_an_error_not_a_failure(self):
        base = make_run("b", build(flags(100), legacy=True))
        cand = make_run("c", build(["judge_error" if i < 3 else True for i in range(100)], legacy=True))
        c = compare_runs(base, cand)
        self.assertEqual(c["errors"]["candidate"]["n_judge_errors"], 3)
        self.assertEqual(c["n_paired"], 97)
        self.assertEqual(c["newly_failing"], [])

    def test_any_judge_error_makes_a_legacy_case_an_error_even_when_another_judge_passed(self):
        # same rule as types.derive_outcome / C1: an applicable judge that errored leaves no reliable verdict
        recs = build(flags(10), legacy=True, scores=1.0)
        recs[0]["judges"][0]["error"] = "one of two judges failed"
        recs[0]["judges"].append({"judge": "j1", "passed": True, "score": 1.0})
        c = compare_runs(make_run("b", build(flags(10), legacy=True)), make_run("c", recs), gate={"max_error_rate": None})
        self.assertEqual(c["errors"]["candidate"]["n_judge_errors"], 1)
        self.assertEqual(c["n_paired"], 9)

    def test_explicit_outcome_wins_over_passed(self):
        recs = build(flags(10))
        recs[0]["outcome"] = "error"
        recs[0]["error_source"] = "judge"
        c = compare_runs(make_run("b", build(flags(10))), make_run("c", recs), gate={"max_error_rate": None})
        self.assertEqual(c["errors"]["candidate"]["n_judge_errors"], 1)
        self.assertEqual(c["n_paired"], 9)

    def test_missing_error_source_is_derived(self):
        recs = build(flags(10, error=[0]))
        for r in recs:
            r.pop("error_source")
        c = compare_runs(make_run("b", build(flags(10))), make_run("c", recs), gate={"max_error_rate": None})
        self.assertEqual(c["errors"]["candidate"]["n_target_errors"], 1)


# ------------------------------------------------------------------ small-set warning


class SmallSetWarning(unittest.TestCase):
    def test_states_the_detectable_drop_below_100_cases(self):
        c = compare(flags(45), flags(45))
        [w] = warning_with(c, "small test set")
        self.assertIn("(n = 45)", w)
        self.assertIn("net loss of 3 cases (6.7 p.p.)", w)
        self.assertRegex(w, r"only drops of ≥ (1\d|9) p\.p\. are reliably detected")
        self.assertIn("≥ 80% of runs at ~5% noise", w)

    def test_detectable_drop_shrinks_as_the_test_set_grows(self):
        import re

        def detectable(n):
            w = warning_with(compare(flags(n), flags(n)), "small test set")[0]
            return int(re.search(r"drops of ≥ (\d+) p\.p\.", w).group(1))

        values = [detectable(n) for n in (20, 45, 70, 99)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[-1], values[0])

    def test_appears_below_100_paired_cases_only(self):
        self.assertEqual(SMALL_SET, 100)
        self.assertTrue(warning_with(compare(flags(99), flags(99)), "small test set"))
        self.assertFalse(warning_with(compare(flags(100), flags(100)), "small test set"))

    def test_counts_paired_cases_not_the_size_of_the_run(self):
        c = compare(flags(105), ["judge_error" if i < 6 else True for i in range(105)], gate={"max_error_rate": None})
        self.assertEqual(c["n_paired"], 99)
        self.assertTrue(warning_with(c, "small test set"))

    def test_singular_case_wording(self):
        # the auto rule never trips below 3 cases; a fixed 5 p.p. limit on 10 cases trips at 1
        w = warning_with(compare(flags(10), flags(10), gate={"max_pass_rate_drop": 0.05}), "small test set")[0]
        self.assertIn("net loss of 1 case (10.0 p.p.)", w)
        self.assertIn("net loss of 3 cases (30.0 p.p.)", warning_with(compare(flags(10), flags(10)), "small test set")[0])

    def test_no_warning_without_paired_cases(self):
        c = compare_runs(make_run("b", []), make_run("c", []))
        self.assertEqual(warning_with(c, "small test set"), [])

    def test_a_limit_that_can_never_fire_is_stated(self):
        w = warning_with(compare(flags(20), flags(20), gate={"max_pass_rate_drop": 1.0}), "small test set")[0]
        self.assertIn("with a drop limit of 100.0 p.p. no pass-rate drop can trip the gate", w)

    def test_detectable_drop_matches_an_independent_power_calculation(self):
        # exact power of the auto rule at n = 45 (net loss >= 3 and one-sided p < alpha(45)) when both runs are noisy (5%)
        # readings of the same latent verdicts, written from scratch (no evalsprint code)
        import math

        def power(n, drop_pp, noise=0.05):
            d = drop_pp / 100
            ub = noise * (1 - noise) * (1 - d) + d * (1 - noise) ** 2
            uc = noise * (1 - noise) * (1 - d) + d * noise**2
            alpha = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)
            total = 0.0
            for b in range(n + 1):
                for c in range(n - b + 1):
                    if b - c >= max(3, math.ceil(n / 50)) and sum(math.comb(b + c, i) for i in range(b, b + c + 1)) / 2 ** (b + c) < alpha:
                        rest = n - b - c
                        total += math.factorial(n) // (math.factorial(b) * math.factorial(c) * math.factorial(rest)) * ub**b * uc**c * (1 - ub - uc) ** rest
            return total

        self.assertLess(power(45, 13), 0.8)
        self.assertGreaterEqual(power(45, 14), 0.8)
        self.assertIn("only drops of ≥ 14 p.p.", warning_with(compare(flags(45), flags(45)), "small test set")[0])

    def test_off_rule_is_stated(self):
        w = warning_with(compare(flags(20), flags(20), gate={"max_pass_rate_drop": None}), "small test set")[0]
        self.assertIn("pass-rate rule is off", w)


# ----------------------------------------------------------------- calibration link


class CalibrationLink(unittest.TestCase):
    """The gate reads the estimand calibrate() decides on: the WEIGHTED kappa of the whole run and its kappa_decision."""

    # what calibration.calibrate() writes (the unweighted pair describes the stratified label sample and is deliberately different)
    GOOD = {
        "judge": "quality", "kappa": 0.55, "kappa_ci": [0.30, 0.75],
        "kappa_weighted": 0.85, "kappa_weighted_ci": [0.72, 0.95], "kappa_decision": "usable", "kappa_decision_basis": "weighted",
    }  # fmt: skip
    WEAK = {
        "judge": "quality", "kappa": 0.95, "kappa_ci": [0.90, 0.99],
        "kappa_weighted": 0.70, "kappa_weighted_ci": [0.45, 0.90], "kappa_decision": "inconclusive", "kappa_decision_basis": "weighted",
    }  # fmt: skip
    LEGACY = {"judge": "quality", "kappa": 0.85, "kappa_ci": [0.72, 0.95], "kappa_decision": "usable"}

    def weighted(self, kappa=0.85, ci=(0.72, 0.95), decision="usable", judge="quality", **extra):
        return {
            "judge": judge, "kappa": 0.1, "kappa_ci": [0.0, 0.2], "kappa_weighted": kappa,
            "kappa_weighted_ci": list(ci) if ci else None, "kappa_decision": decision, **extra,
        }  # fmt: skip

    def run_llm(self, calibration=None, gate=None, llm=True):
        return compare(BASE_100, BASE_100, gate=gate, calibration=calibration, base_kw={"llm": llm}, cand_kw={"llm": llm})

    def test_uncalibrated_llm_judge_warns(self):
        c = self.run_llm()
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(len(warning_with(c, "LLM judge not calibrated")), 1)
        self.assertIsNone(c["calibration_status"])

    def test_no_warning_for_heuristic_judges(self):
        self.assertEqual(warning_with(self.run_llm(llm=False), "calibrated"), [])

    def test_no_warning_when_the_run_has_no_judges_at_all(self):
        self.assertEqual(warning_with(compare(BASE_100, BASE_100), "calibrated"), [])

    def test_llm_judge_is_recognised_by_token_usage_even_without_details(self):
        recs = build(flags(10), llm=True)
        for r in recs:
            r["judges"][0]["details"] = {}
        c = compare_runs(make_run("b", build(flags(10))), make_run("c", recs))
        self.assertEqual(len(warning_with(c, "LLM judge not calibrated")), 1)

    def test_good_calibration_is_ok_and_silent(self):
        c = self.run_llm(self.GOOD)
        self.assertEqual(
            c["calibration_status"],
            {"judge": "quality", "kappa": 0.85, "kappa_ci": [0.72, 0.95], "kappa_decision": "usable", "basis": "weighted", "ok": True},
        )
        self.assertEqual(warning_with(c, "calibrat"), [])
        self.assertEqual(c["verdict"], "pass")

    def test_the_weighted_kappa_is_used_not_the_unweighted_one(self):
        # the unweighted sample kappa looks great (0.95, CI 0.90-0.99) but the run-level weighted one is what the decision uses
        c = self.run_llm(self.WEAK)
        self.assertEqual(c["calibration_status"]["kappa"], 0.70)
        self.assertEqual(c["calibration_status"]["kappa_ci"], [0.45, 0.90])
        self.assertEqual(c["calibration_status"]["basis"], "weighted")
        self.assertFalse(c["calibration_status"]["ok"])
        # and the other way round: a poor unweighted kappa does not sink a judge whose weighted kappa is fine
        self.assertTrue(self.run_llm(self.GOOD)["calibration_status"]["ok"])

    def test_low_kappa_lower_bound_warns_by_default(self):
        c = self.run_llm(self.WEAK)  # point estimate 0.70 is fine, the LOWER bound 0.45 is not
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "judge calibration is weak")
        self.assertIn("weighted kappa lower bound 0.45 is below the required 0.60 (weighted kappa 0.70)", w)
        self.assertIn("calibration decision is 'inconclusive'", w)

    def test_the_lower_bound_decides_not_the_point_estimate(self):
        self.assertTrue(self.run_llm(self.weighted(0.65, (0.60, 0.70)))["calibration_status"]["ok"])  # lower == threshold passes
        self.assertFalse(self.run_llm(self.weighted(0.95, (0.59, 1.0)))["calibration_status"]["ok"])

    def test_threshold_is_configurable_but_the_decision_still_counts(self):
        c = self.run_llm(self.weighted(0.70, (0.45, 0.90), decision="usable"), gate={"min_judge_kappa": 0.4})
        self.assertTrue(c["calibration_status"]["ok"])
        c = self.run_llm(self.weighted(0.70, (0.45, 0.90), decision="usable"), gate={"min_judge_kappa": None})
        self.assertTrue(c["calibration_status"]["ok"])
        self.assertEqual(warning_with(c, "calibrat"), [])
        # an 'inconclusive' decision is not a threshold: lowering min_judge_kappa does not release the judge
        c = self.run_llm(self.WEAK, gate={"min_judge_kappa": 0.4})
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertIn("calibration decision is 'inconclusive'", warning_with(c, "judge calibration")[0])
        self.assertNotIn("below the required", warning_with(c, "judge calibration")[0])
        c = self.run_llm(self.WEAK, gate={"min_judge_kappa": None})
        self.assertFalse(c["calibration_status"]["ok"])

    def test_decision_must_be_usable_or_trustworthy(self):
        for decision, ok in (("usable", True), ("trustworthy", True), ("inconclusive", False), ("not_trustworthy", False), ("insufficient_data", False), ("mystery", False)):
            with self.subTest(decision=decision):
                c = self.run_llm(self.weighted(0.9, (0.8, 0.97), decision=decision))  # a fine interval, so only the decision differs
                self.assertIs(c["calibration_status"]["ok"], ok)
                self.assertEqual(c["calibration_status"]["kappa_decision"], decision)

    def test_a_decision_that_holds_the_judge_back_despite_a_good_interval_is_named(self):
        # e.g. fewer labels than the release minimum: the interval clears 0.6 but the decision stays inconclusive
        c = self.run_llm(self.weighted(0.9, (0.8, 0.97), decision="inconclusive"))
        [w] = warning_with(c, "judge calibration is weak")
        self.assertEqual(
            w,
            "judge calibration is weak: calibration decision is 'inconclusive' (the weighted kappa interval straddles the usable "
            "threshold, or there are too few labels to release the judge)",
        )
        self.assertIn("'mystery' (only usable or trustworthy release a judge)", warning_with(self.run_llm(self.weighted(decision="mystery")), "weak")[0])
        [nt] = warning_with(self.run_llm(self.weighted(0.2, (0.1, 0.4), decision="not_trustworthy")), "weak")
        self.assertIn("weighted kappa lower bound 0.10 is below the required 0.60", nt)
        self.assertIn("calibration decision is 'not_trustworthy' (the weighted kappa interval lies below the usable threshold)", nt)

    def test_require_calibration_fails_on_the_decision_even_with_a_high_lower_bound(self):
        c = self.run_llm(self.weighted(0.9, (0.8, 0.97), decision="inconclusive"), gate={"require_calibration": True})
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("calibration decision is 'inconclusive'", c["reasons"][0])

    def test_require_calibration_fails_on_the_lower_bound_even_when_the_decision_is_usable(self):
        c = self.run_llm(self.weighted(0.85, (0.72, 0.95), decision="usable"), gate={"require_calibration": True, "min_judge_kappa": 0.8})
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("weighted kappa lower bound 0.72 is below the required 0.80", c["reasons"][0])

    def test_require_calibration_turns_the_missing_calibration_into_a_regression(self):
        c = self.run_llm(gate={"require_calibration": True})
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(len(c["reasons"]), 1)
        self.assertIn("LLM judge not calibrated", c["reasons"][0])
        self.assertEqual(warning_with(c, "LLM judge not calibrated"), [])

    def test_require_calibration_turns_a_weak_calibration_into_a_regression(self):
        c = self.run_llm(self.WEAK, gate={"require_calibration": True})
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("judge calibration is weak", c["reasons"][0])

    def test_require_calibration_with_good_calibration_passes(self):
        self.assertEqual(self.run_llm(self.GOOD, gate={"require_calibration": True})["verdict"], "pass")

    def test_require_calibration_does_not_bite_runs_without_llm_judges(self):
        self.assertEqual(self.run_llm(gate={"require_calibration": True}, llm=False)["verdict"], "pass")

    def test_insufficient_labels_are_not_ok(self):
        cal = self.weighted(None, None, decision="insufficient_data")
        c = self.run_llm(cal)
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertEqual(c["calibration_status"]["basis"], "weighted")
        self.assertEqual(len(warning_with(c, "calibration")), 1)
        self.assertIn("too few human labels", warning_with(c, "calibration")[0])

    def test_calibration_without_ci_cannot_meet_a_threshold(self):
        c = self.run_llm(self.weighted(0.9, None, decision="usable"))
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertIn("no kappa confidence interval", warning_with(c, "calibration")[0])
        self.assertIsNone(c["calibration_status"]["kappa_ci"])

    def test_a_weighted_dict_never_falls_back_to_the_unweighted_ci(self):
        # the weighted keys exist (their values are null: too few labels); the unweighted pair must not stand in for them
        cal = {"judge": "q", "kappa": 0.9, "kappa_ci": [0.8, 1.0], "kappa_weighted": None, "kappa_weighted_ci": None, "kappa_decision": "usable"}
        c = self.run_llm(cal)
        self.assertEqual(c["calibration_status"]["basis"], "weighted")
        self.assertIsNone(c["calibration_status"]["kappa"])
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertEqual(warning_with(c, "legacy calibration"), [])

    # ---- calibration files written before the weighted estimate existed

    def test_a_legacy_calibration_falls_back_to_the_unweighted_pair_and_says_so(self):
        c = self.run_llm(self.LEGACY)
        self.assertEqual(c["calibration_status"]["basis"], "legacy")
        self.assertEqual((c["calibration_status"]["kappa"], c["calibration_status"]["kappa_ci"]), (0.85, [0.72, 0.95]))
        self.assertTrue(c["calibration_status"]["ok"])
        self.assertEqual(c["verdict"], "pass")
        [w] = warning_with(c, "legacy calibration: unweighted kappa used")
        self.assertIn("re-run `evalsprint calibrate", w)

    def test_a_legacy_calibration_is_judged_on_the_unweighted_lower_bound(self):
        c = self.run_llm({"judge": "q", "kappa": 0.7, "kappa_ci": [0.45, 0.9]})
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertIn("unweighted kappa lower bound 0.45 is below the required 0.60 (unweighted kappa 0.70)", warning_with(c, "judge calibration is weak")[0])
        self.assertEqual(len(warning_with(c, "legacy calibration")), 1)

    def test_a_legacy_calibration_without_a_decision_is_not_held_back_by_it(self):
        c = self.run_llm({"judge": "q", "kappa": 0.65, "kappa_ci": [0.60, 0.70]})
        self.assertIsNone(c["calibration_status"]["kappa_decision"])
        self.assertTrue(c["calibration_status"]["ok"])

    def test_a_legacy_decision_is_still_honoured(self):
        c = self.run_llm({"judge": "q", "kappa": 0.9, "kappa_ci": [0.7, 1.0], "kappa_decision": "insufficient_data"})
        self.assertFalse(c["calibration_status"]["ok"])
        self.assertIn("too few human labels", warning_with(c, "judge calibration")[0])

    def test_the_legacy_note_is_a_warning_even_when_calibration_is_required(self):
        c = self.run_llm(self.LEGACY, gate={"require_calibration": True})
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(len(warning_with(c, "legacy calibration")), 1)

    def test_the_legacy_note_appears_without_llm_judges_too(self):
        self.assertEqual(len(warning_with(self.run_llm(self.LEGACY, llm=False), "legacy calibration")), 1)

    def test_reads_what_the_real_calibrate_writes(self):
        from evalsprint import calibration
        from tests import test_calibration as tc

        for counts, decision, ok in (((30, 2, 3, 25), "usable", True), ((4, 0, 0, 4), "insufficient_data", False)):
            with self.subTest(decision=decision):
                run, labels = tc.confusion_scenario(*counts)
                cal = calibration.calibrate(run, labels, **tc.FAST)
                c = self.run_llm(cal)
                status = c["calibration_status"]
                self.assertEqual(cal["kappa_decision"], decision)
                self.assertEqual((status["basis"], status["kappa_decision"], status["ok"]), ("weighted", decision, ok))
                self.assertEqual(status["kappa"], cal["kappa_weighted"])
                self.assertEqual(status["kappa_ci"], None if cal["kappa_weighted_ci"] is None else list(cal["kappa_weighted_ci"]))
                self.assertEqual(warning_with(c, "legacy calibration"), [])

    def test_a_calibration_status_survives_json(self):
        c = self.run_llm(self.GOOD)
        self.assertEqual(json.loads(json.dumps(c))["calibration_status"], c["calibration_status"])

    def test_markdown_shows_the_calibration_line(self):
        self.assertIn(
            "Judge calibration `quality`: κ = 0.85 (95% CI 0.72–0.95), decision `usable` — meets the gate's requirement.",
            to_markdown(self.run_llm(self.GOOD)),
        )
        weak = to_markdown(self.run_llm(self.WEAK))
        self.assertIn("decision `inconclusive` — does not meet the gate's requirement", weak)
        self.assertIn("Judge calibration: κ", to_markdown(self.run_llm(self.weighted(0.9, (0.8, 1.0), judge="overall"))))
        self.assertNotIn("Judge calibration", to_markdown(self.run_llm()))
        legacy = to_markdown(self.run_llm(self.LEGACY))
        self.assertIn("κ = 0.85 (95% CI 0.72–0.95), decision `usable` (unweighted, legacy file) — meets", legacy)
        self.assertIn("legacy calibration: unweighted kappa used", legacy)


# ------------------------------------------------------------------ judge configuration


class JudgeConfigurationRule(unittest.TestCase):
    """summary.judges_sha: a PR must not change the judges (config, rubric, pass_rule) and get its verdicts in one go."""

    REASON = "judge configuration or rubric changed: update the baseline in a reviewed PR"

    def test_same_judges_sha_is_fine_and_silent(self):
        c = compare(BASE_100, BASE_100)
        self.assertIs(c["judges_changed"], False)
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(warning_with(c, "judges_sha"), [])

    def test_a_different_judges_sha_is_a_regression_with_an_explicit_reason(self):
        c = compare(BASE_100, BASE_100, base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb")
        self.assertIs(c["judges_changed"], True)
        self.assertEqual(c["verdict"], "regression")
        [reason] = c["reasons"]
        self.assertTrue(reason.startswith(self.REASON))
        self.assertIn("judges_sha aaaaaaaaaaaa → bbbbbbbbbbbb", reason)

    def test_the_judge_change_is_a_regression_even_when_every_number_improved(self):
        # a PR that loosens the rubric and lifts the pass rate must not sail through
        c = compare(BASE_100, flags(100), base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb")
        self.assertEqual(c["newly_passing"], [f"c{i:03d}" for i in range(90, 100)])
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(len(c["reasons"]), 1)

    def test_allow_judge_changes_turns_it_into_a_warning(self):
        c = compare(BASE_100, BASE_100, base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb", gate={"allow_judge_changes": True})
        self.assertIs(c["judges_changed"], True)
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual(c["reasons"], [])
        [w] = warning_with(c, "judge configuration or rubric changed")
        self.assertIn("allowed by gate.allow_judge_changes", w)
        self.assertIn("judges_sha aaaaaaaaaaaa → bbbbbbbbbbbb", w)

    def test_default_is_not_to_allow_changes(self):
        self.assertIs(DEFAULT_GATE["allow_judge_changes"], False)
        self.assertIs(compare(BASE_100, BASE_100)["gate"]["allow_judge_changes"], False)

    def test_a_missing_sha_is_a_warning_not_a_regression_and_judges_changed_is_null(self):
        for base_sha, cand_sha, who in (
            (None, JUDGES_SHA, "the baseline summary carries no judges_sha"),
            (JUDGES_SHA, None, "the candidate summary carries no judges_sha"),
            (None, None, "the baseline and candidate summaries carry no judges_sha"),
        ):
            with self.subTest(who=who):
                c = compare(BASE_100, BASE_100, base_sha=base_sha, cand_sha=cand_sha)
                self.assertIsNone(c["judges_changed"])
                self.assertEqual(c["verdict"], "pass")
                [w] = warning_with(c, "judges_sha")
                self.assertIn(who, w)
                self.assertIn("older evalsprint", w)

    def test_an_empty_sha_counts_as_missing(self):
        c = compare(BASE_100, BASE_100, base_sha="", cand_sha=JUDGES_SHA)
        self.assertIsNone(c["judges_changed"])

    def test_a_missing_sha_does_not_hide_a_real_regression(self):
        c = compare(BASE_100, drop_by(20), base_sha=None, cand_sha=None)
        self.assertEqual(c["verdict"], "regression")
        self.assertIn("pass rate dropped", c["reasons"][0])

    def test_judge_change_and_pass_rate_drop_report_both_reasons(self):
        c = compare(BASE_100, drop_by(20), base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb")
        self.assertEqual(len(c["reasons"]), 2)

    def test_the_explicit_gate_decides_not_the_candidate_summary(self):
        # the PR sets allow_judge_changes in its own config; the trusted gate does not
        base = make_run("base", build(BASE_100), judges_sha="aaaaaaaaaaaa")
        cand = make_run("cand", build(BASE_100), gate={"allow_judge_changes": True}, judges_sha="bbbbbbbbbbbb")
        self.assertEqual(compare_runs(base, cand, gate={})["verdict"], "regression")
        self.assertEqual(compare_runs(base, cand)["verdict"], "pass")  # gate=None reads the candidate's own gate

    def test_allow_judge_changes_must_be_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "allow_judge_changes must be true or false"):
            resolve_gate({}, {"allow_judge_changes": "yes"})
        self.assertIs(resolve_gate({}, {"allow_judge_changes": True})["allow_judge_changes"], True)

    def test_markdown_shows_the_reason_the_gate_line_and_the_warning(self):
        md = to_markdown(compare(BASE_100, BASE_100, base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb"))
        self.assertIn(f"- {self.REASON}", md)
        self.assertIn("judges unchanged", md)
        md = to_markdown(compare(BASE_100, BASE_100, base_sha="aaaaaaaaaaaa", cand_sha="bbbbbbbbbbbb", gate={"allow_judge_changes": True}))
        self.assertIn("- ⚠️ judge configuration or rubric changed", md)
        self.assertNotIn("judges unchanged", md)
        self.assertIn("no judges_sha", to_markdown(compare(BASE_100, BASE_100, base_sha=None)))

    def test_judges_changed_survives_json(self):
        for sha in ("bbbbbbbbbbbb", None):
            c = compare(BASE_100, BASE_100, base_sha="aaaaaaaaaaaa", cand_sha=sha)
            self.assertEqual(json.loads(json.dumps(c))["judges_changed"], c["judges_changed"])


class Pluralisation(unittest.TestCase):
    """Every count interpolated into a reason or warning agrees with its noun and verb: 1 case differs, 2 cases differ."""

    def edited(self, k):
        base = make_run("main", build(flags(30)))
        cand = make_run("pr", build(flags(30), sha=lambda i: "x" if i < k else f"c{i:03d}-v1"))
        return compare_runs(base, cand)

    def test_edited_cases(self):
        self.assertIn("test cases were edited: 1 case differs from the baseline (c000)", self.edited(1)["reasons"][0])
        self.assertIn("test cases were edited: 2 cases differ from the baseline (c000, c001)", self.edited(2)["reasons"][0])

    def test_removed_cases(self):
        base = make_run("main", build(flags(30)))
        self.assertTrue(compare_runs(base, make_run("pr", build(flags(29))))["reasons"][0].startswith("1 baseline case missing from the candidate"))
        self.assertTrue(compare_runs(base, make_run("pr", build(flags(28))))["reasons"][0].startswith("2 baseline cases missing from the candidate"))

    def test_nothing_comparable(self):
        one = compare_runs(make_run("b", build(flags(1))), make_run("c", build(flags(1), sha=lambda i: "x")))
        self.assertIn("no comparable cases: 1 case in common, but 1 was edited and 0 errored in at least one run", one["reasons"][0])
        two = compare_runs(make_run("b", build(flags(2))), make_run("c", build(flags(2), sha=lambda i: f"x{i}")))
        self.assertIn("no comparable cases: 2 cases in common, but 2 were edited and 0 errored in at least one run", two["reasons"][0])
        none = compare_runs(make_run("b", build(flags(1), prefix="a")), make_run("c", build(flags(1), prefix="z")))
        self.assertEqual(none["reasons"][0], "no cases in common between baseline (1 case) and candidate (1 case); nothing to compare")

    def test_errored_cases_and_the_unhealthy_baseline(self):
        one = compare(flags(100, error=[0]), flags(100))
        self.assertTrue(warning_with(one, "left out")[0].startswith("1 case left out of the comparison because it errored in the baseline"))
        two = compare(flags(100, error=[0, 1]), flags(100))
        self.assertTrue(warning_with(two, "left out")[0].startswith("2 cases left out of the comparison because they errored in the baseline"))
        one_left = compare(["error"] * 12 + [True], [True] * 13)  # 1 case with a verdict in the baseline
        [w] = warning_with(one_left, "baseline run is unhealthy")
        self.assertIn("only the 1 case with a verdict in the baseline is compared", w)
        both = compare(["error"] * 5 + [True] * 2, [True] * 7)
        self.assertIn("only the 2 cases with a verdict in the baseline are compared", warning_with(both, "baseline run is unhealthy")[0])

    def test_crashed_cases(self):
        one = compare(flags(100), flags(100, error=[0]), gate={"max_error_rate": None})
        [w] = warning_with(one, "crashed in the candidate")
        self.assertTrue(w.startswith("1 case that passed in the baseline crashed in the candidate (target error) and counts as a failure"))
        self.assertIn("but check it", w)
        two = compare(flags(100), flags(100, error=[0, 1]), gate={"max_error_rate": None})
        [w] = warning_with(two, "crashed in the candidate")
        self.assertTrue(w.startswith("2 cases that passed in the baseline crashed in the candidate (target error) and count as failures"))
        self.assertIn("but check them", w)
        big = compare(flags(100), flags(100, error=range(10)), gate={"max_error_rate": None})
        self.assertIn("of these, 10 cases crashed in the candidate (target error) and count as failures", " ".join(big["reasons"]))
        single = compare(flags(30), flags(30, error=range(5)), gate={"max_error_rate": None})
        self.assertIn("of these, 5 cases crashed", " ".join(single["reasons"]))
        one_reason = compare(flags(10), flags(10, fail=[1, 2], error=[0]), gate={"max_error_rate": None})
        self.assertIn("of these, 1 case crashed in the candidate (target error) and counts as a failure: c000", " ".join(one_reason["reasons"]))

    def test_critical_cases_that_ran_too_few_times(self):
        gate = {"critical_tags": ["safety"]}
        tags = lambda i: ["safety"] if i < 1 else ["general"]  # noqa: E731
        one = compare(flags(100), flags(100), gate=gate, base_kw={"tags": tags}, cand_kw={"tags": tags})
        [w] = warning_with(one, "fewer than 5 times")
        self.assertTrue(w.startswith("critical case ran fewer than 5 times"))
        self.assertIn("(1 of 1 paired)", w)
        self.assertIn("false alarms with 1 critical case at ~5% noise", w)
        tags2 = lambda i: ["safety"] if i < 2 else ["general"]  # noqa: E731
        two = compare(flags(100), flags(100), gate=gate, base_kw={"tags": tags2}, cand_kw={"tags": tags2})
        [w] = warning_with(two, "fewer than 5 times")
        self.assertTrue(w.startswith("critical cases ran fewer than 5 times"))
        self.assertIn("false alarms with 2 critical cases", w)

    def test_no_reason_or_warning_ever_says_1_cases_or_1_were(self):
        import re

        bad = re.compile(r"(?<![\d.])1 (cases|were|differ|baseline cases|critical cases|judge errors|target errors)\b")
        comparisons = [
            self.edited(1),
            compare_runs(make_run("main", build(flags(30))), make_run("pr", build(flags(29)))),
            compare(flags(100), flags(100, error=[0])),
            compare(flags(10), flags(10, error=[0]), gate={"max_error_rate": None}),
            compare(flags(100, fail=[0]), flags(100, error=[0])),
            compare(["error"] * 12 + [True], [True] * 13),
            compare(flags(100), ["judge_error"] + [True] * 99, gate={"max_error_rate": 0.0}),
            compare(flags(100), ["error"] + [True] * 99, gate={"max_error_rate": 0.0}),
            compare(flags(1), flags(1)),
        ]
        for c in comparisons:
            for text in [*c["reasons"], *c["warnings"], to_markdown(c)]:
                with self.subTest(text=text[:80]):
                    self.assertIsNone(bad.search(text), text)


# ------------------------------------------------------------------ gate resolution


class GateResolution(unittest.TestCase):
    def test_defaults_when_nothing_is_configured(self):
        self.assertEqual(compare(BASE_100, BASE_100)["gate"], DEFAULT_GATE)

    def test_default_gate_matches_contract_c3(self):
        self.assertEqual(
            DEFAULT_GATE,
            {
                "max_pass_rate_drop": "auto",
                "alpha": 0.05,
                "require_significance": False,
                "min_pass_rate": None,
                "critical_tags": [],
                "max_cost_increase_pct": None,
                "max_p95_latency_increase_pct": None,
                "max_error_rate": 0.05,
                "max_removed_cases": 0,
                "max_changed_cases": 0,
                "min_judge_kappa": 0.6,
                "require_calibration": False,
                "allow_judge_changes": False,
                "allow_repeat_change": False,
                "allow_target_retries_change": False,  # loop-back 1 (C22): a changed target_retries is a protocol change, like the repeat block
                "retry_share_warn": 0.10,  # measured: 0.12% false warnings at N=100 and 5% transient crashes (retry_share_warn_measure.py)
                "require_same_kit_version": False,
                "tag_drop_warning_pp": 20,
                "tag_drop_min_cases": 20,  # cycle 4 (C17): with 10 tags of 10 cases the critic measured 43% spurious warnings
            },
        )
        self.assertEqual(set(GATE_KEYS), set(DEFAULT_GATE))

    def test_summary_gate_is_used_only_when_no_gate_is_passed(self):
        c = compare(BASE_100, BASE_100, summary_gate={"max_pass_rate_drop": 0.10, "min_pass_rate": 0.5})
        self.assertEqual(c["gate"]["max_pass_rate_drop"], 0.10)
        self.assertEqual(c["gate"]["min_pass_rate"], 0.5)
        self.assertEqual(c["gate"]["max_error_rate"], DEFAULT_GATE["max_error_rate"])  # untouched default

    def test_an_explicit_gate_is_authoritative_and_never_merged_with_the_summary_gate(self):
        # the PR's own run says "no rules"; the trusted gate only sets min_pass_rate: nothing leaks in from the summary
        loosened = {"max_pass_rate_drop": 1.0, "max_error_rate": 1.0, "max_removed_cases": None, "max_changed_cases": None}
        c = compare(BASE_100, drop_by(20), gate={"min_pass_rate": 0.7}, summary_gate=loosened)
        self.assertEqual(c["gate"], {**DEFAULT_GATE, "min_pass_rate": 0.7})
        self.assertEqual(c["verdict"], "regression")

    def test_an_empty_explicit_gate_means_defaults_not_the_summary_gate(self):
        c = compare(BASE_100, drop_by(20), gate={}, summary_gate={"max_pass_rate_drop": None})
        self.assertEqual(c["gate"], DEFAULT_GATE)
        self.assertEqual(c["verdict"], "regression")

    def test_explicit_none_disables_an_optional_rule(self):
        c = compare(BASE_100, BASE_100, gate={"min_pass_rate": None})
        self.assertIsNone(c["gate"]["min_pass_rate"])
        self.assertEqual(c["verdict"], "pass")

    def test_inputs_are_not_mutated(self):
        gate = {"critical_tags": "safety"}
        compare(BASE_100, BASE_100, gate=gate)
        self.assertEqual(gate, {"critical_tags": "safety"})
        self.assertEqual(DEFAULT_GATE["critical_tags"], [])

    def test_unknown_key_raises_so_typos_cannot_disable_rules(self):
        with self.assertRaisesRegex(ValueError, "unknown gate setting.*max_pass_rate_dop"):
            compare(BASE_100, BASE_100, gate={"max_pass_rate_dop": 0.01})
        with self.assertRaisesRegex(ValueError, "unknown gate setting"):
            compare(BASE_100, BASE_100, summary_gate={"min_pas_rate": 0.9})

    def test_auto_is_accepted_case_insensitively_and_normalised(self):
        self.assertEqual(resolve_gate({}, {"max_pass_rate_drop": " Auto "})["max_pass_rate_drop"], "auto")

    def test_other_strings_are_rejected(self):
        for bad in ("3%", "0.03", "automatic", ""):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "must be 'auto', a fraction"):
                resolve_gate({}, {"max_pass_rate_drop": bad})

    def test_percent_style_fractions_are_rejected(self):
        for key in ("max_pass_rate_drop", "min_pass_rate", "max_error_rate", "alpha"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fraction between 0 and 1"):
                resolve_gate({}, {key: 3})

    def test_non_numeric_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a number"):
            resolve_gate({}, {"max_cost_increase_pct": "25%"})
        with self.assertRaisesRegex(ValueError, "max_pass_rate_drop must be"):
            resolve_gate({}, {"max_pass_rate_drop": True})
        with self.assertRaisesRegex(ValueError, "alpha must be a number"):
            resolve_gate({}, {"alpha": None})
        with self.assertRaisesRegex(ValueError, "must be a number"):
            resolve_gate({}, {"min_pass_rate": float("nan")})

    def test_case_count_limits_must_be_non_negative_integers_or_null(self):
        for key in ("max_removed_cases", "max_changed_cases"):
            for bad in (-1, 1.5, "3", True):
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, "non-negative integer"):
                    resolve_gate({}, {key: bad})
            self.assertIsNone(resolve_gate({}, {key: None})[key])
            self.assertEqual(resolve_gate({}, {key: 4})[key], 4)

    def test_kappa_threshold_must_be_a_kappa(self):
        for bad in (1.5, -2, "0.6", True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "min_judge_kappa"):
                resolve_gate({}, {"min_judge_kappa": bad})
        self.assertIsNone(resolve_gate({}, {"min_judge_kappa": None})["min_judge_kappa"])

    def test_booleans_must_be_booleans(self):
        for key in ("require_significance", "require_calibration"):
            for bad in ("false", 1, None):
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, "true or false"):
                    resolve_gate({}, {key: bad})

    def test_gate_must_be_a_mapping(self):
        with self.assertRaisesRegex(ValueError, "mapping"):
            resolve_gate({}, ["max_error_rate"])


# ------------------------------------------------------------------- score and tags


class ScoreComparison(unittest.TestCase):
    def test_score_delta_and_paired_bootstrap_ci(self):
        base_scores = [0.5 + 0.02 * (i % 20) for i in range(40)]
        cand_scores = [s - 0.05 - 0.01 * (i % 3) for i, s in enumerate(base_scores)]
        c = compare(flags(40), flags(40), base_kw={"scores": base_scores}, cand_kw={"scores": cand_scores})
        s = c["score"]
        self.assertEqual(s["n"], 40)
        self.assertAlmostEqual(s["baseline_mean"], sum(base_scores) / 40)
        self.assertAlmostEqual(s["candidate_mean"], sum(cand_scores) / 40)
        self.assertAlmostEqual(s["delta"], s["candidate_mean"] - s["baseline_mean"])
        lo, hi = s["ci"]
        self.assertLessEqual(lo, s["delta"])
        self.assertGreaterEqual(hi, s["delta"])
        self.assertLess(hi, 0)  # the drop is consistent, so the interval excludes zero

    def test_case_score_is_the_mean_of_its_judge_scores_ignoring_none(self):
        base = make_run("b", build(flags(2), scores=[(0.5, 1.0), (0.4, None)]))
        cand = make_run("c", build(flags(2), scores=[(0.75,), (0.4,)]))
        s = compare_runs(base, cand)["score"]
        self.assertAlmostEqual(s["baseline_mean"], (0.75 + 0.4) / 2)
        self.assertAlmostEqual(s["candidate_mean"], (0.75 + 0.4) / 2)
        self.assertAlmostEqual(s["delta"], 0.0)

    def test_score_is_null_when_no_case_has_scores(self):
        self.assertIsNone(compare(BASE_100, BASE_100)["score"])

    def test_score_is_null_when_only_one_side_has_scores(self):
        c = compare(flags(10), flags(10), base_kw={"scores": 0.8})
        self.assertIsNone(c["score"])

    def test_cases_scored_on_one_side_only_are_excluded(self):
        c = compare(
            flags(10),
            flags(10),
            base_kw={"scores": 0.8},
            cand_kw={"scores": lambda i: 0.6 if i < 4 else None},
        )
        self.assertEqual(c["score"]["n"], 4)
        self.assertAlmostEqual(c["score"]["delta"], -0.2)

    def test_single_scored_pair_has_no_ci(self):
        c = compare(flags(5), flags(5), base_kw={"scores": lambda i: 0.8 if i == 0 else None}, cand_kw={"scores": lambda i: 0.6 if i == 0 else None})
        self.assertEqual(c["score"]["n"], 1)
        self.assertIsNone(c["score"]["ci"])
        self.assertAlmostEqual(c["score"]["delta"], -0.2)

    def test_score_does_not_trigger_the_gate(self):
        c = compare(flags(20), flags(20), base_kw={"scores": 0.9}, cand_kw={"scores": 0.1})
        self.assertEqual(c["verdict"], "pass")

    def test_comparison_is_deterministic(self):
        kw = {"base_kw": {"scores": [0.1 * (i % 9) for i in range(30)]}, "cand_kw": {"scores": [0.05 * (i % 7) for i in range(30)]}}
        self.assertEqual(compare(flags(30), flags(30), **kw), compare(flags(30), flags(30), **kw))


class ByTag(unittest.TestCase):
    def test_rates_per_tag_on_paired_cases(self):
        tags = lambda i: ["billing"] if i < 10 else (["shipping"] if i < 30 else [])  # noqa: E731
        c = compare(flags(40), flags(40, fail=[0, 1, 2, 3, 10]), base_kw={"tags": tags}, cand_kw={"tags": tags})
        self.assertEqual(
            c["by_tag"],
            {
                "(untagged)": {"n": 10, "baseline": 1.0, "candidate": 1.0, "delta": 0.0},
                "billing": {"n": 10, "baseline": 1.0, "candidate": 0.6, "delta": -0.4},
                "shipping": {"n": 20, "baseline": 1.0, "candidate": 0.95, "delta": -0.05},
            },
        )

    def test_multi_tag_case_counts_in_each_tag(self):
        tags = lambda i: ["a", "b"]  # noqa: E731
        c = compare(flags(4), flags(4, fail=[0]), base_kw={"tags": tags}, cand_kw={"tags": tags})
        self.assertEqual(c["by_tag"]["a"]["n"], 4)
        self.assertEqual(c["by_tag"]["b"]["n"], 4)


class Serialization(unittest.TestCase):
    def test_result_round_trips_through_json(self):
        c = compare(
            BASE_100,
            drop_by(6),
            gate={"critical_tags": ["safety"], "max_cost_increase_pct": 20},
            base_kw={"scores": 0.8, "cost": 0.01, "latency": 1.0, "tags": lambda i: ["safety"], "llm": True},
            cand_kw={"scores": 0.7, "cost": 0.02, "latency": 2.0, "tags": lambda i: ["safety"], "llm": True},
            calibration={"judge": "q", "kappa": 0.8, "kappa_ci": [0.7, 0.9]},
        )
        self.assertEqual(json.loads(json.dumps(c)), c)

    def test_contract_keys_are_present(self):
        c = compare(BASE_100, drop_by(6))
        contract = {
            "verdict", "reasons", "warnings", "baseline_run_id", "candidate_run_id", "n_paired", "unpaired", "pass_rate", "score",
            "by_tag", "newly_failing", "newly_passing", "critical_regressions", "cost", "latency_p95", "error_rate", "gate",
            "removed_cases", "changed_cases", "errors", "effective_max_pass_rate_drop", "gate_mode", "calibration_status",
            "judges_changed", "effective_alpha", "effective_min_net_loss", "tag_warnings", "flaky_crashes", "repeat_changed",
            "repeat_protocol", "kit_version",
        }  # fmt: skip
        self.assertLessEqual(contract, set(c))
        self.assertEqual((c["baseline_run_id"], c["candidate_run_id"]), ("base", "cand"))
        self.assertLessEqual(
            {"baseline", "candidate", "delta", "baseline_ci", "candidate_ci", "b_pass_to_fail", "c_fail_to_pass", "p_value", "p_value_two_sided"},
            set(c["pass_rate"]),
        )
        for side in ("baseline", "candidate"):
            extra = {"paired_error_rate", "paired_n_errors", "paired_n_cases"} if side == "candidate" else set()
            attempts = {"n_attempts", "n_attempt_errors", "n_attempt_target_errors", "n_attempt_judge_errors", "attempt_error_rate"}
            extra = extra | {"paired_attempt_error_rate", "paired_n_attempt_errors", "paired_n_attempts"} if side == "candidate" else extra
            self.assertEqual(set(c["errors"][side]), {"n_errors", "n_judge_errors", "n_target_errors", "error_rate"} | attempts | extra)

    def test_removed_and_changed_are_lists_of_ids(self):
        c = compare(BASE_100, BASE_100)
        self.assertEqual((c["removed_cases"], c["changed_cases"]), ([], []))
        self.assertIsNone(c["calibration_status"])


# ------------------------------------------------------------------------- markdown


class Markdown(unittest.TestCase):
    def setUp(self):
        self.tags = lambda i: ["safety"] if i < 10 else ["general"]  # noqa: E731
        kw = {"tags": self.tags}
        self.regression = compare(
            BASE_100,
            drop_by(6),
            gate={"critical_tags": ["general"], "max_cost_increase_pct": 25},
            base_kw={**kw, "scores": 0.80, "cost": 0.0100, "latency": 2.0},
            cand_kw={**kw, "scores": 0.70, "cost": 0.0138, "latency": 2.4},
        )
        self.md = to_markdown(self.regression)

    def test_regression_headline_and_reasons(self):
        self.assertTrue(self.md.startswith("## ❌ Regression detected\n"))
        for reason in self.regression["reasons"]:
            self.assertIn(f"- {reason}", self.md)
        self.assertIn("pass rate dropped 6.0 p.p.", self.md)
        self.assertIn("mean cost per request +38% (limit +25%)", self.md)

    def test_pass_headline_has_no_reason_list(self):
        md = to_markdown(compare(BASE_100, BASE_100))
        self.assertTrue(md.startswith("## ✅ No regression\n"))
        self.assertNotIn("Regression detected", md)
        self.assertNotIn("\n- ", md)
        self.assertNotIn("Intentional change?", md)

    def test_metrics_table(self):
        md = self.md
        self.assertIn("| Metric | Baseline | Candidate | Δ |", md)
        self.assertRegex(md, r"\| Pass rate \(95% CI\) \| 90\.0% \(\d+–\d+%\) \| 84\.0% \(\d+–\d+%\) \| -6\.0 p\.p\. \|")
        self.assertRegex(md, r"\| Mean judge score \| 0\.800 \| 0\.700 \| -0\.100 \(95% CI -0\.100 to -0\.100\) \|")
        self.assertIn("| Cost / request | $0.0100 | $0.0138 | +38% |", md)
        self.assertIn("| Cost / 1k requests | $10.00 | $13.80 | +38% |", md)
        self.assertIn("| p95 latency | 2.00 s | 2.40 s | +20% |", md)
        self.assertIn("| Error rate | 0.0% | 0.0% | 0.0 p.p. |", md)

    def test_rows_without_data_are_omitted(self):
        md = to_markdown(compare(BASE_100, BASE_100))
        self.assertNotIn("Cost / request", md)
        self.assertNotIn("p95 latency", md)
        self.assertNotIn("Mean judge score", md)
        self.assertIn("| Pass rate (95% CI) |", md)

    def test_by_tag_table_shows_all_tags_when_few_and_marks_critical(self):
        md = self.md
        self.assertIn("| Tag | n | Baseline | Candidate | Δ |", md)
        self.assertRegex(md, r"\| `general` 🔒 \| 90 \| 88\.9% \| ")
        self.assertRegex(md, r"\| `safety` \| 10 \| 100\.0% \| ")

    def test_by_tag_table_lists_only_changed_tags_when_many(self):
        tags = lambda i: [f"t{i % 13:02d}"]  # noqa: E731  13 tags
        kw = {"tags": tags}
        c = compare(flags(130), flags(130, fail=[0, 13]), base_kw=kw, cand_kw=kw, gate={"max_pass_rate_drop": None})
        md = to_markdown(c)
        self.assertIn("| `t00` |", md)
        self.assertNotIn("| `t05` |", md)
        c = compare(flags(130), flags(130), base_kw=kw, cand_kw=kw)
        self.assertIn("No per-tag pass rate changed (13 tags)", to_markdown(c))
        self.assertNotIn("| Tag |", to_markdown(c))

    def test_tag_names_cannot_break_the_table(self):
        tags = lambda i: ["a|b"]  # noqa: E731
        c = compare(flags(4), flags(4), base_kw={"tags": tags}, cand_kw={"tags": tags})
        self.assertIn("| `a`&#124;`b` | 4 |", to_markdown(c))  # C23: a code span per piece, the pipe between them is an entity

    def test_newly_failing_is_truncated_after_15_ids(self):
        c = compare(flags(60), flags(60, fail=range(20)), gate={"max_pass_rate_drop": None})
        md = to_markdown(c)
        self.assertIn("**Newly failing (20):**", md)
        self.assertIn("`c014`", md)
        self.assertNotIn("`c015`", md)
        self.assertIn(f"and {20 - MAX_LISTED_IDS} more", md)

    def test_newly_passing_is_listed_and_empty_sections_are_omitted(self):
        md = to_markdown(compare(BASE_100, flags(100)))
        self.assertIn("**Newly passing (10):**", md)
        self.assertNotIn("Newly failing", md)

    def test_significance_line_shows_both_p_values(self):
        self.assertIn("One-sided McNemar exact p = 0.016 (two-sided p = 0.031) on 6 discordant pairs (6 pass→fail, 0 fail→pass).", self.md)
        self.assertIn("1 discordant pair (", to_markdown(compare(flags(10), flags(10, fail=[0]))))
        self.assertIn("No discordant pairs", to_markdown(compare(BASE_100, BASE_100)))
        tiny = to_markdown(compare(flags(100), flags(100, fail=range(20))))
        self.assertIn("p < 0.001", tiny)

    def test_small_set_warning_is_shown_once_with_the_detectable_drop(self):
        md = to_markdown(compare(flags(45), flags(45)))
        self.assertEqual(md.count("small test set (n = 45)"), 1)
        self.assertIn("- ⚠️ small test set (n = 45): the gate trips at a net loss of 3 cases (6.7 p.p.)", md)
        self.assertIn("are reliably detected", md)
        self.assertNotIn("small test set", self.md)  # n = 100
        self.assertNotIn("within noise", md)

    def test_warnings_are_shown(self):
        md = to_markdown(compare(flags(100), flags(100), gate={"max_cost_increase_pct": 10}))
        self.assertIn("⚠️ cost limit not evaluated: the baseline run has no cost data", md)

    def test_gate_line_for_the_auto_rule(self):
        self.assertIn(
            "<sub>Gate: pass-rate drop: net loss ≥ 3 cases and significant (one-sided α=0.120) · auto, 100 paired cases · critical tags: `general`"
            " · cost ≤ +25% · error rate ≤ 5.0% · removed cases ≤ 0 · edited cases ≤ 0 · judges unchanged · repeat protocol unchanged · target retries unchanged</sub>",
            self.md,
        )
        large = to_markdown(compare(flags(200), flags(200)))
        self.assertIn("pass-rate drop: net loss ≥ 4 cases and significant (one-sided α=0.062) · auto, 200 paired cases", large)

    def test_gate_line_of_a_comparison_file_written_by_0_1_still_reads(self):
        for mode, limit, text in (
            ("auto-small", 0.05, "pass-rate drop ≤ 5.0 p.p. · auto (0.1 rule): < 150 cases"),
            ("auto-large", 0.03, "pass-rate drop ≤ 3.0 p.p. and significant (one-sided α=0.05) · auto (0.1 rule): ≥ 150 cases"),
        ):
            with self.subTest(mode=mode):
                old = compare(BASE_100, BASE_100)
                old.update(gate_mode=mode, effective_max_pass_rate_drop=limit)
                for key in ("effective_alpha", "effective_min_net_loss", "repeat_changed", "kit_version", "tag_warnings"):
                    old.pop(key)  # keys 0.1 did not write
                self.assertIn(text, to_markdown(old))
                self.assertEqual(normalize_gate_mode(mode), "auto")
        self.assertEqual([normalize_gate_mode(m) for m in ("auto", "fixed", "off", None)], ["auto", "fixed", "off", None])

    def test_gate_line_for_fixed_mode_and_options(self):
        self.assertIn("Update eval baseline", self.md)
        gate = {
            "max_pass_rate_drop": 0.05,
            "require_significance": True,
            "min_pass_rate": 0.8,
            "max_p95_latency_increase_pct": 12.5,
            "require_calibration": True,
            "allow_judge_changes": True,
        }
        line = to_markdown(compare(BASE_100, BASE_100, gate=gate))
        self.assertIn("pass-rate drop ≤ 5.0 p.p. (if significant, one-sided α=0.05)", line)
        self.assertIn("pass rate ≥ 80.0%", line)
        self.assertIn("p95 latency ≤ +12.5%", line)
        self.assertIn("calibrated judge required (κ lower bound ≥ 0.6)", line)
        self.assertNotIn("judges unchanged", line)  # allow_judge_changes: the rule is off
        off = to_markdown(compare(BASE_100, BASE_100, gate={"max_pass_rate_drop": None, "max_error_rate": None, "max_removed_cases": None, "max_changed_cases": None, "allow_judge_changes": True, "allow_repeat_change": True, "allow_target_retries_change": True}))
        self.assertIn("Gate: no rules enabled", off)
        self.assertIn("Gate: target retries unchanged", to_markdown(compare(BASE_100, BASE_100, gate={"max_pass_rate_drop": None, "max_error_rate": None, "max_removed_cases": None, "max_changed_cases": None, "allow_judge_changes": True, "allow_repeat_change": True})))

    def test_run_ids_and_size_are_in_the_header_line(self):
        self.assertIn("Baseline `base` → candidate `cand` · 100 paired cases", self.md)

    def test_errors_note_separates_the_errors_that_count_from_the_ones_that_do_not(self):
        c = compare(flags(100), ["judge_error" if i < 3 else ("error" if i < 5 else True) for i in range(100)])
        md = to_markdown(c)
        self.assertIn("ℹ️ Errors: baseline 0 (0 judge, 0 target), candidate 5 (3 judge, 2 target). ", md)
        self.assertIn("2 cases that passed in the baseline crashed in the candidate (target error) and count as failures: `c003`, `c004`.", md)
        self.assertIn("Judge errors and cases that errored in the baseline are infrastructure problems, not quality failures, and are left out of the pass rates.", md)
        self.assertIn("**Newly failing (2):** `c003`, `c004` — 2 of them crashed in the candidate: `c003`, `c004`", md)
        self.assertIn("(2 pass→fail, 2 of them crashes, 0 fail→pass)", md)
        self.assertNotIn("ℹ️ Errors:", to_markdown(compare(BASE_100, BASE_100)))

    def test_errors_note_without_crashes_does_not_mention_them(self):
        md = to_markdown(compare(flags(100), ["judge_error"] * 3 + [True] * 97))
        self.assertIn("candidate 3 (3 judge, 0 target)", md)
        self.assertNotIn("crashed", md)
        self.assertNotIn("Newly failing", md)

    def test_error_rate_row_shows_the_rate_on_the_baselines_cases_when_it_differs(self):
        base = make_run("b", build(flags(20)))
        cand = make_run("c", build(flags(20, error=[0], fail=[])) + build(flags(80), prefix="new"))
        c = compare_runs(base, cand, gate={"max_pass_rate_drop": None})
        self.assertIn("| Error rate | 0.0% | 1.0% (5.0% on the baseline's cases) | +1.0 p.p. |", to_markdown(c))
        self.assertNotIn("on the baseline's cases", to_markdown(compare(flags(20), flags(20))))

    def test_run_unhealthy_reason_is_shown_in_the_headline_list(self):
        md = to_markdown(compare(flags(100), flags(100, error=range(9))))
        self.assertIn("- run unhealthy: 0 judge errors, 9 target errors (9.0% of cases, limit 5.0%) — not a quality signal", md)

    def test_removed_and_edited_cases_are_visible(self):
        base = make_run("main", build(flags(30)))
        cand = make_run("pr", build(flags(25), sha=lambda i: "x" if i < 2 else f"c{i:03d}-v1"))
        md = to_markdown(compare_runs(base, cand))
        self.assertIn("- 5 baseline cases missing from the candidate", md)
        self.assertIn("- test cases were edited: 2 cases differ from the baseline (`c000`, `c001`)", md)
        self.assertIn("**Edited test cases (2):** `c000`, `c001`", md)
        self.assertIn("Baseline only (5)", md)
        self.assertIn("2 shared cases are left out (edited or errored)", md)

    def test_usd_formatting_uses_four_decimals_below_one_dollar(self):
        c = compare(flags(30), flags(30), base_kw={"cost": 0.05}, cand_kw={"cost": 0.06})
        md = to_markdown(c)
        self.assertIn("| Cost / request | $0.0500 | $0.0600 |", md)
        self.assertIn("| Cost / 1k requests | $50.00 | $60.00 |", md)

    def test_markdown_renders_when_metrics_are_missing_on_one_side(self):
        c = compare(flags(30), flags(30), base_kw={"cost": None, "latency": None}, cand_kw={"cost": 0.01, "latency": 1.0})
        md = to_markdown(c)
        self.assertIn("| Cost / request | n/a | $0.0100 | n/a |", md)
        self.assertIn("| p95 latency | n/a | 1.00 s | n/a |", md)

    def test_sub_second_latency_is_shown_in_milliseconds(self):
        c = compare(flags(30), flags(30), base_kw={"latency": 0.25}, cand_kw={"latency": 0.30})
        self.assertIn("| p95 latency | 250 ms | 300 ms | +20% |", to_markdown(c))

    def test_markdown_survives_a_comparison_from_before_cycle_2(self):
        c = compare(BASE_100, drop_by(6))
        for key in ("errors", "removed_cases", "changed_cases", "excluded_errors", "calibration_status", "gate_mode", "effective_max_pass_rate_drop"):
            c.pop(key)
        c["pass_rate"].pop("p_value_two_sided")
        c["gate"] = {**DEFAULT_GATE, "max_pass_rate_drop": 0.03}
        md = to_markdown(c)
        self.assertIn("One-sided McNemar exact p = 0.016 on 6 discordant pairs", md)
        self.assertIn("pass-rate drop ≤ 3.0 p.p.", md)


# ------------------------------------------------------- C10: attempt-level errors


class AttemptErrors(unittest.TestCase):
    """A repeated case takes its majority verdict, which can hide a crashed attempt: count attempts too."""

    def repeated(self, n=100, repeated_ids=(), attempts=None, flag=True):
        """Baseline all pass; candidate records are all pass, with `attempts` on the cases in `repeated_ids`."""
        atts = {cid: attempts for cid in repeated_ids} if not isinstance(attempts, dict) else attempts
        return compare(
            flags(n),
            [flag] * n,
            cand_kw={"attempts": lambda i: atts.get(i)},
            gate=None,
        )

    def test_a_flaky_crash_in_a_case_that_passed_is_named_even_though_the_majority_is_pass(self):
        c = self.repeated(attempts={5: ["pass", "error", "pass"]})
        self.assertEqual(c["verdict"], "pass")  # 1 of 102 attempts errored: under the 5% ceiling
        self.assertEqual(c["flaky_crashes"], ["c005"])
        [w] = warning_with(c, "flaky crash")
        self.assertEqual(
            w,
            "1 case that passed in the baseline hit a target error in some candidate attempts although the majority outcome is pass "
            "(flaky crash): c005 (1/3 attempts); the pass rate does not show it, check it",
        )
        self.assertEqual(c["target_error_regressions"], [])  # it is not a failure: the majority verdict is pass
        self.assertEqual(c["errors"]["candidate"]["n_errors"], 0)  # no case verdict is an error ...
        self.assertEqual(c["errors"]["candidate"]["n_attempt_errors"], 1)  # ... but one attempt was

    def test_several_flaky_cases_are_listed_and_pluralised(self):
        c = self.repeated(attempts={3: ["pass", "error", "pass"], 7: ["error", "pass", "pass", "pass", "pass"], 9: ["pass", "pass", "error"]})
        self.assertEqual(c["flaky_crashes"], ["c003", "c007", "c009"])
        [w] = warning_with(c, "flaky crash")
        self.assertTrue(w.startswith("3 cases that passed in the baseline hit a target error"))
        self.assertIn("c003 (1/3 attempts), c007 (1/5 attempts), c009 (1/3 attempts)", w)
        self.assertTrue(w.endswith("check them"))

    def test_what_is_not_a_flaky_crash(self):
        # a judge error in an attempt, a case that already failed in the baseline, a majority-fail case and a clean repeat
        base = flags(20, fail=[1])
        cand_attempts = {
            0: ["pass", "judge_error", "pass"],  # judge outage, not a target crash
            1: ["pass", "error", "pass"],  # already failing in the baseline: nothing left to regress
            2: ["fail", "error", "fail"],  # majority fail: a counted failure, not a hidden crash
            3: ["pass", "pass", "pass"],
        }
        c = compare(base, flags(20, fail=[2]), cand_kw={"attempts": lambda i: cand_attempts.get(i)}, gate={"max_error_rate": None})
        self.assertEqual(c["flaky_crashes"], [])
        self.assertEqual(warning_with(c, "flaky crash"), [])

    def test_a_case_that_crashed_in_most_attempts_is_a_crash_not_a_flaky_one(self):
        # 2 of 3 attempts errored: the majority verdict is error -> paired as a failure like any crash
        c = compare(flags(100), flags(100, error=[4]), cand_kw={"attempts": lambda i: ["error", "pass", "error"] if i == 4 else None}, gate={"max_error_rate": None})
        self.assertEqual(c["target_error_regressions"], ["c004"])
        self.assertEqual(c["flaky_crashes"], [])

    def test_an_edited_case_is_not_judged_for_flaky_crashes(self):
        base = build(flags(20))
        cand = build(flags(20), attempts=lambda i: ["pass", "error", "pass"] if i == 0 else None, sha=lambda i: "edited" if i == 0 else None)
        c = compare_runs(make_run("b", base), make_run("c", cand))
        self.assertEqual(c["changed_cases"], ["c000"])
        self.assertEqual(c["flaky_crashes"], [])

    def test_the_attempt_error_rate_is_a_reason_with_the_not_a_quality_signal_wording(self):
        # 20 repeated cases (3 attempts) and 20 single ones: 80 attempts, 5 crashed attempts = 6.2% > 5%, yet no case verdict is an error
        atts = {i: ["pass", "error", "pass"] for i in range(5)} | {i: ["pass", "pass", "pass"] for i in range(5, 20)}
        c = compare(flags(40), flags(40), cand_kw={"attempts": lambda i: atts.get(i)})
        self.assertEqual(c["errors"]["candidate"]["error_rate"], 0.0)
        self.assertAlmostEqual(c["errors"]["candidate"]["attempt_error_rate"], 5 / 80)
        self.assertEqual((c["errors"]["candidate"]["n_attempts"], c["errors"]["candidate"]["n_attempt_target_errors"]), (80, 5))
        self.assertEqual(c["verdict"], "regression")
        [reason] = c["reasons"]
        self.assertEqual(
            reason,
            "run unhealthy: 5 of 80 attempts errored (5 target errors, 0 judge errors; 6.2% of attempts over the whole run, limit 5.0%), "
            "although only 0 case verdicts ended in an error: repeated cases hide crashed attempts behind their majority verdict — not a quality signal",
        )
        self.assertEqual(len(c["flaky_crashes"]), 5)  # the cases are named too
        # the ceiling is configurable like the per-case one
        self.assertEqual(compare(flags(40), flags(40), cand_kw={"attempts": lambda i: atts.get(i)}, gate={"max_error_rate": 0.10})["reasons"], [])
        self.assertEqual(compare(flags(40), flags(40), cand_kw={"attempts": lambda i: atts.get(i)}, gate={"max_error_rate": None})["reasons"], [])

    def test_judge_and_target_attempt_errors_are_counted_apart(self):
        atts = {0: ["error", "judge_error", "pass"], 1: ["judge_error", "pass", "pass"]}
        c = compare(flags(10), flags(10), cand_kw={"attempts": lambda i: atts.get(i)}, gate={"max_error_rate": None})
        e = c["errors"]["candidate"]
        self.assertEqual((e["n_attempts"], e["n_attempt_errors"], e["n_attempt_target_errors"], e["n_attempt_judge_errors"]), (14, 3, 1, 2))
        self.assertAlmostEqual(e["attempt_error_rate"], 3 / 14)
        self.assertEqual(c["errors"]["baseline"]["n_attempts"], 10)

    def test_a_suite_without_repeats_gets_one_reason_not_two(self):
        c = compare(flags(100), flags(100, error=range(6)), gate={"max_pass_rate_drop": None})
        self.assertEqual(len(c["reasons"]), 1)
        self.assertTrue(c["reasons"][0].startswith("run unhealthy: 0 judge errors, 6 target errors (6.0% of cases"))
        self.assertEqual(c["errors"]["candidate"]["attempt_error_rate"], c["errors"]["candidate"]["error_rate"])

    def test_added_cases_cannot_dilute_the_attempt_error_rate(self):
        atts = {i: ["pass", "error", "pass"] for i in range(3)}
        base = make_run("b", build(flags(10)))
        cand_records = build(flags(10), attempts=lambda i: atts.get(i)) + build(flags(200), prefix="new")
        c = compare_runs(base, make_run("c", cand_records))
        self.assertLess(c["errors"]["candidate"]["attempt_error_rate"], 0.05)  # over 236 attempts: 1.3%
        self.assertGreater(c["errors"]["candidate"]["paired_attempt_error_rate"], 0.05)  # on the 10 baseline cases: 3 of 16 attempts
        self.assertEqual(len(c["reasons"]), 1)
        self.assertIn("on the cases that also exist in the baseline", c["reasons"][0])
        self.assertIn("not a quality signal", c["reasons"][0])

    def test_an_unhealthy_baseline_is_noted_per_attempt(self):
        atts = {i: ["pass", "error", "pass"] for i in range(5)}
        c = compare(flags(40), flags(40), base_kw={"attempts": lambda i: atts.get(i)})
        [w] = warning_with(c, "baseline run is unhealthy")
        self.assertIn("per attempt", w)

    def test_attempt_entries_without_error_source_still_count(self):
        base, cand = build(flags(10)), build(flags(10))
        cand[2]["attempts"] = [{"outcome": "pass", "passed": True}, {"outcome": "error", "passed": False}, {"passed": True}]  # an older writer
        c = compare_runs(make_run("b", base), make_run("c", cand), gate={"max_error_rate": None})
        self.assertEqual(c["errors"]["candidate"]["n_attempt_target_errors"], 1)
        self.assertEqual(c["flaky_crashes"], ["c002"])


# --------------------------------------------- C11: protocol and version integrity


class RepeatProtocol(unittest.TestCase):
    """C11 + C17: summary.repeat = {"n", "per_tag"}; the cycle-3 shape {"n", "tags"} reads as the same protocol."""

    def cmp(self, base_repeat, cand_repeat, gate=None, **kw):
        base = {} if base_repeat is False else {"repeat": base_repeat}
        cand = {} if cand_repeat is False else {"repeat": cand_repeat}
        return compare(flags(100), flags(100), gate=gate, base_summary=base, cand_summary=cand, **kw)

    def test_the_same_protocol_is_unchanged(self):
        for repeat in (
            {"n": 1, "per_tag": {}},
            {"n": 3, "per_tag": {}},
            {"n": 1, "per_tag": {"safety": 5}},
            {"n": 3, "per_tag": {"safety": 5, "billing": 4}},
            {"n": 1, "tags": None},  # cycle-3 shape
            {"n": 3, "tags": None},
            {"n": 5, "tags": ["safety", "billing"]},
        ):
            with self.subTest(repeat=repeat):
                c = self.cmp(repeat, dict(repeat))
                self.assertIs(c["repeat_changed"], False)
                self.assertEqual((c["reasons"], warning_with(c, "protocol")), ([], []))

    def test_tag_and_key_order_do_not_matter(self):
        c = self.cmp({"n": 3, "per_tag": {"a": 5, "b": 4}}, {"n": 3, "per_tag": {"b": 4, "a": 5}})
        self.assertIs(c["repeat_changed"], False)
        self.assertEqual(c["repeat_protocol"]["baseline"], {"n": 3, "per_tag": {"a": 5, "b": 4}})
        self.assertEqual(list(c["repeat_protocol"]["candidate"]["per_tag"]), ["a", "b"])  # sorted, like summary.repeat
        self.assertIs(self.cmp({"n": 3, "tags": ["a", "b"]}, {"n": 3, "tags": ["b", "a"]})["repeat_changed"], False)

    def test_the_old_tags_key_is_the_same_protocol_as_per_tag(self):
        # {n: 5, tags: [safety]} means: safety cases 5 times, every other case once
        for old, new in (
            ({"n": 5, "tags": ["safety"]}, {"n": 1, "per_tag": {"safety": 5}}),
            ({"n": 3, "tags": ["b", "a"]}, {"n": 1, "per_tag": {"a": 3, "b": 3}}),
            ({"n": 3, "tags": None}, {"n": 3, "per_tag": {}}),
            ({"n": 1, "tags": ["x"]}, {"n": 1, "per_tag": {}}),  # nothing is repeated: the tags say nothing
        ):
            with self.subTest(old=old):
                c = self.cmp(old, new)
                self.assertIs(c["repeat_changed"], False)
                self.assertEqual(c["repeat_protocol"]["baseline"], c["repeat_protocol"]["candidate"])
                self.assertEqual(c["repeat_protocol"]["baseline"], {"n": new["n"], "per_tag": new["per_tag"]})

    def test_a_per_tag_entry_equal_to_n_changes_nothing_but_a_smaller_one_does(self):
        self.assertIs(self.cmp({"n": 3, "per_tag": {}}, {"n": 3, "per_tag": {"safety": 3}})["repeat_changed"], False)
        self.assertIs(self.cmp({"n": 1, "per_tag": {}}, {"n": 1, "per_tag": {"safety": 1}})["repeat_changed"], False)
        self.assertIs(self.cmp({"n": 3, "per_tag": {}}, {"n": 3, "per_tag": {"safety": 1}})["repeat_changed"], True)  # safety once
        self.assertEqual(self.cmp({"n": 3}, {"n": 3, "per_tag": {"safety": 3, "x": 4}})["repeat_protocol"]["candidate"], {"n": 3, "per_tag": {"x": 4}})

    def test_a_pr_that_sets_repeat_is_a_regression_reason(self):  # critic W11 / W12b
        c = self.cmp({"n": 1, "per_tag": {}}, {"n": 3, "per_tag": {}})
        self.assertIs(c["repeat_changed"], True)
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(
            c["reasons"],
            ["evaluation protocol changed (repeat: baseline every case once (n: 1), candidate n: 3 on every case): update the baseline in a reviewed PR"],
        )
        self.assertEqual(c["repeat_protocol"], {"baseline": {"n": 1, "per_tag": {}}, "candidate": {"n": 3, "per_tag": {}}})

    def test_a_pr_that_adds_or_lowers_per_tag_repeats_is_a_regression_reason(self):
        c = self.cmp({"n": 3, "per_tag": {}}, {"n": 3, "per_tag": {"safety": 5}})
        self.assertIs(c["repeat_changed"], True)
        self.assertEqual(
            c["reasons"],
            ["evaluation protocol changed (repeat: baseline n: 3 on every case, candidate n: 3, per_tag: {safety: 5}): update the baseline in a reviewed PR"],
        )
        lowered = self.cmp({"n": 3, "per_tag": {"safety": 5}}, {"n": 3, "per_tag": {"safety": 1}})  # the PR makes the critical cases easier to pass
        self.assertIs(lowered["repeat_changed"], True)
        self.assertIn("baseline n: 3, per_tag: {safety: 5}, candidate n: 3, per_tag: {safety: 1}", lowered["reasons"][0])

    def test_a_baseline_from_0_1_without_the_key_ran_every_case_once(self):
        c = self.cmp(False, {"n": 3, "per_tag": {}})  # 0.1 wrote `repeat` only when n > 1
        self.assertIs(c["repeat_changed"], True)
        self.assertEqual(c["verdict"], "regression")
        self.assertIs(self.cmp(False, {"n": 1, "per_tag": {}})["repeat_changed"], False)
        self.assertIs(self.cmp(False, False)["repeat_changed"], False)
        self.assertIs(self.cmp({"n": 3, "per_tag": {}}, False)["repeat_changed"], True)  # the PR removed repeat
        self.assertIs(self.cmp({"n": 1, "per_tag": {"safety": 5}}, False)["repeat_changed"], True)

    def test_changing_n_or_the_per_tag_map_counts_as_a_change(self):
        for base, cand in (
            ({"n": 3, "per_tag": {"safety": 5}}, {"n": 3, "per_tag": {}}),
            ({"n": 3, "per_tag": {"safety": 5}}, {"n": 3, "per_tag": {"safety": 5, "billing": 5}}),
            ({"n": 3, "per_tag": {"safety": 5}}, {"n": 3, "per_tag": {"safety": 4}}),
            ({"n": 3, "per_tag": {}}, {"n": 5, "per_tag": {}}),
            ({"n": 3, "per_tag": {"safety": 5}}, {"n": 4, "per_tag": {"safety": 5}}),
            ({"n": 5, "tags": ["safety"]}, {"n": 5, "tags": None}),  # cycle-3 shape: every case 5 times instead of the safety ones only
        ):
            with self.subTest(base=base, cand=cand):
                self.assertIs(self.cmp(base, cand)["repeat_changed"], True)
        text = self.cmp({"n": 3, "tags": ["safety"]}, {"n": 5, "tags": None})["reasons"][0]
        self.assertIn("baseline n: 1, per_tag: {safety: 3}, candidate n: 5 on every case", text)

    def test_per_tag_wins_over_a_stray_tags_key(self):
        self.assertEqual(self.cmp({"n": 3, "per_tag": {"safety": 5}, "tags": ["billing"]}, False)["repeat_protocol"]["baseline"], {"n": 3, "per_tag": {"safety": 5}})

    def test_a_bare_integer_is_n(self):
        self.assertIs(self.cmp(3, {"n": 3, "per_tag": {}})["repeat_changed"], False)
        self.assertIs(self.cmp(1, {"n": 3, "per_tag": {}})["repeat_changed"], True)

    def test_allow_repeat_change_turns_the_reason_into_a_warning(self):
        c = self.cmp({"n": 1, "per_tag": {}}, {"n": 1, "per_tag": {"safety": 5}}, gate={"allow_repeat_change": True})
        self.assertIs(c["repeat_changed"], True)
        self.assertEqual((c["verdict"], c["reasons"]), ("pass", []))
        [w] = warning_with(c, "evaluation protocol changed")
        self.assertIn("allowed by gate.allow_repeat_change", w)
        self.assertIn("update the baseline in a reviewed PR", w)

    def test_summaries_that_predate_both_keys_are_unknown_not_a_regression(self):
        base = make_run("b", build(flags(100)), judges_sha=None)
        cand = make_run("c", build(flags(100)), judges_sha=None)
        c = compare_runs(base, cand)
        self.assertIsNone(c["repeat_changed"])
        self.assertEqual(c["reasons"], [])
        self.assertEqual(c["repeat_protocol"], {"baseline": None, "candidate": None})

    def test_a_malformed_repeat_is_unknown_and_never_crashes(self):
        for bad in (
            "three",
            {"n": "3"},
            {"n": True},
            {"n": 0},
            [3],
            2.5,
            {"n": 3, "per_tag": "safety"},
            {"n": 3, "per_tag": ["safety"]},
            {"n": 3, "per_tag": {"safety": "5"}},
            {"n": 3, "per_tag": {"safety": 0}},
            {"n": 3, "per_tag": {"safety": True}},
            {"n": 3, "per_tag": {"safety": 2.5}},
        ):
            with self.subTest(bad=bad):
                c = self.cmp(bad, {"n": 1, "per_tag": {}})
                self.assertIsNone(c["repeat_changed"])
                self.assertEqual(c["reasons"], [])

    def test_gate_line_and_report_key(self):
        c = self.cmp({"n": 1, "per_tag": {}}, {"n": 1, "per_tag": {}})
        self.assertIn("repeat protocol unchanged", to_markdown(c))
        c = self.cmp({"n": 1, "per_tag": {}}, {"n": 1, "per_tag": {}}, gate={"allow_repeat_change": True})
        self.assertNotIn("repeat protocol unchanged", to_markdown(c))


class KitVersion(unittest.TestCase):
    """C11 + C20: any version difference is a warning (patch included); require_same_kit_version makes it a reason."""

    def cmp(self, base_v, cand_v, gate=None, key="evalsprint_version"):
        base = {} if base_v is None else {key: base_v}
        cand = {} if cand_v is None else {key: cand_v}
        return compare(flags(100), flags(100), gate=gate, base_summary=base, cand_summary=cand)

    def test_the_same_release_is_unchanged_however_it_is_spelled(self):
        for base, cand in (("0.2.0", "0.2.0"), ("0.2", "0.2.0"), ("v0.2.0", "0.2.0"), (" 0.2.0 ", "V0.2.0"), ("0.2.0.0", "0.2")):
            with self.subTest(base=base, cand=cand):
                c = self.cmp(base, cand)
                self.assertEqual(c["kit_version"], {"baseline": base, "candidate": cand, "changed": False})
                self.assertEqual(warning_with(c, "version"), [])

    def test_a_patch_release_is_a_warning(self):  # C20
        for base, cand in (("0.2.0", "0.2.7"), ("0.2", "0.2.1"), ("v0.2.0", "0.2.4"), ("0.2.3", "0.2.0")):
            with self.subTest(base=base, cand=cand):
                c = self.cmp(base, cand)
                self.assertEqual(c["kit_version"], {"baseline": base, "candidate": cand, "changed": True})
                self.assertEqual((c["verdict"], c["reasons"]), ("pass", []))
                [w] = warning_with(c, "evalsprint patch version differs")
                self.assertIn(f"baseline {base}, candidate {cand}", w)
                self.assertIn("scoring is expected to be the same", w)
                self.assertEqual(warning_with(c, "scoring may differ"), [])

    def test_a_different_minor_or_major_version_is_a_stronger_warning(self):
        for base, cand in (("0.1.0", "0.2.0"), ("0.2.5", "0.3.0"), ("1.9.0", "2.0.0")):
            with self.subTest(base=base, cand=cand):
                c = self.cmp(base, cand)
                self.assertEqual(c["kit_version"], {"baseline": base, "candidate": cand, "changed": True})
                self.assertEqual(c["verdict"], "pass")
                [w] = warning_with(c, "evalsprint version differs")
                self.assertIn(f"baseline {base}, candidate {cand}", w)
                self.assertIn("scoring may differ across versions", w)
                self.assertEqual(warning_with(c, "patch version"), [])

    def test_require_same_kit_version_makes_any_difference_a_reason(self):
        c = self.cmp("0.1.0", "0.2.0", gate={"require_same_kit_version": True})
        self.assertEqual(c["verdict"], "regression")
        self.assertEqual(
            c["reasons"],
            ["evalsprint version changed (baseline 0.1.0, candidate 0.2.0; gate.require_same_kit_version is on): update the baseline in a reviewed PR"],
        )
        self.assertEqual(warning_with(c, "evalsprint version differs"), [])
        patch = self.cmp("0.2.0", "0.2.3", gate={"require_same_kit_version": True})  # C20: the patch counts as well
        self.assertEqual(patch["reasons"], ["evalsprint version changed (baseline 0.2.0, candidate 0.2.3; gate.require_same_kit_version is on): update the baseline in a reviewed PR"])
        self.assertEqual(warning_with(patch, "patch version"), [])
        self.assertEqual(self.cmp("0.2.0", "v0.2", gate={"require_same_kit_version": True})["reasons"], [])

    def test_kit_version_wins_over_evalsprint_version_when_both_exist(self):
        base = make_run("b", build(flags(50)), kit_version="0.3.0", evalsprint_version="0.1.0")
        cand = make_run("c", build(flags(50)), kit_version="0.3.1", evalsprint_version="0.1.0")
        self.assertEqual(compare_runs(base, cand)["kit_version"], {"baseline": "0.3.0", "candidate": "0.3.1", "changed": True})
        self.assertEqual(self.cmp("0.1.0", "0.2.0", key="kit_version")["kit_version"]["changed"], True)

    def test_a_missing_version_is_unknown_and_silent_unless_required(self):
        c = self.cmp(None, "0.2.0")
        self.assertEqual(c["kit_version"], {"baseline": None, "candidate": "0.2.0", "changed": None})
        self.assertEqual((c["reasons"], c["warnings"]), ([], []))
        c = self.cmp(None, "0.2.0", gate={"require_same_kit_version": True})
        self.assertEqual(c["reasons"], [])
        [w] = warning_with(c, "no evalsprint version")
        self.assertIn("baseline summary carries", w)
        self.assertIn("could not be checked", w)
        c = self.cmp(None, None, gate={"require_same_kit_version": True})
        self.assertIn("baseline and candidate summaries carry no evalsprint version", c["warnings"][0])

    def test_versions_that_do_not_look_like_versions_compare_by_equality(self):
        self.assertIs(self.cmp("dev", "dev")["kit_version"]["changed"], False)
        self.assertIs(self.cmp("dev", "main")["kit_version"]["changed"], True)
        [w] = warning_with(self.cmp("dev", "main"), "evalsprint version differs")  # not a patch: nothing says they are the same series
        self.assertIn("baseline dev, candidate main", w)
        self.assertIs(self.cmp("v0.2.0", "0.2.4")["kit_version"]["changed"], True)
        self.assertIs(self.cmp("0.3.0rc1", "0.3.0")["kit_version"]["changed"], True)  # a pre-release is another build


class TagWarnings(unittest.TestCase):
    """C11 + C17: a non-critical tag whose pass rate fell by >= tag_drop_warning_pp with >= tag_drop_min_cases (default 20) paired cases."""

    N = 200
    # c000..c019 carry tag "search" (20 cases), c020..c049 "billing" (30), c050..c057 "rare" (8), the rest "general"
    TAGS = staticmethod(lambda i: ["search"] if i < 20 else ["billing"] if i < 50 else ["rare"] if i < 58 else ["general"])

    def cmp(self, cand_fail, gate=None, base_fail=(), base_kw=None, cand_kw=None):
        kw = {"tags": self.TAGS}
        return compare(
            flags(self.N, fail=base_fail),
            flags(self.N, fail=cand_fail),
            gate=gate,
            base_kw={**kw, **(base_kw or {})},
            cand_kw={**kw, **(cand_kw or {})},
        )

    def test_the_default_minimum_is_20_paired_cases(self):
        self.assertEqual(DEFAULT_GATE["tag_drop_min_cases"], 20)
        self.assertEqual(resolve_gate({}, None)["tag_drop_min_cases"], 20)

    def test_a_big_drop_in_a_tag_warns_and_never_blocks(self):
        # 8 of the 20 "search" cases regress while 8 baseline failures in "general" recover: overall unchanged
        c = self.cmp(cand_fail=range(8), base_fail=range(100, 108))
        self.assertEqual(c["verdict"], "pass")
        [w] = c["tag_warnings"]
        self.assertEqual((w["tag"], w["n"], w["baseline"], w["candidate"]), ("search", 20, 1.0, 0.6))
        self.assertAlmostEqual(w["delta"], -0.4)
        [text] = warning_with(c, "tag 'search'")
        self.assertEqual(
            text, "tag 'search': pass rate fell 40.0 p.p. (100.0% → 60.0%, 20 paired cases); not a gate reason (the tag is not critical), but check it"
        )
        self.assertEqual(c["reasons"], [])

    def test_the_threshold_is_inclusive_and_configurable(self):
        four_of_twenty = dict(cand_fail=range(4), base_fail=range(100, 104))  # exactly -20 p.p.
        self.assertEqual([w["tag"] for w in self.cmp(**four_of_twenty)["tag_warnings"]], ["search"])
        two_of_twenty = dict(cand_fail=range(2), base_fail=range(100, 102))  # -10 p.p.
        self.assertEqual(self.cmp(**two_of_twenty)["tag_warnings"], [])
        self.assertEqual([w["tag"] for w in self.cmp(**two_of_twenty, gate={"tag_drop_warning_pp": 10})["tag_warnings"]], ["search"])
        self.assertEqual(self.cmp(**four_of_twenty, gate={"tag_drop_warning_pp": 25})["tag_warnings"], [])
        self.assertEqual(self.cmp(**four_of_twenty, gate={"tag_drop_warning_pp": None})["tag_warnings"], [])

    def test_a_tag_with_too_few_paired_cases_is_not_reported(self):
        # "rare" has 8 cases: all 8 regress (-100 p.p.) but the default minimum is 20
        c = self.cmp(cand_fail=range(50, 58), base_fail=range(100, 108))
        self.assertEqual(c["tag_warnings"], [])
        c = self.cmp(cand_fail=range(50, 58), base_fail=range(100, 108), gate={"tag_drop_min_cases": 8})
        self.assertEqual([(w["tag"], w["n"], w["delta"]) for w in c["tag_warnings"]], [("rare", 8, -1.0)])

    def test_the_minimum_is_inclusive(self):
        drop = dict(cand_fail=range(10), base_fail=range(100, 110))  # 10 of the 20 "search" cases: -50 p.p.
        self.assertEqual([w["n"] for w in self.cmp(**drop)["tag_warnings"]], [20])  # exactly 20 paired cases: reported
        self.assertEqual(self.cmp(**drop, gate={"tag_drop_min_cases": 21})["tag_warnings"], [])

    def test_ten_tags_of_ten_cases_do_not_warn_by_default(self):
        # the critic's 10 tags x 10 cases: 2 flipped cases in one tag is -20 p.p. and happens by chance 43% of the time
        tags = lambda i: [f"t{i // 10}"]  # noqa: E731
        c = compare(flags(100), flags(100, fail=[0, 1]), base_kw={"tags": tags}, cand_kw={"tags": tags}, gate={"max_pass_rate_drop": None})
        self.assertEqual(c["tag_warnings"], [])
        c = compare(flags(100), flags(100, fail=[0, 1]), base_kw={"tags": tags}, cand_kw={"tags": tags}, gate={"max_pass_rate_drop": None, "tag_drop_min_cases": 10})
        self.assertEqual([w["tag"] for w in c["tag_warnings"]], ["t0"])

    def test_critical_and_untagged_cases_are_left_to_their_own_rules(self):
        c = self.cmp(cand_fail=range(8), base_fail=range(100, 108), gate={"critical_tags": ["search"]})
        self.assertEqual(c["tag_warnings"], [])
        self.assertEqual(c["verdict"], "regression")  # the critical rule fires instead
        base = flags(60)
        c = compare(base, flags(60, fail=range(30)), gate={"max_pass_rate_drop": None})  # everything is "(untagged)"
        self.assertEqual(c["tag_warnings"], [])

    def test_worst_tag_first_and_crashes_count_as_failures(self):
        # the gate may block as well (here it does): the tag warnings are reported regardless
        crash = compare(
            flags(self.N),
            flags(self.N, error=range(20, 40)),  # 20 of the 30 "billing" cases crash: -67 p.p.
            gate={"max_error_rate": None},
            base_kw={"tags": self.TAGS},
            cand_kw={"tags": self.TAGS},
        )
        [w] = crash["tag_warnings"]
        self.assertEqual((w["tag"], w["n"], w["baseline"]), ("billing", 30, 1.0))
        self.assertAlmostEqual(w["candidate"], 10 / 30)
        worst_first = self.cmp(cand_fail=[*range(0, 10), *range(20, 50)])  # search -50, billing -100
        self.assertEqual([w["tag"] for w in worst_first["tag_warnings"]], ["billing", "search"])

    def test_no_warning_when_nothing_dropped(self):
        c = self.cmp(cand_fail=(), base_fail=[0, 1, 2])
        self.assertEqual(c["tag_warnings"], [])
        self.assertEqual(warning_with(c, "tag '"), [])

    def test_a_case_in_two_tags_counts_in_both(self):
        both = lambda i: ["a", "b"] if i < 24 else ["c"]  # noqa: E731
        c = compare(
            flags(100), flags(100, fail=range(12)), base_kw={"tags": both}, cand_kw={"tags": both}, gate={"max_pass_rate_drop": None}
        )
        self.assertEqual(sorted(w["tag"] for w in c["tag_warnings"]), ["a", "b"])


class PlaceholderBaseline(unittest.TestCase):
    """C18 compare side: a baseline recorded from a placeholder run is flagged; the new summary keys change nothing else."""

    def test_a_placeholder_baseline_is_a_warning_and_never_a_reason(self):
        c = compare(BASE_100, BASE_100, base_summary={"placeholder": True})
        self.assertEqual((c["verdict"], c["reasons"]), ("pass", []))
        [w] = warning_with(c, "placeholder")
        self.assertTrue(w.startswith("the baseline run is a placeholder"))
        self.assertIn("simulated target or mock judge", w)
        self.assertIn("record a baseline from a real run", w)
        self.assertIn("⚠️ the baseline run is a placeholder", to_markdown(c))

    def test_no_warning_without_the_key_or_when_it_is_not_true(self):
        for summary in ({}, {"placeholder": False}, {"placeholder": None}, {"placeholder": "true"}, {"placeholder": 1}):
            with self.subTest(summary=summary):
                self.assertEqual(warning_with(compare(BASE_100, BASE_100, base_summary=summary), "placeholder"), [])

    def test_only_the_baseline_is_judged_here(self):
        self.assertEqual(warning_with(compare(BASE_100, BASE_100, cand_summary={"placeholder": True}), "placeholder"), [])
        both = compare(BASE_100, BASE_100, base_summary={"placeholder": True}, cand_summary={"placeholder": True})
        self.assertEqual(len(warning_with(both, "placeholder")), 1)

    def test_a_placeholder_baseline_does_not_change_the_verdict(self):
        clean = compare(BASE_100, drop_by(6))
        flagged = compare(BASE_100, drop_by(6), base_summary={"placeholder": True})
        self.assertEqual((flagged["verdict"], flagged["reasons"]), (clean["verdict"], clean["reasons"]))
        self.assertEqual(len(flagged["warnings"]), len(clean["warnings"]) + 1)

    def test_the_new_summary_keys_are_ignored_by_the_comparison(self):
        plain = compare(BASE_100, drop_by(6))
        extra = {"n_target_retries": 7, "placeholder": False, "n_oversize": 3, "target_retries": 1}
        with_keys = compare(BASE_100, drop_by(6), base_summary=extra, cand_summary=extra)
        # the retry counts are reported (C22: `target_retries`), every verdict, reason, warning and statistic is the same
        self.assertEqual({k: v for k, v in with_keys.items() if k != "target_retries"}, {k: v for k, v in plain.items() if k != "target_retries"})
        self.assertEqual(with_keys["target_retries"]["baseline"]["retried_calls"], 7)
        self.assertEqual(with_keys["target_retries"]["baseline"]["configured"], 1)
        self.assertEqual(plain["target_retries"]["baseline"]["retried_calls"], None)

    def test_records_that_needed_a_target_retry_are_ordinary_records(self):
        def retried(records):
            for i, rec in enumerate(records):
                if i % 7 == 0:
                    rec["target_retries_used"] = 1
            return records

        base = make_run("b", retried(build(BASE_100)))
        cand = make_run("c", retried(build(drop_by(6))))
        plain = compare(BASE_100, drop_by(6))
        c = compare_runs(base, cand)
        self.assertEqual((c["verdict"], c["reasons"], c["errors"]), (plain["verdict"], plain["reasons"], plain["errors"]))
        self.assertEqual(c["pass_rate"], plain["pass_rate"])

    def test_a_judge_result_too_large_to_judge_is_an_ordinary_failure(self):
        base = make_run("b", build(flags(100), scores=1.0))
        recs = build(flags(100, fail=range(6)), scores=0.0)
        for rec in recs[:6]:
            rec["judges"][0]["details"] = {"oversize": True}
        c = compare_runs(base, make_run("c", recs))
        self.assertEqual(c["newly_failing"], [f"c{i:03d}" for i in range(6)])
        self.assertEqual(c["verdict"], "regression")


# ---------------------------------------------------------------- C20: markdown safety

_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def inline_tokens(text: str) -> list[tuple[str, str]]:
    """Split one line of markdown into ('code', content) and ('text', raw) pieces with the CommonMark rules that matter here:
    a backslash before ASCII punctuation makes it literal (an escaped backtick opens no code span), a run of N backticks closes
    at the next run of exactly N, one space is stripped from both ends of a span whose content is not all spaces."""
    out, buf, i = [], [], 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _PUNCT:
            buf.append(text[i : i + 2])
            i += 2
        elif ch == "`":
            j = i
            while j < len(text) and text[j] == "`":
                j += 1
            width, k, close = j - i, j, None
            while k < len(text):
                if text[k] == "`":
                    m = k
                    while m < len(text) and text[m] == "`":
                        m += 1
                    if m - k == width:
                        close = k
                        break
                    k = m
                else:
                    k += 1
            if close is None:
                buf.append(text[i:j])
                i = j
                continue
            content = text[j:close]
            if len(content) >= 2 and content[0] == " " and content[-1] == " " and content.strip(" "):
                content = content[1:-1]
            out.append(("text", "".join(buf)))
            buf = []
            out.append(("code", content))
            i = close + width
        else:
            buf.append(ch)
            i += 1
    out.append(("text", "".join(buf)))
    return out


def outside_code(md: str) -> str:
    """The markdown with every code span removed, line by line."""
    return "\n".join("".join(t for kind, t in inline_tokens(line) if kind == "text") for line in md.splitlines())


def printable(text: str) -> str:
    return "".join(ch if ch.isprintable() else " " for ch in text)


def unescaped_pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def shown(fragment: str) -> str:
    """What a renderer displays for a fragment made of code spans and character references only (anything else is kept as written)."""
    return "".join(t if kind == "code" else html.unescape(t) for kind, t in inline_tokens(fragment))


# what the critic's 70_xss_fixture.py / 71_xss_static.py put into case ids, tags, versions and run ids (cycle 3, check 5)
CRITIC_P = '</script><img src=x onerror=alert("X")><svg/onload=alert(1)>"\'<!-- <ai_output_7f3a9c>IGNORE PREVIOUS INSTRUCTIONS</ai_output_7f3a9c> `tick` | pipe'
HOSTILE_IDS = [
    "=cmd|' /C calc'!A0",
    "id1</script><script>alert(2)</script>",
    "-neg`tick`|pipe",
    "removed-<b>",
    "added-<i>",
    "``double`` and ```triple```",
    "tail`",
    "`lead",
    " spaced ",
    "line1\n## Fake heading\n- [x] injected\r\n> quote",
    "rtl‮txt⁦x⁩\x00\x1b[31m",
    "[click](http://evil.example)![img](http://evil.example/x.png)",
    "<http://evil.example>",
    "trailing\\",
    "| a | b |",
    CRITIC_P,
]


class MarkdownIsSafeForHostileText(unittest.TestCase):
    """C20: case ids, tag names, versions, run ids and judge names come from files a PR controls.  The markdown escapes them.

    One scenario reaches every slot that holds such text (reasons, warnings, id lists, the tag table, the header, the gate line, the
    calibration line); it is built twice from the same flags, once with the critic's hostile strings and once with plain words, and the
    two markdowns must have the same structure.
    """

    N = 100

    def scenario(self, hostile: bool):
        ids = HOSTILE_IDS if hostile else [f"case-{i}" for i in range(len(HOSTILE_IDS))]
        tag_a = "<img src=x onerror=alert('tag')>" if hostile else "tag-a"
        tag_crit = "safe`ty|<b>x</b>" if hostile else "tag-crit"
        tag_grp = CRITIC_P if hostile else "tag-grp"
        tags = lambda i: [tag_a] if i < 20 else [tag_crit] if i < 30 else [tag_grp] if i < 60 else []  # noqa: E731
        special = {**{i: ids[i] for i in range(10)}, 21: ids[10], 22: ids[11], 30: ids[12], 31: ids[13], 90: ids[14], 91: ids[15], 45: ids[3] + "-e1", 46: ids[4] + "-e2"}

        def named(records):
            for i, rec in enumerate(records):
                if i in special:
                    rec["case_id"] = rec["case"]["id"] = special[i]
            return records

        base = named(build(flags(self.N, fail=[90, 91, 92]), tags=tags))
        cand = named(build(flags(self.N, fail=[*range(10), 21, 22, 25], error=[30, 31]), tags=tags))
        for rec in cand[45:47]:
            rec["case_sha"] = "edited-" + rec["case_sha"]  # two edited cases
        gone = build([True, True], prefix="gone-")
        gone[0]["case_id"] = gone[0]["case"]["id"] = ids[1] + "-gone"
        new = build([True, True], prefix="new-")
        new[0]["case_id"] = new[0]["case"]["id"] = ids[6] + "-new"
        version = "0.2.0" + CRITIC_P if hostile else "0.2.1"
        run_id = "20260930-" + CRITIC_P if hostile else "20260930-cand"
        sha = "bb<b>`|" if hostile else "bbbbbbbbbbbb"
        judge = "j<img src=x>`|" if hostile else "judge"
        calibration = {"judge": judge, "kappa_weighted": 0.9, "kappa_weighted_ci": [0.8, 1.0], "kappa_decision": "trustworthy"}
        gate = {"critical_tags": [tag_crit, "nope-" + tag_a]}
        b = make_run("base", base + gone, judges_sha="aaaaaaaaaaaa", evalsprint_version="0.2.0", repeat={"n": 1, "per_tag": {}})
        c = make_run(run_id, cand + new, judges_sha=sha, evalsprint_version=version, repeat={"n": 1, "per_tag": {tag_a: 5}})
        return compare_runs(b, c, gate=gate, calibration=calibration), {"tag_a": tag_a, "tag_crit": tag_crit, "version": version, "run_id": run_id}

    def setUp(self):
        self.h, self.hostile = self.scenario(True)
        self.b, _ = self.scenario(False)
        self.hmd, self.bmd = to_markdown(self.h), to_markdown(self.b)

    def test_the_scenario_reaches_every_slot_that_holds_user_text(self):
        self.assertEqual(self.h["verdict"], "regression")
        text = "\n".join(self.h["reasons"] + self.h["warnings"])
        for what in ("critical tag", "edited", "missing from the candidate", "judge configuration", "evaluation protocol changed", "tag '", "crashed", "evalsprint patch version"):
            self.assertIn(what, text, what)
        self.assertTrue(self.h["newly_failing"] and self.h["newly_passing"] and self.h["removed_cases"] and self.h["unpaired"]["only_candidate"])
        self.assertTrue(self.h["target_error_regressions"] and self.h["changed_cases"] and self.h["tag_warnings"])
        for what in ("| Tag | n |", "Judge calibration", "critical tags:", "Baseline ", "**Edited test cases", "Candidate only"):
            self.assertIn(what, self.hmd, what)

    def test_the_json_keeps_the_raw_text_and_only_the_markdown_is_escaped(self):
        self.assertIn("id1</script><script>alert(2)</script>", self.h["newly_failing"])
        self.assertEqual(self.h["kit_version"]["candidate"], self.hostile["version"])
        self.assertEqual(self.h["candidate_run_id"], self.hostile["run_id"])
        self.assertIn("safe`ty|<b>x</b>", self.h["gate"]["critical_tags"])
        self.assertTrue(any("<img src=x onerror=alert('tag')>" in w for w in self.h["warnings"]))
        self.assertTrue(any(self.hostile["version"] in w for w in self.h["warnings"]))

    def test_the_hostile_markdown_has_the_same_structure_as_the_benign_one(self):
        hl, bl = self.hmd.splitlines(), self.bmd.splitlines()
        self.assertEqual(len(hl), len(bl), "hostile text created or removed a line")
        for k, (x, y) in enumerate(zip(hl, bl)):
            self.assertEqual(x[:2], y[:2], f"line {k} changed its markdown block kind: {x[:60]!r} vs {y[:60]!r}")

    def test_no_injected_block_or_control_character_survives(self):
        md = self.hmd
        for ch in ("\x00", "\x1b", "‮", "⁦", "⁩", "\r"):
            self.assertNotIn(ch, md)
        lines = md.splitlines()
        self.assertEqual([l for l in lines if l.startswith(("## Fake", "- [x]", ">"))], [])
        self.assertEqual([l for l in lines if l.startswith("#")], [lines[0]])  # only the kit's own headline

    def test_no_raw_html_link_or_unbalanced_backtick_outside_a_code_span(self):
        text, benign = outside_code(self.hmd), outside_code(self.bmd)
        self.assertEqual(text.count("<"), benign.count("<"))  # the kit's own <sub> and the "p < 0.001" of its own sentence
        self.assertEqual(text.count(">"), benign.count(">"))
        self.assertEqual(re.findall(r"</?[A-Za-z!][^>]*>", text), re.findall(r"</?[A-Za-z!][^>]*>", benign))
        self.assertEqual(set(re.findall(r"</?[A-Za-z!][^>]*>", text)), {"<sub>", "</sub>"})  # only the kit's own tags
        for raw in ("`", "](", "[", "]", "<script", "<img", "<svg", "<!--", "<ai_output", "<http", "<b>", "<i>"):
            self.assertNotIn(raw, text)

    def test_the_critics_static_check_passes_too(self):
        # 71_xss_static.py: `re.sub(r"`[^`\n]*`", "", md)` must leave no <script / <img / <svg, and the tag table keeps its column count
        naive = re.sub(r"`[^`\n]*`", "", self.hmd)
        self.assertNotRegex(naive, r"<script|<img|<svg")
        lines = self.hmd.splitlines()
        start = next(k for k, l in enumerate(lines) if l.startswith("| Tag |"))
        rows = [l for l in lines[start:] if l.startswith("|")]
        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual({unescaped_pipes(l) for l in rows}, {6})

    def test_every_id_in_a_list_round_trips_through_its_code_span(self):
        line = next(l for l in self.hmd.splitlines() if l.startswith("**Newly failing"))
        ids = self.h["newly_failing"][:MAX_LISTED_IDS]
        crashed = self.h["target_error_regressions"]
        for i in ids + crashed:
            # every id is one run of single-backtick code spans (an id with a backtick is cut there and the backtick is a reference): it
            # reads back as the id (control characters as spaces), and the text between the spans is only those references
            self.assertIn(_code(i), line, i)
            fragment = _code(i)
            self.assertEqual(shown(fragment), printable(i), i)
            self.assertTrue(all(re.fullmatch(r"(&#96;)+", t) for kind, t in inline_tokens(fragment) if kind == "text" and t), i)
        # nothing of an id leaks outside its spans, and the naive "pairs of backticks" reading agrees with CommonMark on this line
        self.assertEqual(re.sub(r"`[^`\n]*`", "", line), outside_code(line))

    def test_the_header_and_the_gate_line_keep_ids_in_code_spans(self):
        # C23: every artifact value is a run of single-backtick code spans (a backtick of the text is a character reference between two spans)
        header = next(l for l in self.hmd.splitlines() if l.startswith("Baseline "))
        self.assertEqual(header, f"Baseline `base` → candidate {_code(self.hostile['run_id'])} · {self.h['n_paired']} paired cases")
        self.assertEqual(shown(_code(self.hostile["run_id"])), printable(self.hostile["run_id"]))
        gate_line = next(l for l in self.hmd.splitlines() if l.startswith("<sub>Gate:"))
        self.assertIn(f"critical tags: {_code(self.hostile['tag_crit'])}, {_code('nope-' + self.hostile['tag_a'])}", gate_line)
        benign_line = next(l for l in self.bmd.splitlines() if l.startswith("<sub>Gate:"))
        self.assertEqual(outside_code(gate_line).count("<"), outside_code(benign_line).count("<"))  # only the kit's own <sub> tags
        calibration = next(l for l in self.hmd.splitlines() if l.startswith("Judge calibration"))
        self.assertTrue(calibration.startswith(f"Judge calibration {_code('j<img src=x>`|')}: "), calibration)
        self.assertEqual([t for kind, t in inline_tokens(calibration) if kind == "code"], ["j<img src=x>", "|", "trustworthy"])  # judge (cut at its backtick), decision

    def test_a_tag_name_cannot_break_out_of_its_table_cell(self):
        lines = self.hmd.splitlines()
        rows = [l for l in lines[next(k for k, l in enumerate(lines) if l.startswith("| Tag |")) :] if l.startswith("|")]
        self.assertIn(f"| {_code('safe`ty|<b>x</b>', table=True)} 🔒 |", "\n".join(rows))
        self.assertIn("| `safe`&#96;`ty`&#124;`<b>x</b>` 🔒 |", "\n".join(rows))
        for row in rows:
            self.assertEqual(unescaped_pipes(row), 6, row)


class MarkdownHelpers(unittest.TestCase):
    def test_a_code_span_holds_any_text_literally(self):
        from evalsprint.compare import _code

        for text in ("plain", " ", "  ", " a", "a ", " a ", "<script>", "a|b", "x\ny", "x\u202ey", "\u202e", "[a](b)", "\\"):
            with self.subTest(text=text):
                line = f"before {_code(text)} after"
                tokens = inline_tokens(line)
                self.assertEqual([t for kind, t in tokens if kind == "code"], [printable(text)], line)
                self.assertEqual("".join(t for kind, t in tokens if kind == "text"), "before  after")
        self.assertEqual([t for kind, t in inline_tokens(_code("")) if kind == "code"], [" "])  # an empty id still shows a span

    def test_the_fence_is_always_one_backtick_and_a_backtick_of_the_text_is_a_reference_between_two_spans(self):
        self.assertEqual(_code("id"), "`id`")
        self.assertEqual(_code(" id"), "`  id `")  # padded: CommonMark strips one space from each side again
        self.assertEqual(_code("a`b"), "`a`&#96;`b`")
        self.assertEqual(_code("`"), "&#96;")
        for text in ("a`b", "``two``", "```three```", "`", "tail`", "`lead", "`a``b```c`"):
            with self.subTest(text=text):
                out = _code(text)
                self.assertEqual(shown(out), text)  # CommonMark reads it back as the text
                # the naive "pairs of single backticks" reading leaves only the references, so it agrees with CommonMark
                self.assertEqual(re.sub(r"`[^`\n]*`", "", out), "&#96;" * text.count("`"))
                self.assertEqual(re.sub(r"`[^`\n]*`", "", out), outside_code(out))

    def test_md_text_turns_markup_into_character_references(self):
        from evalsprint.compare import _md_text

        self.assertEqual(_md_text("a`b|c<d>e[f]g\\h"), "a&#96;b&#124;c&lt;d&gt;e&#91;f&#93;g&#92;h")
        self.assertEqual(_md_text("a|b", table=True), "a\\|b")
        self.assertEqual(_md_text("a\nb\rc\td\x00e‮f"), "a b c d e f")
        self.assertEqual(_md_text("p < 0.001 ≥ → α"), "p &lt; 0.001 ≥ → α")
        self.assertEqual(_md_text(12), "12")

    def test_md_text_leaves_nothing_a_parser_could_act_on(self):
        from evalsprint.compare import _md_text

        for text in HOSTILE_IDS:
            with self.subTest(text=text[:30]):
                out = _md_text(text)
                self.assertFalse(set(out) & set("`<>[]\\|"), out)
                self.assertEqual(len(out.splitlines()), 1)
                self.assertEqual(outside_code(f"- {out}"), f"- {out}")  # it cannot open a code span

    def test_a_backslash_in_an_id_cannot_unescape_the_pipe_that_follows(self):
        from evalsprint.compare import _cell

        # the pipe is cut out of the span and written as a reference: no `\|` (which a table unescapes even inside a code span) is ever built
        self.assertEqual(_cell("a\\|b"), "`a\\`&#124;`b`")
        self.assertEqual(unescaped_pipes(f"| {_cell('a' + chr(92) + '|b')} | 4 |"), 3)
        self.assertNotIn("\\|", _cell("a\\|b\\\\|c"))
        self.assertEqual(shown(_cell("a\\|b")), "a\\|b")


class NewGateKeyValidation(unittest.TestCase):
    def test_bool_keys_must_be_booleans(self):
        for key in ("allow_repeat_change", "require_same_kit_version"):
            for bad in ("yes", 1, None):
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, f"gate.{key} must be true or false"):
                    resolve_gate({}, {key: bad})
            self.assertIs(resolve_gate({}, {key: True})[key], True)

    def test_tag_drop_warning_pp_is_a_percentage_above_zero(self):
        for bad in (0, -5, 100.5, "20", True, float("nan")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "gate.tag_drop_warning_pp"):
                resolve_gate({}, {"tag_drop_warning_pp": bad})
        for good in (None, 0.5, 20, 100):
            self.assertEqual(resolve_gate({}, {"tag_drop_warning_pp": good})["tag_drop_warning_pp"], good)

    def test_tag_drop_min_cases_is_a_positive_integer(self):
        for bad in (0, -1, 2.5, "10", True, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "gate.tag_drop_min_cases must be a positive integer"):
                resolve_gate({}, {"tag_drop_min_cases": bad})
        self.assertEqual(resolve_gate({}, {"tag_drop_min_cases": 1})["tag_drop_min_cases"], 1)

    def test_the_old_auto_mode_names_are_not_gate_values(self):
        for old in ("auto-small", "auto-large"):
            with self.assertRaisesRegex(ValueError, "must be 'auto'"):
                resolve_gate({}, {"max_pass_rate_drop": old})
        self.assertEqual(resolve_gate({}, {"max_pass_rate_drop": " AUTO "})["max_pass_rate_drop"], "auto")

    def test_the_summary_gate_cannot_weaken_the_new_rules_either(self):
        # a PR's own summary gate is ignored when a trusted gate is passed
        base = make_run("b", build(flags(100)), repeat={"n": 1, "tags": None})
        cand = make_run("c", build(flags(100)), gate={"allow_repeat_change": True}, repeat={"n": 3, "tags": None})
        self.assertEqual(compare_runs(base, cand, gate={})["verdict"], "regression")
        self.assertEqual(compare_runs(base, cand)["verdict"], "pass")  # no trusted gate: the summary's own gate is used (documented)


# ---------------------------------------------------------------- loop-back 1, C22: target_retries is part of the compared protocol


def retried(records: list[dict], k: int, used: int = 1) -> list[dict]:
    """The record key the runner always writes: 0 everywhere, except that the first ``k`` records needed ``used`` target retries each."""
    for i, rec in enumerate(records):
        rec["target_retries_used"] = used if i < k else 0
    return records


def retry_compare(base_cfg=1, cand_cfg=1, base_used=0, cand_used=0, n=100, gate=None, **cand_extra):
    base = make_run("base", retried(build(flags(n)), base_used), target_retries=base_cfg)
    cand = make_run("cand", retried(build(flags(n)), cand_used), target_retries=cand_cfg, **cand_extra)
    return compare_runs(base, cand, gate=gate)


class TargetRetries(unittest.TestCase):
    def test_the_same_value_is_unchanged_and_silent(self):
        c = retry_compare()
        self.assertIs(c["target_retries_changed"], False)
        self.assertEqual(c["verdict"], "pass")
        self.assertEqual([w for w in c["warnings"] if "retr" in w], [])
        self.assertEqual(c["target_retries"]["changed"], False)
        self.assertEqual(c["target_retries"]["baseline"]["configured"], 1)
        self.assertEqual(c["target_retries"]["candidate"]["configured"], 1)

    def test_a_different_value_blocks_in_both_directions(self):
        for base_cfg, cand_cfg in ((1, 3), (3, 1), (0, 1), (1, 0)):
            with self.subTest(base=base_cfg, cand=cand_cfg):
                c = retry_compare(base_cfg, cand_cfg)
                self.assertIs(c["target_retries_changed"], True)
                self.assertEqual(c["verdict"], "regression")
                self.assertEqual(len(c["reasons"]), 1)
                self.assertIn(f"target_retries: baseline {base_cfg}, candidate {cand_cfg}", c["reasons"][0])
                self.assertIn("update the baseline in a reviewed PR", c["reasons"][0])
                self.assertTrue(c["target_retries"]["changed"])

    def test_the_allow_flag_turns_the_reason_into_a_warning_that_says_so(self):
        c = retry_compare(1, 3, gate={"allow_target_retries_change": True})
        self.assertEqual((c["verdict"], c["reasons"]), ("pass", []))
        self.assertIs(c["target_retries_changed"], True)  # the fact is still reported
        (warning,) = warning_with(c, "target retries changed")
        self.assertIn("allowed by gate.allow_target_retries_change", warning)
        self.assertIn("baseline 1, candidate 3", warning)

    def test_the_pr_cannot_allow_it_in_its_own_summary(self):
        # a trusted gate is passed: the candidate summary's own gate block is ignored (as for allow_repeat_change)
        base = make_run("base", build(flags(100)), target_retries=1)
        cand = make_run("cand", build(flags(100)), gate={"allow_target_retries_change": True}, target_retries=3)
        self.assertEqual(compare_runs(base, cand, gate={})["verdict"], "regression")
        self.assertEqual(compare_runs(base, cand)["verdict"], "pass")  # no trusted gate: the summary's own gate is used (documented)

    def test_a_summary_without_the_field_warns_and_does_not_block(self):
        for which, base_cfg, cand_cfg, who in (
            ("baseline", None, 1, "baseline summary does not record"),
            ("candidate", 1, None, "candidate summary does not record"),
            ("both", None, None, "baseline and candidate summaries do not record"),
        ):
            with self.subTest(missing=which):
                c = retry_compare(base_cfg, cand_cfg)
                self.assertEqual((c["verdict"], c["reasons"]), ("pass", []))
                self.assertIsNone(c["target_retries_changed"])
                (warning,) = warning_with(c, "target_retries")
                self.assertIn(who, warning)
                self.assertIn("re-create the baseline", warning)
                self.assertIsNone(c["target_retries"][which if which != "both" else "baseline"]["configured"])

    def test_a_summary_written_before_the_field_existed_has_no_key_at_all(self):
        base = RunResult(run_dir=None, summary={"run_id": "b", "judges_sha": JUDGES_SHA}, results=build(flags(100)))
        cand = make_run("cand", build(flags(100)), target_retries=1)
        c = compare_runs(base, cand)
        self.assertIsNone(c["target_retries_changed"])
        self.assertEqual(len(warning_with(c, "target_retries")), 1)

    def test_a_forged_value_is_not_a_recorded_value(self):
        for bad in ("3", 3.0, True, False, -1, [], {}):
            with self.subTest(bad=bad):
                c = retry_compare(1, bad)
                self.assertIsNone(c["target_retries_changed"])
                self.assertEqual(c["verdict"], "pass")
                self.assertEqual(len(warning_with(c, "does not record target_retries")), 1)

    def test_the_gate_json_block(self):
        c = retry_compare(1, 3, base_used=5, cand_used=20)
        self.assertEqual(
            c["target_retries"],
            {
                "baseline": {"configured": 1, "retried_calls": 5, "attempts": 100, "share": 0.05},
                "candidate": {"configured": 3, "retried_calls": 20, "attempts": 100, "share": 0.2},
                "changed": True,
            },
        )
        self.assertEqual(json.loads(json.dumps(c))["target_retries"], c["target_retries"])
        self.assertIs(retry_compare()["target_retries"]["changed"], False)
        self.assertEqual(retry_compare()["target_retries"]["baseline"]["retried_calls"], 0)  # the runner writes 0 on every record
        bare = compare_runs(make_run("b", build(flags(100))), make_run("c", build(flags(100))))
        self.assertIsNone(bare["target_retries"]["baseline"]["retried_calls"])  # no record carries the count, no summary total: unknown, not 0

    def test_the_count_comes_from_the_records_else_from_the_summary_total(self):
        base = make_run("base", build(flags(100)), target_retries=1, n_target_retries=4)
        cand = make_run("cand", build(flags(100)), target_retries=1, n_target_retries=9)
        c = compare_runs(base, cand)
        self.assertEqual((c["target_retries"]["baseline"]["retried_calls"], c["target_retries"]["candidate"]["retried_calls"]), (4, 9))
        self.assertAlmostEqual(c["target_retries"]["candidate"]["share"], 0.09)
        records = make_run("cand", retried(build(flags(100)), 12), target_retries=1, n_target_retries=9)  # the records win
        self.assertEqual(compare_runs(base, records)["target_retries"]["candidate"]["retried_calls"], 12)
        for bad in ("4", -1, True, None):
            forged = make_run("cand", build(flags(100)), target_retries=1, n_target_retries=bad)
            self.assertIsNone(compare_runs(base, forged)["target_retries"]["candidate"]["retried_calls"])

    def test_retries_are_counted_per_attempt_so_the_denominator_is_the_attempts(self):
        attempts = lambda i: ["pass", "pass", "pass"]  # noqa: E731
        base = make_run("base", build(flags(40), attempts=attempts), target_retries=1, n_target_retries=0)
        cand_records = retried(build(flags(40), attempts=attempts), 40, used=1)
        cand = make_run("cand", cand_records, target_retries=1)
        c = compare_runs(base, cand)
        self.assertEqual(c["target_retries"]["candidate"]["attempts"], 120)
        self.assertAlmostEqual(c["target_retries"]["candidate"]["share"], 40 / 120)

    # ---- the retry share warning

    def test_the_share_warning_fires_at_the_limit_not_below_it(self):
        self.assertEqual(DEFAULT_GATE["retry_share_warn"], 0.10)
        below = retry_compare(base_used=5, cand_used=14)  # +9 p.p.
        self.assertEqual(warning_with(below, "retried its target"), [])
        at = retry_compare(base_used=5, cand_used=15)  # +10 p.p.: at the limit counts
        (warning,) = warning_with(at, "retried its target")
        self.assertEqual(at["verdict"], "pass")  # a warning, never a reason
        self.assertIn("15 retries in 100 calls (15.0% of the calls) against 5.0% in the baseline (+10.0 p.p., limit 10.0)", warning)
        self.assertIn("check the crash counts", warning)

    def test_fewer_retries_than_the_baseline_never_warn(self):
        self.assertEqual(warning_with(retry_compare(base_used=40, cand_used=2), "retried its target"), [])

    def test_the_limit_is_a_gate_setting_and_none_turns_it_off(self):
        self.assertEqual(len(warning_with(retry_compare(base_used=0, cand_used=6, gate={"retry_share_warn": 0.05}), "retried its target")), 1)
        self.assertEqual(warning_with(retry_compare(base_used=0, cand_used=60, gate={"retry_share_warn": None}), "retried its target"), [])
        self.assertEqual(warning_with(retry_compare(base_used=0, cand_used=60, gate={"retry_share_warn": 1}), "retried its target"), [])  # a share of 100%: never

    def test_the_share_warning_needs_both_counts(self):
        base = make_run("base", build(flags(100)), target_retries=1)  # no count anywhere
        cand = make_run("cand", retried(build(flags(100)), 50), target_retries=1)
        self.assertEqual(warning_with(compare_runs(base, cand), "retried its target"), [])

    def test_a_retry_hides_a_crash_so_the_warning_does_not_depend_on_the_pass_rate(self):
        # the crashes were all absorbed by the retry: same pass rate, 30 p.p. more retries
        c = retry_compare(base_used=0, cand_used=30)
        self.assertEqual(c["pass_rate"]["delta"], 0.0)
        self.assertEqual(len(warning_with(c, "retried its target")), 1)

    def test_markdown_shows_the_line_when_either_run_retried_or_the_values_differ(self):
        self.assertNotIn("Target retries", to_markdown(retry_compare()))  # nothing retried, nothing changed
        md = to_markdown(retry_compare(base_used=5, cand_used=20))
        self.assertIn("↻ Target retries: baseline `target_retries: 1`, 5 retries after a crash in 100 calls (5.0%); candidate `target_retries: 1`, "
                      "20 retries after a crash in 100 calls (20.0%).", md)
        self.assertIn("hides a transient crash from the pass rate and from the error rate", md)
        changed = to_markdown(retry_compare(1, 3))
        self.assertIn("baseline `target_retries: 1`, 0 retries after a crash in 100 calls (0.0%); candidate `target_retries: 3`, 0 retries after a crash in 100 calls (0.0%).", changed)
        self.assertIn("- target retries changed (", changed)  # the reason, in the regression list
        one = to_markdown(retry_compare(base_used=1, cand_used=1))
        self.assertIn("1 retry after a crash in 100 calls (1.0%)", one)
        missing = to_markdown(retry_compare(None, 1, cand_used=3))
        self.assertIn("baseline `target_retries` not recorded, 0 retries after a crash in 100 calls (0.0%); candidate `target_retries: 1`, 3 retries after a crash in 100 calls (3.0%)", missing)

    def test_the_gate_line_names_the_rule_only_while_it_is_on_and_both_runs_recorded_the_value(self):
        self.assertIn("· target retries unchanged</sub>", to_markdown(retry_compare()))
        self.assertNotIn("target retries unchanged", to_markdown(retry_compare(gate={"allow_target_retries_change": True})))
        self.assertNotIn("target retries unchanged", to_markdown(retry_compare(None, 1)))  # the rule cannot be evaluated

    def test_a_comparison_file_written_before_the_field_still_renders(self):
        old = retry_compare()
        for key in ("target_retries", "target_retries_changed"):
            old.pop(key)
        md = to_markdown(old)
        self.assertNotIn("Target retries", md)
        self.assertNotIn("target retries unchanged", md)

    def test_the_new_gate_keys_are_validated(self):
        for bad in ("yes", 1, None):
            with self.subTest(allow=bad), self.assertRaisesRegex(ValueError, "gate.allow_target_retries_change must be true or false"):
                resolve_gate({}, {"allow_target_retries_change": bad})
        self.assertIs(resolve_gate({}, {"allow_target_retries_change": True})["allow_target_retries_change"], True)
        for bad in (3, -0.1, 1.5, "0.1", True):
            with self.subTest(share=bad), self.assertRaises(ValueError):
                resolve_gate({}, {"retry_share_warn": bad})
        for good in (None, 0, 0.05, 1):
            self.assertEqual(resolve_gate({}, {"retry_share_warn": good})["retry_share_warn"], good)
        self.assertEqual(resolve_gate({}, None)["retry_share_warn"], 0.10)

    def test_the_other_verdicts_do_not_move(self):
        plain = compare(BASE_100, drop_by(6))
        with_retries = compare(BASE_100, drop_by(6), base_summary={"target_retries": 1}, cand_summary={"target_retries": 1})
        self.assertEqual((with_retries["verdict"], with_retries["reasons"], with_retries["warnings"]), (plain["verdict"], plain["reasons"], plain["warnings"]))


if __name__ == "__main__":
    unittest.main()
