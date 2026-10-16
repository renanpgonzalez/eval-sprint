"""Tests for ``compare.expected_gate_rates_protocol``: the exact false-alarm and detection rates of the WHOLE gate for a suite whose
cases ran a different number of times and whose critical cases have their own rule (cycle 4, loop-back 1, contract C21).

Three independent readings of the same numbers:

* the methodology critic's exact figures (``cycle4/critic-methodology/gate_common.py``, numpy 2-D convolution), pasted below as
  constants and compared to 1e-9;
* a brute force over every outcome of a tiny suite, with each case's probabilities computed by enumerating its attempts (no
  ``majority_noise``, no closed form), which also covers the even number of attempts and the committed-baseline lottery;
* ``expected_gate_rates`` (the homogeneous function of cycle 3), which the protocol function must equal when every case ran once.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
import unittest
from unittest import mock

import evalsprint.compare as compare_module
from evalsprint.compare import expected_gate_rates, expected_gate_rates_protocol
from evalsprint.stats import majority_noise

from tests.test_gate_rule import contract_trips

# ---------------------------------------------------------------- the critic's figures
# 45 cases, 8 of them critical, 5% noise per attempt, a healthy suite (every case passes).  Key: (attempts of the 37 others, attempts of the
# 8 critical cases).  Value: (whole gate, pass-rate rule, critical-case rule), printed by the critic's gate_common.py
# (`gate_fa_with_critical`, `fa_groups`, `1 - (1 - (1 - qc) * qc) ** 8`) at full precision.
CRITIC = {
    (1, 1): (0.3667000329315089, 0.10574436534944064, 0.32248350241802226),  # repeat: {n: 1}
    (3, 3): (0.05785905470494246, 0.003182887690722241, 0.05614970582337564),  # repeat: {n: 3}
    (5, 5): (0.009227914501312218, 2.018578788159688e-05, 0.009216888369955933),  # repeat: {n: 5}
    (3, 5): (0.011095595535370317, 0.0020963793212607153, 0.009216888369955933),  # repeat: {n: 3, per_tag: {critical: 5}}
    (1, 5): (0.09366671370701374, 0.08629380159575457, 0.009216888369955933),  # repeat: {n: 1, per_tag: {critical: 5}}
}


def close(test: unittest.TestCase, a, b, delta: float = 1e-12, path: str = "") -> None:
    """``a`` equals ``b`` leaf by leaf, floats to ``delta`` (summing the same terms in another order moves the last digit)."""
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b), path)
        for key in a:
            close(test, a[key], b[key], delta, f"{path}/{key}")
    elif isinstance(a, list):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            close(test, x, y, delta, f"{path}[{i}]")
    elif isinstance(a, float):
        test.assertAlmostEqual(a, b, delta=delta, msg=path)
    else:
        test.assertEqual(a, b, path)


def protocol(n_other: int, k_other: int, n_critical: int, k_critical: int):
    return [k_other] * n_other + [k_critical] * n_critical, [False] * n_other + [True] * n_critical


class CriticFigures(unittest.TestCase):
    def test_every_protocol_equals_the_critics_exact_numbers_to_1e_9(self):
        for (k_other, k_critical), (whole, pass_rule, critical_rule) in CRITIC.items():
            with self.subTest(other=k_other, critical=k_critical):
                attempts, critical = protocol(37, k_other, 8, k_critical)
                fa = expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]
                self.assertAlmostEqual(fa["whole_gate"], whole, delta=1e-9)
                self.assertAlmostEqual(fa["pass_rate_rule"], pass_rule, delta=1e-9)
                self.assertAlmostEqual(fa["critical_rule"], critical_rule, delta=1e-9)

    def test_the_percentages_the_docs_quote(self):
        shown = {}
        for key in ((1, 1), (3, 3), (1, 5), (3, 5)):
            attempts, critical = protocol(37, *key[:1], 8, key[1])
            shown[key] = round(100 * expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]["whole_gate"], 2)
        self.assertEqual(shown, {(1, 1): 36.67, (3, 3): 5.79, (1, 5): 9.37, (3, 5): 1.11})
        for key, want in (((1, 5), 8.63), ((3, 5), 0.21)):
            attempts, critical = protocol(37, key[0], 8, key[1])
            self.assertEqual(round(100 * expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]["pass_rate_rule"], 2), want)

    def test_the_order_of_the_cases_does_not_matter(self):
        attempts, critical = protocol(37, 3, 8, 5)
        pairs = list(zip(attempts, critical))
        random.Random(7).shuffle(pairs)
        a2, c2 = [p[0] for p in pairs], [p[1] for p in pairs]
        first, second = expected_gate_rates_protocol(attempts, critical, 0.05), expected_gate_rates_protocol(a2, c2, 0.05)
        close(self, first, second)

    def test_the_whole_gate_sits_between_its_two_rules_and_their_sum(self):
        for key in CRITIC:
            attempts, critical = protocol(37, key[0], 8, key[1])
            fa = expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]
            self.assertGreaterEqual(fa["whole_gate"], max(fa["pass_rate_rule"], fa["critical_rule"]) - 1e-15)
            self.assertLessEqual(fa["whole_gate"], fa["pass_rate_rule"] + fa["critical_rule"] + 1e-15)

    def test_more_attempts_on_the_critical_cases_lower_every_false_alarm(self):
        rates = []
        for k_critical in (1, 3, 5, 7):
            attempts, critical = protocol(37, 3, 8, k_critical)
            rates.append(expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"])
        for a, b in zip(rates, rates[1:]):
            self.assertGreater(a["critical_rule"], b["critical_rule"])
            self.assertGreater(a["whole_gate"], b["whole_gate"])


class Shape(unittest.TestCase):
    def setUp(self):
        attempts, critical = protocol(37, 3, 8, 5)
        self.r = expected_gate_rates_protocol(attempts, critical, 0.05)

    def test_the_contract_keys(self):
        r = self.r
        self.assertEqual({"n", "verdict_noise_mean", "false_alarm", "detection", "baseline_lottery"} - set(r), set())
        self.assertEqual(r["n"], 45)
        self.assertEqual(set(r["false_alarm"]), {"pass_rate_rule", "critical_rule", "whole_gate"})
        self.assertEqual(set(r["detection"]), {0.05, 0.10})
        for entry in r["detection"].values():
            self.assertEqual({"pass_rate_rule", "critical_rule", "whole_gate"} - set(entry), set())
        self.assertEqual({"p_fa_over_15", "median_fa"} - set(r["baseline_lottery"]), set())
        for value in (*r["false_alarm"].values(), *r["baseline_lottery"].values(), *(v for e in r["detection"].values() for v in e.values())):
            self.assertTrue(0.0 <= value <= 1.0, value)

    def test_the_verdict_noise_is_the_mean_over_the_cases_of_their_majority_noise(self):
        want = (37 * majority_noise(0.05, 3) + 8 * majority_noise(0.05, 5)) / 45
        self.assertAlmostEqual(self.r["verdict_noise_mean"], want, places=15)

    def test_it_is_json_serialisable(self):
        back = json.loads(json.dumps(self.r))
        self.assertEqual(back["false_alarm"], self.r["false_alarm"])
        self.assertEqual(set(back["detection"]), {"0.05", "0.1"})

    def test_a_real_drop_is_detected_more_than_noise_alarms(self):
        for drop, entry in self.r["detection"].items():
            self.assertGreater(entry["whole_gate"], self.r["false_alarm"]["whole_gate"], drop)
            self.assertGreaterEqual(entry["whole_gate"], entry["pass_rate_rule"])
        self.assertGreater(self.r["detection"][0.10]["whole_gate"], self.r["detection"][0.05]["whole_gate"])

    def test_the_groups_are_reported(self):
        self.assertEqual(
            sorted((g["attempts"], g["critical"], g["cases"]) for g in self.r["groups"]),
            [(3, False, 37), (5, True, 8)],
        )
        self.assertEqual(self.r["n_critical"], 8)

    def test_a_drop_above_the_baseline_pass_rate_is_none(self):
        attempts, critical = protocol(37, 3, 8, 5)
        r = expected_gate_rates_protocol(attempts, critical, 0.05, baseline_pass=0.05, drops=(0.10,))
        self.assertIsNone(r["detection"][0.10])


# ---------------------------------------------------------------- brute force over a tiny suite


def case_cells(k: int, noise: float, pi: float, d: float) -> tuple[float, float]:
    """(P(pass -> fail), P(fail -> pass)) of one case run ``k`` times in each of two runs, by enumerating every attempt outcome.

    A latent pass reads pass with probability 1 - noise on each attempt, a latent fail with probability noise; the verdict is a pass
    iff a strict majority of the attempts read pass (a tie is a fail).  Latent state: pass in both runs with probability pi - d,
    regressed (pass in the baseline, fail in the candidate) with probability d, fail in both with probability 1 - pi.
    """

    def verdict_pass(latent_pass: bool) -> float:
        total = 0.0
        for reads in itertools.product((0, 1), repeat=k):
            prob = math.prod((1 - noise) if bool(r) == latent_pass else noise for r in reads)
            if sum(reads) > k / 2:
                total += prob
        return total

    vp, vf = verdict_pass(True), verdict_pass(False)
    p_b = (pi - d) * vp * (1 - vp) + d * vp * (1 - vf) + (1 - pi) * vf * (1 - vf)
    p_c = (pi - d) * (1 - vp) * vp + d * (1 - vp) * vf + (1 - pi) * (1 - vf) * vf
    return p_b, p_c


def brute_force_rates(attempts, critical, noise, pi, d) -> dict:
    """Every (pass -> fail, fail -> pass, unchanged) outcome of every case: 3 ** n outcomes, the contract's rule on each."""
    n = len(attempts)
    cells = [case_cells(k, noise, pi, d) for k in attempts]
    pass_rule = critical_rule = whole = 0.0
    for outcome in itertools.product((0, 1, 2), repeat=n):  # 0: pass -> fail, 1: fail -> pass, 2: unchanged
        prob = math.prod(c[0] if o == 0 else c[1] if o == 1 else 1 - c[0] - c[1] for o, c in zip(outcome, cells))
        b, c = outcome.count(0), outcome.count(1)
        crit_flip = any(o == 0 and is_critical for o, is_critical in zip(outcome, critical))
        trips = contract_trips(b, c, n)
        pass_rule += prob * trips
        critical_rule += prob * crit_flip
        whole += prob * (trips or crit_flip)
    return {"pass_rate_rule": pass_rule, "critical_rule": critical_rule, "whole_gate": whole}


class BruteForce(unittest.TestCase):
    SUITES = {
        "mixed odd": ([1, 1, 3, 3, 5, 5], [False, False, False, True, True, True]),
        "even attempts": ([2, 2, 4, 1, 3, 2], [False, True, False, True, False, False]),
        "no critical case": ([1, 3, 3, 2, 1, 5], [False] * 6),
        "all critical": ([3, 3, 1, 1, 2, 4], [True] * 6),
        "a single group": ([2] * 6, [False, False, False, False, True, True]),
    }

    def test_false_alarm_and_detection_equal_the_enumeration_of_every_outcome(self):
        for name, (attempts, critical) in self.SUITES.items():
            for noise in (0.05, 0.2):
                with self.subTest(suite=name, noise=noise):
                    got = expected_gate_rates_protocol(attempts, critical, noise, baseline_pass=0.8, drops=(0.1, 0.3))
                    for d, rates in ((0.0, got["false_alarm"]), (0.1, got["detection"][0.1]), (0.3, got["detection"][0.3])):
                        want = brute_force_rates(attempts, critical, noise, 0.8, d)
                        self.assertAlmostEqual(rates["pass_rate_rule"], want["pass_rate_rule"], delta=1e-12, msg=f"pass rate rule, drop {d}")
                        self.assertAlmostEqual(rates["whole_gate"], want["whole_gate"], delta=1e-12, msg=f"whole gate, drop {d}")
                        if d == 0.0:
                            self.assertAlmostEqual(rates["critical_rule"], want["critical_rule"], delta=1e-12)

    def test_the_critical_rule_of_a_real_drop(self):
        attempts, critical = [1, 3, 3, 5, 2, 1], [False, False, True, True, True, False]
        got = expected_gate_rates_protocol(attempts, critical, 0.1, drops=(0.2,))
        want = brute_force_rates(attempts, critical, 0.1, 0.8, 0.2)
        self.assertAlmostEqual(got["detection"][0.2]["whole_gate"], want["whole_gate"], delta=1e-12)

    def test_the_committed_baseline_lottery_equals_the_enumeration_of_every_baseline(self):
        # 6 cases, 4 latent passes (2 in each attempt group), noise 0.3 so that the false alarm has real spread
        for attempts in ([1, 1, 1, 3, 3, 3], [1, 1, 1, 2, 2, 2], [2, 2, 2, 4, 4, 4]):
            with self.subTest(attempts=attempts):
                noise = 0.3
                got = expected_gate_rates_protocol(attempts, [False] * 6, noise, baseline_pass=4 / 6)["baseline_lottery"]
                latent = [True, True, False, True, True, False]
                p_pass = [case_verdict_pass(k, noise, lp) for k, lp in zip(attempts, latent)]
                draws = []
                for base in itertools.product((0, 1), repeat=6):  # the baseline verdicts: 1 = pass
                    weight = math.prod(p if v else 1 - p for v, p in zip(base, p_pass))
                    fa = 0.0
                    for cand in itertools.product((0, 1), repeat=6):
                        w = math.prod(p if v else 1 - p for v, p in zip(cand, p_pass))
                        b = sum(1 for x, y in zip(base, cand) if x == 1 and y == 0)
                        c = sum(1 for x, y in zip(base, cand) if x == 0 and y == 1)
                        fa += w * contract_trips(b, c, 6)
                    draws.append((fa, weight))
                draws.sort()
                total = sum(w for _, w in draws)
                self.assertAlmostEqual(total, 1.0, places=12)

                def quantile(q):
                    acc = 0.0
                    for fa, w in draws:
                        acc += w
                        if acc >= q * total:
                            return fa
                    return draws[-1][0]

                self.assertAlmostEqual(got["mean_fa"], sum(fa * w for fa, w in draws), delta=1e-9)
                self.assertAlmostEqual(got["median_fa"], quantile(0.5), delta=1e-9)
                self.assertAlmostEqual(got["p_fa_over_15"], sum(w for fa, w in draws if fa > 0.15), delta=1e-9)
                self.assertAlmostEqual(got["p_fa_over_25"], sum(w for fa, w in draws if fa > 0.25), delta=1e-9)
                self.assertAlmostEqual(got["p95_fa"], quantile(0.95), delta=1e-9)
                self.assertGreater(got["p_fa_over_15"], 0.01)  # the spread is real, so this is not a comparison of zeros


def case_verdict_pass(k: int, noise: float, latent_pass: bool) -> float:
    total = 0.0
    for reads in itertools.product((0, 1), repeat=k):
        if sum(reads) > k / 2:
            total += math.prod((1 - noise) if bool(r) == latent_pass else noise for r in reads)
    return total


class EvenAttempts(unittest.TestCase):
    """A tie is a fail, so with an even number of attempts a latent pass is lost more often than a latent fail is gained."""

    def test_two_attempts_are_priced_exactly_not_with_the_mean_of_the_two_directions(self):
        exact = expected_gate_rates_protocol([2] * 45, [False] * 45, 0.05)["false_alarm"]["pass_rate_rule"]
        with_the_mean = expected_gate_rates(45, majority_noise(0.05, 2))["false_alarm"]  # majority_noise(0.05, 2) = 0.05
        self.assertAlmostEqual(exact, 0.13880640121629448, delta=1e-9)
        self.assertAlmostEqual(with_the_mean, 0.10574436534943936, delta=1e-9)
        self.assertEqual((round(100 * exact, 1), round(100 * with_the_mean, 1)), (13.9, 10.6))  # the figures of the stats docstring
        self.assertGreater(exact, with_the_mean)

    def test_two_attempts_are_noisier_than_one_for_the_gate(self):
        once = expected_gate_rates_protocol([1] * 45, [False] * 45, 0.05)["false_alarm"]["pass_rate_rule"]
        twice = expected_gate_rates_protocol([2] * 45, [False] * 45, 0.05)["false_alarm"]["pass_rate_rule"]
        thrice = expected_gate_rates_protocol([3] * 45, [False] * 45, 0.05)["false_alarm"]["pass_rate_rule"]
        self.assertGreater(twice, once)  # a tie fails the case: repeating twice hurts
        self.assertLess(thrice, once)  # the docs recommend an odd number


class Homogeneous(unittest.TestCase):
    def test_every_case_run_once_equals_expected_gate_rates(self):
        for n in (45, 100, 200):
            for noise in (0.0, 0.02, 0.05, 0.10):
                with self.subTest(n=n, noise=noise):
                    want = expected_gate_rates(n, noise)
                    got = expected_gate_rates_protocol([1] * n, [False] * n, noise)
                    self.assertAlmostEqual(got["false_alarm"]["pass_rate_rule"], want["false_alarm"], delta=1e-12)
                    self.assertEqual(got["false_alarm"]["critical_rule"], 0.0)
                    self.assertAlmostEqual(got["false_alarm"]["whole_gate"], want["false_alarm"], delta=1e-12)
                    for d in (0.05, 0.10):
                        self.assertAlmostEqual(got["detection"][d]["pass_rate_rule"], want["detection"][d], delta=1e-12)
                        self.assertAlmostEqual(got["detection"][d]["whole_gate"], want["detection"][d], delta=1e-12)
                    for key, value in want["baseline_lottery"].items():
                        self.assertAlmostEqual(got["baseline_lottery"][key], value, delta=1e-12, msg=key)

    def test_a_homogeneous_odd_protocol_equals_expected_gate_rates_at_its_verdict_noise(self):
        for k in (3, 5):
            want = expected_gate_rates(100, majority_noise(0.05, k))
            got = expected_gate_rates_protocol([k] * 100, [False] * 100, 0.05)
            self.assertAlmostEqual(got["false_alarm"]["pass_rate_rule"], want["false_alarm"], delta=1e-12)
            self.assertAlmostEqual(got["detection"][0.10]["pass_rate_rule"], want["detection"][0.10], delta=1e-12)

    def test_the_critic_reference_points_of_the_homogeneous_function(self):
        for n, want in ((45, 10.57), (100, 7.05), (200, 4.01)):
            self.assertEqual(round(100 * expected_gate_rates_protocol([1] * n, [False] * n, 0.05)["false_alarm"]["whole_gate"], 2), want)


class ExactCriticalRule(unittest.TestCase):
    def test_without_noise_a_critical_case_trips_the_gate_when_it_regresses(self):
        # noise 0: a case flips pass -> fail exactly when it regressed (probability d), so P(some critical case regressed) = 1 - (1 - d) ** m
        attempts, critical = protocol(37, 3, 8, 5)
        r = expected_gate_rates_protocol(attempts, critical, 0.0, drops=(0.05, 0.10))
        self.assertEqual(r["false_alarm"], {"pass_rate_rule": 0.0, "critical_rule": 0.0, "whole_gate": 0.0})
        for d in (0.05, 0.10):
            self.assertAlmostEqual(r["detection"][d]["critical_rule"], 1 - (1 - d) ** 8, delta=1e-12)
            self.assertGreaterEqual(r["detection"][d]["whole_gate"], r["detection"][d]["critical_rule"])
        self.assertEqual(r["baseline_lottery"]["p_fa_over_15"], 0.0)

    def test_the_critical_rule_alone_is_the_closed_form(self):
        for k in (1, 3, 5):
            q = majority_noise(0.05, k)
            attempts, critical = protocol(37, 3, 8, k)
            got = expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]["critical_rule"]
            self.assertAlmostEqual(got, 1 - (1 - (1 - q) * q) ** 8, delta=1e-12)


class GateSettings(unittest.TestCase):
    def test_with_the_pass_rate_rule_off_the_whole_gate_is_the_critical_rule(self):
        attempts, critical = protocol(37, 1, 8, 3)
        r = expected_gate_rates_protocol(attempts, critical, 0.05, gate={"max_pass_rate_drop": None})
        self.assertEqual(r["false_alarm"]["pass_rate_rule"], 0.0)
        self.assertEqual(r["false_alarm"]["whole_gate"], r["false_alarm"]["critical_rule"])
        self.assertGreater(r["false_alarm"]["critical_rule"], 0.0)
        self.assertEqual(r["gate_mode"], "off")
        self.assertEqual(set(r["baseline_lottery"].values()), {0.0})
        for d, entry in r["detection"].items():
            self.assertEqual(entry["pass_rate_rule"], 0.0)
            self.assertGreater(entry["whole_gate"], 0.0)

    def test_a_fixed_limit_is_priced_like_expected_gate_rates_does(self):
        gate = {"max_pass_rate_drop": 0.05}
        want = expected_gate_rates(100, 0.05, gate=gate)
        got = expected_gate_rates_protocol([1] * 100, [False] * 100, 0.05, gate=gate)
        self.assertEqual(got["gate_mode"], "fixed")
        self.assertAlmostEqual(got["false_alarm"]["pass_rate_rule"], want["false_alarm"], delta=1e-12)
        self.assertAlmostEqual(got["detection"][0.10]["pass_rate_rule"], want["detection"][0.10], delta=1e-12)

    def test_the_critical_rule_does_not_depend_on_the_gate_config(self):
        attempts, critical = protocol(37, 3, 8, 5)
        a = expected_gate_rates_protocol(attempts, critical, 0.05)["false_alarm"]["critical_rule"]
        b = expected_gate_rates_protocol(attempts, critical, 0.05, gate={"critical_tags": ["x"], "max_pass_rate_drop": 0.05})["false_alarm"]["critical_rule"]
        self.assertEqual(a, b)

    def test_a_bad_gate_is_rejected_like_compare_does(self):
        with self.assertRaises(ValueError):
            expected_gate_rates_protocol([1] * 10, [False] * 10, 0.05, gate={"max_pass_rate_drop": "sometimes"})


class Validation(unittest.TestCase):
    def test_bad_arguments(self):
        ok = ([3] * 10, [False] * 10)
        bad = [
            dict(attempts=[], critical=[]),
            dict(attempts=[3] * 10, critical=[False] * 9),
            dict(attempts=[3] * 9 + [0], critical=ok[1]),
            dict(attempts=[3] * 9 + [2.0], critical=ok[1]),
            dict(attempts=[3] * 9 + [True], critical=ok[1]),
            dict(attempts=[3] * 9 + [None], critical=ok[1]),
            dict(attempts=ok[0], critical=[False] * 9 + [1]),
            dict(attempts=ok[0], critical=[False] * 9 + [None]),
            dict(attempts=ok[0], critical=ok[1], noise=-0.01),
            dict(attempts=ok[0], critical=ok[1], noise=0.51),
            dict(attempts=ok[0], critical=ok[1], noise=True),
            dict(attempts=ok[0], critical=ok[1], noise="0.05"),
            dict(attempts=ok[0], critical=ok[1], noise=None),
            dict(attempts=ok[0], critical=ok[1], baseline_pass=1.1),
            dict(attempts=ok[0], critical=ok[1], baseline_pass=-0.1),
            dict(attempts=ok[0], critical=ok[1], drops=(1.5,)),
            dict(attempts=ok[0], critical=ok[1], drops=("0.1",)),
        ]
        for kwargs in bad:
            kwargs = {"noise": 0.05, **kwargs}
            with self.subTest(**{k: (v if not isinstance(v, list) else v[-2:]) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                expected_gate_rates_protocol(**kwargs)

    def test_any_sequence_is_accepted(self):
        a = expected_gate_rates_protocol((3,) * 20 + (5,) * 5, tuple([False] * 20 + [True] * 5), 0.05)
        b = expected_gate_rates_protocol([3] * 20 + [5] * 5, [False] * 20 + [True] * 5, 0.05)
        self.assertEqual(a, b)  # the same order of cases: the same floats


class Budget(unittest.TestCase):
    def test_the_lottery_draw_budget_moves_the_figures_by_less_than_a_hundredth_of_a_point(self):
        attempts = [1] * 50 + [3] * 40 + [5] * 10
        critical = [False] * 90 + [True] * 10
        fast = expected_gate_rates_protocol(attempts, critical, 0.05)
        with mock.patch.object(compare_module, "_LOTTERY_DRAWS", 10**9):  # no budget: every draw heavier than 1e-13
            exact = expected_gate_rates_protocol(attempts, critical, 0.05)
        self.assertEqual(fast["false_alarm"], exact["false_alarm"])  # the budget only concerns the lottery
        self.assertEqual(fast["detection"], exact["detection"])
        for key, value in exact["baseline_lottery"].items():
            self.assertAlmostEqual(fast["baseline_lottery"][key], value, delta=1e-4, msg=key)

    def test_a_suite_of_four_hundred_cases_with_three_groups_takes_seconds(self):
        attempts = [1] * 200 + [3] * 160 + [5] * 40
        critical = [False] * 360 + [True] * 40
        start = time.perf_counter()
        r = expected_gate_rates_protocol(attempts, critical, 0.05)
        self.assertLess(time.perf_counter() - start, 30.0)  # about 1 s on the machine that wrote the test
        self.assertTrue(0.0 < r["false_alarm"]["whole_gate"] < 0.5)


if __name__ == "__main__":
    unittest.main()
