"""C9: the `auto` pass-rate rule is ONE rule for every n, exact, and cliff-free.

The rule (evalsprint.compare.auto_trips): the gate trips when the net loss b - c is at least max(3, ceil(n / 50)) cases
AND the one-sided exact McNemar p-value is below alpha(n) = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5).

This file checks it three ways, each independent of the others:

* structure: the shipped rule against a from-scratch reading of the formulas; more evidence never un-trips the gate;
* the acceptance numbers of the cycle-3 contract, by EXACT enumeration under model B (both runs independent noisy
  readings of the same latent verdicts; the enumeration below is written here, from the multinomial, and only asks
  `auto_trips` which (b, c) pairs trip);
* end to end: a seeded Monte Carlo through `compare_runs` lands on the exact false-alarm rate.

The full curve for every n in 30..500 (and the old-vs-new comparison) is in
orchestration/2026-09-29_kit-v0.2/cycle3/eval-scientist-gate/gate_rates.md.

C19 (cycle 4) adds `compare.expected_gate_rates`: the same enumeration as a public function (false alarm, detection of a
real drop, and the committed-baseline lottery).  `ExpectedGateRates` checks it against the enumeration above, against a
brute force over every reading of a tiny suite, and against the numbers the cycle-3 methodology critic computed with its own code.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import unittest

from evalsprint.compare import auto_alpha, auto_min_net_loss, auto_trips, compare_runs, expected_gate_rates
from evalsprint.runner import RunResult

# ------------------------------------------------------------------ an independent reading of the rule


def contract_trips(b: int, c: int, n: int) -> bool:
    floor = max(3, -(-n // 50))
    alpha = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)
    m = b + c
    p = sum(math.comb(m, i) for i in range(b, m + 1)) / 2**m if m else 1.0
    return b - c >= floor and p < alpha


# ------------------------------------------------------------------ exact enumeration (model B)

_LF = [math.lgamma(i + 1) for i in range(1201)]


def _probs(noise: float, d: float) -> tuple[float, float]:
    """P(pass->fail), P(fail->pass) per case: independent reads flip with prob `noise`; a fraction `d` of all cases regressed."""
    return d * (1 - noise) ** 2 + (1 - d) * noise * (1 - noise), d * noise**2 + (1 - d) * noise * (1 - noise)


def _pmf(k: int, m: int, q: float) -> float:
    if k < 0 or k > m:
        return 0.0
    if q <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(_LF[m] - _LF[k] - _LF[m - k] + k * math.log(q) + (m - k) * math.log1p(-q))


_CMAX: dict[int, dict[int, int]] = {}


def _cmax(n: int, b: int) -> int:
    """Largest c with auto_trips(b, c, n), or -1.  auto_trips is monotone (see MoreEvidenceNeverUntrips), so cmax is
    non-decreasing in b and each b only needs to look at the boundary."""
    table = _CMAX.setdefault(n, {})
    if b not in table:
        c = min(_cmax(n, b - 1), n - b) if b > 0 else -1
        if c >= 0 and not auto_trips(b, c, n):  # cannot happen when the region is monotone; stay correct anyway
            c = -1
        while c + 1 <= n - b and auto_trips(b, c + 1, n):
            c += 1
        table[b] = c
    return table[b]


def _window(n: int, q: float, z: float = 12.0) -> range:
    mu, sd = n * q, math.sqrt(max(n * q * (1 - q), 1e-12))
    return range(max(0, int(mu - z * sd - 3)), min(n, int(mu + z * sd + 4)) + 1)


def trip_probability(n: int, noise: float, d: float = 0.0) -> float:
    pb, pc = _probs(noise, d)
    q = pc / (1 - pb)
    total = 0.0
    for b in _window(n, pb):
        pr = _pmf(b, n, pb)
        cm = min(_cmax(n, b), n - b)
        if pr < 1e-18 or cm < 0:
            continue
        m2 = n - b
        term = _pmf(0, m2, q)  # then P(c + 1) = P(c) * (m2 - c) / (c + 1) * q / (1 - q)
        acc = term
        for c in range(cm):
            term *= (m2 - c) / (c + 1) * q / (1 - q)
            acc += term
        total += pr * min(1.0, acc)
    return total


def _full_dist(m: int, noise: float, d: float) -> dict:
    pb, pc = _probs(noise, d)
    q = pc / (1 - pb)
    out = {}
    for b in _window(m, pb, 10.0):
        pr = _pmf(b, m, pb)
        if pr < 1e-15:
            continue
        for c in _window(m - b, q, 10.0):
            p = pr * _pmf(c, m - b, q)
            if p > 1e-15:
                out[(b, c)] = p
    return out


def heterogeneous_trip_probability(n: int, d: float = 0.0) -> float:
    """20% of the cases flip at 20%, the rest at 1% (exact split), by convolving the two classes' (b, c) distributions."""
    high = round(0.2 * n)
    dist = {(0, 0): 1.0}
    for m, noise in ((high, 0.20), (n - high, 0.01)):
        step = _full_dist(m, noise, d)
        nxt: dict = {}
        for (b1, c1), p1 in dist.items():
            for (b2, c2), p2 in step.items():
                nxt[(b1 + b2, c1 + c2)] = nxt.get((b1 + b2, c1 + c2), 0.0) + p1 * p2
        dist = nxt
    return sum(p for (b, c), p in dist.items() if c <= _cmax(n, b))


def drawdown(values: list[float]) -> float:
    """Largest drop below the running maximum, in the units of `values`."""
    run, worst = -1.0, 0.0
    for v in values:
        run = max(run, v)
        worst = max(worst, run - v)
    return worst


# ------------------------------------------------------------------ structure


class RuleMatchesTheFormulas(unittest.TestCase):
    def test_shipped_rule_equals_a_from_scratch_reading(self):
        for n in [*range(1, 61), 75, 99, 100, 101, 105, 106, 149, 150, 151, 199, 200, 201, 250, 251, 300, 400, 500, 501]:
            for b in range(0, min(n, 30) + 1):
                for c in range(0, min(n - b, 30) + 1):
                    self.assertEqual(auto_trips(b, c, n), contract_trips(b, c, n), (n, b, c))

    def test_floor_and_alpha_values(self):
        self.assertEqual([auto_min_net_loss(n) for n in (1, 45, 100, 150, 151, 200, 201, 250, 251, 500)], [3, 3, 3, 3, 4, 4, 5, 5, 6, 10])
        for n, alpha in ((45, 0.2396), (90, 0.1414), (100, 0.1200), (150, 0.0714), (200, 0.0615), (300, 0.0582), (500, 0.0576)):
            self.assertAlmostEqual(auto_alpha(n), alpha, places=3, msg=n)
        alphas = [auto_alpha(n) for n in range(1, 2001)]
        self.assertTrue(all(a >= b for a, b in zip(alphas, alphas[1:])), "alpha(n) must never grow with n")
        self.assertGreater(min(alphas), 0.0575)
        self.assertLess(max(alphas), 0.25)

    def test_nothing_trips_without_paired_cases_or_below_the_floor(self):
        self.assertFalse(auto_trips(0, 0, 0))
        self.assertFalse(auto_trips(5, 0, 0))
        for n in (10, 45, 200, 500):
            k = auto_min_net_loss(n)
            self.assertFalse(auto_trips(k - 1, 0, n), n)  # even a perfectly clean loss of k - 1 cases

    def test_an_all_clean_loss_of_the_floor_trips_until_alpha_shrinks_past_its_p_value(self):
        self.assertTrue(auto_trips(3, 0, 45))  # p = 1/8 < alpha(45)
        self.assertTrue(auto_trips(3, 0, 97))
        self.assertFalse(auto_trips(3, 0, 98))  # alpha(98) = 0.124 < 1/8: one more case needed
        self.assertTrue(auto_trips(4, 0, 150))  # p = 1/16
        self.assertTrue(auto_trips(4, 0, 190))
        self.assertFalse(auto_trips(4, 0, 191))  # alpha(191) = 0.0624 < 1/16
        self.assertTrue(auto_trips(6, 0, 300))  # the floor (2 p.p. = 6 cases) now binds
        self.assertFalse(auto_trips(5, 0, 300))
        self.assertTrue(auto_trips(10, 0, 500))


class MoreEvidenceNeverUntrips(unittest.TestCase):
    """The region is an up-set in b and a down-set in c: one more regression, or one fewer recovery, never clears the gate.
    (Also what the exact enumeration above relies on.)"""

    def test_monotone_in_b_and_c(self):
        for n in (12, 30, 45, 80, 100, 150, 151, 200, 320, 500):
            for b in range(0, min(n, 70)):
                for c in range(0, min(n - b, 50)):
                    if auto_trips(b, c, n):
                        self.assertTrue(auto_trips(b + 1, c, n) or b + c + 1 > n, (n, b, c))
                        if c > 0:
                            self.assertTrue(auto_trips(b, c - 1, n), (n, b, c))

    def test_the_scan_used_by_the_enumeration_equals_brute_force(self):
        for n in (30, 45, 100, 150, 200, 500):
            for b in range(0, min(n, 70) + 1):
                brute = max((c for c in range(0, n - b + 1) if auto_trips(b, c, n)), default=-1)
                self.assertEqual(_cmax(n, b), brute, (n, b))


# ------------------------------------------------------------------ the C9 acceptance numbers, exactly


class AcceptanceNumbers(unittest.TestCase):
    """Model B, independent reads, baseline pass rate 80% (b and c do not depend on it)."""

    def test_false_alarm_at_45_cases_with_5_percent_noise(self):
        self.assertLessEqual(trip_probability(45, 0.05), 0.11)
        self.assertAlmostEqual(trip_probability(45, 0.05), 0.1057, places=3)

    def test_false_alarm_from_150_to_500_cases_with_5_percent_noise(self):
        worst = max(trip_probability(n, 0.05) for n in range(150, 501))
        self.assertLessEqual(worst, 0.05)
        self.assertGreater(worst, 0.04)  # the rule is not merely conservative: it spends about its alpha

    def test_false_alarm_from_150_to_500_cases_stays_within_5_percent_up_to_10_percent_noise(self):
        """Beyond the C9 number (5% noise): the same bound holds for noisier judges, so the curve can be quoted without a noise caveat."""
        for noise in (0.02, 0.075, 0.10):
            self.assertLessEqual(max(trip_probability(n, noise) for n in range(150, 501)), 0.05, noise)

    def test_detection_of_a_10pp_drop(self):
        self.assertGreaterEqual(trip_probability(45, 0.05, 0.10), 0.65)
        self.assertGreaterEqual(trip_probability(200, 0.05, 0.10), 0.90)

    def test_detection_is_monotone_in_n_at_5_and_10_percent_noise(self):
        """detection at n + k is never more than 3 p.p. below detection at n, for every n in 30..500 (no cliff)."""
        for noise in (0.05, 0.10):
            curve = [100 * trip_probability(n, noise, 0.10) for n in range(30, 501)]
            self.assertLessEqual(drawdown(curve), 3.0, f"noise {noise}")

    def test_no_step_at_the_old_switch(self):
        for noise in (0.05, 0.10):
            for d in (0.0, 0.10):
                self.assertLess(abs(trip_probability(150, noise, d) - trip_probability(149, noise, d)), 0.02, (noise, d))
        self.assertGreater(trip_probability(150, 0.05, 0.10), trip_probability(149, 0.05, 0.10) - 0.01)  # v0.1 lost 7 p.p. here

    def test_heterogeneous_noise_at_45_cases(self):
        """20% of the cases flip at 20%, the rest at 1%: false alarm still <= 11%."""
        self.assertLessEqual(heterogeneous_trip_probability(45), 0.11)
        self.assertAlmostEqual(heterogeneous_trip_probability(45), 0.0899, places=3)

    def test_false_alarm_never_exceeds_alpha_whatever_the_noise(self):
        """The exact one-sided test is valid under model B for any noise: b | b + c ~ Bin(b + c, 1/2) under no change."""
        for n in (30, 45, 60, 100, 150, 300, 500):
            for noise in (0.01, 0.05, 0.10, 0.20, 0.35):
                self.assertLessEqual(trip_probability(n, noise), auto_alpha(n) + 1e-12, (n, noise))

    def test_a_real_drop_is_caught_more_often_than_no_change_everywhere(self):
        for n in (30, 60, 100, 150, 250, 500):
            self.assertGreater(trip_probability(n, 0.05, 0.10), trip_probability(n, 0.05) + 0.3, n)

    def test_the_enumeration_matches_a_full_double_sum(self):
        """Guards the enumeration itself: the recurrence-based sums equal a plain multinomial sum over every (b, c)."""
        for n, noise, d in ((12, 0.05, 0.0), (45, 0.05, 0.10), (60, 0.10, 0.0)):
            pb, pc = _probs(noise, d)
            p0 = 1 - pb - pc
            full = sum(
                math.exp(_LF[n] - _LF[b] - _LF[c] - _LF[n - b - c] + b * math.log(pb) + c * math.log(pc) + (n - b - c) * math.log(p0))
                for b in range(n + 1)
                for c in range(n - b + 1)
                if auto_trips(b, c, n)
            )
            self.assertAlmostEqual(trip_probability(n, noise, d), full, places=12)


# ------------------------------------------------------------------ end to end


def _records(prefix: str, reads: list[bool]) -> list[dict]:
    return [
        {
            "case_id": f"{prefix}{i}",
            "case": {"id": f"{prefix}{i}", "tags": []},
            "case_sha": f"sha{i}",
            "outcome": "pass" if ok else "fail",
            "passed": ok,
            "output": {"error": None},
            "judges": [],
        }
        for i, ok in enumerate(reads)
    ]


def monte_carlo_false_alarm(n: int, noise: float, runs: int, seed: int) -> float:
    rng = random.Random(seed)
    latent = [i < int(0.8 * n) for i in range(n)]  # baseline pass rate 80%, no real change
    trips = 0
    for _ in range(runs):
        base = _records("c", [v != (rng.random() < noise) for v in latent])
        cand = _records("c", [v != (rng.random() < noise) for v in latent])
        summary = {"judges_sha": "x", "repeat": {"n": 1, "per_tag": {}}}
        result = compare_runs(
            RunResult(run_dir=None, summary={"run_id": "b", **summary}, results=base),
            RunResult(run_dir=None, summary={"run_id": "c", **summary}, results=cand),
            gate={},
        )
        trips += result["verdict"] == "regression"
    return trips / runs


class EndToEnd(unittest.TestCase):
    def test_compare_runs_false_alarm_rate_matches_the_exact_enumeration(self):
        for n, runs in ((45, 2000), (200, 1000)):
            exact = trip_probability(n, 0.05)
            measured = monte_carlo_false_alarm(n, 0.05, runs, seed=20260929)
            sigma = math.sqrt(exact * (1 - exact) / runs)
            self.assertLess(abs(measured - exact), 4 * sigma, (n, measured, exact))

    def test_the_reported_effective_numbers_are_the_rules(self):
        for n in (10, 45, 100, 150, 200, 500):
            records = _records("c", [True] * n)
            base = RunResult(run_dir=None, summary={"run_id": "b", "judges_sha": "x"}, results=records)
            cand = RunResult(run_dir=None, summary={"run_id": "c", "judges_sha": "x"}, results=_records("c", [True] * n))
            c = compare_runs(base, cand, gate={})
            k = auto_min_net_loss(n)
            self.assertEqual((c["gate_mode"], c["effective_min_net_loss"]), ("auto", k))
            self.assertAlmostEqual(c["effective_alpha"], auto_alpha(n))
            self.assertAlmostEqual(c["effective_max_pass_rate_drop"], (k - 1) / n)


# ------------------------------------------------------------------ C19: compare.expected_gate_rates


def brute_force(n: int, noise: float, baseline_pass: float, trips, d: float = 0.0) -> dict:
    """Every reading of a tiny suite, written from the model and nothing else: no windows, no convolution, no cmax table.

    Latent verdicts: the first round(baseline_pass * n) cases pass.  Each latent-pass case regresses (pass -> fail in the candidate's
    latent verdict) independently with probability d / baseline_pass, i.e. a fraction d of ALL cases.  The baseline and the candidate
    read every case with flip probability `noise`.  Returns the false-alarm / detection rate (over baseline draws) and, for d = 0, the
    distribution of the false-alarm rate given the baseline.
    """
    n_pass = round(baseline_pass * n)
    latent = [i < n_pass for i in range(n)]
    rho = d / baseline_pass if n_pass else 0.0

    def readings(truth):
        for flips in itertools.product((False, True), repeat=n):
            yield [t != f for t, f in zip(truth, flips)], math.prod(noise if f else 1 - noise for f in flips)

    candidate_reads = {}
    for regress in itertools.product((False, True), repeat=n_pass):
        weight = math.prod(rho if r else 1 - rho for r in regress)
        if weight == 0:
            continue
        truth = [(v and not (i < n_pass and regress[i])) for i, v in enumerate(latent)]
        candidate_reads[regress] = (weight, list(readings(truth)))
    draws, total = [], 0.0
    for base, p_base in readings(latent):
        fa = 0.0
        for weight, reads in candidate_reads.values():
            for cand, p_cand in reads:
                b = sum(1 for x, y in zip(base, cand) if x and not y)
                c = sum(1 for x, y in zip(base, cand) if not x and y)
                if trips(b, c, n):
                    fa += weight * p_cand
        draws.append((fa, p_base))
        total += p_base * fa
    return {"rate": total, "draws": sorted(draws)}


def brute_force_iid(n: int, noise: float, baseline_pass: float, trips, d: float) -> float:
    """Detection rate of model B, enumerated case by case with no closed form.

    One case: latent pass with probability baseline_pass (and then regressed with probability d / baseline_pass, so a fraction d of ALL cases
    really regressed) or latent fail; the baseline and the candidate each read the latent verdict and flip it with probability `noise`.
    That gives four (baseline, candidate) outcomes per case; a suite is n independent cases, enumerated outcome by outcome (4 ** n).
    """
    rho = d / baseline_pass if baseline_pass else 0.0
    cell = {(x, y): 0.0 for x in (False, True) for y in (False, True)}
    for base_latent, cand_latent, p_state in (
        (True, True, baseline_pass * (1 - rho)),
        (True, False, baseline_pass * rho),
        (False, False, 1 - baseline_pass),
    ):
        for base_flip, cand_flip in itertools.product((False, True), repeat=2):
            weight = (noise if base_flip else 1 - noise) * (noise if cand_flip else 1 - noise)
            cell[(base_latent != base_flip, cand_latent != cand_flip)] += p_state * weight
    rate = 0.0
    for outcome in itertools.product(cell, repeat=n):
        b = sum(1 for x, y in outcome if x and not y)
        c = sum(1 for x, y in outcome if not x and y)
        if trips(b, c, n):
            rate += math.prod(cell[o] for o in outcome)
    return rate


def quantile(draws, q: float) -> float:
    acc = 0.0
    for fa, w in draws:
        acc += w
        if acc >= q * sum(x for _, x in draws):
            return fa
    return draws[-1][0]


class ExpectedGateRates(unittest.TestCase):
    def test_the_shape_is_the_contract(self):
        r = expected_gate_rates(45, 0.05)
        self.assertEqual({"false_alarm", "detection", "baseline_lottery"} - set(r), set())
        self.assertEqual(set(r["detection"]), {0.05, 0.10})
        self.assertEqual({"p_fa_over_15", "median_fa"} - set(r["baseline_lottery"]), set())
        for value in (r["false_alarm"], *r["detection"].values(), *r["baseline_lottery"].values()):
            self.assertTrue(0.0 <= value <= 1.0, value)
        self.assertEqual((r["n"], r["noise"], r["baseline_pass"], r["gate_mode"]), (45, 0.05, 0.8, "auto"))
        self.assertEqual((r["effective_min_net_loss"], r["effective_max_pass_rate_drop"]), (3, 2 / 45))
        self.assertAlmostEqual(r["effective_alpha"], auto_alpha(45))
        json.dumps(r)  # float keys become strings: the CLI can dump it as it is

    def test_it_equals_the_independent_enumeration_of_this_file(self):
        for n in (12, 30, 45, 100, 150, 151, 200, 300):
            for noise in (0.02, 0.05, 0.10):
                r = expected_gate_rates(n, noise, drops=(0.05, 0.10))
                self.assertAlmostEqual(r["false_alarm"], trip_probability(n, noise), places=9, msg=(n, noise))
                for d in (0.05, 0.10):
                    self.assertAlmostEqual(r["detection"][d], trip_probability(n, noise, d), places=9, msg=(n, noise, d))

    def test_the_numbers_the_methodology_critic_computed_with_its_own_code(self):
        """cycle3/critic-methodology/02e_lottery_v01_vs_v02.log (own enumerator cm_gate.py / cm_lottery.py, 5% noise, 80% pass)."""
        for n, fa, over15, median, p95, over25 in (
            (45, 10.57, 28.8, 7.05, 42.9, 8.2),
            (100, 7.05, 14.0, 1.34, 31.3, 7.6),
            (200, 4.01, 7.2, 0.60, 22.6, 3.9),
        ):
            with self.subTest(n=n):
                r = expected_gate_rates(n, 0.05)
                lot = r["baseline_lottery"]
                self.assertEqual(round(100 * r["false_alarm"], 2), fa)
                self.assertEqual(round(100 * lot["p_fa_over_15"], 1), over15)
                self.assertEqual(round(100 * lot["median_fa"], 2), median)
                self.assertEqual(round(100 * lot["p95_fa"], 1), p95)
                self.assertEqual(round(100 * lot["p_fa_over_25"], 1), over25)

    def test_the_c9_acceptance_numbers_are_the_same_function_of_it(self):
        self.assertLessEqual(expected_gate_rates(45, 0.05)["false_alarm"], 0.11)
        self.assertGreaterEqual(expected_gate_rates(45, 0.05)["detection"][0.10], 0.65)
        self.assertGreaterEqual(expected_gate_rates(200, 0.05)["detection"][0.10], 0.90)
        self.assertLessEqual(max(expected_gate_rates(n, 0.05, drops=())["false_alarm"] for n in (150, 200, 300, 400, 500)), 0.05)

    def test_the_mean_over_baseline_draws_is_the_false_alarm_rate(self):
        for n, noise in ((20, 0.05), (45, 0.10), (100, 0.02), (150, 0.05)):
            r = expected_gate_rates(n, noise)
            self.assertAlmostEqual(r["baseline_lottery"]["mean_fa"], r["false_alarm"], places=9, msg=(n, noise))

    def test_the_lottery_spreads_the_rate_a_lot_and_orders_its_quantiles(self):
        lot = expected_gate_rates(100, 0.05)["baseline_lottery"]
        self.assertLess(lot["median_fa"], 0.02)  # the typical baseline is benign ...
        self.assertGreater(lot["p95_fa"], 0.25)  # ... an unlucky one makes a quarter of unchanged PRs fail
        self.assertLessEqual(lot["median_fa"], lot["mean_fa"])
        self.assertLessEqual(lot["mean_fa"], lot["p95_fa"])
        self.assertLessEqual(lot["p_fa_over_25"], lot["p_fa_over_15"])

    def test_brute_force_over_every_reading_of_a_tiny_suite(self):
        for n, noise, baseline_pass in ((8, 0.10, 0.75), (8, 0.05, 0.5), (7, 0.20, 0.8)):
            with self.subTest(n=n, noise=noise):
                want = brute_force(n, noise, baseline_pass, auto_trips)
                got = expected_gate_rates(n, noise, baseline_pass)
                self.assertAlmostEqual(got["false_alarm"], want["rate"], places=9)
                lot, draws = got["baseline_lottery"], want["draws"]
                total = sum(w for _, w in draws)
                self.assertAlmostEqual(lot["p_fa_over_15"], sum(w for fa, w in draws if fa > 0.15) / total, places=9)
                self.assertAlmostEqual(lot["p_fa_over_25"], sum(w for fa, w in draws if fa > 0.25) / total, places=9)
                self.assertAlmostEqual(lot["median_fa"], quantile(draws, 0.5), places=9)
                self.assertAlmostEqual(lot["p95_fa"], quantile(draws, 0.95), places=9)
                self.assertAlmostEqual(lot["mean_fa"], want["rate"], places=9)

    def test_brute_force_detection_of_a_real_drop(self):
        """Model B with a real drop: every case is, independently, latent-pass (baseline_pass) and then regresses with d / baseline_pass."""
        for n, noise, baseline_pass, d in ((7, 0.10, 0.5, 0.25), (7, 0.05, 0.5, 1 / 3), (8, 0.15, 0.75, 0.2)):
            with self.subTest(n=n, d=d):
                want = brute_force_iid(n, noise, baseline_pass, auto_trips, d)
                got = expected_gate_rates(n, noise, baseline_pass, drops=(d,))["detection"][d]
                self.assertAlmostEqual(got, want, places=9)
        # with no drop the per-case cell does not depend on the latent pass rate, so this is the false-alarm rate too
        for n, noise in ((7, 0.10), (8, 0.20)):
            self.assertAlmostEqual(brute_force_iid(n, noise, 0.75, auto_trips, 0.0), expected_gate_rates(n, noise, 0.75)["false_alarm"], places=9)

    def test_a_fixed_gate_is_enumerated_with_its_own_predicate(self):
        for gate, trips in (
            ({"max_pass_rate_drop": 0.2}, lambda b, c, n: b - c > 0.2 * n + 1e-9),
            ({"max_pass_rate_drop": 0.2, "require_significance": True, "alpha": 0.1}, lambda b, c, n: b - c > 0.2 * n + 1e-9 and contract_p(b, c) < 0.1),
            ({"max_pass_rate_drop": 0.0}, lambda b, c, n: b - c > 0),
        ):
            with self.subTest(gate=gate):
                want = brute_force(8, 0.10, 0.75, trips)
                got = expected_gate_rates(8, 0.10, 0.75, gate=gate)
                self.assertEqual(got["gate_mode"], "fixed")
                self.assertAlmostEqual(got["false_alarm"], want["rate"], places=9)
                self.assertAlmostEqual(got["baseline_lottery"]["median_fa"], quantile(want["draws"], 0.5), places=9)
        fixed = expected_gate_rates(100, 0.05, gate={"max_pass_rate_drop": 0.03})
        self.assertEqual((fixed["effective_max_pass_rate_drop"], fixed["effective_alpha"], fixed["effective_min_net_loss"]), (0.03, None, None))

    def test_a_fixed_gate_without_significance_false_alarms_more_than_auto_at_large_n(self):
        auto = expected_gate_rates(300, 0.05, drops=())["false_alarm"]
        fixed = expected_gate_rates(300, 0.05, drops=(), gate={"max_pass_rate_drop": 0.01})["false_alarm"]
        self.assertGreater(fixed, 5 * auto)

    def test_gate_may_be_partial_and_unknown_keys_raise(self):
        self.assertEqual(expected_gate_rates(45, 0.05, gate={})["false_alarm"], expected_gate_rates(45, 0.05)["false_alarm"])
        self.assertEqual(expected_gate_rates(45, 0.05, gate={"min_pass_rate": 0.9})["gate_mode"], "auto")
        with self.assertRaisesRegex(ValueError, "unknown gate setting"):
            expected_gate_rates(45, 0.05, gate={"max_pass_drop": 0.03})

    def test_with_the_pass_rate_rule_off_nothing_trips(self):
        r = expected_gate_rates(45, 0.05, gate={"max_pass_rate_drop": None})
        self.assertEqual(r["gate_mode"], "off")
        self.assertEqual((r["false_alarm"], r["detection"], r["effective_alpha"]), (0.0, {0.05: 0.0, 0.10: 0.0}, None))
        self.assertEqual(set(r["baseline_lottery"].values()), {0.0})

    def test_rates_move_the_right_way(self):
        for n in (45, 100, 200):
            fa = [expected_gate_rates(n, noise, drops=())["false_alarm"] for noise in (0.0, 0.02, 0.05, 0.10)]
            self.assertEqual(fa[0], 0.0)
            self.assertEqual(fa, sorted(fa), n)
            det = expected_gate_rates(n, 0.05, drops=(0.0, 0.02, 0.05, 0.10, 0.20))["detection"]
            self.assertEqual(list(det.values()), sorted(det.values()), n)
            self.assertAlmostEqual(det[0.0], expected_gate_rates(n, 0.05, drops=())["false_alarm"], places=12)

    def test_no_noise_means_a_real_drop_is_caught_or_missed_by_the_rule_alone(self):
        r = expected_gate_rates(45, 0.0, drops=(0.10,))
        self.assertEqual(r["false_alarm"], 0.0)
        self.assertEqual(r["baseline_lottery"]["p_fa_over_15"], 0.0)
        # b ~ Bin(45, 0.10), c = 0: the gate trips on 3 or more regressions (p = 1/8 < alpha(45)) ...
        self.assertAlmostEqual(r["detection"][0.10], 1 - sum(math.comb(45, k) * 0.1**k * 0.9 ** (45 - k) for k in range(3)), places=12)

    def test_a_drop_larger_than_the_pass_rate_cannot_happen(self):
        r = expected_gate_rates(45, 0.05, baseline_pass=0.08, drops=(0.05, 0.10))
        self.assertIsNotNone(r["detection"][0.05])
        self.assertIsNone(r["detection"][0.10])
        json.dumps(r)

    def test_tiny_and_huge_suites_do_not_break_it(self):
        self.assertEqual(expected_gate_rates(1, 0.05)["false_alarm"], 0.0)  # the rule needs a net loss of 3 cases
        self.assertEqual(expected_gate_rates(2, 0.5, drops=(0.5,))["detection"][0.5], 0.0)
        big = expected_gate_rates(1200, 0.05, drops=(0.05,))
        self.assertAlmostEqual(big["baseline_lottery"]["mean_fa"], big["false_alarm"], places=8)
        self.assertGreater(big["detection"][0.05], 0.95)  # 60 of 1200 cases is a net loss of 2.4x the floor of 24

    def test_bad_arguments_are_refused(self):
        bad = [
            dict(n=0, noise=0.05),
            dict(n=-3, noise=0.05),
            dict(n=45.0, noise=0.05),
            dict(n=True, noise=0.05),
            dict(n=45, noise=-0.01),
            dict(n=45, noise=0.51),
            dict(n=45, noise="0.05"),
            dict(n=45, noise=float("nan")),
            dict(n=45, noise=0.05, baseline_pass=1.5),
            dict(n=45, noise=0.05, baseline_pass=-0.1),
            dict(n=45, noise=0.05, drops=(0.1, 1.2)),
            dict(n=45, noise=0.05, drops=(-0.1,)),
            dict(n=45, noise=0.05, drops=("x",)),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                expected_gate_rates(**kwargs)

    def test_compare_runs_false_alarm_and_detection_land_on_the_exact_numbers(self):
        n, noise = 45, 0.05
        for d in (0.0, 0.10):
            exact = expected_gate_rates(n, noise, drops=(d,))["detection"][d]
            measured = monte_carlo_rate(n, noise, d, runs=2000, seed=20260930)
            sigma = math.sqrt(exact * (1 - exact) / 2000)
            self.assertLess(abs(measured - exact), 4 * sigma, (d, measured, exact))


def contract_p(b: int, c: int) -> float:
    m = b + c
    return sum(math.comb(m, i) for i in range(b, m + 1)) / 2**m if m else 1.0


def monte_carlo_rate(n: int, noise: float, d: float, runs: int, seed: int) -> float:
    """Trip rate of compare_runs on two noisy readings; a fraction d of all cases (of the 80% latent passes) really regressed."""
    rng = random.Random(seed)
    n_pass = int(0.8 * n)
    base_latent = [i < n_pass for i in range(n)]
    trips = 0
    for _ in range(runs):
        cand_latent = [v and not (rng.random() < d / 0.8) for v in base_latent]
        base = _records("c", [v != (rng.random() < noise) for v in base_latent])
        cand = _records("c", [v != (rng.random() < noise) for v in cand_latent])
        summary = {"judges_sha": "x", "repeat": {"n": 1, "per_tag": {}}}
        result = compare_runs(
            RunResult(run_dir=None, summary={"run_id": "b", **summary}, results=base),
            RunResult(run_dir=None, summary={"run_id": "c", **summary}, results=cand),
            gate={},
        )
        trips += result["verdict"] == "regression"
    return trips / runs


if __name__ == "__main__":
    unittest.main()
