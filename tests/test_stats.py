import math
import unittest
from fractions import Fraction

from evalsprint import stats


class WilsonTests(unittest.TestCase):
    def test_known_value_8_of_10(self):
        lo, hi = stats.wilson_ci(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=4)
        self.assertAlmostEqual(hi, 0.9433, places=4)

    def test_known_value_50_of_100(self):
        lo, hi = stats.wilson_ci(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=4)
        self.assertAlmostEqual(hi, 0.5962, places=4)

    def test_all_fail_and_all_pass_stay_inside_unit_interval(self):
        lo, hi = stats.wilson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = stats.wilson_ci(10, 10)
        self.assertAlmostEqual(lo, 0.7225, places=4)
        self.assertEqual(hi, 1.0)

    def test_n_zero_is_uninformative(self):
        self.assertEqual(stats.wilson_ci(0, 0), (0.0, 1.0))

    def test_interval_contains_point_estimate_and_shrinks_with_n(self):
        for k, n in [(1, 5), (7, 20), (60, 100), (300, 500)]:
            lo, hi = stats.wilson_ci(k, n)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)
        w_small = stats.wilson_ci(8, 10)
        w_big = stats.wilson_ci(80, 100)
        self.assertLess(w_big[1] - w_big[0], w_small[1] - w_small[0])

    def test_custom_z(self):
        lo90, hi90 = stats.wilson_ci(8, 10, z=1.645)
        lo95, hi95 = stats.wilson_ci(8, 10, z=1.96)
        self.assertGreater(lo90, lo95)
        self.assertLess(hi90, hi95)


class PercentileMeanTests(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertAlmostEqual(stats.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(stats.percentile([1, 2, 3, 4, 5], 95), 4.8)
        self.assertAlmostEqual(stats.percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_extremes_and_unsorted_input(self):
        self.assertEqual(stats.percentile([3, 1, 2], 0), 1.0)
        self.assertEqual(stats.percentile([3, 1, 2], 100), 3.0)
        self.assertEqual(stats.percentile([9, 1, 5, 3], 100), 9.0)

    def test_single_value_and_empty(self):
        self.assertEqual(stats.percentile([10], 95), 10.0)
        self.assertIsNone(stats.percentile([], 50))

    def test_none_values_are_ignored(self):
        self.assertAlmostEqual(stats.percentile([None, 1, 3, None], 50), 2.0)
        self.assertIsNone(stats.percentile([None, None], 50))

    def test_mean(self):
        self.assertAlmostEqual(stats.mean([1, 2, 3, 6]), 3.0)
        self.assertAlmostEqual(stats.mean([None, 2, 4]), 3.0)
        self.assertIsNone(stats.mean([]))
        self.assertIsNone(stats.mean([None]))


class BootstrapTests(unittest.TestCase):
    DATA = [0.1 * i for i in range(30)]

    def test_deterministic_with_seed(self):
        a = stats.bootstrap_ci(self.DATA, seed=7)
        b = stats.bootstrap_ci(self.DATA, seed=7)
        self.assertEqual(a, b)

    def test_different_seed_changes_result(self):
        self.assertNotEqual(stats.bootstrap_ci(self.DATA, seed=1), stats.bootstrap_ci(self.DATA, seed=2))

    def test_interval_brackets_the_mean_and_is_ordered(self):
        lo, hi = stats.bootstrap_ci(self.DATA, seed=3)
        m = sum(self.DATA) / len(self.DATA)
        self.assertLess(lo, m)
        self.assertGreater(hi, m)
        self.assertLessEqual(lo, hi)

    def test_constant_data_gives_degenerate_interval(self):
        self.assertEqual(stats.bootstrap_ci([0.5] * 10), (0.5, 0.5))

    def test_empty_returns_none(self):
        self.assertIsNone(stats.bootstrap_ci([]))

    def test_custom_statistic(self):
        lo, hi = stats.bootstrap_ci(self.DATA, stat=max, seed=0)
        self.assertLessEqual(hi, max(self.DATA))

    def test_paired_diff_ci(self):
        baseline = [0] * 10
        candidate = [1] * 10
        self.assertEqual(stats.paired_bootstrap_diff_ci(baseline, candidate), (1.0, 1.0))
        self.assertEqual(stats.paired_bootstrap_diff_ci(candidate, baseline), (-1.0, -1.0))

    def test_paired_diff_is_candidate_minus_baseline_and_seeded(self):
        a = [0, 1, 1, 0, 1, 0, 0, 1]
        b = [1, 1, 1, 1, 1, 0, 1, 1]
        r1 = stats.paired_bootstrap_diff_ci(a, b, seed=4)
        r2 = stats.paired_bootstrap_diff_ci(a, b, seed=4)
        self.assertEqual(r1, r2)
        self.assertGreaterEqual(r1[0], 0.0)  # candidate never worse pair-wise

    def test_paired_diff_requires_equal_length(self):
        with self.assertRaises(ValueError):
            stats.paired_bootstrap_diff_ci([1, 2], [1])


class McNemarTests(unittest.TestCase):
    def test_exact_values(self):
        self.assertAlmostEqual(stats.mcnemar_exact(1, 9), 0.021484375)
        self.assertAlmostEqual(stats.mcnemar_exact(2, 8), 0.109375)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(stats.mcnemar_exact(3, 0), 0.25)

    def test_symmetric_in_b_and_c(self):
        self.assertEqual(stats.mcnemar_exact(2, 9), stats.mcnemar_exact(9, 2))

    def test_no_discordant_pairs_or_balanced_pairs(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(stats.mcnemar_exact(5, 5), 1.0)

    def test_p_value_capped_at_one(self):
        self.assertLessEqual(stats.mcnemar_exact(4, 5), 1.0)

    def test_more_evidence_smaller_p(self):
        self.assertLess(stats.mcnemar_exact(0, 10), stats.mcnemar_exact(0, 5))

    def test_binom_two_sided_matches_direct_sum(self):
        n, k = 12, 3
        direct = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        self.assertAlmostEqual(stats.binom_two_sided_p(k, n), direct)
        self.assertEqual(stats.binom_two_sided_p(0, 0), 1.0)


def _binom_cdf(k: int, n: int) -> Fraction:
    """P(X <= k), X ~ Binomial(n, 1/2), by the pmf recurrence (independent of math.comb sums)."""
    if k < 0:
        return Fraction(0)
    pmf, total = Fraction(1, 2**n), Fraction(0)
    for i in range(min(k, n) + 1):
        total += pmf
        pmf = pmf * (n - i) / (i + 1)
    return total


class McNemarAlternativeTests(unittest.TestCase):
    """`alternative` picks the tail of the exact binomial test on the discordant pairs."""

    def test_default_is_two_sided(self):
        self.assertEqual(stats.mcnemar_exact(2, 9), stats.mcnemar_exact(2, 9, alternative="two-sided"))
        self.assertEqual(stats.mcnemar_exact(2, 9), stats.mcnemar_exact(2, 9, "two-sided"))

    def test_greater_hand_computed(self):
        # candidate worse: P(X >= b) with n = b + c discordant pairs
        self.assertAlmostEqual(stats.mcnemar_exact(6, 0, "greater"), 1 / 64)
        self.assertAlmostEqual(stats.mcnemar_exact(4, 0, "greater"), 1 / 16)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 3, "greater"), 93 / 256)  # (56 + 28 + 8 + 1) / 256
        self.assertAlmostEqual(stats.mcnemar_exact(9, 1, "greater"), 11 / 1024)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5, "greater"), 1.0)  # candidate better: no evidence it is worse

    def test_less_hand_computed(self):
        self.assertAlmostEqual(stats.mcnemar_exact(0, 6, "less"), 1 / 64)
        self.assertAlmostEqual(stats.mcnemar_exact(3, 5, "less"), (1 + 8 + 28 + 56) / 256)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 0, "less"), 1.0)

    def test_matches_exact_binomial_on_a_grid(self):
        for n in range(0, 41):
            for b in range(n + 1):
                c = n - b
                with self.subTest(b=b, c=c):
                    greater = 1 - _binom_cdf(b - 1, n)  # P(X >= b)
                    less = _binom_cdf(b, n)  # P(X <= b)
                    self.assertAlmostEqual(stats.mcnemar_exact(b, c, "greater"), float(greater), places=12)
                    self.assertAlmostEqual(stats.mcnemar_exact(b, c, "less"), float(less), places=12)
                    self.assertAlmostEqual(stats.mcnemar_exact(b, c), float(min(Fraction(1), 2 * min(greater, less))), places=12)

    def test_greater_and_less_are_mirror_images(self):
        for b, c in ((0, 0), (1, 9), (7, 2), (6, 6), (30, 12)):
            with self.subTest(b=b, c=c):
                self.assertAlmostEqual(stats.mcnemar_exact(b, c, "greater"), stats.mcnemar_exact(c, b, "less"), places=12)

    def test_greater_and_less_overlap_only_on_the_observed_count(self):
        # P(X >= b) + P(X <= b - 1) = 1 for the same n
        for b, c in ((3, 9), (10, 2), (1, 1)):
            self.assertAlmostEqual(
                stats.mcnemar_exact(b, c, "greater") + stats.mcnemar_exact(b - 1, c + 1, "less"), 1.0, places=12
            )

    def test_one_sided_is_half_the_two_sided_when_the_direction_matches(self):
        for b, c in ((6, 0), (9, 2), (1, 9), (2, 12)):
            direction = "greater" if b > c else "less"
            with self.subTest(b=b, c=c):
                self.assertAlmostEqual(stats.mcnemar_exact(b, c, direction) * 2, stats.mcnemar_exact(b, c), places=12)

    def test_no_discordant_pairs(self):
        for alt in ("two-sided", "greater", "less"):
            self.assertEqual(stats.mcnemar_exact(0, 0, alt), 1.0)

    def test_p_values_are_probabilities(self):
        for alt in ("two-sided", "greater", "less"):
            for b, c in ((0, 1), (1, 0), (4, 5), (100, 3)):
                self.assertTrue(0 < stats.mcnemar_exact(b, c, alt) <= 1.0)

    def test_greater_shrinks_with_more_pass_to_fail_flips(self):
        values = [stats.mcnemar_exact(b, 0, "greater") for b in range(1, 12)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(stats.mcnemar_exact(10, 0, "greater"), 0.001)

    def test_large_counts_do_not_overflow(self):
        self.assertAlmostEqual(stats.mcnemar_exact(600, 600, "greater"), 0.5 + 0.5 * (math.comb(1200, 600) / 2**1200), places=9)
        self.assertLess(stats.mcnemar_exact(900, 100, "greater"), 1e-100)

    def test_bad_arguments_raise(self):
        with self.assertRaisesRegex(ValueError, "alternative"):
            stats.mcnemar_exact(1, 2, "worse")
        with self.assertRaisesRegex(ValueError, "alternative"):
            stats.mcnemar_exact(1, 2, alternative=None)
        for b, c in ((-1, 2), (1, -2), (1.5, 2), (True, 2)):
            with self.subTest(b=b, c=c), self.assertRaisesRegex(ValueError, "non-negative integer"):
                stats.mcnemar_exact(b, c)


class KappaTests(unittest.TestCase):
    def test_hand_computed_example(self):
        # a: 6 pass / 4 fail, b: 5 pass / 5 fail, 7 of 10 items agree.
        # po = 0.7, pe = 0.6*0.5 + 0.4*0.5 = 0.5, kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
        a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        b = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        self.assertAlmostEqual(stats.cohen_kappa(a, b), 0.4)

    def test_perfect_and_opposite_agreement(self):
        a = ["pass", "fail", "pass", "fail"]
        self.assertAlmostEqual(stats.cohen_kappa(a, list(a)), 1.0)
        b = ["fail", "pass", "fail", "pass"]
        self.assertAlmostEqual(stats.cohen_kappa(a, b), -1.0)

    def test_chance_level_is_zero(self):
        a = [1, 1, 0, 0]
        b = [1, 0, 1, 0]
        self.assertAlmostEqual(stats.cohen_kappa(a, b), 0.0)

    def test_works_with_booleans_and_strings(self):
        self.assertAlmostEqual(stats.cohen_kappa([True, False, True], [True, False, True]), 1.0)

    def test_constant_raters(self):
        self.assertEqual(stats.cohen_kappa([1, 1, 1], [1, 1, 1]), 1.0)
        # one rater constant, the other not: pe < 1, agreement no better than chance
        self.assertAlmostEqual(stats.cohen_kappa([1, 1, 1, 1], [1, 1, 1, 0]), 0.0)

    def test_empty_and_length_mismatch(self):
        self.assertIsNone(stats.cohen_kappa([], []))
        with self.assertRaises(ValueError):
            stats.cohen_kappa([1], [1, 0])


class RoganGladenTests(unittest.TestCase):
    def test_known_value(self):
        # (0.6 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.4 / 0.7
        self.assertAlmostEqual(stats.rogan_gladen(0.6, 0.9, 0.8), 0.4 / 0.7)

    def test_perfect_judge_returns_observed(self):
        self.assertAlmostEqual(stats.rogan_gladen(0.9, 1.0, 1.0), 0.9)

    def test_no_better_than_chance_returns_none(self):
        self.assertIsNone(stats.rogan_gladen(0.5, 0.5, 0.5))  # J == 0
        self.assertIsNone(stats.rogan_gladen(0.5, 0.3, 0.4))  # J < 0

    def test_clipped_to_unit_interval(self):
        self.assertEqual(stats.rogan_gladen(1.0, 0.6, 0.6), 1.0)  # raw (1+0.6-1)/0.2 = 3
        self.assertEqual(stats.rogan_gladen(0.0, 0.9, 0.9), 0.0)  # raw (0-0.1)/0.8 < 0

    def test_lenient_judge_is_corrected_downwards(self):
        # judge passes 90% of everything, but only catches 70% of real failures
        corrected = stats.rogan_gladen(0.9, sensitivity=0.98, specificity=0.7)
        self.assertLess(corrected, 0.9)


class KappaLabelTests(unittest.TestCase):
    def test_bands(self):
        cases = {
            -0.3: "poor",
            0.0: "poor",
            0.19: "poor",
            0.2: "fair",
            0.39: "fair",
            0.4: "moderate",
            0.59: "moderate",
            0.6: "substantial",
            0.79: "substantial",
            0.8: "almost perfect",
            1.0: "almost perfect",
        }
        for value, label in cases.items():
            self.assertEqual(stats.kappa_label(value), label, value)

    def test_none(self):
        self.assertEqual(stats.kappa_label(None), "n/a")


class ExactTailsAreTheOldSumsTests(unittest.TestCase):
    """The tails come from an integer recurrence; they must be bit-identical to the plain sum of math.comb they replaced."""

    @staticmethod
    def old_sum(m: int, k: int) -> int:
        return sum(math.comb(m, i) for i in range(max(k, 0), m + 1))

    def test_half_tail_count_equals_the_comb_sum_everywhere(self):
        for m in range(0, 70):
            for k in range(-2, m + 4):
                self.assertEqual(stats._half_tail_count(m, k), self.old_sum(m, k), (m, k))

    def test_half_tail_count_edges(self):
        self.assertEqual(stats._half_tail_count(0, 0), 1)
        self.assertEqual(stats._half_tail_count(0, 1), 0)
        self.assertEqual(stats._half_tail_count(10, 0), 2**10)
        self.assertEqual(stats._half_tail_count(10, -5), 2**10)
        self.assertEqual(stats._half_tail_count(10, 11), 0)
        self.assertEqual(stats._half_tail_count(10, 10), 1)
        self.assertEqual(stats._half_tail_count(1000, 500), self.old_sum(1000, 500))

    def test_mcnemar_greater_and_less_are_bit_identical_to_the_old_sums(self):
        for b in range(0, 41):
            for c in range(0, 41):
                m = b + c
                if m == 0:
                    continue
                self.assertEqual(stats.mcnemar_exact(b, c, "greater"), self.old_sum(m, b) / 2**m, (b, c))
                self.assertEqual(
                    stats.mcnemar_exact(b, c, "less"), sum(math.comb(m, i) for i in range(0, b + 1)) / 2**m, (b, c)
                )

    def test_two_sided_is_bit_identical_to_the_old_sum(self):
        for n in range(0, 121):
            for k in range(0, n + 1):
                kk = min(k, n - k)
                want = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2**n)
                self.assertEqual(stats.binom_two_sided_p(k, n), want, (k, n))

    def test_a_big_discordant_count_is_fast_and_right(self):
        self.assertAlmostEqual(stats.mcnemar_exact(600, 500, "greater"), float(1 - _binom_cdf(599, 1100)), places=12)
        self.assertEqual(stats.mcnemar_exact(0, 3000, "greater"), 1.0)
        self.assertEqual(stats.mcnemar_exact(3000, 0, "greater"), 2.0 ** -3000 if 2.0 ** -3000 > 0 else 0.0)


class MajorityNoiseTests(unittest.TestCase):
    def test_one_attempt_is_the_attempt_noise(self):
        for p in (0.0, 0.01, 0.2, 0.5, 1.0):
            self.assertEqual(stats.majority_noise(p, 1), p)

    def test_the_number_the_docs_quote(self):
        self.assertAlmostEqual(stats.majority_noise(0.05, 3), 3 * 0.05**2 * 0.95 + 0.05**3, places=15)
        self.assertAlmostEqual(stats.majority_noise(0.05, 3), 0.00725, places=12)

    def test_more_odd_attempts_help_while_the_attempt_beats_a_coin(self):
        values = [stats.majority_noise(0.05, k) for k in (1, 3, 5, 7, 9)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(stats.majority_noise(0.05, 5), 10 * 0.05**3 * 0.95**2 + 5 * 0.05**4 * 0.95 + 0.05**5, places=15)

    def test_a_coin_stays_a_coin_and_a_liar_gets_worse(self):
        for k in (1, 3, 5, 9):
            self.assertAlmostEqual(stats.majority_noise(0.5, k), 0.5, places=12)
        self.assertGreater(stats.majority_noise(0.8, 5), 0.8)

    def test_even_attempts_average_the_two_tie_directions(self):
        p = 0.1
        # two attempts: a latent pass is misread when at least one attempt fails (tie = fail); a latent fail only when both pass
        latent_pass_misread = 1 - (1 - p) ** 2
        latent_fail_misread = p**2
        self.assertAlmostEqual(stats.majority_noise(p, 2), (latent_pass_misread + latent_fail_misread) / 2, places=15)

    def test_bad_arguments(self):
        for bad in (0, -1, 2.0, True, "3", None):
            with self.assertRaises(ValueError, msg=bad):
                stats.majority_noise(0.05, bad)
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError, msg=bad):
                stats.majority_noise(bad, 3)


class NoiseFromFlipRateTests(unittest.TestCase):
    @staticmethod
    def expected_flip_rate(p, ks):
        return sum(1 - p**k - (1 - p) ** k for k in ks) / len(ks)

    def test_round_trip_for_uniform_protocols(self):
        for k in (2, 3, 5, 7):
            for p in (0.001, 0.02, 0.05, 0.10, 0.25, 0.4):
                flip = self.expected_flip_rate(p, [k])
                self.assertAlmostEqual(stats.noise_from_flip_rate(flip, k), p, places=10, msg=(k, p))

    def test_round_trip_for_a_mixed_protocol(self):
        ks = [5] * 10 + [3] * 40  # safety x5, the rest x3
        for p in (0.01, 0.05, 0.15):
            flip = self.expected_flip_rate(p, ks)
            self.assertAlmostEqual(stats.noise_from_flip_rate(flip, ks), p, places=10)
            self.assertAlmostEqual(stats.noise_from_flip_rate(flip, tuple(ks)), p, places=10)

    def test_a_known_reading(self):
        # 3 attempts, 5% per attempt: 1 - 0.05^3 - 0.95^3 = 14.25% of the cases disagree with themselves
        self.assertAlmostEqual(self.expected_flip_rate(0.05, [3]), 0.1425, places=12)
        self.assertAlmostEqual(stats.noise_from_flip_rate(0.1425, 3), 0.05, places=10)

    def test_no_flips_means_no_noise_and_coin_flips_cap_at_one_half(self):
        self.assertEqual(stats.noise_from_flip_rate(0.0, 3), 0.0)
        self.assertEqual(stats.noise_from_flip_rate(1 - 2 * 0.5**3, 3), 0.5)
        self.assertEqual(stats.noise_from_flip_rate(0.9, 3), 0.5)
        self.assertEqual(stats.noise_from_flip_rate(1.0, 5), 0.5)

    def test_monotone_in_the_flip_rate(self):
        values = [stats.noise_from_flip_rate(f / 100, 3) for f in range(0, 76, 5)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(values))

    def test_the_flip_rate_is_not_the_noise(self):
        # the trap this function exists for: a 14% flip rate is 5% noise per attempt, and 0.7% noise per majority verdict
        p = stats.noise_from_flip_rate(0.1425, 3)
        self.assertLess(p, 0.06)
        self.assertLess(stats.majority_noise(p, 3), 0.01)

    def test_bad_arguments(self):
        for flip in (-0.1, 1.5, float("nan"), "0.1", None, True):
            with self.assertRaises(ValueError, msg=flip):
                stats.noise_from_flip_rate(flip, 3)
        for attempts in (1, 0, [], [3, 1], [True], 2.5, True):
            with self.assertRaises(ValueError, msg=attempts):
                stats.noise_from_flip_rate(0.1, attempts)


class BinomPmfWindowTests(unittest.TestCase):
    @staticmethod
    def exact(m, q, k):
        return math.comb(m, k) * q**k * (1 - q) ** (m - k)

    def test_matches_the_plain_pmf_and_carries_all_the_mass(self):
        for m, q in ((1, 0.3), (10, 0.5), (45, 0.05), (200, 0.0725), (500, 0.9), (1200, 0.05)):
            lo, probs = stats.binom_pmf_window(m, q)
            self.assertAlmostEqual(sum(probs), 1.0, places=12, msg=(m, q))
            for i, v in enumerate(probs):
                if m <= 200:
                    self.assertAlmostEqual(v, self.exact(m, q, lo + i), places=13, msg=(m, q, lo + i))
            self.assertGreaterEqual(lo, 0)
            self.assertLessEqual(lo + len(probs) - 1, m)

    def test_the_window_is_narrow_for_a_big_m(self):
        lo, probs = stats.binom_pmf_window(100000, 0.01)
        self.assertLess(len(probs), 1200)
        self.assertAlmostEqual(sum(probs), 1.0, places=8)  # lgamma of 1e5 costs a few 1e-10
        self.assertAlmostEqual(lo + sum(i * v for i, v in enumerate(probs)), 1000.0, places=6)

    def test_degenerate_probabilities(self):
        self.assertEqual(stats.binom_pmf_window(7, 0.0), (0, (1.0,)))
        self.assertEqual(stats.binom_pmf_window(7, 1.0), (7, (1.0,)))
        self.assertEqual(stats.binom_pmf_window(0, 0.4), (0, (1.0,)))

    def test_it_returns_an_immutable_cached_value(self):
        a = stats.binom_pmf_window(45, 0.05)
        self.assertIs(a, stats.binom_pmf_window(45, 0.05))
        self.assertIsInstance(a[1], tuple)

    def test_bad_arguments(self):
        for m, q in ((-1, 0.5), (3.0, 0.5), (True, 0.5), (3, -0.1), (3, 1.1)):
            with self.assertRaises(ValueError, msg=(m, q)):
                stats.binom_pmf_window(m, q)


class MajorityNoiseDirectionsTests(unittest.TestCase):
    """Loop-back 1 (C21, even k): a tie is a fail, so a latent pass is lost more often than a latent fail is gained."""

    def test_odd_attempts_have_one_noise_in_both_directions(self):
        for k in (1, 3, 5, 7):
            for p in (0.0, 0.01, 0.05, 0.2, 0.5):
                lost_pass, lost_fail = stats.majority_noise_directions(p, k)
                self.assertEqual(lost_pass, lost_fail)
                self.assertEqual(lost_pass, stats.majority_noise(p, k))

    def test_two_attempts_at_five_percent_against_the_hand_computation(self):
        lost_pass, lost_fail = stats.majority_noise_directions(0.05, 2)
        self.assertAlmostEqual(lost_pass, 1 - 0.95**2, places=15)  # one wrong attempt of two is a tie, and a tie fails
        self.assertAlmostEqual(lost_fail, 0.05**2, places=15)  # both attempts must be wrong to read a fail as a pass
        self.assertAlmostEqual(lost_pass, 0.0975, places=15)
        self.assertAlmostEqual(lost_fail, 0.0025, places=15)

    def test_even_attempts_are_asymmetric_and_majority_noise_is_their_mean(self):
        for k in (2, 4, 6):
            for p in (0.01, 0.05, 0.2):
                lost_pass, lost_fail = stats.majority_noise_directions(p, k)
                self.assertGreater(lost_pass, lost_fail)
                self.assertAlmostEqual(stats.majority_noise(p, k), (lost_pass + lost_fail) / 2, places=15)

    def test_against_a_brute_force_sum_over_the_attempt_outcomes(self):
        import itertools

        for k in range(1, 7):
            for p in (0.03, 0.05, 0.3):
                lost_pass = lost_fail = 0.0
                for wrong in itertools.product((0, 1), repeat=k):
                    prob = math.prod(p if w else 1 - p for w in wrong)
                    # latent pass: each attempt reads fail when wrong; the verdict passes with a strict majority of passes
                    if k - sum(wrong) <= k / 2:
                        lost_pass += prob
                    # latent fail: each attempt reads pass when wrong; the verdict passes with a strict majority of passes
                    if sum(wrong) > k / 2:
                        lost_fail += prob
                got = stats.majority_noise_directions(p, k)
                self.assertAlmostEqual(got[0], lost_pass, places=14, msg=(k, p))
                self.assertAlmostEqual(got[1], lost_fail, places=14, msg=(k, p))

    def test_bad_arguments(self):
        for p, k in ((0.05, 0), (0.05, -1), (0.05, 2.0), (0.05, True), (-0.1, 3), (1.1, 3), (True, 3), ("0.05", 3), (None, 3)):
            with self.assertRaises(ValueError, msg=(p, k)):
                stats.majority_noise_directions(p, k)


class ClopperPearsonUpperTests(unittest.TestCase):
    # scipy.stats.beta.ppf(0.95, k + 1, n - k), checked when the tests were written
    KNOWN = [
        (0, 20, 0.13910834066826516),
        (1, 20, 0.2161061642068473),
        (3, 20, 0.3436638043142818),
        (5, 45, 0.21953609544733202),
        (0, 45, 0.06440428880778348),
        (10, 100, 0.16371762327581477),
        (19, 20, 0.9974386212234697),
    ]

    def test_known_values(self):
        for k, n, want in self.KNOWN:
            self.assertAlmostEqual(stats.clopper_pearson_upper(k, n), want, places=12, msg=(k, n))

    def test_no_success_has_the_closed_form(self):
        for n in (1, 5, 20, 45, 300):
            self.assertAlmostEqual(stats.clopper_pearson_upper(0, n), 1 - 0.05 ** (1 / n), places=12)
            self.assertAlmostEqual(stats.clopper_pearson_upper(0, n, 0.9), 1 - 0.10 ** (1 / n), places=12)

    def test_all_successes_is_one(self):
        self.assertEqual(stats.clopper_pearson_upper(7, 7), 1.0)

    def test_the_bound_solves_its_defining_equation_and_is_monotone(self):
        previous = 0.0
        for k in range(0, 30):
            bound = stats.clopper_pearson_upper(k, 30)
            cdf = sum(math.comb(30, i) * bound**i * (1 - bound) ** (30 - i) for i in range(k + 1))
            self.assertAlmostEqual(cdf, 0.05, places=10, msg=k)
            self.assertGreater(bound, previous)
            self.assertGreater(bound, k / 30)  # an upper bound sits above the observed rate
            previous = bound

    def test_it_holds_for_thousands_of_trials(self):
        bound = stats.clopper_pearson_upper(40, 2000)
        self.assertGreater(bound, 0.02)
        self.assertLess(bound, 0.03)

    def test_bad_arguments(self):
        for k, n, c in ((-1, 10, 0.95), (11, 10, 0.95), (1, 0, 0.95), (1.0, 10, 0.95), (True, 10, 0.95), (1, 10, 0.0), (1, 10, 1.0), (1, 10, True)):
            with self.assertRaises(ValueError, msg=(k, n, c)):
                stats.clopper_pearson_upper(k, n, c)


class AttemptNoiseEstimateTests(unittest.TestCase):
    def test_the_estimate_is_the_flip_rate_inverted_and_the_upper_bound_goes_through_the_same_function(self):
        est = stats.attempt_noise_estimate(6, [3] * 40)
        self.assertEqual((est["n_repeated"], est["n_flipped"]), (40, 6))
        self.assertAlmostEqual(est["flip_rate"], 0.15)
        self.assertAlmostEqual(est["noise"], stats.noise_from_flip_rate(0.15, 3), places=14)
        self.assertAlmostEqual(est["flip_rate_upper95"], stats.clopper_pearson_upper(6, 40), places=14)
        self.assertAlmostEqual(est["noise_upper95"], stats.noise_from_flip_rate(est["flip_rate_upper95"], 3), places=14)
        self.assertGreater(est["noise_upper95"], est["noise"])

    def test_no_flipped_case_has_noise_zero_and_a_positive_upper_bound(self):
        est = stats.attempt_noise_estimate(0, [3] * 20)
        self.assertEqual(est["noise"], 0.0)
        self.assertAlmostEqual(est["flip_rate_upper95"], 1 - 0.05 ** (1 / 20), places=12)
        self.assertAlmostEqual(est["noise_upper95"], stats.noise_from_flip_rate(1 - 0.05 ** (1 / 20), 3), places=12)
        self.assertGreater(est["noise_upper95"], 0.02)  # 20 quiet cases do not prove a noise free run

    def test_a_mixed_protocol_uses_one_attempt_count_per_case(self):
        ks = [5] * 8 + [3] * 32
        est = stats.attempt_noise_estimate(5, ks)
        self.assertAlmostEqual(est["noise"], stats.noise_from_flip_rate(5 / 40, ks), places=14)
        self.assertNotAlmostEqual(est["noise"], stats.noise_from_flip_rate(5 / 40, 3), places=4)

    def test_all_flipped_cases_hit_the_ceiling(self):
        est = stats.attempt_noise_estimate(20, [3] * 20)
        self.assertEqual((est["noise"], est["noise_upper95"]), (0.5, 0.5))

    def test_no_repeated_case_is_not_estimated(self):
        est = stats.attempt_noise_estimate(0, [])
        self.assertEqual((est["noise"], est["noise_upper95"], est["n_repeated"]), (None, None, 0))

    def test_the_keys_of_the_contract(self):
        est = stats.attempt_noise_estimate(2, [3] * 25)
        self.assertTrue({"noise", "noise_upper95", "n_repeated", "n_flipped"} <= set(est))

    def test_it_accepts_any_iterable_of_counts(self):
        self.assertEqual(stats.attempt_noise_estimate(1, (3 for _ in range(30)))["n_repeated"], 30)

    def test_bad_arguments(self):
        for flipped, ks in ((-1, [3] * 5), (6, [3] * 5), (1.0, [3] * 5), (True, [3] * 5), (1, [3, 1, 3]), (1, [3, 2.0]), (1, [True, 3]), (1, 5), (0, [0])):
            with self.assertRaises(ValueError, msg=(flipped, ks)):
                stats.attempt_noise_estimate(flipped, ks)


if __name__ == "__main__":
    unittest.main()
