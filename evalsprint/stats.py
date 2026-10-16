"""Small, dependency-free statistics used across the kit.

Everything here is deliberately simple and well-known, so results can be
explained to a client in one sentence each.
"""

from __future__ import annotations

import functools
import math
import random
from typing import Callable, Sequence


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    vals = list(values)
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return (boots[int(iters * alpha / 2)], boots[min(iters - 1, int(iters * (1 - alpha / 2)))])


def paired_bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(b - a) over paired observations (candidate minus baseline)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [y - x for x, y in zip(a, b)]
    return bootstrap_ci(diffs, iters=iters, alpha=alpha, seed=seed)


def _half_tail_count(m: int, k: int) -> int:
    """``sum(comb(m, i) for i in range(k, m + 1))`` as an exact integer, from the shorter side of the binomial row.

    Consecutive terms come from one multiplication and one exact division each (C(m, i+1) = C(m, i) * (m - i) / (i + 1)), so
    it costs O(m) small steps instead of m calls of ``math.comb``; the value is identical, which matters: p-values are compared
    with a threshold.
    """
    if k <= 0:
        return 1 << m
    if k > m:
        return 0
    if k > m // 2:  # fewer terms above k than below it
        term, total = math.comb(m, k), 0
        for i in range(k, m + 1):
            total += term
            term = term * (m - i) // (i + 1)
        return total
    term, below = 1, 0
    for i in range(k):
        below += term
        term = term * (m - i) // (i + 1)
    return (1 << m) - below


def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes out of n with p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = ((1 << n) - _half_tail_count(n, k + 1)) / 2**n  # P(X <= k)
    return min(1.0, 2 * tail)


MCNEMAR_ALTERNATIVES = ("two-sided", "greater", "less")


def mcnemar_exact(b: int, c: int, alternative: str = "two-sided") -> float:
    """Exact McNemar test on the discordant pairs of a paired pass/fail comparison.

    b = cases that passed in baseline and fail in candidate
    c = cases that failed in baseline and pass in candidate

    Under H0 (no change) each discordant pair is equally likely to go either way,
    so ``b ~ Binomial(b + c, 1/2)``.  ``alternative`` picks the tail:

    * ``"two-sided"`` (default): the candidate differs from the baseline in either direction;
    * ``"greater"``: the candidate is *worse* (b larger than chance allows), ``P(X >= b)``.
      This is the test the regression gate uses;
    * ``"less"``: the candidate is *better*, ``P(X <= b)``.
    """
    if alternative not in MCNEMAR_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {MCNEMAR_ALTERNATIVES}, got {alternative!r}")
    for name, value in (("b", b), ("c", c)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer count, got {value!r}")
    n = b + c
    if alternative == "two-sided":
        return binom_two_sided_p(b, n)
    if n == 0:
        return 1.0
    if alternative == "greater":
        return _half_tail_count(n, b) / 2**n
    return ((1 << n) - _half_tail_count(n, b + 1)) / 2**n


def cohen_kappa(a: Sequence, b: Sequence) -> float | None:
    """Cohen's kappa between two raters over the same items."""
    if len(a) != len(b):
        raise ValueError("rater sequences must have equal length")
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((list(a).count(label) / n) * (list(b).count(label) / n) for label in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def rogan_gladen(p_obs: float, sensitivity: float, specificity: float) -> float | None:
    """Correct an observed pass rate for a known-imperfect judge.

    sensitivity = P(judge says pass | truly pass)
    specificity = P(judge says fail | truly fail)
    Returns the estimated true pass rate clipped to [0, 1], or None when the
    judge is no better than chance (Youden's J <= 0).
    """
    j = sensitivity + specificity - 1
    if j <= 0:
        return None
    return min(1.0, max(0.0, (p_obs + specificity - 1) / j))


def kappa_label(kappa: float | None) -> str:
    """Landis & Koch style reading of kappa, used in reports."""
    if kappa is None:
        return "n/a"
    if kappa < 0.2:
        return "poor"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


# ----------------------------------------------------------------------------- run-to-run noise
#
# The regression gate's false-alarm and detection rates (``compare.expected_gate_rates``) are stated for a PER-VERDICT flip
# probability ``noise``: how often a case's verdict in one run differs from its latent verdict.  What a run records is
# different: ``summary.stability.flip_rate``, the share of repeated cases whose attempts did not all agree.  The two
# helpers below convert between the two, under the model the gate numbers use (every attempt of a case is wrong
# independently with the same probability).


def majority_noise_directions(attempt_noise: float, attempts: int) -> tuple[float, float]:
    """``(P(latent pass read as fail), P(latent fail read as pass))`` for the majority verdict of ``attempts`` attempts.

    Each attempt is wrong with probability ``attempt_noise``; the harness counts a tie as a fail.  For odd ``attempts`` the two
    numbers are equal.  For even ``attempts`` they are not: a latent pass is lost on a tie (``attempts / 2`` failures), a latent fail
    needs a strict majority of passes to be misread.  The pass -> fail direction is the one the regression gate trips on, so for
    even ``attempts`` the symmetric ``majority_noise`` understates the gate's false alarm (5% per attempt, 2 attempts, 45 cases:
    13.9% exact, 10.6% with the mean); ``compare.expected_gate_rates_protocol`` uses this pair.
    """
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"attempts must be a positive integer, got {attempts!r}")
    if isinstance(attempt_noise, bool) or not isinstance(attempt_noise, (int, float)) or not 0 <= attempt_noise <= 1:
        raise ValueError(f"attempt_noise must be a probability between 0 and 1, got {attempt_noise!r}")

    def tail(k_min: int) -> float:
        return sum(math.comb(attempts, j) * attempt_noise**j * (1 - attempt_noise) ** (attempts - j) for j in range(k_min, attempts + 1))

    return tail((attempts + 1) // 2), tail(attempts // 2 + 1)


def majority_noise(attempt_noise: float, attempts: int) -> float:
    """Probability that the majority verdict of ``attempts`` independent attempts is wrong.

    Each attempt is wrong with probability ``attempt_noise``.  Odd ``attempts``: the exact binomial tail.  Even
    ``attempts``: a tie counts as a fail (the harness rule), so the two directions differ and their MEAN is returned: a
    summary number, not the probability the gate's pass -> fail count sees (it is higher: ``majority_noise_directions``
    has both, and ``compare.expected_gate_rates_protocol`` uses them; the docs recommend odd ``attempts``).
    ``attempts = 1`` returns ``attempt_noise``.  Example: 5% per attempt becomes 0.725% with 3 attempts.
    """
    lost_pass, lost_fail = majority_noise_directions(attempt_noise, attempts)
    return lost_pass if attempts % 2 else (lost_pass + lost_fail) / 2  # odd: the two are the same tail (no rounding difference)


def noise_from_flip_rate(flip_rate: float, attempts: int | Sequence[int]) -> float:
    """Per-attempt flip probability that explains an observed ``flip_rate``: the share of repeated cases whose attempts disagree.

    If every attempt of a case is wrong independently with probability p (relative to the case's latent verdict), a case run
    k times disagrees with itself with probability ``1 - p**k - (1 - p)**k``.  ``attempts`` is k, or one k per repeated case
    (a protocol like safety x5, rest x3); the expected flip rate is their mean.  Returns the p in [0, 0.5] that matches
    ``flip_rate`` (0.5, the noisiest a verdict can be, when the flip rate is at least what coin flips give).

    This is the per-ATTEMPT noise.  The noise of a case VERDICT is ``majority_noise(p, k)`` when the gate runs every case k times
    (``p`` itself when it runs once).  Real cases are not equally noisy (a few flaky cases carry most of the flips), so read the
    result as an average, not a per-case guarantee.
    """
    if isinstance(flip_rate, bool) or not isinstance(flip_rate, (int, float)) or not 0 <= flip_rate <= 1 or math.isnan(flip_rate):
        raise ValueError(f"flip_rate must be a share between 0 and 1, got {flip_rate!r}")
    try:
        ks = [attempts] if isinstance(attempts, int) and not isinstance(attempts, bool) else list(attempts)
    except TypeError:
        ks = []
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in ks):
        raise ValueError(f"attempts must be an integer >= 2 (or a list of them): a case run once cannot flip, got {attempts!r}")

    def expected(p: float) -> float:
        return sum(1 - p**k - (1 - p) ** k for k in ks) / len(ks)

    if flip_rate <= 0:
        return 0.0
    if flip_rate >= expected(0.5):
        return 0.5
    lo, hi = 0.0, 0.5  # expected() increases on [0, 0.5]
    for _ in range(100):
        mid = (lo + hi) / 2
        if expected(mid) < flip_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _binom_cdf_log(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p), summed in log space (n up to many thousands)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq, lf = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    return min(1.0, sum(math.exp(lf - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)))


def clopper_pearson_upper(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound for a proportion: the p at which ``P(X <= successes) = 1 - confidence``.

    ``successes`` out of ``n`` trials; the bound is 1.0 when every trial succeeded and ``1 - (1 - confidence) ** (1 / n)`` when none did.
    """
    for name, value in (("successes", successes), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if n < 1 or successes > n:
        raise ValueError(f"need 0 <= successes <= n and n >= 1, got {successes} of {n}")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}")
    if successes == n:
        return 1.0
    target = 1.0 - confidence
    lo, hi = successes / n, 1.0  # P(X <= successes) falls as p grows: 0.5-ish at p = x/n, 0 at p = 1
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf_log(successes, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def attempt_noise_estimate(n_flipped: int, attempts_of_repeated: Sequence[int]) -> dict:
    """The per-attempt noise a run measured on its repeated cases, and an upper 95% bound for it.

    ``n_flipped`` of the repeated cases had attempts that disagreed (``summary.stability``: ``flip_rate * n_repeated``);
    ``attempts_of_repeated`` holds one attempt count (>= 2) per repeated case.  ``noise`` is ``noise_from_flip_rate`` at the observed flip
    rate; ``noise_upper95`` maps the one-sided 95% Clopper-Pearson upper bound of the flip rate through the same function: a
    statement about how noisy the repeated cases could still be, which matters most when none of them flipped (``noise`` is then 0
    and says nothing for a handful of cases).  Both are averages over the repeated cases: applying them to cases that were not repeated
    is an assumption.  Returns ``{"noise", "noise_upper95", "n_repeated", "n_flipped", "flip_rate", "flip_rate_upper95"}``; with no
    repeated case the noise entries are None.
    """
    try:
        ks = list(attempts_of_repeated)
    except TypeError:
        raise ValueError(f"attempts_of_repeated must be a list of attempt counts, got {attempts_of_repeated!r}") from None
    m = len(ks)
    if isinstance(n_flipped, bool) or not isinstance(n_flipped, int) or not 0 <= n_flipped <= m:
        raise ValueError(f"n_flipped must be an integer between 0 and the number of repeated cases ({m}), got {n_flipped!r}")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in ks):
        raise ValueError(f"every repeated case ran at least 2 attempts, got {ks[:5]!r}")
    if m == 0:
        return {"noise": None, "noise_upper95": None, "n_repeated": 0, "n_flipped": 0, "flip_rate": None, "flip_rate_upper95": None}
    flip = n_flipped / m
    upper = clopper_pearson_upper(n_flipped, m, 0.95)
    return {
        "noise": noise_from_flip_rate(flip, ks),
        "noise_upper95": noise_from_flip_rate(upper, ks),
        "n_repeated": m,
        "n_flipped": n_flipped,
        "flip_rate": flip,
        "flip_rate_upper95": upper,
    }


def binom_pmf_window(m: int, q: float, cutoff: float = 1e-18) -> tuple[int, tuple[float, ...]]:
    """Binomial(m, q) probabilities, as ``(lo, probs)``: ``probs[i]`` is P(X = lo + i), for the values above ``cutoff``.

    The mass outside the window is below ``m * cutoff`` (negligible: the gate rates are quoted to a fraction of a percent).
    Built from the mode outwards, so it costs the width of the window, not m, and never underflows.  Cached: callers
    get an immutable tuple.
    """
    # Validate before the cache: lru_cache keys compare equal for 3 / 3.0 / True and 1, so a bad argument would hit the entry of a good one.
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError(f"m must be a non-negative integer, got {m!r}")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q <= 1:
        raise ValueError(f"q must be a probability between 0 and 1, got {q!r}")
    return _binom_pmf_window(m, float(q), float(cutoff))


@functools.lru_cache(maxsize=8192)
def _binom_pmf_window(m: int, q: float, cutoff: float) -> tuple[int, tuple[float, ...]]:
    if q <= 0.0:
        return 0, (1.0,)
    if q >= 1.0:
        return m, (1.0,)
    mode = min(m, int((m + 1) * q))
    at_mode = math.exp(
        math.lgamma(m + 1) - math.lgamma(mode + 1) - math.lgamma(m - mode + 1) + mode * math.log(q) + (m - mode) * math.log1p(-q)
    )
    odds = q / (1 - q)
    up: list[float] = []
    value, k = at_mode, mode
    while k < m:
        value *= (m - k) / (k + 1) * odds
        k += 1
        if value < cutoff:
            break
        up.append(value)
    down: list[float] = []
    value, k = at_mode, mode
    while k > 0:
        value *= k / (m - k + 1) / odds
        k -= 1
        if value < cutoff:
            break
        down.append(value)
    return mode - len(down), (*reversed(down), at_mode, *up)
