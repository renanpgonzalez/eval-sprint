"""Exact false-alarm / detection rates of the default gate (`max_pass_rate_drop: auto`, contract C9), for the docs tables.

Independent re-derivation (stdlib only, no evalsprint import) of the cycle-1 critic's model B:
  * a fixed set of n cases, baseline pass rate 80% (latent verdicts);
  * baseline and candidate are two INDEPENDENT noisy readings of the latent verdicts: every case verdict flips
    with probability `noise` in each reading (no drift; this is the physical null: a committed baseline vs a
    fresh candidate run of the same system);
  * detection: the candidate system truly loses `d` of the cases (latent pass -> fail), then the same noise applies;
  * repeat: a case run k times takes the majority verdict (a tie fails). With independent attempts a case's verdict
    flips with probability q(k, noise) = P(at least ceil(k/2) of k attempts flip) instead of `noise`, so every
    number for `repeat: {n: k}` is the same sum with noise replaced by q. The independence of the attempts is an
    ASSUMPTION: noise that is the same on every attempt of a case (a case the judge always gets wrong) is not
    reduced by repeating it, so this is the best case for repeat.
Rule (contract C9), applied to the paired counts b (pass->fail) and c (fail->pass) among n paired cases:
  regression  <=>  b - c >= k(n) = max(3, ceil(n / 50))   AND   P(X >= b | X ~ Bin(b + c, 1/2)) < alpha(n)
  alpha(n) = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)
Errors do not occur in this model (they are excluded from pairing before the rule is applied).

Further exact derivations (v0.2.0 docs, every figure they quote is recomputed by tests/test_templates.py):
  * `flag01`, `rate_rule`, `v01_better_sizes`, `v01_means`: the rule of v0.1 (net loss above 5 p.p. under 150 cases; above 3 p.p. and
    p < 0.05 from 150), to say where v0.1 was better (60 to 149 cases: lower false alarm AND higher detection at 84 of 90 sizes AT 5%
    NOISE; at 10% noise at none of them, v0.2 is the stricter rule: mean false alarm 8.3% against 11.5%, mean detection of a
    10 p.p. drop 61.2% against 72.9%);
  * `mixed_false_alarm`, `critical_rule_false_alarm`, `whole_gate_false_alarm`: cases with different attempts (repeat: {n: 3, per_tag:
    {critical: 5}}): the pass-rate rule over the sum of independent classes, the critical-tag rule (any critical case that goes
    pass -> fail), and the two together, EXACT (the joint distribution of the paired counts; the docs quote this one:
    36.67 / 5.79 / 0.92 / 1.11 / 9.37% at 45 cases, 8 critical, 5% noise). `total_false_alarm` is the independence approximation
    the docs used before (39.41 / 5.92 / 0.92 / 1.13 / 9.47%), kept to show the size of the difference;
  * `crash_false_alarm`, `crash_blocked_or_unhealthy`: a target that crashes on a fraction of the candidate's cases; a crash on a case
    that passed in the baseline counts as a failure (contract C10), so crashes are extra pass -> fail pairs; the second one adds the
    unhealthy runs (more than `max_error_rate` of the cases crashed), which are never compared;
  * `retry_share_warning_rate`: the warning of `compare` when the candidate retried at least `retry_share_warn` (10 p.p.) more of its target
    calls than the baseline did (C22), exact: every call crashes independently, a crashed call is repeated up to `target_retries` times,
    so a run of N calls retries the sum of N such counts (the 0.12% false warning at 100 calls against a baseline that retries on 5% of
    them, and the 11.2 / 88.8 / 100% of a candidate that crashes on 10 / 20 / 50% of its calls);
  * `lottery`: the baseline is ONE noisy draw that stays committed; the false alarm of the later PRs, given that baseline, varies a lot.

Re-run:  python3 scripts/gate_numbers.py                      (prints the tables)
         python3 scripts/gate_numbers.py --json numbers.json  (also writes them as JSON)
Cross-check: agrees with the gate executor's exact enumeration
(orchestration/2026-09-29_kit-v0.2/cycle3/eval-scientist-gate/gate_rates.md) and, through tests/test_templates.py,
with `evalsprint.compare.auto_trips` and `evalsprint.compare.expected_gate_rates`. The 2, 5 and 10% noise levels are an
ASSUMPTION of the model, not a measurement. Standard library only; takes under a minute.
"""
import argparse
import json
import math
import pathlib
from functools import lru_cache

EPS = 1e-12


def min_net_loss(n: int) -> int:
    """Smallest net loss (b - c) that can trip the gate: 3 cases, and 2 p.p. of the paired cases from n = 150 on."""
    return max(3, -(-n // 50))


def alpha(n: int) -> float:
    """One-sided significance level: 0.25 for tiny suites, falling smoothly to 0.0575."""
    return 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)


@lru_cache(maxsize=None)
def p_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar: P(X >= b) for X ~ Binomial(b + c, 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, sum(math.comb(m, k) for k in range(b, m + 1)) / 2**m)


def flag(b: int, c: int, n: int) -> bool:
    return b - c >= min_net_loss(n) and p_one_sided(b, c) < alpha(n) - EPS


def majority_flip(attempts: int, noise: float) -> float:
    """P(the majority verdict of `attempts` independent attempts differs from the latent one), for an odd number of attempts.

    With an even number a tie fails, which flips a latent pass but not a latent fail: the two directions then differ and
    model B's single flip probability no longer describes them, so this function refuses them (the docs recommend 3 and 5).
    """
    if attempts % 2 == 0:
        raise ValueError("majority_flip: use an odd number of attempts (a tie fails, which is not symmetric)")
    return sum(math.comb(attempts, j) * noise**j * (1 - noise) ** (attempts - j) for j in range(attempts // 2 + 1, attempts + 1))


def dist(n: int, noise: float, d: float):
    """Joint distribution of (b, c) under model B with latent drop d (fraction of all cases)."""
    p = noise
    ub = p * (1 - p) * (1 - d) + d * (1 - p) ** 2  # P(pass in baseline reading, fail in candidate reading)
    uc = p * (1 - p) * (1 - d) + d * p**2  # P(fail in baseline reading, pass in candidate reading)
    z = 1 - ub - uc
    out = {}
    for b in range(n + 1):
        for c in range(n - b + 1):
            rest = n - b - c
            if (ub == 0 and b) or (uc == 0 and c) or (z <= 0 and rest):
                continue
            lg = math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(rest + 1)
            lg += (b * math.log(ub) if b else 0) + (c * math.log(uc) if c else 0) + (rest * math.log(z) if rest else 0)
            pr = math.exp(lg)
            if pr > 1e-16:
                out[(b, c)] = pr
    return out


@lru_cache(maxsize=None)
def rate(n: int, noise: float, d: float) -> float:
    """P(the gate trips) for n paired cases, per-reading flip probability `noise`, and a true drop `d`."""
    return sum(pr for (b, c), pr in dist(n, noise, d).items() if flag(b, c, n))


def repeat_rate(n: int, noise: float, d: float, attempts: int) -> float:
    """The same with every case repeated `attempts` times (independent attempts, majority verdict)."""
    return rate(n, majority_flip(attempts, noise), d)


# ---------------------------------------------------------------------------------------------- v0.1 rule


def flag01(b: int, c: int, n: int) -> bool:
    """The default gate of v0.1: net loss above 5 p.p. under 150 paired cases; from 150, above 3 p.p. and one-sided McNemar p < 0.05."""
    if n < 150:
        return 20 * (b - c) > n
    return 100 * (b - c) > 3 * n and p_one_sided(b, c) < 0.05 - EPS


@lru_cache(maxsize=None)
def rate_rule(n: int, noise: float, d: float, rule) -> float:
    """P(the gate trips) under model B for any rule `rule(b, c, n) -> bool` (the current one is `flag`)."""
    return sum(pr for (b, c), pr in dist(n, noise, d).items() if rule(b, c, n))


def v01_better_sizes(noise: float = 0.05, sizes=range(60, 150), drop: float = 0.10) -> list[int]:
    """Sizes at which the rule of v0.1 has a LOWER false alarm and a HIGHER detection of a `drop` than the current rule (strict, both)."""
    return [n for n in sizes if rate_rule(n, noise, 0.0, flag01) < rate(n, noise, 0.0) and rate_rule(n, noise, drop, flag01) > rate(n, noise, drop)]


def v01_means(noise: float, sizes=range(60, 150), drop: float = 0.10) -> dict:
    """Means over `sizes` (default 60 to 149 cases) of the false alarm and the detection of a `drop`, for the current rule and for v0.1."""
    sizes = list(sizes)
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    return {
        "false_alarm_v02": mean([rate(n, noise, 0.0) for n in sizes]),
        "false_alarm_v01": mean([rate_rule(n, noise, 0.0, flag01) for n in sizes]),
        "detection_v02": mean([rate(n, noise, drop) for n in sizes]),
        "detection_v01": mean([rate_rule(n, noise, drop, flag01) for n in sizes]),
    }


# -------------------------------------------------------------------------------- cases with different attempts


def _convolve_pairs(x: dict, y: dict) -> dict:
    out: dict = {}
    for (b1, c1), p1 in x.items():
        for (b2, c2), p2 in y.items():
            key = (b1 + b2, c1 + c2)
            out[key] = out.get(key, 0.0) + p1 * p2
    return out


def mixed_false_alarm(classes: list[tuple[int, float]]) -> float:
    """False alarm of the pass-rate rule when the cases come in classes `(count, per-verdict flip probability)`.

    Independent classes: the paired counts (b, c) add up. A class run with k attempts has flip probability `majority_flip(k, noise)`.
    """
    n = sum(count for count, _ in classes)
    total: dict = {(0, 0): 1.0}
    for count, q in classes:
        if count:
            total = _convolve_pairs(total, dist(count, q, 0.0))
    return sum(pr for (b, c), pr in total.items() if flag(b, c, n))


def critical_rule_false_alarm(cases: int, attempts: int, noise: float = 0.05) -> float:
    """P(at least one of `cases` critical cases goes pass -> fail in an unchanged system), each run with `attempts` attempts."""
    q = majority_flip(attempts, noise) if attempts > 1 else noise
    return 1 - (1 - q * (1 - q)) ** cases


def total_false_alarm(n: int, critical: int, critical_attempts: int, other_attempts: int, noise: float = 0.05) -> float:
    """False alarm of the pass-rate rule OR the critical-tag rule (independence APPROXIMATION) for `critical` of `n` cases run with
    `critical_attempts` attempts and the rest with `other_attempts`: `repeat: {n: other_attempts, per_tag: {critical: critical_attempts}}`.

    The two rules share the critical cases (a critical case that flips to fail also counts in the pass-rate rule), so they are not
    independent; the product below overstates the whole gate slightly. v0.2.0 docs quote `whole_gate_false_alarm`, which is exact;
    this one is kept to show how far the approximation was (5.92% vs 5.79%, 39.41% vs 36.67%)."""
    flip = lambda k: majority_flip(k, noise) if k > 1 else noise  # noqa: E731
    pass_rate_rule = mixed_false_alarm([(n - critical, flip(other_attempts)), (critical, flip(critical_attempts))])
    return 1 - (1 - pass_rate_rule) * (1 - critical_rule_false_alarm(critical, critical_attempts, noise))


def whole_gate_false_alarm(n: int, critical: int, critical_attempts: int, other_attempts: int, noise: float = 0.05) -> float:
    """EXACT false alarm of the whole default gate on an unchanged system: P(pass-rate rule trips OR any critical case goes pass -> fail).

    The cases come in two independent classes (critical, with `critical_attempts` attempts; the rest, with `other_attempts`), each
    with its own flip probability. The pass-rate rule counts the critical cases too, so the exact value tracks, over the joint
    distribution of the paired counts (b, c), the event "no critical case went pass -> fail" and asks the pass-rate rule only on it:
        quiet = P(no critical b  AND  pass-rate rule quiet);   whole = 1 - quiet.
    Matches the methodology critic's enumeration of cycle 4 (36.67 / 5.79 / 0.92 / 1.11 / 9.37% for 45 cases, 8 critical, 5% noise).
    """
    flip = lambda k: majority_flip(k, noise) if k > 1 else noise  # noqa: E731
    pc = dist(critical, flip(critical_attempts), 0.0) if critical else {(0, 0): 1.0}
    pn = dist(n - critical, flip(other_attempts), 0.0) if n - critical else {(0, 0): 1.0}
    quiet = 0.0
    for (b_crit, c_crit), p_crit in pc.items():
        if b_crit:  # a critical case went pass -> fail: the critical rule has tripped already
            continue
        for (b_oth, c_oth), p_oth in pn.items():
            if not flag(b_oth, c_oth + c_crit, n):
                quiet += p_crit * p_oth
    return 1 - quiet


# --------------------------------------------------------------------------------------- target that crashes


def crash_false_alarm(n: int, noise: float, crash: float, latent_pass: float = 0.8) -> float:
    """False alarm when an unchanged target crashes on a fraction `crash` of the candidate's cases.

    A crash on a case that passed in the baseline counts as a failure (b + 1); a crash on a case that failed changes nothing. Per
    case, with latent verdict pass or fail and independent readings that flip with probability `noise`:
    b (baseline pass -> candidate fail) and c (baseline fail -> candidate pass) are exclusive events of the case.
    """
    p, r = noise, crash
    n_pass = round(latent_pass * n)
    total: dict = {(0, 0): 1.0}
    for count, is_pass in ((n_pass, True), (n - n_pass, False)):
        pb = (1 - p) * (r + (1 - r) * p) if is_pass else p * (r + (1 - r) * (1 - p))
        pc = p * (1 - r) * (1 - p) if is_pass else (1 - p) * (1 - r) * p
        for _ in range(count):
            step: dict = {}
            for (b, c), w in total.items():
                step[(b + 1, c)] = step.get((b + 1, c), 0.0) + w * pb
                step[(b, c + 1)] = step.get((b, c + 1), 0.0) + w * pc
                step[(b, c)] = step.get((b, c), 0.0) + w * (1 - pb - pc)
            total = step
    return sum(w for (b, c), w in total.items() if flag(b, c, n))


def crash_blocked_or_unhealthy(
    n: int, noise: float, crash: float, latent_pass: float = 0.8, max_error_rate: float = 0.05, cutoff: float = 1e-16
) -> dict:
    """Unchanged system, a target that crashes on a fraction `crash` of the candidate's cases (independent per call).

    Returns `unhealthy` (more than `max_error_rate` of the cases crashed: the run is never compared), `trips` (the run is healthy and
    the gate trips on it, a crash on a case that passed counting as a failure) and `blocked` (their sum: what stops the pull request).
    One retry inside the run (`target_retries: 1`) makes a case end in an error only if two calls in a row crash, so its effect is the
    same table with `crash` squared (independent calls).
    """
    allowed = math.floor(max_error_rate * n + 1e-9)  # errors that keep the run healthy
    n_pass = round(latent_pass * n)
    state: dict = {(0, 0, 0): 1.0}  # (b, c, crashed cases) -> probability, healthy runs only
    unhealthy = 0.0
    for count, is_pass in ((n_pass, True), (n - n_pass, False)):
        base_pass = (1 - noise) if is_pass else noise  # P(the baseline reading of the case is a pass)
        cand_pass = base_pass  # same latent verdict, same flip probability
        for _ in range(count):
            step: dict = {}

            def add(key, weight):
                step[key] = step.get(key, 0.0) + weight

            for (b, c, e), w in state.items():
                for baseline_passed, pb in ((True, base_pass), (False, 1 - base_pass)):
                    wc = w * crash * pb  # the target crashes on the candidate's case
                    if wc >= cutoff:
                        if e + 1 > allowed:
                            unhealthy += wc
                        else:
                            add((b + (1 if baseline_passed else 0), c, e + 1), wc)
                    for candidate_passed, pc in ((True, cand_pass), (False, 1 - cand_pass)):
                        wn = w * (1 - crash) * pb * pc
                        if wn >= cutoff:
                            add((b + (baseline_passed and not candidate_passed), c + (candidate_passed and not baseline_passed), e), wn)
            state = step
    trips = sum(w for (b, c, _e), w in state.items() if flag(b, c, n))
    return {"unhealthy": unhealthy, "trips": trips, "blocked": unhealthy + trips}


# --------------------------------------------------------------------------------- the retry-share warning (C22)


def retries_pmf(n: int, crash: float, retries: int) -> list[float]:
    """P(a run of `n` target calls repeats k calls in all), k = 0 .. n * retries.

    Each call crashes independently with probability `crash` and is repeated until it answers or `retries` repeats were made, so one call
    uses j repeats with probability crash**j * (1 - crash) for j < retries and crash**retries for j = retries."""
    one = [crash**j * (1 - crash) for j in range(retries)] + [crash**retries]
    dist = [1.0]
    for _ in range(n):
        step = [0.0] * (len(dist) + retries)
        for i, weight in enumerate(dist):
            if weight:
                for j, w in enumerate(one):
                    step[i + j] += weight * w
        dist = step
    return dist


def retry_share_warning_rate(n: int, base_crash: float, cand_crash: float, retries: int = 1, limit: float = 0.10) -> float:
    """P(`compare` warns that the candidate retried more): its share of repeated calls (repeats over the n calls) is at least `limit` above
    the baseline's. `base_crash` and `cand_crash` are the per-call crash probabilities of the two runs (equal: an unchanged target)."""
    base, cand = retries_pmf(n, base_crash, retries), retries_pmf(n, cand_crash, retries)
    need = limit * n - EPS  # share difference >= limit  <=>  repeats_candidate - repeats_baseline >= limit * n
    cumulative, acc = [], 0.0
    for weight in base:
        acc += weight
        cumulative.append(acc)
    total = 0.0
    for repeats, weight in enumerate(cand):
        reach = repeats - need  # the baseline must have repeated at most this many calls
        if weight > 0 and reach >= 0:
            total += weight * cumulative[min(int(reach + EPS), len(cumulative) - 1)]
    return total


# --------------------------------------------------------------------------------- the committed-baseline lottery


def _pmf(m: int, p: float) -> list[float]:
    return [math.comb(m, k) * p**k * (1 - p) ** (m - k) for k in range(m + 1)]


def _convolve(x: list[float], y: list[float]) -> list[float]:
    out = [0.0] * (len(x) + len(y) - 1)
    for i, u in enumerate(x):
        if u < 1e-300:
            continue
        for j, v in enumerate(y):
            out[i + j] += u * v
    return out


def lottery(n: int, noise: float, latent_pass: float = 0.8, cutoff: float = 1e-13) -> dict:
    """The committed baseline is one noisy draw; every later PR is a fresh one. Given the baseline the false alarm is exact and varies.

    The system is unchanged and has round(latent_pass * n) latent passes. The baseline misread `e` of them as fail and `a` latent fails
    as pass (Binomial draws). Given (e, a): b = Bin(n_pass - e, noise) + Bin(a, 1 - noise), c = Bin(e, 1 - noise) + Bin(n_fail - a,
    noise). Returns {"p_over_15": P(that false alarm is above 15%), "median", "p95", "mean"} over baseline draws.
    """
    n_pass = round(latent_pass * n)
    n_fail = n - n_pass
    # largest c that still trips, for each b (the rule is monotone in c)
    cmax = []
    for b in range(n + 1):
        best = -1
        for c in range(0, n - b + 1):
            if flag(b, c, n):
                best = c
            elif c > 0 and b - c < min_net_loss(n):
                break
        cmax.append(best)
    draws = []
    for e, we in enumerate(_pmf(n_pass, noise)):
        for a, wa in enumerate(_pmf(n_fail, noise)):
            weight = we * wa
            if weight < cutoff:
                continue
            pb = _convolve(_pmf(n_pass - e, noise), _pmf(a, 1 - noise))
            pc = _convolve(_pmf(e, 1 - noise), _pmf(n_fail - a, noise))
            cdf, acc = [], 0.0
            for v in pc:
                acc += v
                cdf.append(acc)
            fa = sum(v * cdf[min(cmax[b], len(cdf) - 1)] for b, v in enumerate(pb) if b <= n and cmax[b] >= 0)
            draws.append((min(fa, 1.0), weight))
    draws.sort()
    mass = sum(w for _, w in draws)

    def quantile(q: float) -> float:
        acc = 0.0
        for fa, w in draws:
            acc += w
            if acc >= q * mass:
                return fa
        return draws[-1][0]

    return {
        "p_over_15": sum(w for fa, w in draws if fa > 0.15) / mass,
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "mean": sum(fa * w for fa, w in draws) / mass,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="False-alarm and detection rates of the default regression gate (model B).")
    ap.add_argument("--json", metavar="PATH", help="also write the tables to this JSON file")
    args = ap.parse_args(argv)
    res = {"false_alarm": [], "detection": [], "repeat": []}
    print("FALSE ALARM (identical system), %")
    print("   n | noise 2% | noise 5% | noise 10%")
    for n in (45, 100, 149, 150, 200, 400, 800):
        row = {"n": n, **{f"noise_{int(p * 100)}": rate(n, p, 0.0) for p in (0.02, 0.05, 0.10)}}
        res["false_alarm"].append(row)
        print(f"{n:4d} | " + " | ".join(f"{row[f'noise_{k}'] * 100:7.1f}%" for k in (2, 5, 10)))
    print("\nDETECTION of a true drop, noise 5%, baseline pass rate 80%, %")
    print("   n |  3 p.p. |  5 p.p. | 10 p.p. | 15 p.p. | 20 p.p.")
    for n in (45, 100, 200, 400, 800):
        row = {"n": n, **{f"drop_{int(d * 100)}": rate(n, 0.05, d) for d in (0.03, 0.05, 0.10, 0.15, 0.20)}}
        res["detection"].append(row)
        print(f"{n:4d} | " + " | ".join(f"{row[f'drop_{k}'] * 100:6.1f}%" for k in (3, 5, 10, 15, 20)))
    print("\nREPEAT: every case run once vs the majority of 3 attempts (independent attempts), noise 5% per attempt, %")
    print("   n | false alarm 1x -> 3x | detection 10 p.p. 1x -> 3x | detection 5 p.p. 1x -> 3x")
    for n in (30, 45, 60, 100, 149):
        row = {"n": n}
        for label, d in (("false_alarm", 0.0), ("detect_10", 0.10), ("detect_5", 0.05)):
            row[label] = [rate(n, 0.05, d), repeat_rate(n, 0.05, d, 3)]
        res["repeat"].append(row)
        cells = [f"{row[k][0] * 100:5.1f}% -> {row[k][1] * 100:5.1f}%" for k in ("false_alarm", "detect_10", "detect_5")]
        print(f"{n:4d} | " + " | ".join(cells))
    print(f"\nflip probability of a repeated case (majority of 3) at 5% noise per attempt: {majority_flip(3, 0.05):.5f}")
    print("\nWHOLE GATE (pass-rate rule OR critical rule), 45 cases, 8 critical, 5% noise per attempt, exact, %")
    print("  critical x | others x | critical rule | pass-rate rule | whole gate")
    res["whole_gate"] = []
    for critical_attempts, other_attempts in ((1, 1), (3, 3), (5, 5), (5, 3), (5, 1)):
        flip = lambda k: majority_flip(k, 0.05) if k > 1 else 0.05  # noqa: E731
        row = {
            "critical_attempts": critical_attempts,
            "other_attempts": other_attempts,
            "critical_rule": critical_rule_false_alarm(8, critical_attempts),
            "pass_rate_rule": mixed_false_alarm([(37, flip(other_attempts)), (8, flip(critical_attempts))]),
            "whole_gate": whole_gate_false_alarm(45, 8, critical_attempts, other_attempts),
        }
        res["whole_gate"].append(row)
        print(f"  {critical_attempts:10d} | {other_attempts:8d} | {row['critical_rule'] * 100:12.2f}% | {row['pass_rate_rule'] * 100:13.2f}% | {row['whole_gate'] * 100:9.2f}%")
    print("\nRETRY SHARE WARNING (retry_share_warn 0.10, target_retries 1, 100 calls), % of the comparisons that warn")
    print("  baseline crashes on | candidate crashes on | warns")
    res["retry_share_warning"] = []
    for base_crash, cand_crash in ((0.05, 0.05), (0.05, 0.10), (0.05, 0.20), (0.05, 0.50), (0.0, 0.10), (0.0, 0.20), (0.0, 0.50)):
        row = {"baseline_crash": base_crash, "candidate_crash": cand_crash, "warns": retry_share_warning_rate(100, base_crash, cand_crash)}
        res["retry_share_warning"].append(row)
        print(f"  {base_crash:18.0%} | {cand_crash:20.0%} | {row['warns'] * 100:5.2f}%")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
