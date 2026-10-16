"""Baseline-vs-candidate comparison and the CI regression gate.

`compare_runs` answers one question: *did this change make the LLM feature
worse?*  It pairs the two runs case by case, computes what moved (pass rate,
judge score, cost, latency, errors), and applies the gate rules from
`eval.yaml` to produce a verdict.  `to_markdown` renders the result for a
GitHub PR comment / job summary.

Design decisions worth knowing before changing anything here:

* **Paired, not pooled.**  Baseline and candidate answer the *same* cases, so
  the right test is McNemar's exact test on the discordant pairs, and the right
  interval for a score change is a paired bootstrap.  Pooled two-sample tests
  throw away the pairing and are much less sensitive on small test sets.
* **Only comparable cases feed the deltas.**  A case is paired when it exists in
  both runs with the same ``case_sha`` (same content) and has a verdict in the
  baseline.  Removed and edited cases are reported and gated separately, never
  mixed into a delta, so a delta compares like with like and a PR cannot improve
  its numbers by deleting or rewriting the cases it fails.
* **Errors are not all alike.**  A *judge* error (outage, unparsable verdict) and a
  case that already errored in the *baseline* say nothing about quality: they are
  left out of the pass rates and counted in ``errors``.  A *target* error the
  candidate introduces on a case that passed in the baseline is a product
  regression (the feature crashed where it used to answer): it is paired and
  counts as a failure in the delta, McNemar's b count, by-tag rates and critical
  tags, and is listed in ``target_error_regressions``.  Otherwise a PR could turn
  regressions into "errors" and stay under every limit.
* **The default drop rule is ONE rule for every n** (``auto``, ``gate_mode: "auto"``), so quoting its false-alarm
  and detection rates never needs a footnote about a switch.  It trips when BOTH hold: the net loss
  (b - c, cases that went pass->fail minus fail->pass) is at least ``auto_min_net_loss(n) = max(3, ceil(n/50))``
  cases ("at least 3 cases and at least 2 p.p."), AND the one-sided exact McNemar p-value is below
  ``auto_alpha(n) = 0.0575 + 0.1925 / (1 + (n/85)**4.5)``.  A small suite has little data, so it gets a generous alpha
  (0.24 at n = 45) and the 3-case floor; a large one a strict alpha (0.0575 as n grows) and the 2 p.p. floor.
  It is exact (a finite set of (b, c) pairs per n), so the rates in
  orchestration/2026-09-29_kit-v0.2/cycle3/eval-scientist-gate/gate_rates.md can be recomputed with
  ``auto_trips``.  ``max_pass_rate_drop: 0.03`` (a number) is the fixed mode of v0.1, unchanged.
* **Absolute rules cannot be diluted.**  The pass-rate floor and the error-rate
  ceiling are checked on the whole candidate run *and* on the cases that also
  exist in the baseline (target errors count as failures there), so appending
  easy cases cannot lift the first or push the second under its limit.
* **Records are the source of truth.**  Everything is recomputed from
  ``results.jsonl`` records; only the run id, the ``gate`` block and the protocol
  fields (``judges_sha``, ``repeat``, the evalsprint version) are read from
  ``summary.json``.  Records written before ``outcome``/``case_sha``
  existed still work (outcome is derived, pairing falls back to ids, with a warning).
* **A gate must not silently switch itself off.**  Unknown gate keys raise,
  and rules that are configured but cannot be evaluated (no cost data,
  a baseline cost of zero, a critical tag no case carries) surface as
  ``warnings`` instead of passing quietly.
* **The evaluation protocol is part of the baseline.**  Custom judge code is inside ``judges_sha``; the ``repeat``
  protocol (``summary.repeat``: ``{"n": 3, "per_tag": {"safety": 5}}``, the legacy ``{"n": 5, "tags": ["safety"]}``
  being the same protocol as ``{"n": 1, "per_tag": {"safety": 5}}``) and the evalsprint version are compared between the
  two runs, so a PR cannot get an easier majority vote (``repeat``) or a different scoring code path by changing the
  harness instead of the feature.  A baseline recorded from a placeholder run (``summary.placeholder``: the scaffold's
  offline stub or a mock judge) is flagged: its verdicts say nothing about the real product.
* **Attempts are counted, not only cases.**  A repeated case whose majority verdict is pass can hide a target error
  in one attempt; ``errors.*.attempt_error_rate`` and the flaky-crash warning make that visible.
* **The gate's own error rates are computable.**  ``expected_gate_rates(n, noise)`` enumerates exactly (no simulation) how
  often the rule false-alarms on an unchanged system and how often it catches a real drop, for n paired cases and a
  per-verdict noise, plus the "baseline lottery": the committed baseline is ONE noisy reading, so the false-alarm rate of
  every later PR depends on how lucky that draw was (``evalsprint noise`` prints it).
* **Text that embeds user data is escaped in the markdown, not in the JSON.**  Case ids, tag names, run ids and the
  evalsprint version come from files a PR controls: ``to_markdown`` escapes backticks, ``|``, ``<``, ``>``, ``[``, ``]`` and
  backslashes in every reason and warning, puts ids in single-backtick code spans (an id with a backtick in it is shown
  escaped instead), and turns control characters into spaces, so an id can neither inject markup nor start a new block.
* **An explicit ``gate`` is authoritative.**  The candidate summary's own gate
  (written by whoever ran the candidate, i.e. the PR) is only consulted when no
  ``gate`` is passed, so a PR cannot loosen the rules it is judged by.
"""

from __future__ import annotations

import functools
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .gate_defaults import DEFAULT_GATE
from .stats import (
    binom_pmf_window,
    majority_noise,
    majority_noise_directions,
    mcnemar_exact,
    mean,
    paired_bootstrap_diff_ci,
    percentile,
    wilson_ci,
)

if TYPE_CHECKING:  # avoid pulling the runner (and every provider) in for a type hint
    from .runner import RunResult

# Guards float noise: 0.94 - 0.91 is 0.030000000000000027, which must not count
# as "more than 3 p.p.".
_EPS = 1e-9

UNTAGGED = "(untagged)"  # same label runner.summarize uses

# `max_pass_rate_drop: auto`: one rule for every number of paired cases n (see auto_trips).
AUTO_MIN_NET_LOSS = 3  # never trip on fewer than 3 net lost cases ...
AUTO_MIN_DROP_DENOMINATOR = 50  # ... nor on a net loss below 1/50 of the paired cases (2 p.p.)
AUTO_ALPHA_LARGE = 0.0575  # alpha(n) tends to this for large suites
AUTO_ALPHA_SPAN = 0.1925  # alpha(0) = AUTO_ALPHA_LARGE + AUTO_ALPHA_SPAN = 0.25
AUTO_ALPHA_MIDPOINT = 85  # n at which alpha is halfway between its two extremes
AUTO_ALPHA_STEEPNESS = 4.5
LEGACY_GATE_MODES = {"auto-small": "auto", "auto-large": "auto"}  # values written by evalsprint 0.1 into comparison files

SMALL_SET = 100  # below this many paired cases, tell the reader which drops can be detected
_RELIABLE_POWER = 0.8  # "reliably detected" = caught at least this often
_ASSUMED_NOISE = 0.05  # per-case run-to-run flip probability behind that statement
RECOMMENDED_CRITICAL_REPEATS = 5  # attempts per critical case that keep false alarms ~1% for 8 cases at 5% noise
MAX_LISTED_IDS = 15  # case ids shown per list in the markdown
MAX_TAG_ROWS = 12  # by-tag table: show every tag up to this many, else only changed ones
_MAX_REASON_IDS = 5  # case ids quoted inside a single gate reason

_PCT_KEYS = ("max_cost_increase_pct", "max_p95_latency_increase_pct")
_FRACTION_KEYS = ("alpha", "min_pass_rate", "max_error_rate", "retry_share_warn")  # fractions in [0, 1]; a value like `3` would mean "3 p.p."
_COUNT_KEYS = ("max_removed_cases", "max_changed_cases")
_BOOL_KEYS = ("require_significance", "require_calibration", "allow_judge_changes", "allow_repeat_change", "allow_target_retries_change", "require_same_kit_version")


# --------------------------------------------------------------------------- gate


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resolve_gate(candidate_summary: dict, gate: dict | None = None) -> dict:
    """Effective gate: DEFAULT_GATE overlaid with ONE source.

    The source is the explicit ``gate`` when given (even ``{}``: "defaults only"),
    otherwise the candidate summary's own ``gate``.  They are never merged: the
    summary's gate is produced by the run under test, so letting it fill in keys
    the trusted gate left out would let a PR loosen its own rules.

    Within the source, an explicit ``None`` disables an optional rule.  Raises
    ValueError on unknown keys or malformed values so a typo in eval.yaml fails
    the build loudly instead of turning a rule off.
    """
    effective = dict(DEFAULT_GATE)
    layer = gate if gate is not None else candidate_summary.get("gate")
    if layer:
        if not isinstance(layer, dict):
            raise ValueError(f"gate must be a mapping of settings, got {layer!r}")
        unknown = sorted(set(layer) - set(DEFAULT_GATE))
        if unknown:
            raise ValueError(f"unknown gate setting(s) {unknown}; valid keys: {sorted(DEFAULT_GATE)}")
        effective.update(layer)

    drop = effective["max_pass_rate_drop"]
    if isinstance(drop, str):
        if drop.strip().lower() != "auto":
            raise ValueError(f"gate.max_pass_rate_drop must be 'auto', a fraction between 0 and 1, or null, got {drop!r}")
        effective["max_pass_rate_drop"] = "auto"
    elif drop is not None:
        if not _is_number(drop):
            raise ValueError(f"gate.max_pass_rate_drop must be 'auto', a fraction between 0 and 1, or null, got {drop!r}")
        if not 0 <= drop <= 1:
            raise ValueError(f"gate.max_pass_rate_drop is a fraction between 0 and 1 (e.g. 0.03 for 3 p.p.), got {drop!r}")

    if not _is_number(effective["alpha"]):
        raise ValueError(f"gate.alpha must be a number between 0 and 1, got {effective['alpha']!r}")
    for key in _FRACTION_KEYS:
        value = effective[key]
        if value is None:
            continue
        if not _is_number(value):
            raise ValueError(f"gate.{key} must be a number or null, got {value!r}")
        if not 0 <= value <= 1:
            raise ValueError(f"gate.{key} is a fraction between 0 and 1 (e.g. 0.03 for 3 p.p.), got {value!r}")
    for key in _PCT_KEYS:
        value = effective[key]
        if value is not None and not _is_number(value):
            raise ValueError(f"gate.{key} must be a number or null, got {value!r}")
    for key in _COUNT_KEYS:
        value = effective[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"gate.{key} must be a non-negative integer or null, got {value!r}")
    kappa = effective["min_judge_kappa"]
    if kappa is not None and (not _is_number(kappa) or not -1 <= kappa <= 1):
        raise ValueError(f"gate.min_judge_kappa must be a number between -1 and 1, or null, got {kappa!r}")
    for key in _BOOL_KEYS:
        if not isinstance(effective[key], bool):
            raise ValueError(f"gate.{key} must be true or false, got {effective[key]!r}")
    tag_pp = effective["tag_drop_warning_pp"]
    if tag_pp is not None and (not _is_number(tag_pp) or not 0 < tag_pp <= 100):
        raise ValueError(f"gate.tag_drop_warning_pp is a number of percentage points above 0 and at most 100 (e.g. 20), or null, got {tag_pp!r}")
    tag_min = effective["tag_drop_min_cases"]
    if isinstance(tag_min, bool) or not isinstance(tag_min, int) or tag_min < 1:
        raise ValueError(f"gate.tag_drop_min_cases must be a positive integer, got {tag_min!r}")

    tags = effective["critical_tags"]
    effective["critical_tags"] = [tags] if isinstance(tags, str) else [str(t) for t in tags or []]
    return effective


# ------------------------------------------------------------------------ helpers


def _tags(*records: dict) -> list[str]:
    """Tags of a case, unioned across the records that describe it (baseline and candidate)."""
    tags: list[str] = []
    for rec in records:
        for tag in (rec.get("case") or {}).get("tags") or []:
            if tag not in tags:
                tags.append(tag)
    return tags or [UNTAGGED]


def _outcome(record: dict) -> str:
    """'pass' | 'fail' | 'error'.  Records from before `outcome` existed: derive it, like types.derive_outcome.

    A target error is an error; so is any judge error (the case has no reliable verdict);
    everything else is pass/fail by ``passed``.
    """
    outcome = record.get("outcome")
    if outcome in ("pass", "fail", "error"):
        return outcome
    if (record.get("output") or {}).get("error"):
        return "error"
    if any(j.get("error") for j in record.get("judges") or []):
        return "error"
    return "pass" if record.get("passed") else "fail"


def _error_source(record: dict) -> str | None:
    if _outcome(record) != "error":
        return None
    source = record.get("error_source")
    if source in ("target", "judge"):
        return source
    return "target" if (record.get("output") or {}).get("error") else "judge"


def _attempt_outcomes(record: dict) -> list[tuple[str, str | None]]:
    """(outcome, error_source) of every attempt of a case: its ``attempts`` when repeated, else the record itself.

    Attempt entries written before ``error_source`` existed fall back to the case's own source (a case that ended
    pass/fail but has an errored attempt of unknown origin counts as a target error: the conservative reading).
    """
    attempts = _attempts(record)
    if not attempts:
        return [(_outcome(record), _error_source(record))]
    out = []
    for a in attempts:
        outcome = a.get("outcome")
        if outcome not in ("pass", "fail", "error"):
            outcome = "pass" if a.get("passed") else "fail"
        source = None
        if outcome == "error":
            source = a.get("error_source")
            if source not in ("target", "judge"):
                source = _error_source(record) or "target"
        out.append((outcome, source))
    return out


def _error_stats(records: list[dict]) -> dict:
    target = sum(1 for r in records if _error_source(r) == "target")
    judge = sum(1 for r in records if _error_source(r) == "judge")
    total = target + judge
    attempts = [o for r in records for o in _attempt_outcomes(r)]
    a_target = sum(1 for oc, src in attempts if oc == "error" and src == "target")
    a_judge = sum(1 for oc, src in attempts if oc == "error" and src == "judge")
    return {
        "n_errors": total,
        "n_judge_errors": judge,
        "n_target_errors": target,
        "error_rate": total / len(records) if records else 0.0,
        # The same counts per attempt: a repeated case that ended pass can still have crashed in one attempt.
        "n_attempts": len(attempts),
        "n_attempt_errors": a_target + a_judge,
        "n_attempt_target_errors": a_target,
        "n_attempt_judge_errors": a_judge,
        "attempt_error_rate": (a_target + a_judge) / len(attempts) if attempts else 0.0,
    }


def _attempts(record: dict) -> list[dict] | None:
    """The per-attempt list of a repeated case (2 or more attempts), else None."""
    attempts = record.get("attempts")
    return attempts if isinstance(attempts, list) and len(attempts) >= 2 else None


def _n_attempts(record: dict) -> int:
    """How many times the case ran: the number of attempts of a repeated case, else 1."""
    attempts = _attempts(record)
    return len(attempts) if attempts else 1


def _failed_attempts(record: dict) -> str | None:
    """'2/3' (failed attempts / attempts) for a repeated case, else None."""
    attempts = _attempts(record)
    if attempts is None:
        return None
    failed = sum(1 for a in attempts if (a.get("outcome") == "fail" if a.get("outcome") else not a.get("passed")))
    return f"{failed}/{len(attempts)}"


def _is_llm_judge(judge_record: dict) -> bool:
    """LLM judges write per-criterion details and spend tokens; heuristics do neither."""
    details = judge_record.get("details")
    if isinstance(details, dict) and "criteria" in details:
        return True
    return bool((judge_record.get("usage") or {}).get("input_tokens"))


def _case_score(record: dict) -> float | None:
    """Mean of the case's judge scores that exist, or None when no judge scored it."""
    return mean([j.get("score") for j in record.get("judges") or []])


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _pct_change(baseline: float | None, candidate: float | None) -> float | None:
    """Relative change in percent; undefined (None) when the baseline is missing or zero."""
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return (candidate - baseline) / baseline * 100


def _no_percent_reason(what: str, baseline: float | None, candidate: float | None) -> str:
    """Why a percentage limit could not be evaluated (surfaced as a warning)."""
    if baseline is None:
        return f"{what} limit not evaluated: the baseline run has no {what} data"
    if candidate is None:
        return f"{what} limit not evaluated: the candidate run has no {what} data"
    return f"{what} limit not evaluated: the baseline {what} is zero, so a percentage increase is undefined"


_OPEN, _CLOSE = "\ue000", "\ue001"  # private-use marks around a value that came from a run artifact (see _q)


def _q(value) -> str:
    """Mark ``value`` (a case id, tag, version, sha... taken from a file a PR controls) inside a reason or warning.

    The marks are invisible to everything but ``_message``: the plain text of the message (JSON, CLI, report) is unchanged, and the
    markdown of the message (``_Text.md``) prints the marked value in a code span, where nothing is markup, a mention or a link.
    """
    return f"{_OPEN}{str(value).replace(_OPEN, ' ').replace(_CLOSE, ' ')}{_CLOSE}"


class _Text(str):
    """A reason or warning: the plain text every consumer reads, plus ``.md``, the same text ready for a PR comment.

    It is a ``str`` (compares, serialises and prints as the plain text).  ``to_markdown`` uses ``.md``; a message that did not
    come from ``compare_runs`` (read back from JSON, appended by the CLI) has no ``.md`` and is made inert as a whole instead.
    """

    md: str


def _message(marked: str) -> _Text:
    text = _Text(marked.replace(_OPEN, "").replace(_CLOSE, ""))
    text.md = _render_marked(marked)
    return text


def _limited_ids(ids: list[str], limit: int) -> str:
    """'a, b, c' or 'a, b, c and N more' (every id marked as artifact text, see ``_q``)."""
    shown = ", ".join(i if _OPEN in i else _q(i) for i in ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _count(k: int, noun: str) -> str:
    """'1 judge error' / '3 judge errors'."""
    return f"{k} {noun}" if k == 1 else f"{k} {noun}s"


def _retries(k: int) -> str:
    """'1 retry' / '3 retries'."""
    return "1 retry" if k == 1 else f"{k} retries"


def _was(k: int) -> str:
    """'was' for exactly one, else 'were' (0 were, 1 was, 2 were)."""
    return "was" if k == 1 else "were"


def _p_expr(p: float, sep: str = "") -> str:
    """'p=0.031' (sep='') or 'p = 0.031' (sep=' '); tiny values collapse to 'p<0.001'."""
    return f"p{sep}<{sep}0.001" if p < 0.001 else f"p{sep}={sep}{p:.3f}"


def _signed(value: float, decimals: int) -> str:
    """'+1.5' / '-1.5', but plain '0.0' when the value rounds to zero (never '+0.0' or '-0.0')."""
    return f"{0.0:.{decimals}f}" if abs(value) < 0.5 * 10**-decimals else f"{value:+.{decimals}f}"


def _fmt_limit(limit: float) -> str:
    """'+25%' or '+12.5%': limits are shown exactly as configured."""
    return f"{limit:+g}%"


def _fmt_change(pct: float) -> str:
    """'+38%' from 10 up, '+4.5%' below."""
    return _signed(pct, 0 if abs(pct) >= 10 else 1) + "%"


# ------------------------------------------------------------- pass-rate drop rule


def auto_min_net_loss(n: int) -> int:
    """Smallest net loss (cases pass->fail minus fail->pass) the ``auto`` rule can trip on, for n paired cases.

    ``max(3, ceil(n / 50))``: never fewer than 3 cases, and from n = 150 on never below 2 p.p.  Integer arithmetic,
    so there is no float edge at the boundaries.
    """
    return max(AUTO_MIN_NET_LOSS, -(-n // AUTO_MIN_DROP_DENOMINATOR))


def auto_alpha(n: int) -> float:
    """One-sided significance level of the ``auto`` rule for n paired cases, decreasing smoothly with n.

    ``0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)``: 0.25 at n = 0, 0.24 at n = 45, 0.12 at n = 100, 0.072 at n = 150,
    0.061 at n = 250 and 0.0575 as n grows.  With an exact McNemar test the false-alarm rate is at most alpha whatever the
    noise, and about 0.6 to 0.77 alpha in practice (the test is discrete): at most 5% for n >= 150 with noise up to 10%.
    """
    return AUTO_ALPHA_LARGE + AUTO_ALPHA_SPAN / (1 + (n / AUTO_ALPHA_MIDPOINT) ** AUTO_ALPHA_STEEPNESS)


def auto_trips(b: int, c: int, n: int) -> bool:
    """The whole ``auto`` rule: does b pass->fail and c fail->pass flips among n paired cases trip the gate?

    True when ``b - c >= auto_min_net_loss(n)`` and the one-sided exact McNemar p-value ``P(X >= b | X ~ Bin(b + c, 1/2))``
    is below ``auto_alpha(n)``.  Single source of truth: the gate, the small-set warning, the tests and the simulations in
    eval-scientist-gate/ all call it.
    """
    if n <= 0 or b - c < auto_min_net_loss(n):
        return False
    return mcnemar_exact(b, c, "greater") < auto_alpha(n)


@dataclass(frozen=True)
class _DropRule:
    """The pass-rate rule in force for n paired cases (hashable: `_reliable_drop_pp` is cached on it).

    ``mode`` is "auto" or "fixed" (the off mode has no rule).  ``limit`` is the ``effective_max_pass_rate_drop``: a net
    loss strictly above it trips (auto: ``(k - 1) / n`` with k = ``min_net_loss``, so the trip point is exactly a net
    loss of k cases).  ``alpha`` is None when no significance test is used.
    """

    mode: str
    n: int
    limit: float | None
    alpha: float | None
    min_net_loss: int | None = None

    def verdict(self, b: int, c: int) -> tuple[bool, bool | None, bool]:
        """(exceeded, significant, tripped): is the loss big enough, is the test below alpha (None: no test), does it trip."""
        if self.n == 0 or self.limit is None:
            return False, None, False
        if self.mode == "auto":
            big_enough = b - c >= self.min_net_loss
            significant = mcnemar_exact(b, c, "greater") < self.alpha
            return big_enough, significant, big_enough and significant
        return _drop_verdict(b, c, self.n, self.limit, self.alpha is not None, self.alpha)

    def trips(self, b: int, c: int) -> bool:
        return self.verdict(b, c)[2]


def _drop_verdict(b: int, c: int, n: int, limit: float | None, needs_sig: bool, alpha: float):
    """Apply a FIXED pass-rate limit to b pass->fail and c fail->pass flips among n paired cases.

    Returns ``(exceeded, significant, tripped)``: whether the drop is larger than
    ``limit``, whether the one-sided McNemar test is below ``alpha`` (None when it
    was not needed) and whether the rule fires.  This is the v0.1 semantics that `fixed` mode keeps;
    the ``auto`` rule is `auto_trips`.
    """
    if n == 0 or limit is None:
        return False, None, False
    if not (c - b) / n < -limit - _EPS:
        return False, None, False
    if not needs_sig:
        return True, None, True
    significant = mcnemar_exact(b, c, "greater") < alpha
    return True, significant, significant


def _drop_setup(setting, n: int, require_significance: bool = False, alpha: float = 0.05) -> tuple[str, _DropRule | None]:
    """(gate_mode, rule) for the configured ``max_pass_rate_drop`` and n paired cases; the rule is None when it is off."""
    if setting is None:
        return "off", None
    if setting == "auto":
        if n <= 0:
            return "auto", _DropRule("auto", 0, None, None, None)
        k = auto_min_net_loss(n)
        return "auto", _DropRule("auto", n, (k - 1) / n, auto_alpha(n), k)
    return "fixed", _DropRule("fixed", n, setting, alpha if require_significance else None)


def normalize_gate_mode(mode):
    """``gate_mode`` as of evalsprint 0.2: the old "auto-small" / "auto-large" of comparison files written by 0.1 both read as "auto"."""
    return LEGACY_GATE_MODES.get(mode, mode)


@functools.lru_cache(maxsize=128)
def _reliable_drop_pp(rule: _DropRule) -> int | None:
    """Smallest whole-number *true* drop (p.p.) the rule catches in >= 80% of runs.

    Exact enumeration of (b, c) when both runs are independent noisy readings
    (each verdict flips with probability 5%) of the same latent verdicts, and a
    fraction d of the cases really regressed (b and c do not depend on the baseline pass rate).
    None when no drop up to 50 p.p. qualifies.
    """
    n, p = rule.n, _ASSUMED_NOISE
    log_fact = [math.lgamma(i + 1) for i in range(n + 1)]
    trips = [(b, c) for b in range(n + 1) for c in range(n - b + 1) if rule.trips(b, c)]
    for pp in range(1, 51):
        d = pp / 100
        p_b = p * (1 - p) * (1 - d) + d * (1 - p) ** 2  # baseline reads pass, candidate reads fail
        p_c = p * (1 - p) * (1 - d) + d * p**2  # baseline reads fail, candidate reads pass
        p_0 = 1 - p_b - p_c
        power = 0.0
        for b, c in trips:
            rest = n - b - c
            log_prob = log_fact[n] - log_fact[b] - log_fact[c] - log_fact[rest]
            log_prob += b * math.log(p_b) + c * math.log(p_c) + rest * math.log(p_0)
            power += math.exp(log_prob)
        if power >= _RELIABLE_POWER:
            return pp
    return None


def _critical_false_alarm(m: int, attempts: int, noise: float = _ASSUMED_NOISE) -> float:
    """P(at least one of m healthy critical cases flips pass -> fail by noise alone).

    Each attempt of a healthy case fails with probability `noise`; the case verdict is the
    majority (ties -> fail, C2), taken independently in the baseline and the candidate.
    """
    fails_needed = attempts // 2 + 1 if attempts % 2 else attempts // 2
    q = sum(math.comb(attempts, j) * noise**j * (1 - noise) ** (attempts - j) for j in range(fails_needed, attempts + 1))
    return 1 - (1 - (1 - q) * q) ** m


def _fmt_rate(x: float) -> str:
    return f"{x * 100:.0f}%" if x >= 0.095 else f"{x * 100:.1f}%"


def _small_set_warning(n: int, rule: _DropRule | None) -> str:
    """What a test set of n paired cases can and cannot detect."""
    head = f"small test set (n = {n})"
    if rule is None:
        return f"{head}: the pass-rate rule is off, so pass-rate drops are not gated"
    trip = next((k for k in range(1, n + 1) if rule.trips(k, 0)), None)
    if trip is None:
        what = "the auto rule" if rule.mode == "auto" else f"a drop limit of {rule.limit * 100:.1f} p.p."
        return f"{head}: with {what} no pass-rate drop can trip the gate; add cases" + ("" if rule.mode == "auto" else " or lower the limit")
    reliable = _reliable_drop_pp(rule)
    reliable_txt = (
        f"only drops of ≥ {reliable} p.p. are reliably detected (≥ {_RELIABLE_POWER:.0%} of runs at ~{_ASSUMED_NOISE:.0%} noise)"
        if reliable is not None
        else "even a 50 p.p. drop is not reliably detected"
    )
    return (
        f"{head}: the gate trips at a net loss of {_count(trip, 'case')} ({trip / n * 100:.1f} p.p.); {reliable_txt}; "
        "add cases to catch smaller drops"
    )


def _drop_message(delta: float, limit: float, p_one_sided: float, significant: bool | None, alpha: float) -> str:
    """Fixed mode.  `significant` is None when no test was needed, else whether the one-sided p is below alpha."""
    detail = f"limit {limit * 100:.1f} p.p.; one-sided McNemar {_p_expr(p_one_sided)}"
    if significant:
        detail += f", significant at α={alpha:g}"
    return f"pass rate dropped {-delta * 100:.1f} p.p. ({detail})"


def _auto_drop_message(delta: float, rule: _DropRule, net_loss: int, p_one_sided: float) -> str:
    """Auto mode: the numbers behind the two conditions."""
    return (
        f"pass rate dropped {-delta * 100:.1f} p.p. (auto rule for {rule.n} paired cases: net loss {_count(net_loss, 'case')} "
        f"vs {rule.min_net_loss} needed, one-sided McNemar {_p_expr(p_one_sided)} vs α={rule.alpha:.3f})"
    )


# ---------------------------------------------------------------------- comparison


_Protocol = tuple  # (n, ((tag, attempts), ...)): see _repeat_protocol


def _is_attempts(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _repeat_protocol(summary: dict) -> "_Protocol | None":
    """The repeat protocol a run ran with, normalised to ``(n, sorted (tag, attempts) pairs)``; None when it cannot be told.

    ``summary["repeat"]`` is ``{"n": int, "per_tag": {tag: attempts}}``: a case runs ``max(per_tag[t] for its tags)`` times, else
    ``n`` times.  The shape of cycle 3, ``{"n": 5, "tags": ["safety"]}`` (n attempts on the listed tags only), is the same
    protocol as ``{"n": 1, "per_tag": {"safety": 5}}`` and is read as that.  A ``per_tag`` entry equal to ``n`` changes
    nothing and is dropped, so two spellings of one protocol compare equal.  evalsprint 0.1 wrote the key only when n > 1, so a
    summary that carries ``judges_sha`` (0.1 or later) without it ran every case once.  A summary with neither predates both and is
    unknown (its missing ``judges_sha`` is already warned about), as is a malformed ``repeat``.
    """
    repeat = summary.get("repeat")
    if repeat is None:
        return (1, ()) if summary.get("judges_sha") else None
    if isinstance(repeat, int) and not isinstance(repeat, bool):
        repeat = {"n": repeat}
    if not isinstance(repeat, dict):
        return None
    n = repeat.get("n", 1)
    if not _is_attempts(n):
        return None
    per_tag_raw = repeat.get("per_tag")
    if isinstance(per_tag_raw, dict):
        per_tag = {}
        for tag, attempts in per_tag_raw.items():
            if not _is_attempts(attempts):
                return None
            per_tag[str(tag)] = attempts
    elif per_tag_raw is not None:
        return None
    else:
        tags = repeat.get("tags")  # legacy: `n` attempts on the listed tags, every other case once
        per_tag = {}
        if n > 1 and isinstance(tags, (list, tuple)) and tags:
            per_tag = {str(t): n for t in tags}
            n = 1
    return (n, tuple(sorted((t, k) for t, k in per_tag.items() if k != n)))


def _protocol_dict(protocol: "_Protocol | None") -> dict | None:
    """The protocol as ``summary["repeat"]`` writes it: ``{"n": int, "per_tag": {tag: attempts}}``."""
    return None if protocol is None else {"n": protocol[0], "per_tag": dict(protocol[1])}


def _repeat_text(protocol: "_Protocol | None") -> str:
    if protocol is None:
        return "unknown"
    n, per_tag = protocol
    if not per_tag:
        return "every case once (n: 1)" if n <= 1 else f"n: {n} on every case"
    return f"n: {n}, per_tag: {{{', '.join(f'{_q(tag)}: {attempts}' for tag, attempts in per_tag)}}}"


def _kit_version(summary: dict) -> str | None:
    """The evalsprint version that wrote a run (``kit_version``, else ``evalsprint_version``)."""
    version = summary.get("kit_version") or summary.get("evalsprint_version")
    return str(version) if version else None


def _kit_series(version: str) -> tuple:
    """(major, minor) of a version string; the whole string when it does not start like one (then only equality counts)."""
    m = re.match(r"\s*v?(\d+)\.(\d+)", version, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else (version,)


def _kit_release(version: str):
    """A comparable key for a version: ``v0.2.0``, ``0.2.0`` and ``0.2`` are the same release, ``0.2.1`` is another one.

    Numeric dotted versions compare as integer tuples without trailing zeros; anything else (``dev``, ``0.3.0rc1``) by its text.
    """
    text = re.sub(r"^v(?=\d)", "", version.strip(), flags=re.IGNORECASE)
    if not re.fullmatch(r"\d+(\.\d+)*", text):
        return text
    parts = [int(x) for x in text.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _before_v02(version: str | None) -> bool:
    """True when ``version`` is older than evalsprint 0.2, missing or not a version (the runs whose ``judges_sha`` did not cover custom-judge code)."""
    m = re.match(r"\s*v?(\d+)\.(\d+)", version or "", re.IGNORECASE)
    return not m or (int(m.group(1)), int(m.group(2))) < (0, 2)


_DECISION_OK = ("usable", "trustworthy")  # calibration.kappa_decision values that release a judge
_DECISION_WHY = {
    "inconclusive": "the weighted kappa interval straddles the usable threshold, or there are too few labels to release the judge",
    "not_trustworthy": "the weighted kappa interval lies below the usable threshold",
}


def _calibration_status(calibration: dict | None, min_kappa: float | None) -> tuple[dict | None, str | None, list[str]]:
    """(calibration_status, problem, notes) for a calibration dict; problem is None when it is fine.

    The estimand is the one ``calibration.calibrate`` decides on: the Horvitz-Thompson WEIGHTED kappa of the whole
    run (``kappa_weighted`` / ``kappa_weighted_ci``); the unweighted ``kappa`` / ``kappa_ci`` describe the stratified
    label sample, not the run.  Only a calibration dict without the weighted keys (written by an older evalsprint) falls
    back to the unweighted pair, which is said in ``basis`` ("legacy") and in ``notes`` (surfaced as a warning).
    A judge is fine when ``kappa_decision`` is usable or trustworthy AND the kappa lower bound reaches ``min_kappa``.
    A decision that is missing altogether (legacy dicts) is not held against the judge.
    """
    if calibration is None:
        return None, None, []
    weighted = "kappa_weighted_ci" in calibration or "kappa_weighted" in calibration
    if weighted:
        basis, kappa, ci = "weighted", calibration.get("kappa_weighted"), calibration.get("kappa_weighted_ci")
    else:
        basis, kappa, ci = "legacy", calibration.get("kappa"), calibration.get("kappa_ci")
    lower = ci[0] if ci else None
    decision = calibration.get("kappa_decision")
    if decision is not None and not isinstance(decision, str):
        decision = str(decision)  # a forged file: an unknown decision (not released), never a crash
    kind = "weighted" if weighted else "unweighted"
    shown = f"{kappa:.2f}" if kappa is not None else "n/a"

    problem = None
    if decision == "insufficient_data":
        problem = "judge calibration rests on too few human labels to say whether the judge can be trusted"
    else:
        parts = []
        if min_kappa is not None:
            if lower is None:
                if decision in (None, *_DECISION_OK):
                    problem = "judge calibration has no kappa confidence interval (too few labels)"
            elif lower < min_kappa - _EPS:
                parts.append(f"{kind} kappa lower bound {lower:.2f} is below the required {min_kappa:.2f} ({kind} kappa {shown})")
        if decision is not None and decision not in _DECISION_OK:
            why = _DECISION_WHY.get(decision, f"only {' or '.join(_DECISION_OK)} release a judge")
            parts.append(f"calibration decision is '{_q(decision)}' ({why})")
        if parts:
            problem = "judge calibration is weak: " + "; ".join(parts)
    status = {
        "judge": calibration.get("judge"),
        "kappa": kappa,
        "kappa_ci": list(ci) if ci else None,
        "kappa_decision": decision,
        "basis": basis,
        "ok": problem is None,
    }
    notes = [] if weighted else [
        "legacy calibration: unweighted kappa used (the calibration file predates the weighted estimate the judge decision is based on); "
        "re-run `evalsprint calibrate ... --json` with the current version"
    ]
    return status, problem, notes


def _retry_stats(run: "RunResult", errors: dict) -> dict:
    """The retry side of a run: ``configured`` (``summary.target_retries``, None when not recorded), ``retried_calls`` (the extra target calls made
    after a crash, ``n_target_retries``: the sum of the records' ``target_retries_used``, else ``summary.n_target_retries``; None when neither exists), ``attempts`` (target
    calls that started an attempt, from the records) and ``share`` = retried calls / attempts (None when unknown)."""
    value = run.summary.get("target_retries")
    configured = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    used = [r.get("target_retries_used") for r in run.results]
    if any(isinstance(u, int) and not isinstance(u, bool) and u >= 0 for u in used):
        retried = sum(u for u in used if isinstance(u, int) and not isinstance(u, bool) and u > 0)
    else:
        summed = run.summary.get("n_target_retries")
        retried = summed if isinstance(summed, int) and not isinstance(summed, bool) and summed >= 0 else None
    attempts = errors["n_attempts"]
    return {
        "configured": configured,
        "retried_calls": retried,
        "attempts": attempts,
        "share": (retried / attempts) if retried is not None and attempts else None,
    }


def compare_runs(baseline: "RunResult", candidate: "RunResult", gate: dict | None = None, calibration: dict | None = None) -> dict:
    """Compare two runs and apply the regression gate.

    ``gate`` is the trusted gate (see `resolve_gate`); ``calibration`` is the
    dict from ``calibration.calibrate`` for the candidate's judge, when there is one.

    Returns the dict documented in docs/ARCHITECTURE.md, including ``warnings``,
    ``removed_cases``, ``changed_cases``, ``errors``, ``pass_rate.p_value`` (one-sided:
    "the candidate is worse") next to ``pass_rate.p_value_two_sided``,
    ``gate_mode`` (``"auto"`` | ``"fixed"`` | ``"off"``; comparison files of evalsprint 0.1 say ``"auto-small"`` / ``"auto-large"``,
    read them with `normalize_gate_mode`), ``effective_max_pass_rate_drop`` (a net loss strictly above it trips: in ``auto`` mode
    ``(k - 1) / n``, i.e. exactly a net loss of ``effective_min_net_loss`` = k cases), ``effective_alpha`` (the one-sided McNemar
    level the rule tests at; ``None`` when no test is used) and ``calibration_status`` (``judge``, ``kappa``, ``kappa_ci`` of the estimand
    used, ``kappa_decision``, ``basis``: "weighted" or "legacy", ``ok``), plus
    ``pass_rate.candidate_overall`` (whole candidate run, errors left out) and
    ``pass_rate.candidate_paired_pessimistic`` (the cases that also exist, unchanged, in the baseline, target
    errors counted as failures; ``..._n`` is its denominator): ``min_pass_rate`` is checked against both.
    ``errors.candidate.paired_error_rate`` (candidate errors among the cases present in the baseline;
    ``paired_n_errors`` / ``paired_n_cases``) sits next to the whole-run ``error_rate``: ``max_error_rate``
    is checked against both.  ``target_error_regressions`` lists the cases that passed in the baseline and hit a
    target error in the candidate: they are paired as failures, so they are also in ``newly_failing`` and
    ``pass_rate.b_pass_to_fail``.  Judge errors and baseline errors stay in ``excluded_errors``.  Also
    ``score.n``, ``critical_unverified`` (critical cases whose candidate judge errored) and ``judges_changed`` (True / False from
    the two summaries' ``judges_sha``, None when either side predates it), ``repeat_changed`` (True / False from the two
    summaries' ``repeat`` protocol, None when either side predates it) with ``repeat_protocol`` (``baseline`` and ``candidate``, each
    ``{"n": int, "per_tag": {tag: attempts}}`` or None: the normalised protocol, the cycle-3 ``tags`` spelling already folded into
    ``per_tag``), ``kit_version`` (``baseline``, ``candidate``, ``changed``: the two versions differ at all, patch included; None when
    a side has no version),
    ``target_retries_changed`` (True / False from the two summaries' ``target_retries``, None when either side does not record it: a
    warning then; True is a reason unless ``allow_target_retries_change``) with ``target_retries`` (``baseline`` and ``candidate``, each
    ``{"configured": int | None, "retried_calls": int | None, "attempts": int, "share": float | None}``, and ``changed``: bool; ``share`` =
    retried calls / attempts, and a candidate share at least ``retry_share_warn`` above the baseline's is a warning),
    ``tag_warnings`` (non-critical tags with at least ``tag_drop_min_cases`` paired cases whose pass rate fell by at least
    ``tag_drop_warning_pp`` p.p.; warnings only) and ``flaky_crashes`` (cases that passed in the baseline and hit a target
    error in some candidate attempt although their majority outcome is pass).  ``errors.*`` also carry the per-attempt
    counts ``n_attempts``, ``n_attempt_errors``, ``n_attempt_target_errors``, ``n_attempt_judge_errors`` and
    ``attempt_error_rate`` (recomputed from the records' ``attempts``), checked against ``max_error_rate`` like ``error_rate``.
    ``reasons`` and ``warnings`` are ``str`` subclasses: the plain text, plus ``.md`` (what ``to_markdown`` prints: every value taken from
    a run artifact in a code span, so a tag, version or sha cannot notify anyone or build a link in a PR comment).
    Raises ValueError for an invalid gate.
    """
    eff = resolve_gate(candidate.summary, gate)

    base_by, cand_by = baseline.by_case(), candidate.by_case()
    removed = [cid for cid in base_by if cid not in cand_by]  # baseline (dataset) order
    added = [cid for cid in cand_by if cid not in base_by]
    common = [cid for cid in cand_by if cid in base_by]  # candidate (dataset) order

    # Pairing: same id, same content, and a verdict on the baseline side.  On the candidate side an
    # error is one of three things:
    #   * target error on a case that PASSED in the baseline -> paired, counts as a failure (a crash
    #     introduced by the change is a regression, not "no signal");
    #   * judge error (any baseline outcome), or target error on a case that already failed in the
    #     baseline -> left out (infrastructure / nothing left to regress), still counted in the error rate;
    #   * anything the baseline itself errored on -> left out.
    changed: list[str] = []
    errored: list[str] = []  # left out of the comparison
    crashed: list[str] = []  # paired as failures: baseline pass -> candidate target error
    paired_ids: list[str] = []
    no_sha_baseline = no_sha_candidate = False
    for cid in common:
        b, c = base_by[cid], cand_by[cid]
        sha_b, sha_c = b.get("case_sha"), c.get("case_sha")
        if sha_b and sha_c:
            if sha_b != sha_c:
                changed.append(cid)
                continue
        else:
            no_sha_baseline |= not sha_b
            no_sha_candidate |= not sha_c
        if _outcome(b) == "error":
            errored.append(cid)
        elif _outcome(c) != "error":
            paired_ids.append(cid)
        elif _outcome(b) == "pass" and _error_source(c) == "target":
            crashed.append(cid)
            paired_ids.append(cid)
        else:
            errored.append(cid)

    crashed_set = set(crashed)
    pairs = [(base_by[cid], cand_by[cid]) for cid in paired_ids]
    # Verdicts used by every paired statistic: a crash counts as a failure.
    verdicts = [(_outcome(b), "fail" if cid in crashed_set else _outcome(c)) for cid, (b, c) in zip(paired_ids, pairs)]
    case_tags = {cid: _tags(b, c) for cid, (b, c) in zip(paired_ids, pairs)}
    n = len(pairs)

    # -- pass rate, McNemar, by tag ------------------------------------------------
    newly_failing = [cid for cid, (vb, vc) in zip(paired_ids, verdicts) if vb == "pass" and vc == "fail"]
    newly_passing = [cid for cid, (vb, vc) in zip(paired_ids, verdicts) if vb == "fail" and vc == "pass"]
    k_base = sum(1 for vb, _ in verdicts if vb == "pass")
    k_cand = sum(1 for _, vc in verdicts if vc == "pass")
    rate_base, rate_cand = _rate(k_base, n), _rate(k_cand, n)
    delta = rate_cand - rate_base if n else None
    n_b, n_c = len(newly_failing), len(newly_passing)
    p_value = mcnemar_exact(n_b, n_c, "greater") if n else None  # one-sided: the candidate is worse
    p_two_sided = mcnemar_exact(n_b, n_c) if n else None

    # Whole candidate run (errors left out), and the same rate on the cases that also exist,
    # unchanged, in the baseline with every target error counted as a failure: appended cases
    # cannot dilute the second one.
    cand_scored = [r for r in candidate.results if _outcome(r) != "error"]
    cand_overall = _rate(sum(1 for r in cand_scored if _outcome(r) == "pass"), len(cand_scored))
    changed_set = set(changed)
    pessimistic = [
        cid for cid in common if cid not in changed_set and not (_outcome(cand_by[cid]) == "error" and _error_source(cand_by[cid]) == "judge")
    ]
    cand_paired_pessimistic = _rate(sum(1 for cid in pessimistic if _outcome(cand_by[cid]) == "pass"), len(pessimistic))

    tag_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, baseline passes, candidate passes
    for cid, (vb, vc) in zip(paired_ids, verdicts):
        for tag in case_tags[cid]:
            counts = tag_counts[tag]
            counts[0] += 1
            counts[1] += int(vb == "pass")
            counts[2] += int(vc == "pass")
    by_tag = {
        tag: {"n": k, "baseline": kb / k, "candidate": kc / k, "delta": (kc - kb) / k}
        for tag, (k, kb, kc) in sorted(tag_counts.items())
    }

    # Score, cost and latency compare successful outputs: a crashed candidate case has none.
    clean_pairs = [(b, c) for cid, (b, c) in zip(paired_ids, pairs) if cid not in crashed_set]

    # -- score: per-case mean over judges, paired bootstrap ------------------------
    case_scores = [(_case_score(b), _case_score(c)) for b, c in clean_pairs]
    scored = [(s_b, s_c) for s_b, s_c in case_scores if s_b is not None and s_c is not None]
    score = None
    if scored:
        s_base, s_cand = [s_b for s_b, _ in scored], [s_c for _, s_c in scored]
        ci = paired_bootstrap_diff_ci(s_base, s_cand) if len(scored) >= 2 else None  # 1 pair: the CI would be a fake point
        score = {
            "baseline_mean": mean(s_base),
            "candidate_mean": mean(s_cand),
            "delta": mean(s_cand) - mean(s_base),
            "ci": list(ci) if ci else None,
            "n": len(scored),
        }

    # -- cost, latency, errors ------------------------------------------------------
    base_recs, cand_recs = [b for b, _ in clean_pairs], [c for _, c in clean_pairs]
    cost_base = mean([(r.get("output") or {}).get("cost_usd") for r in base_recs])
    cost_cand = mean([(r.get("output") or {}).get("cost_usd") for r in cand_recs])
    lat_base = percentile([(r.get("output") or {}).get("latency_s") for r in base_recs], 95)
    lat_cand = percentile([(r.get("output") or {}).get("latency_s") for r in cand_recs], 95)
    cost_pct = _pct_change(cost_base, cost_cand)
    lat_pct = _pct_change(lat_base, lat_cand)
    err_base, err_cand = _error_stats(baseline.results), _error_stats(candidate.results)
    # The candidate's errors among the cases that also exist in the baseline: added cases cannot dilute this rate.
    err_paired = _error_stats([cand_by[cid] for cid in common])

    # -- gate ---------------------------------------------------------------------
    reasons: list[str] = []
    warnings: list[str] = []

    if no_sha_baseline or no_sha_candidate:
        side = "baseline and candidate" if no_sha_baseline and no_sha_candidate else ("baseline" if no_sha_baseline else "candidate")
        warnings.append(
            f"the {side} run has records without case_sha (written by an older evalsprint): cases are paired by id only, "
            "so edited test cases cannot be detected; re-create the baseline with the current version"
        )

    if n == 0:
        # Nothing was compared, so nothing was verified: fail closed rather than
        # let a renamed dataset, an empty run or an all-error baseline sail through CI.
        if not common:
            reasons.append(
                f"no cases in common between baseline ({_count(len(base_by), 'case')}) and candidate ({_count(len(cand_by), 'case')}); "
                "nothing to compare"
            )
        else:
            reasons.append(
                f"no comparable cases: {_count(len(common), 'case')} in common, but {len(changed)} {_was(len(changed))} edited "
                f"and {len(errored)} errored in at least one run; nothing to compare"
            )

    # Rule 1: pass-rate drop.  `auto` is one rule for every n (auto_trips); a number is the fixed limit of v0.1.
    mode, rule = _drop_setup(eff["max_pass_rate_drop"], n, eff["require_significance"], eff["alpha"])
    limit = rule.limit if rule else None
    exceeded, significant, tripped = rule.verdict(n_b, n_c) if rule else (False, None, False)
    net_loss = n_b - n_c
    one_crash = len(crashed) == 1
    crash_note = (
        f"; of these, {_count(len(crashed), 'case')} crashed in the candidate (target error) and "
        f"{'counts' if one_crash else 'count'} as {'a failure' if one_crash else 'failures'}: {_limited_ids(crashed, _MAX_REASON_IDS)}"
        if crashed
        else ""
    )
    if tripped:
        if mode == "auto":
            reasons.append(_auto_drop_message(delta, rule, net_loss, p_value) + crash_note)
        else:
            reasons.append(_drop_message(delta, limit, p_value, significant, eff["alpha"]) + crash_note)
    elif mode == "auto" and n and net_loss >= 1 and (exceeded or significant):
        if exceeded:  # big enough, but the exact test says it could be noise
            warnings.append(
                f"{_auto_drop_message(delta, rule, net_loss, p_value)}; not counted because the drop is not significant "
                f"at this suite size (p is not below α={rule.alpha:.3f}), so it could be run-to-run noise{crash_note}"
            )
        else:  # significant, but too small to be worth blocking
            warnings.append(
                f"pass rate dropped {-delta * 100:.1f} p.p. and it is significant (one-sided McNemar {_p_expr(p_value)} < α={rule.alpha:.3f}), "
                f"but the net loss is only {_count(net_loss, 'case')} and the auto rule needs at least {rule.min_net_loss} "
                f"(≥ 3 cases and ≥ 2 p.p.): not counted{crash_note}"
            )
    elif exceeded and mode == "fixed":
        warnings.append(
            f"{_drop_message(delta, limit, p_value, significant, eff['alpha'])}; "
            f"not counted because require_significance is on and p is not below alpha={eff['alpha']:g}{crash_note}"
        )
    elif crashed:
        # Counted in the delta above, but the drop is within the limit: say so instead of passing quietly.
        warnings.append(
            f"{_count(len(crashed), 'case')} that passed in the baseline crashed in the candidate (target error) and "
            f"{'counts' if one_crash else 'count'} as {'a failure' if one_crash else 'failures'} in the pass rate "
            f"({_limited_ids(crashed, _MAX_REASON_IDS)}); the drop ({-delta * 100:.1f} p.p.) is within the limit, but check {'it' if one_crash else 'them'}"
        )
    if eff["max_pass_rate_drop"] == "auto":
        if eff["require_significance"]:
            warnings.append("require_significance is ignored while max_pass_rate_drop is 'auto' (auto always applies its own significance test)")
        if eff["alpha"] != DEFAULT_GATE["alpha"]:
            warnings.append(
                f"gate.alpha ({eff['alpha']:g}) is ignored while max_pass_rate_drop is 'auto' "
                "(auto uses alpha(n) = 0.0575 + 0.1925 / (1 + (n / 85)^4.5), which shrinks as the test set grows)"
            )

    # Rule 2: absolute floor.  Checked on the whole candidate run (errors left out) and on the cases that
    # also exist in the baseline (target errors count as failures), so appended cases cannot dilute it.
    floor = eff["min_pass_rate"]
    if floor is not None:
        below_overall = cand_overall is not None and cand_overall < floor - _EPS
        below_paired = cand_paired_pessimistic is not None and cand_paired_pessimistic < floor - _EPS
        if below_overall:
            reasons.append(f"candidate pass rate {cand_overall * 100:.1f}% is below the minimum {floor * 100:.1f}%")
        if below_paired and not (below_overall and abs(cand_overall - cand_paired_pessimistic) <= _EPS):
            reasons.append(
                f"candidate pass rate on the {_count(len(pessimistic), 'case')} that also exist in the baseline is "
                f"{cand_paired_pessimistic * 100:.1f}%, below the minimum {floor * 100:.1f}% "
                "(target errors count as failures; added cases do not count towards it)"
            )

    # Rule 3: critical tags must never flip pass -> fail (repeated cases: by majority, see C2).
    critical_tags = eff["critical_tags"]
    critical = set(critical_tags)
    critical_regressions = [cid for cid in newly_failing if critical & set(case_tags[cid])]
    critical_unverified = [
        cid
        for cid in errored
        if _outcome(base_by[cid]) == "pass" and _outcome(cand_by[cid]) == "error" and critical & set(_tags(base_by[cid], cand_by[cid]))
    ]
    known_tags = {t for cid in common for t in _tags(base_by[cid], cand_by[cid])}
    for tag in critical_tags:
        if tag not in known_tags:
            warnings.append(f"critical tag '{_q(tag)}' matches no paired case; check the spelling in gate.critical_tags")
            continue
        flipped = [cid for cid in critical_regressions if tag in case_tags[cid]]
        if flipped:
            shown = [
                f"{_q(cid)} (crashed in the candidate)"
                if cid in crashed_set
                else (f"{_q(cid)} ({_failed_attempts(cand_by[cid])} attempts failed)" if _attempts(cand_by[cid]) else _q(cid))
                for cid in flipped
            ]
            reasons.append(
                f"critical tag '{_q(tag)}': {_count(len(flipped), 'case')} went pass→fail ({_limited_ids(shown, _MAX_REASON_IDS)})"
            )
        unverified = [cid for cid in critical_unverified if tag in _tags(base_by[cid], cand_by[cid])]
        if unverified:
            reasons.append(
                f"critical tag '{_q(tag)}': {_count(len(unverified), 'case')} passed in the baseline but the judge errored in the candidate, "
                f"so could not be verified ({_limited_ids(unverified, _MAX_REASON_IDS)}) — not a quality signal, but a critical case must be re-checked"
            )
    critical_paired = [cid for cid in paired_ids if critical & set(case_tags[cid])]
    under_repeated = [
        cid for cid in critical_paired if min(_n_attempts(base_by[cid]), _n_attempts(cand_by[cid])) < RECOMMENDED_CRITICAL_REPEATS
    ]
    if under_repeated:
        need = [t for t in critical_tags if any(t in case_tags[cid] for cid in under_repeated)]
        odds = ", ".join(f"{_fmt_rate(_critical_false_alarm(len(critical_paired), k))} with n: {k}" for k in (1, 3, RECOMMENDED_CRITICAL_REPEATS))
        per_tag = ", ".join(f"{_q(t)}: {RECOMMENDED_CRITICAL_REPEATS}" for t in need)
        warnings.append(
            f"{'critical case' if len(under_repeated) == 1 else 'critical cases'} ran fewer than {RECOMMENDED_CRITICAL_REPEATS} times in the baseline or candidate "
            f"({len(under_repeated)} of {len(critical_paired)} paired): "
            f"one noisy verdict can trip the gate (false alarms with {_count(len(critical_paired), 'critical case')} at ~{_ASSUMED_NOISE:.0%} noise: {odds}); "
            f"add per_tag: {{{per_tag}}} under repeat in eval.yaml (repeat: {{n: 1, per_tag: {{{per_tag}}}}} when there is no repeat block yet; "
            "keep your n otherwise, the other cases must not change) and refresh the baseline"
        )

    # Warnings on the non-critical tags: a big drop in one tag can hide behind a stable overall rate.  Never a reason:
    # a tag is a slice of the suite, so with many tags a few will move by chance; critical tags have their own rule.
    tag_warnings: list[dict] = []
    tag_pp, tag_min = eff["tag_drop_warning_pp"], eff["tag_drop_min_cases"]
    if tag_pp is not None:
        for tag, v in by_tag.items():
            if tag == UNTAGGED or tag in critical or v["n"] < tag_min:
                continue
            if -v["delta"] * 100 >= tag_pp - _EPS:
                tag_warnings.append({"tag": tag, "n": v["n"], "baseline": v["baseline"], "candidate": v["candidate"], "delta": v["delta"]})
        tag_warnings.sort(key=lambda w: (w["delta"], w["tag"]))  # worst first
        for w in tag_warnings:
            warnings.append(
                f"tag '{_q(w['tag'])}': pass rate fell {-w['delta'] * 100:.1f} p.p. ({w['baseline'] * 100:.1f}% → {w['candidate'] * 100:.1f}%, "
                f"{_count(w['n'], 'paired case')}); not a gate reason (the tag is not critical), but check it"
            )

    # Rule 4 and 5: relative increases in mean cost and p95 latency.
    for limit_pct, pct, what, label, base_val, cand_val in (
        (eff["max_cost_increase_pct"], cost_pct, "cost", "mean cost per request", cost_base, cost_cand),
        (eff["max_p95_latency_increase_pct"], lat_pct, "latency", "p95 latency", lat_base, lat_cand),
    ):
        if limit_pct is None:
            continue
        if pct is None:
            warnings.append(_no_percent_reason(what, base_val, cand_val))
        elif pct > limit_pct + _EPS:
            reasons.append(f"{label} {_fmt_change(pct)} (limit {_fmt_limit(limit_pct)})")

    # Rule 6: error-rate ceiling, on the whole candidate run and on the cases that also exist in the baseline
    # (so appended cases cannot dilute it), per case and per attempt.  An unhealthy run says nothing about quality.
    max_err = eff["max_error_rate"]
    unhealthy = False
    if max_err is not None and err_cand["error_rate"] > max_err + _EPS:
        unhealthy = True
        reasons.append(
            f"run unhealthy: {_count(err_cand['n_judge_errors'], 'judge error')}, {_count(err_cand['n_target_errors'], 'target error')} "
            f"({err_cand['error_rate'] * 100:.1f}% of cases, limit {max_err * 100:.1f}%) — not a quality signal"
        )
    elif max_err is not None and err_paired["error_rate"] > max_err + _EPS:
        unhealthy = True
        reasons.append(
            f"run unhealthy on the cases that also exist in the baseline: {_count(err_paired['n_judge_errors'], 'judge error')}, "
            f"{_count(err_paired['n_target_errors'], 'target error')} ({err_paired['error_rate'] * 100:.1f}% of {len(common)} cases, "
            f"limit {max_err * 100:.1f}%; {err_cand['error_rate'] * 100:.1f}% over the whole run, where added cases dilute it) — not a quality signal"
        )
    if max_err is not None and not unhealthy:
        # Repeated cases take a majority verdict, which can hide crashed attempts: count them per attempt.
        for scope, stats, where in (
            ("", err_cand, "over the whole run"),
            (" on the cases that also exist in the baseline", err_paired, "on the baseline's cases"),
        ):
            if stats["attempt_error_rate"] > max_err + _EPS:
                reasons.append(
                    f"run unhealthy{scope}: {stats['n_attempt_errors']} of {_count(stats['n_attempts'], 'attempt')} errored "
                    f"({_count(stats['n_attempt_target_errors'], 'target error')}, {_count(stats['n_attempt_judge_errors'], 'judge error')}; "
                    f"{stats['attempt_error_rate'] * 100:.1f}% of attempts {where}, limit {max_err * 100:.1f}%), "
                    f"although only {stats['n_errors']} case verdict{'s' if stats['n_errors'] != 1 else ''} ended in an error: "
                    "repeated cases hide crashed attempts behind their majority verdict — not a quality signal"
                )
                break
    base_rate = max(err_base["error_rate"], err_base["attempt_error_rate"])
    if max_err is not None and base_rate > max_err + _EPS:
        used = f"; only the {_count(n, 'case')} with a verdict in the baseline {'is' if n == 1 else 'are'} compared" if n else ""
        warnings.append(
            f"the baseline run is unhealthy ({base_rate * 100:.1f}% errors{' per attempt' if base_rate > err_base['error_rate'] + _EPS else ''}){used}; "
            "refresh the baseline from a healthy run"
        )

    # A crashed attempt the majority verdict hid: the case still counts as a pass, so the pass rate never shows it.
    flaky: list[tuple[str, int, int]] = []
    for cid in common:
        if cid in changed_set or cid in crashed_set or _outcome(base_by[cid]) != "pass" or _outcome(cand_by[cid]) != "pass":
            continue
        outcomes = _attempt_outcomes(cand_by[cid])
        crashed_attempts = sum(1 for oc, src in outcomes if oc == "error" and src == "target")
        if crashed_attempts:
            flaky.append((cid, crashed_attempts, len(outcomes)))
    if flaky:
        one = len(flaky) == 1
        shown = _limited_ids([f"{_q(cid)} ({k}/{m} attempts)" for cid, k, m in flaky], _MAX_REASON_IDS)
        warnings.append(
            f"{_count(len(flaky), 'case')} that passed in the baseline hit a target error in some candidate attempts although the "
            f"majority outcome is pass (flaky crash): {shown}; the pass rate does not show {'it' if one else 'them'}, check {'it' if one else 'them'}"
        )

    # Errors left out of the comparison, by kind: a baseline error, a judge error and a crash on an already-failing case
    # mean different things, and none of them is counted as a failure.
    left_baseline = [cid for cid in errored if _outcome(base_by[cid]) == "error"]
    left_judge = [cid for cid in errored if cid not in left_baseline and _error_source(cand_by[cid]) == "judge"]
    left_crash = [cid for cid in errored if cid not in left_baseline and cid not in left_judge]
    if left_baseline:
        one = len(left_baseline) == 1
        warnings.append(
            f"{_count(len(left_baseline), 'case')} left out of the comparison because {'it' if one else 'they'} errored in the baseline "
            f"({_limited_ids(left_baseline, _MAX_REASON_IDS)}): a baseline error says nothing about the candidate"
        )
    if left_judge:
        one = len(left_judge) == 1
        warnings.append(
            f"{_count(len(left_judge), 'case')} left out of the comparison because the judge errored in the candidate run "
            f"({_limited_ids(left_judge, _MAX_REASON_IDS)}): judge errors are infrastructure failures, not quality failures, "
            f"so {'it is' if one else 'they are'} not counted as {'a failure' if one else 'failures'}"
        )
    if left_crash:
        one = len(left_crash) == 1
        warnings.append(
            f"{_count(len(left_crash), 'case')} that already failed in the baseline crashed in the candidate (target error; "
            f"{_limited_ids(left_crash, _MAX_REASON_IDS)}): left out of the pass rates because there is nothing left to regress, "
            f"but {'it counts' if one else 'they count'} towards the error rate"
        )

    # Rules 7 and 8: test-set integrity.  Deleting or editing cases must not be a way to pass.
    max_removed = eff["max_removed_cases"]
    if max_removed is not None and len(removed) > max_removed:
        reasons.append(
            f"{_count(len(removed), 'baseline case')} missing from the candidate ({_limited_ids(removed, _MAX_REASON_IDS)}): "
            "removing test cases can hide failures; if intentional (or this is a partial run), update the baseline in a reviewed PR"
        )
    max_changed = eff["max_changed_cases"]
    if max_changed is not None and len(changed) > max_changed:
        reasons.append(
            f"test cases were edited: {_count(len(changed), 'case')} {'differs' if len(changed) == 1 else 'differ'} from the baseline ({_limited_ids(changed, _MAX_REASON_IDS)}); "
            "update the baseline in a reviewed PR"
        )

    # Rule 9: the judges themselves (config, rubric, pass_rule) must not change together with the code under test.
    sha_base, sha_cand = baseline.summary.get("judges_sha"), candidate.summary.get("judges_sha")
    judges_changed: bool | None = None
    if sha_base and sha_cand:
        judges_changed = sha_base != sha_cand
    else:
        side = "baseline and candidate summaries carry" if not (sha_base or sha_cand) else (
            "baseline summary carries" if not sha_base else "candidate summary carries"
        )
        warnings.append(
            f"the {side} no judges_sha (written by an older evalsprint): a changed judge configuration or rubric cannot be detected; "
            "re-create the baseline with the current version"
        )
    if judges_changed:
        detail = f"judges_sha {_q(sha_base)} → {_q(sha_cand)}"
        if _before_v02(_kit_version(baseline.summary)):
            detail += (
                "; evalsprint 0.2 also hashes custom-judge code and data_files, so a baseline recorded by evalsprint 0.1 (or by a version "
                "that did not say) with custom judges must be refreshed in a reviewed PR"
            )
        if eff["allow_judge_changes"]:
            warnings.append(
                f"judge configuration or rubric changed ({detail}; allowed by gate.allow_judge_changes): "
                "pass rates are only comparable if the judge did not get stricter or more lenient; update the baseline in a reviewed PR"
            )
        else:
            reasons.append(f"judge configuration or rubric changed: update the baseline in a reviewed PR ({detail})")

    # Rule 11: the repeat protocol must not change together with the code under test: repeating a case takes its
    # majority verdict, so a PR that sets `repeat` (or removes it) moves the noise the gate is calibrated for.
    repeat_base, repeat_cand = _repeat_protocol(baseline.summary), _repeat_protocol(candidate.summary)
    repeat_changed = None if repeat_base is None or repeat_cand is None else repeat_base != repeat_cand
    if repeat_changed:
        detail = f"repeat: baseline {_repeat_text(repeat_base)}, candidate {_repeat_text(repeat_cand)}"
        if eff["allow_repeat_change"]:
            warnings.append(
                f"evaluation protocol changed ({detail}; allowed by gate.allow_repeat_change): "
                "pass rates and the false-alarm rate of the gate are only comparable under the same protocol; update the baseline in a reviewed PR"
            )
        else:
            reasons.append(f"evaluation protocol changed ({detail}): update the baseline in a reviewed PR")

    # Rule 12: the kit version.  ANY different version is worth saying (a patch release can fix a parser that scored a case
    # differently); a different major.minor is the stronger statement.  require_same_kit_version makes either a reason.
    ver_base, ver_cand = _kit_version(baseline.summary), _kit_version(candidate.summary)
    kit_changed = None if ver_base is None or ver_cand is None else _kit_release(ver_base) != _kit_release(ver_cand)
    if kit_changed:
        detail = f"baseline {_q(ver_base)}, candidate {_q(ver_cand)}"
        if eff["require_same_kit_version"]:
            reasons.append(f"evalsprint version changed ({detail}; gate.require_same_kit_version is on): update the baseline in a reviewed PR")
        elif _kit_series(ver_base) != _kit_series(ver_cand):
            warnings.append(
                f"evalsprint version differs between baseline and candidate ({detail}): scoring may differ across versions, "
                "so the pass rates may not be comparable; refresh the baseline with the version you run"
            )
        else:
            warnings.append(
                f"evalsprint patch version differs between baseline and candidate ({detail}): scoring is expected to be the same, "
                "but a patch release can fix how a case is scored; refresh the baseline with the version you run if the pass rates look odd"
            )
    elif kit_changed is None and eff["require_same_kit_version"]:
        side = "baseline and candidate summaries carry" if not (ver_base or ver_cand) else (
            "baseline summary carries" if not ver_base else "candidate summary carries"
        )
        warnings.append(f"the {side} no evalsprint version: gate.require_same_kit_version could not be checked; re-create the baseline with the current version")

    # A baseline recorded from a placeholder run (the scaffold's offline stub, or an LLM judge on the mock provider) proves
    # nothing about the real product: every pass rate compared against it is an artefact.
    if baseline.summary.get("placeholder") is True:
        warnings.append(
            "the baseline run is a placeholder (simulated target or mock judge: its verdicts say nothing about the real product), "
            "so this comparison is meaningless; record a baseline from a real run and commit that one"
        )

    # Rule 13: target_retries.  A failed target call is repeated up to `target_retries` times inside the run, so a retry hides a transient
    # crash from the pass rate and from the error rate: a PR that raises it in its own eval.yaml would hide a crash regression.
    retries_base, retries_cand = _retry_stats(baseline, err_base), _retry_stats(candidate, err_cand)
    cfg_base, cfg_cand = retries_base["configured"], retries_cand["configured"]
    retries_changed = None if cfg_base is None or cfg_cand is None else cfg_base != cfg_cand
    if retries_changed:
        detail = f"target_retries: baseline {cfg_base}, candidate {cfg_cand}"
        if eff["allow_target_retries_change"]:
            warnings.append(
                f"target retries changed ({detail}; allowed by gate.allow_target_retries_change): a retry hides a transient crash from the pass rate, "
                "so runs with different values are only comparable if the target did not start to crash; update the baseline in a reviewed PR"
            )
        else:
            reasons.append(
                f"target retries changed ({detail}): a retry hides a transient crash from the pass rate, so a change of this value can hide "
                "a crash regression; update the baseline in a reviewed PR"
            )
    elif retries_changed is None:
        missing = [side for side, stats in (("baseline", retries_base), ("candidate", retries_cand)) if stats["configured"] is None]
        if missing:
            who = " and ".join(missing)
            warnings.append(
                f"the {who} summar{'ies do' if len(missing) == 2 else 'y does'} not record target_retries (written by an older evalsprint): "
                "a change of the retry setting, which hides transient crashes, cannot be detected; re-create the baseline with the current version"
            )
    share_warn = eff["retry_share_warn"]
    if share_warn is not None and retries_base["share"] is not None and retries_cand["share"] is not None:
        more = retries_cand["share"] - retries_base["share"]
        if more >= share_warn - _EPS:
            warnings.append(
                f"the candidate retried its target {_retries(retries_cand['retried_calls'])} in {_count(retries_cand['attempts'], 'call')} "
                f"({retries_cand['share'] * 100:.1f}% of the calls) against {retries_base['share'] * 100:.1f}% in the baseline "
                f"(+{more * 100:.1f} p.p., limit {share_warn * 100:.1f}): a retry hides a crash from the pass rate, so the target may have started "
                "to fail much more often than the pass rate shows; check the crash counts of the run before trusting a pass"
            )

    # Rule 10: the LLM judge must be calibrated against human labels.
    calibration_status, calibration_problem, calibration_notes = _calibration_status(calibration, eff["min_judge_kappa"])
    warnings.extend(calibration_notes)
    if calibration is None and any(_is_llm_judge(j) for r in candidate.results for j in r.get("judges") or []):
        calibration_problem = "LLM judge not calibrated: no human-label calibration was provided, so its verdicts are unverified"
    if calibration_problem:
        (reasons if eff["require_calibration"] else warnings).append(calibration_problem)

    if 0 < n < SMALL_SET:
        warnings.append(_small_set_warning(n, rule))

    return {
        "verdict": "regression" if reasons else "pass",
        "reasons": [_message(r) for r in reasons],
        "warnings": [_message(w) for w in warnings],
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "n_paired": n,
        "unpaired": {"only_baseline": removed, "only_candidate": added},
        "removed_cases": list(removed),
        "changed_cases": changed,
        "excluded_errors": errored,
        "target_error_regressions": crashed,
        "pass_rate": {
            "baseline": rate_base,
            "candidate": rate_cand,
            "delta": delta,
            "baseline_ci": list(wilson_ci(k_base, n)) if n else None,
            "candidate_ci": list(wilson_ci(k_cand, n)) if n else None,
            "b_pass_to_fail": n_b,
            "c_fail_to_pass": n_c,
            "p_value": p_value,
            "p_value_two_sided": p_two_sided,
            "candidate_overall": cand_overall,
            "candidate_paired_pessimistic": cand_paired_pessimistic,
            "candidate_paired_pessimistic_n": len(pessimistic),
        },
        "score": score,
        "by_tag": by_tag,
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "critical_regressions": critical_regressions,
        "critical_unverified": critical_unverified,
        "cost": {"baseline_mean_usd": cost_base, "candidate_mean_usd": cost_cand, "delta_pct": cost_pct},
        "latency_p95": {"baseline_s": lat_base, "candidate_s": lat_cand, "delta_pct": lat_pct},
        "error_rate": {"baseline": err_base["error_rate"], "candidate": err_cand["error_rate"]},
        "errors": {
            "baseline": err_base,
            "candidate": {
                **err_cand,
                "paired_error_rate": err_paired["error_rate"],
                "paired_n_errors": err_paired["n_errors"],
                "paired_n_cases": len(common),
                "paired_attempt_error_rate": err_paired["attempt_error_rate"],
                "paired_n_attempt_errors": err_paired["n_attempt_errors"],
                "paired_n_attempts": err_paired["n_attempts"],
            },
        },
        "effective_max_pass_rate_drop": limit,
        "effective_alpha": rule.alpha if rule else None,
        "effective_min_net_loss": rule.min_net_loss if rule else None,
        "gate_mode": mode,
        "tag_warnings": tag_warnings,
        "flaky_crashes": [cid for cid, _, _ in flaky],
        "repeat_changed": repeat_changed,
        "repeat_protocol": {"baseline": _protocol_dict(repeat_base), "candidate": _protocol_dict(repeat_cand)},
        "kit_version": {"baseline": ver_base, "candidate": ver_cand, "changed": kit_changed},
        "target_retries_changed": retries_changed,
        "target_retries": {"baseline": retries_base, "candidate": retries_cand, "changed": retries_changed is True},
        "calibration_status": calibration_status,
        "judges_changed": judges_changed,
        "gate": eff,
    }


# ------------------------------------------------------ the gate's own false-alarm and detection rates

LOTTERY_FA_LEVELS = (0.15, 0.25)  # "a committed baseline makes more than this share of unchanged PRs fail"
_LOTTERY_CUTOFF = 1e-13  # baseline draws lighter than this are left out (total mass lost < 1e-9)
_LOTTERY_DRAWS = 4000  # with several attempt groups the cutoff rises until at most this many baseline draws are visited (time stays in seconds)


def _trip_bound(rule: _DropRule):
    """``b -> cmax``: the largest number c of fail->pass flips at which b pass->fail flips still trip the rule, -1 when none.

    Every rule here trips on a prefix of c for fixed b (more recoveries shrink the net loss and raise the exact one-sided p),
    so the region is a table of bounds and a probability is a finite sum.  Found by bisection on the SHIPPED ``rule.trips``,
    memoised per call.
    """
    n, memo = rule.n, {}

    def cmax(b: int) -> int:
        if b not in memo:
            top = n - b
            if top < 0 or not rule.trips(b, 0):
                memo[b] = -1
            else:
                lo, hi = 0, top
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    lo, hi = (mid, hi) if rule.trips(b, mid) else (lo, mid - 1)
                memo[b] = lo
        return memo[b]

    return cmax


def _binom_cdf(m: int, q: float, k: int) -> float:
    """P(X <= k) for X ~ Binomial(m, q)."""
    lo, probs = binom_pmf_window(m, q)
    return 0.0 if k < lo else min(1.0, sum(probs[: k - lo + 1]))


def _trip_probability(n: int, noise: float, d: float, cmax) -> float:
    """Exact P(the rule trips) when baseline and candidate are two independent noisy readings of the same latent verdicts.

    Each verdict flips with probability ``noise``; a fraction ``d`` of all n cases really regressed (latent pass -> fail).
    Per case P(pass->fail) = d(1-noise)^2 + (1-d)noise(1-noise) and P(fail->pass) = d noise^2 + (1-d)noise(1-noise), whatever
    the baseline pass rate, so b ~ Bin(n, pb) and c | b ~ Bin(n - b, pc / (1 - pb)).
    """
    pb = d * (1 - noise) ** 2 + (1 - d) * noise * (1 - noise)
    pc = d * noise**2 + (1 - d) * noise * (1 - noise)
    q = 1.0 if pb >= 1.0 else min(1.0, pc / (1.0 - pb))
    lo, probs = binom_pmf_window(n, pb)
    total = 0.0
    for i, p_b in enumerate(probs):
        b = lo + i
        bound = cmax(b)
        if bound >= 0:
            total += p_b * _binom_cdf(n - b, q, bound)
    return min(1.0, total)


def _trinomial_bc(m: int, pb: float, pc: float) -> dict[tuple[int, int], float]:
    """``{(b, c): probability}`` for ``m`` independent cases that are b-type (``pb``), c-type (``pc``) or neither, cut at 1e-18."""
    lo_b, p_bs = binom_pmf_window(m, pb)
    q = 0.0 if pb >= 1.0 else min(1.0, pc / (1.0 - pb))
    out: dict[tuple[int, int], float] = {}
    for i, w_b in enumerate(p_bs):
        b = lo_b + i
        lo_c, p_cs = binom_pmf_window(m - b, q)
        for j, w_c in enumerate(p_cs):
            if w_b * w_c >= 1e-18:
                out[(b, lo_c + j)] = w_b * w_c
    return out


def _convolve_bc(x: dict, y: dict) -> dict:
    """Joint (b, c) distribution of two independent groups of cases."""
    out: dict[tuple[int, int], float] = {}
    for (b1, c1), w1 in x.items():
        for (b2, c2), w2 in y.items():
            w = w1 * w2
            if w >= 1e-18:
                out[(b1 + b2, c1 + c2)] = out.get((b1 + b2, c1 + c2), 0.0) + w
    return out


def _trip_probability_groups(groups: list[tuple[int, float, float]], cmax) -> float:
    """Exact P(the rule trips) for independent groups of cases ``(m, pb, pc)``: every case is b-type with ``pb``, c-type with ``pc``.

    The (b, c) distribution of the suite is the convolution of the groups' trinomials: no independence shortcut between the
    groups.  The largest group is taken last and conditioned on its own b (c | b is binomial), so the joint table that is
    actually built only spans the small groups: cheap for one big group plus a few critical or extra-repeated cases.
    """
    groups = sorted((g for g in groups if g[0] > 0), key=lambda g: g[0])
    if not groups:
        return 0.0
    *head, (m, pb, pc) = groups
    joint: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for group in head:
        joint = _convolve_bc(joint, _trinomial_bc(*group))
    q = 0.0 if pb >= 1.0 else min(1.0, pc / (1.0 - pb))
    lo_b, p_bs = binom_pmf_window(m, pb)
    total = 0.0
    for i, w_b in enumerate(p_bs):
        b2 = lo_b + i
        lo_c, p_cs = binom_pmf_window(m - b2, q)
        cum = list(itertools.accumulate(p_cs))
        for (b1, c1), w1 in joint.items():
            bound = cmax(b1 + b2) - c1
            if bound >= lo_c:
                total += w1 * w_b * cum[min(bound - lo_c, len(cum) - 1)]
    return min(1.0, total)


def _convolve(x: tuple[int, tuple[float, ...]], y: tuple[int, tuple[float, ...]]) -> tuple[int, list[float]]:
    """Distribution of the sum of two independent windowed variables."""
    (lo_x, px), (lo_y, py) = x, y
    out = [0.0] * (len(px) + len(py) - 1)
    for i, a in enumerate(px):
        for j, b in enumerate(py):
            out[i + j] += a * b
    return lo_x + lo_y, out


def _false_alarm_given_baseline(n_pass: int, n_fail: int, e: int, a: int, noise: float, cmax) -> float:
    """P(an unchanged candidate trips the gate) once the baseline is fixed: e latent-pass cases read fail, a latent-fail read pass.

    b (baseline pass -> candidate fail) = Bin(n_pass - e, noise) + Bin(a, 1 - noise);
    c (baseline fail -> candidate pass) = Bin(e, 1 - noise) + Bin(n_fail - a, noise).
    """
    return _false_alarm_given_parts(
        [binom_pmf_window(n_pass - e, noise), binom_pmf_window(a, 1 - noise)],
        [binom_pmf_window(e, 1 - noise), binom_pmf_window(n_fail - a, noise)],
        cmax,
    )


def _false_alarm_given_parts(parts_b: list, parts_c: list, cmax) -> float:
    """P(trip) when b and c are independent sums of the windowed binomials in ``parts_b`` and ``parts_c``."""
    dist_b, dist_c = (0, [1.0]), (0, [1.0])
    for part in parts_b:
        dist_b = _convolve(dist_b, part)
    for part in parts_c:
        dist_c = _convolve(dist_c, part)
    return _trip_given_sums(dist_b, dist_c, cmax)


def _trip_given_sums(dist_b: tuple, dist_c: tuple, cmax) -> float:
    """P(the rule trips) for independent windowed b and c: ``sum_b P(b) P(c <= cmax(b))`` (a trip region is a prefix in c per b)."""
    (lo_b, p_b), (lo_c, p_c) = dist_b, dist_c
    cdf, acc = [], 0.0
    for v in p_c:
        acc += v
        cdf.append(acc)
    total = 0.0
    for i, v in enumerate(p_b):
        bound = cmax(lo_b + i)
        if bound >= lo_c:
            total += v * cdf[min(bound - lo_c, len(cdf) - 1)]
    return min(1.0, total)


def _baseline_lottery(n: int, noise: float, baseline_pass: float, cmax) -> dict:
    """How the false-alarm rate of later PRs depends on the ONE noisy baseline that was committed.

    The system is unchanged and has ``round(baseline_pass * n)`` latent passes.  The committed baseline reads each case with
    flip probability ``noise``; every later PR is a fresh reading.  Given the baseline the false-alarm rate is exact, and it
    varies a lot with the draw: a baseline that happens to have misread a few passes as fails makes every later PR look like a
    recovery, one that misread fails as passes makes every PR look like a loss.  This returns the distribution of that rate over
    baseline draws (the critic's cycle-3 "committed-baseline lottery").  Every baseline draw that carries at
    least ``_LOTTERY_CUTOFF`` of the probability is visited: exact, no approximation.
    """
    n_pass = round(baseline_pass * n)
    return _lottery_of_groups([(n_pass, n - n_pass, noise, noise)], cmax)


def _lottery_of_groups(groups: list[tuple[int, int, float, float]], cmax, max_draws: int | None = None) -> dict:
    """The committed-baseline lottery for groups of cases ``(latent passes, latent fails, P(pass read fail), P(fail read pass))``.

    A baseline draw is, per group, ``e`` latent passes read as fails and ``a`` latent fails read as passes (independent binomials);
    given it, b and c of an unchanged candidate are sums of binomials (see ``_false_alarm_given_baseline``).  Draws lighter than
    the cutoff are left out; the others are all visited.  The cutoff is ``_LOTTERY_CUTOFF`` (mass lost < 1e-9), except that with
    several groups and ``max_draws`` it rises by factors of 10 until at most ``max_draws`` draws are visited (the number of draws
    grows with the product of the groups' windows; the lost mass is still below ``1e-6`` for the suites the kit sees, see the tests).
    """
    per_group = []
    for n_pass, n_fail, q_pass, q_fail in groups:
        lo_e, p_e = binom_pmf_window(n_pass, q_pass)
        lo_a, p_a = binom_pmf_window(n_fail, q_fail)
        per_group.append([(lo_e + i, lo_a + j, w_e * w_a) for i, w_e in enumerate(p_e) for j, w_a in enumerate(p_a)])

    def count(cutoff: float, g: int = 0, weight: float = 1.0) -> int:
        if g == len(groups):
            return 1
        return sum(count(cutoff, g + 1, weight * w) for _, _, w in per_group[g] if weight * w >= cutoff)

    cutoff = _LOTTERY_CUTOFF
    if max_draws is not None and len(groups) > 1:
        while cutoff < 1e-6 and count(cutoff) > max_draws:
            cutoff *= 10.0
    draws: list[tuple[float, float]] = []

    def visit(g: int, weight: float, dist_b: tuple, dist_c: tuple) -> None:
        if g == len(groups):
            draws.append((_trip_given_sums(dist_b, dist_c, cmax), weight))
            return
        n_pass, n_fail, q_pass, q_fail = groups[g]
        for e, a, w in per_group[g]:
            if weight * w >= cutoff:
                visit(
                    g + 1,
                    weight * w,
                    _convolve(_convolve(dist_b, binom_pmf_window(n_pass - e, q_pass)), binom_pmf_window(a, 1 - q_fail)),
                    _convolve(_convolve(dist_c, binom_pmf_window(e, 1 - q_pass)), binom_pmf_window(n_fail - a, q_fail)),
                )

    visit(0, 1.0, (0, [1.0]), (0, [1.0]))
    draws.sort()
    mass = sum(w for _, w in draws)

    def over(level: float) -> float:
        return sum(w for fa, w in draws if fa > level) / mass

    def quantile(q: float) -> float:
        acc = 0.0
        for fa, w in draws:
            acc += w
            if acc >= q * mass:
                return fa
        return draws[-1][0]

    return {
        "p_fa_over_15": over(0.15),
        "median_fa": quantile(0.5),
        "mean_fa": sum(fa * w for fa, w in draws) / mass,
        "p95_fa": quantile(0.95),
        "p_fa_over_25": over(0.25),
    }


def expected_gate_rates(n: int, noise: float, baseline_pass: float = 0.8, drops=(0.05, 0.10), gate: dict | None = None) -> dict:
    """What the pass-rate rule of the gate will do on a suite of ``n`` paired cases whose verdicts flip with probability ``noise``.

    Exact (a finite sum over the cases the rule trips on, no simulation, no seed).  ``gate`` is a gate config (partial is
    fine; ``None``: the defaults, i.e. the ``auto`` rule); only its pass-rate rule counts here (``max_pass_rate_drop``,
    ``require_significance``, ``alpha``).  The model is the one of the gate curve: baseline and candidate are two independent
    noisy readings of the same latent verdicts, each verdict wrong with probability ``noise`` (per-VERDICT noise: for a
    repeated case that is the noise of its majority verdict, see ``stats.majority_noise`` and ``stats.noise_from_flip_rate``).
    ``baseline_pass`` is the latent pass rate of the suite; it only matters for ``baseline_lottery``.  Cases are independent: a
    "drop" of d means that the PR lowers the pass probability of every case by d (each case regresses with probability d), so
    the number of regressed cases is Binomial(n, d), not exactly n*d.  A PR that breaks exactly d*n cases is a different
    event: at n=45 and 5% noise, 5 regressed cases (d = 1/9) trip the gate 76.8% of the time, this model says 71.5%; across
    n = 30..300 and d = 5..20% the two differ by -2.4 to +7.9 p.p. (``cycle4/eval-scientist-gate/validate_detection_model.py``).
    Quote the detection rates as "about", never as a guarantee.

    Returns::

        {"false_alarm": share of unchanged PRs that trip the gate, averaged over baseline draws,
         "detection": {drop: share of PRs with a real ``drop`` (fraction of ALL cases that went pass -> fail) that trip it;
                       None when the drop is larger than ``baseline_pass``},
         "baseline_lottery": {"p_fa_over_15": chance that the committed baseline makes more than 15% of unchanged PRs fail,
                              "median_fa": the false-alarm rate of the median baseline draw,
                              "mean_fa" (= false_alarm), "p95_fa", "p_fa_over_25"},
         plus "n", "noise", "baseline_pass", "gate_mode" ("auto" | "fixed" | "off") and the rule's effective_* numbers}

    With the pass-rate rule off nothing ever trips: every rate is 0.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n (paired cases) must be a positive integer, got {n!r}")
    if not _is_number(noise) or not 0 <= noise <= 0.5:
        raise ValueError(f"noise is the chance that a verdict flips between runs, a number between 0 and 0.5, got {noise!r}")
    if not _is_number(baseline_pass) or not 0 <= baseline_pass <= 1:
        raise ValueError(f"baseline_pass must be a number between 0 and 1, got {baseline_pass!r}")
    drops = tuple(drops)
    for d in drops:
        if not _is_number(d) or not 0 <= d <= 1:
            raise ValueError(f"every drop is a fraction of the cases between 0 and 1 (e.g. 0.10 for 10 p.p.), got {d!r}")
    eff = resolve_gate({}, gate)
    mode, rule = _drop_setup(eff["max_pass_rate_drop"], n, eff["require_significance"], eff["alpha"])
    out = {
        "n": n,
        "noise": noise,
        "baseline_pass": baseline_pass,
        "gate_mode": mode,
        "effective_max_pass_rate_drop": rule.limit if rule else None,
        "effective_alpha": rule.alpha if rule else None,
        "effective_min_net_loss": rule.min_net_loss if rule else None,
    }
    if rule is None:
        zero = {"p_fa_over_15": 0.0, "median_fa": 0.0, "mean_fa": 0.0, "p95_fa": 0.0, "p_fa_over_25": 0.0}
        return {**out, "false_alarm": 0.0, "detection": {d: (0.0 if d <= baseline_pass else None) for d in drops}, "baseline_lottery": zero}
    cmax = _trip_bound(rule)
    return {
        **out,
        "false_alarm": _trip_probability(n, noise, 0.0, cmax),
        "detection": {d: (_trip_probability(n, noise, d, cmax) if d <= baseline_pass else None) for d in drops},
        "baseline_lottery": _baseline_lottery(n, noise, baseline_pass, cmax),
    }


def _cell_probabilities(q_pass: float, q_fail: float, baseline_pass: float, drop: float) -> tuple[float, float]:
    """Per case ``(P(baseline pass -> candidate fail), P(baseline fail -> candidate pass))``.

    Every case is, independently, a latent pass with probability ``baseline_pass`` (and then, with probability ``drop / baseline_pass``,
    one the change breaks: latent pass -> latent fail, so a fraction ``drop`` of ALL cases regresses) or a latent fail.  The baseline and
    the candidate each read the latent verdict and get it wrong with ``q_pass`` (latent pass read as fail) or ``q_fail`` (latent fail
    read as pass).  With ``q_pass == q_fail == q`` this is the gate curve's Model B: ``drop (1-q)^2 + (1-drop) q (1-q)`` and
    ``drop q^2 + (1-drop) q (1-q)``, whatever ``baseline_pass``.
    """
    stays = baseline_pass - drop  # latent pass, unchanged
    p_b = stays * (1 - q_pass) * q_pass + drop * (1 - q_pass) * (1 - q_fail) + (1 - baseline_pass) * q_fail * (1 - q_fail)
    p_c = stays * q_pass * (1 - q_pass) + drop * q_pass * q_fail + (1 - baseline_pass) * (1 - q_fail) * q_fail
    return p_b, p_c


def expected_gate_rates_protocol(
    attempts,
    critical,
    noise: float,
    baseline_pass: float = 0.8,
    drops=(0.05, 0.10),
    gate: dict | None = None,
) -> dict:
    """What the WHOLE gate does on a suite whose cases ran a different number of times, and whose critical cases have their own rule.

    ``expected_gate_rates`` prices the pass-rate rule for one verdict noise shared by every case.  A run with
    ``repeat: {n: 3, per_tag: {safety: 5}}`` has cases of different noise (the majority of 3 attempts is wrong far less often than a
    single one, the majority of 5 less still), and a critical case trips the gate on any single pass -> fail flip whatever the pass-rate
    rule says.  This function enumerates exactly, over the groups of cases with equal (attempts, critical): the pass-rate rule, the
    critical-case rule, and the gate as a whole (the exact joint probability that either trips: no independence shortcut, no simulation).

    Arguments.  ``attempts[i]``: attempts of case i (1 for a case that ran once); ``critical[i]``: case i carries a critical tag;
    ``noise``: the per-ATTEMPT flip probability (``stats.noise_from_flip_rate`` or ``stats.attempt_noise_estimate`` give it from a
    run's ``summary.stability.flip_rate``); the verdict noise of case i is the wrong-majority probability of its attempts
    (``stats.majority_noise``; for an even number of attempts a tie is a fail, so the pass -> fail and fail -> pass directions get
    their own exact value, ``stats.majority_noise_directions``: the symmetric mean would understate the false alarm).  ``baseline_pass``,
    ``drops`` and ``gate`` as in ``expected_gate_rates`` (only the pass-rate rule of ``gate`` counts; the critical rule is on for the
    cases flagged in ``critical``, whatever the gate says).  The model has the same caveats as ``expected_gate_rates``: cases are
    independent, a drop of d means every case regresses with probability d (quote detection as "about"), the noise measured on
    the repeated cases is assumed to be the noise of every case.

    For the platform (``evalsprint noise``), from a run's ``results.jsonl``: take the cases that were scored (``outcome`` pass or fail)::

        attempts = [len(r["attempts"]) if isinstance(r.get("attempts"), list) and r["attempts"] else 1 for r in scored]
        critical = [bool(set((r.get("case") or {}).get("tags") or []) & set(gate["critical_tags"])) for r in scored]
        noise = stats.attempt_noise_estimate(n_flipped, [a for a in attempts if a >= 2])["noise"]   # or ["noise_upper95"]
        rates = expected_gate_rates_protocol(attempts, critical, noise, baseline_pass, gate=gate)

    where ``gate`` is the gate compare would use (``resolve_gate``) and ``n_flipped`` is ``round(flip_rate * n_repeated)``.

    Returns::

        {"n", "attempt_noise", "verdict_noise_mean" (mean over cases of ``majority_noise(noise, attempts[i])``), "baseline_pass",
         "n_critical", "groups": [{"attempts", "critical", "cases"}], "gate_mode", "effective_*" (the pass-rate rule),
         "false_alarm": {"pass_rate_rule", "critical_rule", "whole_gate"},
         "detection": {drop: {"pass_rate_rule", "critical_rule", "whole_gate"} | None when drop > baseline_pass},
         "baseline_lottery": {"p_fa_over_15", "median_fa", "mean_fa", "p95_fa", "p_fa_over_25"}  # the pass-rate rule only}

    ``baseline_lottery`` assumes ``round(baseline_pass * n)`` latent passes spread over the attempt groups in proportion to their size;
    its ``mean_fa`` equals ``false_alarm["pass_rate_rule"]`` whenever every verdict noise is symmetric (odd attempts), and differs by a
    few hundredths of a point for even attempts (latent verdicts drawn independently in one, fixed in the other).  With the pass-rate
    rule off its entries are 0 and ``whole_gate`` is the critical rule alone.
    """
    attempts, critical = list(attempts), list(critical)
    n = len(attempts)
    if n < 1 or len(critical) != n:
        raise ValueError(f"attempts and critical need one entry per paired case (at least one), got {n} and {len(critical)}")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in attempts):
        raise ValueError("every entry of attempts must be an integer >= 1 (1 for a case that ran once)")
    if any(not isinstance(c, bool) for c in critical):
        raise ValueError("every entry of critical must be True or False")
    if not _is_number(noise) or not 0 <= noise <= 0.5:
        raise ValueError(f"noise is the chance that ONE attempt lands on the wrong side, a number between 0 and 0.5, got {noise!r}")
    if not _is_number(baseline_pass) or not 0 <= baseline_pass <= 1:
        raise ValueError(f"baseline_pass must be a number between 0 and 1, got {baseline_pass!r}")
    drops = tuple(drops)
    for d in drops:
        if not _is_number(d) or not 0 <= d <= 1:
            raise ValueError(f"every drop is a fraction of the cases between 0 and 1 (e.g. 0.10 for 10 p.p.), got {d!r}")
    eff = resolve_gate({}, gate)
    mode, rule = _drop_setup(eff["max_pass_rate_drop"], n, eff["require_significance"], eff["alpha"])

    counts: dict[tuple[int, bool], int] = {}
    for k, is_critical in zip(attempts, critical):
        counts[(k, is_critical)] = counts.get((k, is_critical), 0) + 1
    verdict = {k: majority_noise_directions(noise, k) for k in {k for k, _ in counts}}
    cmax = _trip_bound(rule) if rule else None

    def rates(drop: float) -> dict:
        cells = {key: _cell_probabilities(*verdict[key[0]], baseline_pass, drop) for key in counts}
        crit = [(counts[key], *cells[key]) for key in counts if key[1]]
        rest = [(counts[key], *cells[key]) for key in counts if not key[1]]
        no_critical_flip = math.prod((1.0 - p_b) ** m for m, p_b, _ in crit)
        pass_rule = _trip_probability_groups(crit + rest, cmax) if rule else 0.0
        if rule and no_critical_flip > 0.0:
            # the gate stays quiet iff no critical case flipped pass -> fail AND the pass-rate rule stays quiet:
            # condition the critical cases on b = 0 (their c is then binomial with pc / (1 - pb), weighted by (1 - pb)^m)
            conditioned = [(m, 0.0, p_c / (1.0 - p_b)) for m, p_b, p_c in crit]
            trips_without_critical_flip = no_critical_flip * _trip_probability_groups(rest + conditioned, cmax)
        else:
            trips_without_critical_flip = 0.0
        return {
            "pass_rate_rule": pass_rule,
            "critical_rule": 1.0 - no_critical_flip,
            "whole_gate": min(1.0, 1.0 - no_critical_flip + trips_without_critical_flip),
        }

    out = {
        "n": n,
        "attempt_noise": noise,
        "verdict_noise_mean": sum(majority_noise(noise, k) for k in attempts) / n,
        "baseline_pass": baseline_pass,
        "n_critical": sum(critical),
        "groups": [{"attempts": k, "critical": c, "cases": m} for (k, c), m in sorted(counts.items())],
        "gate_mode": mode,
        "effective_max_pass_rate_drop": rule.limit if rule else None,
        "effective_alpha": rule.alpha if rule else None,
        "effective_min_net_loss": rule.min_net_loss if rule else None,
        "false_alarm": rates(0.0),
        "detection": {d: (rates(d) if d <= baseline_pass else None) for d in drops},
    }
    if rule is None:
        out["baseline_lottery"] = {"p_fa_over_15": 0.0, "median_fa": 0.0, "mean_fa": 0.0, "p95_fa": 0.0, "p_fa_over_25": 0.0}
        return out
    by_attempts: dict[int, int] = {}
    for k in attempts:
        by_attempts[k] = by_attempts.get(k, 0) + 1
    n_pass, shares = round(baseline_pass * n), {}
    for k, m in by_attempts.items():
        shares[k] = (n_pass * m // n, (n_pass * m) % n)
    left = n_pass - sum(base for base, _ in shares.values())
    for k in sorted(shares, key=lambda k: (-shares[k][1], k))[:left]:
        shares[k] = (shares[k][0] + 1, shares[k][1])
    out["baseline_lottery"] = _lottery_of_groups(
        [(shares[k][0], by_attempts[k] - shares[k][0], *verdict[k]) for k in sorted(by_attempts)], cmax, max_draws=_LOTTERY_DRAWS
    )
    return out


# ---------------------------------------------------------------------- markdown


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _pct_ci(x: float | None, ci: list[float] | None) -> str:
    if x is None:
        return "n/a"
    return f"{_pct(x)} ({ci[0] * 100:.0f}–{ci[1] * 100:.0f}%)" if ci else _pct(x)


def _pp(delta: float | None) -> str:
    """Signed percentage-point difference of two fractions."""
    if delta is None:
        return "n/a"
    return f"{_signed(delta * 100, 1)} p.p."


def _usd(x: float | None) -> str:
    """Contract format: 4 decimals below $1, 2 above."""
    if x is None:
        return "n/a"
    return f"${x:.4f}" if abs(x) < 1 else f"${x:,.2f}"


def _per_1k(cost: float | None) -> float | None:
    return None if cost is None else cost * 1000


def _seconds(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s"


def _change(pct: float | None) -> str:
    return "n/a" if pct is None else _fmt_change(pct)


_MD_ENTITIES = {"&": "&amp;", "\\": "&#92;", "`": "&#96;", "<": "&lt;", ">": "&gt;", "[": "&#91;", "]": "&#93;"}

# What GitHub turns into a notification or a link when it finds it in the text of a comment.  Text is made inert by putting these in a code span.
_NOTIFY = re.compile(
    r"""
      [A-Za-z][A-Za-z0-9+.-]*://\S+            # a URL with a scheme
    | www\.[A-Za-z0-9-]\S*                     # www.host...
    | [\w.+-]+@[\w-]+(?:\.[\w-]+)+            # an e-mail address
    | (?<![\w/])@[\w-][\w./-]*                # @user, @org/team
    | (?:[\w.-]+/[\w.-]+)?\#\d+              # #123, owner/repo#123
    | \bGH-\d+
    """,
    re.VERBOSE,
)


def _printable(text) -> str:
    """The text with every control, line-break and invisible formatting character (bidi overrides, ...) turned into a space."""
    return "".join(ch if ch.isprintable() else " " for ch in str(text))


def _md_text(text, *, table: bool = False) -> str:
    """Text safe to put in a paragraph, list item or table cell as it stands: ``&``, backtick, ``<``, ``>``, ``[``, ``]`` and backslash
    become character references (they render as themselves but are never markup, so no tag, link, code span, escape or entity can be built
    from them); ``|`` becomes a reference, or ``\\|`` in a table cell; control characters become spaces, so the text cannot start a
    new block.  It does NOT stop ``@mention``, ``#123`` or a bare URL from being recognised: use ``_inert`` for text that may hold those.
    """
    text = _printable(text)
    pipe = "\\|" if table else "&#124;"
    return "".join(pipe if ch == "|" else _MD_ENTITIES.get(ch, ch) for ch in text)


def _inert(text, *, table: bool = False) -> str:
    """``_md_text`` plus: every mention, issue reference, URL and e-mail address of the text goes into a code span, where GitHub leaves it alone."""
    text = _printable(text)
    out, pos = [], 0
    for m in _NOTIFY.finditer(text):
        out.append(_md_text(text[pos : m.start()], table=table))
        out.append(_code(m.group(0), table=table))
        pos = m.end()
    out.append(_md_text(text[pos:], table=table))
    return "".join(out)


def _render_marked(marked: str) -> str:
    """The markdown of a message whose artifact values are marked with ``_q``: marked parts in code spans, the rest made inert."""
    parts = re.split(f"{_OPEN}(.*?){_CLOSE}", marked, flags=re.S)
    return "".join(_code(part) if i % 2 else _inert(part) for i, part in enumerate(parts))


def _cell(text) -> str:
    """A value for a Markdown table cell (tag names are user-defined): in a code span, with ``|`` and backticks as character references."""
    return _code(text, table=True)


def _code(text, *, table: bool = False) -> str:
    """Inline code that holds ``text`` literally: nothing inside a code span is markup, a mention, a link or an image.

    Every fence is a single backtick, so no reader (CommonMark, GitHub, a naive pair-of-backticks scan) can disagree about where a span
    ends.  A backtick of the text cannot live in such a span: the text is cut there and the backtick is written as a character
    reference between two spans (it renders as a backtick and is never markup).  In a table cell ``|`` is cut the same way.  A leading
    or trailing space gets the one-space padding CommonMark strips again.
    """
    text = _printable(text)
    out = []
    for piece in re.split(r"([`|])" if table else r"(`)", text):
        if piece == "`":
            out.append("&#96;")
        elif piece == "|" and table:
            out.append("&#124;")
        elif piece:
            pad = " " if piece.strip(" ") and (piece[0] == " " or piece[-1] == " ") else ""
            out.append(f"`{pad}{piece}{pad}`")
    return "".join(out) or "` `"


def _md(message) -> str:
    """A reason or warning for markdown: its own ``.md`` when ``compare_runs`` made it, else the whole text made inert."""
    return getattr(message, "md", None) or _inert(message)


def _ids(ids: list[str]) -> str:
    return ", ".join(_code(i) for i in ids[:MAX_LISTED_IDS]) + (
        f" and {len(ids) - MAX_LISTED_IDS} more" if len(ids) > MAX_LISTED_IDS else ""
    )


def _gate_line(c: dict) -> str:
    """One-line summary of the thresholds this verdict was judged against."""
    gate = c["gate"]
    parts = []
    raw_mode, limit = c.get("gate_mode"), c.get("effective_max_pass_rate_drop")
    mode = normalize_gate_mode(raw_mode)
    if gate["max_pass_rate_drop"] == "auto" and mode == "auto" and raw_mode in LEGACY_GATE_MODES:
        # A comparison file written by evalsprint 0.1: describe the rule it was judged by, not today's.
        if limit is not None and raw_mode == "auto-large":
            parts.append(f"pass-rate drop ≤ {limit * 100:.1f} p.p. and significant (one-sided α={gate['alpha']:g}) · auto (0.1 rule): ≥ 150 cases")
        elif limit is not None:
            parts.append(f"pass-rate drop ≤ {limit * 100:.1f} p.p. · auto (0.1 rule): < 150 cases")
    elif gate["max_pass_rate_drop"] == "auto" and limit is not None:
        k, alpha = c.get("effective_min_net_loss"), c.get("effective_alpha")
        need = f"net loss ≥ {k} cases" if k else f"drop > {limit * 100:.1f} p.p."
        parts.append(f"pass-rate drop: {need}" + (f" and significant (one-sided α={alpha:.3f})" if alpha is not None else "") + f" · auto, {c.get('n_paired')} paired cases")
    elif gate["max_pass_rate_drop"] not in (None, "auto"):
        rule = f"pass-rate drop ≤ {gate['max_pass_rate_drop'] * 100:.1f} p.p."
        parts.append(rule + (f" (if significant, one-sided α={gate['alpha']:g})" if gate["require_significance"] else ""))
    if gate["min_pass_rate"] is not None:
        parts.append(f"pass rate ≥ {gate['min_pass_rate'] * 100:.1f}%")
    if gate["critical_tags"]:
        parts.append("critical tags: " + ", ".join(_code(t) for t in gate["critical_tags"]))
    if gate["max_cost_increase_pct"] is not None:
        parts.append(f"cost ≤ {_fmt_limit(gate['max_cost_increase_pct'])}")
    if gate["max_p95_latency_increase_pct"] is not None:
        parts.append(f"p95 latency ≤ {_fmt_limit(gate['max_p95_latency_increase_pct'])}")
    if gate["max_error_rate"] is not None:
        parts.append(f"error rate ≤ {gate['max_error_rate'] * 100:.1f}%")
    if gate["max_removed_cases"] is not None:
        parts.append(f"removed cases ≤ {gate['max_removed_cases']}")
    if gate["max_changed_cases"] is not None:
        parts.append(f"edited cases ≤ {gate['max_changed_cases']}")
    if not gate.get("allow_judge_changes", False):
        parts.append("judges unchanged")
    if not gate.get("allow_repeat_change", False) and c.get("repeat_changed") is not None:
        parts.append("repeat protocol unchanged")
    if not gate.get("allow_target_retries_change", False) and c.get("target_retries_changed") is not None:
        parts.append("target retries unchanged")
    if gate.get("require_same_kit_version"):
        parts.append("same evalsprint version")
    if gate["require_calibration"]:
        parts.append("calibrated judge required" + (f" (κ lower bound ≥ {gate['min_judge_kappa']:g})" if gate["min_judge_kappa"] is not None else ""))
    return "Gate: " + " · ".join(parts) if parts else "Gate: no rules enabled"


def _metrics_table(c: dict) -> list[str]:
    pr, cost, lat, err = c["pass_rate"], c["cost"], c["latency_p95"], c["error_rate"]
    rows = [
        (
            "Pass rate (95% CI)",
            _pct_ci(pr["baseline"], pr["baseline_ci"]),
            _pct_ci(pr["candidate"], pr["candidate_ci"]),
            _pp(pr["delta"]),
        )
    ]
    if c["score"]:
        s = c["score"]
        delta = _signed(s["delta"], 3)
        if s["ci"]:
            delta += f" (95% CI {_signed(s['ci'][0], 3)} to {_signed(s['ci'][1], 3)})"
        rows.append(("Mean judge score", f"{s['baseline_mean']:.3f}", f"{s['candidate_mean']:.3f}", delta))
    if cost["baseline_mean_usd"] is not None or cost["candidate_mean_usd"] is not None:
        b, k = cost["baseline_mean_usd"], cost["candidate_mean_usd"]
        rows.append(("Cost / request", _usd(b), _usd(k), _change(cost["delta_pct"])))
        rows.append(("Cost / 1k requests", _usd(_per_1k(b)), _usd(_per_1k(k)), _change(cost["delta_pct"])))
    if lat["baseline_s"] is not None or lat["candidate_s"] is not None:
        rows.append(("p95 latency", _seconds(lat["baseline_s"]), _seconds(lat["candidate_s"]), _change(lat["delta_pct"])))
    cand_err = _pct(err["candidate"])
    cand_errors = (c.get("errors") or {}).get("candidate") or {}
    paired_err = cand_errors.get("paired_error_rate")
    if paired_err is not None and abs(paired_err - err["candidate"]) > _EPS:
        cand_err += f" ({_pct(paired_err)} on the baseline's cases)"
    attempt_err = cand_errors.get("attempt_error_rate")
    if attempt_err is not None and attempt_err > err["candidate"] + _EPS:
        cand_err += f" ({_pct(attempt_err)} of attempts)"
    rows.append(("Error rate", _pct(err["baseline"]), cand_err, _pp(err["candidate"] - err["baseline"])))

    lines = ["| Metric | Baseline | Candidate | Δ |", "|---|---:|---:|---:|"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return lines


def _tag_table(c: dict) -> list[str]:
    by_tag = c["by_tag"]
    if not by_tag:
        return []
    critical = set(c["gate"]["critical_tags"])
    rows = list(by_tag.items())
    if len(rows) > MAX_TAG_ROWS:
        rows = [(t, v) for t, v in rows if abs(v["delta"]) > _EPS]  # too many tags: only those that moved
        if not rows:
            return [f"No per-tag pass rate changed ({len(by_tag)} tags)."]
    rows.sort(key=lambda item: (item[1]["delta"], item[0]))  # worst first
    hidden = max(0, len(rows) - MAX_TAG_ROWS)
    rows = rows[:MAX_TAG_ROWS]
    lines = ["| Tag | n | Baseline | Candidate | Δ |", "|---|---:|---:|---:|---:|"]
    for tag, v in rows:
        name = _cell(tag) + (" 🔒" if tag in critical else "")
        lines.append(f"| {name} | {v['n']} | {_pct(v['baseline'])} | {_pct(v['candidate'])} | {_pp(v['delta'])} |")
    if hidden:
        lines.append("")
        lines.append(f"…and {hidden} more tags changed.")
    return lines


def _errors_note(c: dict) -> str | None:
    """One line when errors exist, saying which kinds count as failures (a crash the change introduced) and which do not."""
    errors = c.get("errors")
    if not errors or not (errors["baseline"]["n_errors"] or errors["candidate"]["n_errors"]):
        return None

    def side(e: dict) -> str:
        return f"{e['n_errors']} ({e['n_judge_errors']} judge, {e['n_target_errors']} target)"

    text = f"ℹ️ Errors: baseline {side(errors['baseline'])}, candidate {side(errors['candidate'])}. "
    crashed = c.get("target_error_regressions") or []
    if crashed:
        one = len(crashed) == 1
        text += (
            f"{_count(len(crashed), 'case')} that passed in the baseline crashed in the candidate (target error) and "
            f"{'counts' if one else 'count'} as {'a failure' if one else 'failures'}: {_ids(crashed)}. "
        )
    text += "Judge errors and cases that errored in the baseline are infrastructure problems, not quality failures, and are left out of the pass rates."
    return text


def _calibration_line(status: dict | None) -> str | None:
    if not status:
        return None
    kappa, ci = status.get("kappa"), status.get("kappa_ci")
    text = f"κ = {kappa:.2f}" if _is_number(kappa) else "κ n/a"
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_is_number(x) for x in ci):
        text += f" (95% CI {ci[0]:.2f}–{ci[1]:.2f})"
    judge = status.get("judge")
    who = f" {_code(judge)}" if judge and judge != "overall" else ""
    if status.get("kappa_decision"):
        text += f", decision {_code(status['kappa_decision'])}"
    if status.get("basis") == "legacy":
        text += " (unweighted, legacy file)"
    return f"Judge calibration{who}: {text} — {'meets' if status.get('ok') else 'does not meet'} the gate's requirement."


def _retries_line(c: dict) -> str | None:
    """One line about ``target_retries`` whenever either run repeated a crashed call or the two values differ: a retry hides a crash."""
    info = c.get("target_retries")
    if not isinstance(info, dict):
        return None
    sides = {name: info.get(name) if isinstance(info.get(name), dict) else {} for name in ("baseline", "candidate")}

    def whole(x) -> int | None:
        return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else None

    retried = any((whole(s.get("retried_calls")) or 0) > 0 for s in sides.values())
    if not (retried or info.get("changed")):
        return None

    def side(s: dict) -> str:
        cfg, used, calls = whole(s.get("configured")), whole(s.get("retried_calls")), whole(s.get("attempts"))
        text = f"`target_retries: {cfg}`" if cfg is not None else "`target_retries` not recorded"
        if used is not None and calls:
            text += f", {_retries(used)} after a crash in {_count(calls, 'call')} ({_pct(used / calls)})"
        elif used is not None:
            text += f", {_retries(used)} after a crash"
        return text

    return (
        f"↻ Target retries: baseline {side(sides['baseline'])}; candidate {side(sides['candidate'])}. "
        "A repeated call hides a transient crash from the pass rate and from the error rate."
    )


def to_markdown(c: dict) -> str:
    """Render a comparison for a GitHub PR comment or $GITHUB_STEP_SUMMARY."""
    ok = c["verdict"] == "pass"
    out = ["## ✅ No regression" if ok else "## ❌ Regression detected", ""]
    if not ok:
        out += [f"- {_md(reason)}" for reason in c["reasons"]]
        out.append("")
    out.append(f"Baseline {_code(c['baseline_run_id'])} → candidate {_code(c['candidate_run_id'])} · {c['n_paired']} paired cases")
    out.append("")
    if c.get("warnings"):
        out += [f"- ⚠️ {_md(w)}" for w in c["warnings"]]
        out.append("")

    out += _metrics_table(c)
    out.append("")
    note = _errors_note(c)
    if note:
        out += [note, ""]
    retries = _retries_line(c)
    if retries:
        out += [retries, ""]

    tag_lines = _tag_table(c)
    if tag_lines:
        out += ["**By tag**", ""] + tag_lines + [""]

    if c["newly_failing"]:
        line = f"**Newly failing ({len(c['newly_failing'])}):** {_ids(c['newly_failing'])}"
        crashed = c.get("target_error_regressions") or []
        if crashed:
            line += f" — {len(crashed)} of them crashed in the candidate: {_ids(crashed)}"
        out += [line, ""]
    if c["newly_passing"]:
        out += [f"**Newly passing ({len(c['newly_passing'])}):** {_ids(c['newly_passing'])}", ""]
    if c.get("changed_cases"):
        out += [f"**Edited test cases ({len(c['changed_cases'])}):** {_ids(c['changed_cases'])}", ""]

    pr = c["pass_rate"]
    if pr["p_value"] is not None:
        discordant = pr["b_pass_to_fail"] + pr["c_fail_to_pass"]
        if discordant:
            noun = "pair" if discordant == 1 else "pairs"
            two_sided = pr.get("p_value_two_sided")
            both = f" (two-sided {_p_expr(two_sided, ' ')})" if two_sided is not None else ""
            crashed = len(c.get("target_error_regressions") or [])
            flips = f"{pr['b_pass_to_fail']} pass→fail" + (f", {crashed} of them crashes" if crashed else "")
            out.append(
                f"One-sided McNemar exact {_p_expr(pr['p_value'], ' ')}{both} on {discordant} discordant {noun} "
                f"({flips}, {pr['c_fail_to_pass']} fail→pass)."
            )
        else:
            out.append("No discordant pairs: every paired case has the same pass/fail outcome in both runs.")
        out.append("")

    cal = _calibration_line(c.get("calibration_status"))
    if cal:
        out += [cal, ""]

    only_base, only_cand = c["unpaired"]["only_baseline"], c["unpaired"]["only_candidate"]
    excluded = len(c.get("excluded_errors") or []) + len(c.get("changed_cases") or [])
    if only_base or only_cand or excluded:
        note = f"ℹ️ Only cases present in both runs are compared; pass rates above use those {c['n_paired']} paired cases."
        if excluded:
            note += f" {excluded} shared case{'s are' if excluded != 1 else ' is'} left out (edited or errored)."
        if only_base:
            note += f" Baseline only ({len(only_base)}): {_ids(only_base)}."
        if only_cand:
            note += f" Candidate only ({len(only_cand)}): {_ids(only_cand)}."
        out += [note, ""]

    out.append(f"<sub>{_gate_line(c)}</sub>")
    if not ok:
        out.append("")
        out.append(
            "<sub>Intentional change? Refresh the baseline with the *Update eval baseline* workflow "
            "and review the resulting PR.</sub>"
        )
    return "\n".join(out) + "\n"
