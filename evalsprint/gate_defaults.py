"""Default regression-gate settings (the single source of truth).

Kept in its own dependency-free module so both ``config`` (which re-exports it)
and ``compare`` can import it without a cycle.  Semantics live in
``compare.py``; the short version:

* ``max_pass_rate_drop: auto`` is ONE rule for every number of paired cases n, with no switch
  and therefore no cliff: the gate trips when the net loss (cases that went pass->fail minus
  cases that went fail->pass) is at least ``max(3, ceil(n / 50))`` cases AND the one-sided exact
  McNemar p-value is below ``alpha(n) = 0.0575 + 0.1925 / (1 + (n / 85) ** 4.5)``.  Small suites get a
  generous alpha (they have little data), large suites a strict one.  ``gate_mode`` is
  ``"auto"``, ``"fixed"`` or ``"off"``; ``alpha`` and ``require_significance`` below do not apply
  to ``auto``.  Measured false-alarm and detection rates for every n:
  orchestration/2026-09-29_kit-v0.2/cycle3/eval-scientist-gate/gate_rates.md.
* A number instead of ``auto`` is a fixed absolute limit (optionally combined
  with ``require_significance``), exactly as before.
* Errors are never counted as failures, except a target error the candidate introduces on a case
  that passed in the baseline (a crash is a regression): pairing drops any case whose error says
  nothing about quality, and ``max_error_rate`` flags an unhealthy run separately (per case and per attempt).
* A PR must not change the judges (config, rubric, pass_rule, custom judge code), the repeat protocol or the test
  cases' verdicts in one go: runs carry ``summary["judges_sha"]`` and ``summary["repeat"]`` (``{"n": 3, "per_tag":
  {"safety": 5}}``; the old ``{"n": 5, "tags": ["safety"]}`` reads as ``{"n": 1, "per_tag": {"safety": 5}}``), and a
  different sha or protocol is a regression reason unless ``allow_judge_changes`` / ``allow_repeat_change`` is true.
* ``target_retries`` (the config value the run used: a failed target call is repeated up to that many times inside the run,
  so a retry hides a transient crash from the pass rate) is part of the protocol too: a baseline and a candidate that record
  different values are a regression reason unless ``allow_target_retries_change`` is true, a run that does not record it is a
  warning, and ``retry_share_warn`` warns when the candidate retried a larger share of its target calls than the baseline did.
  The 10 p.p. default: for an unchanged target that crashes on 5% of its calls the warning fires in 0.12% of comparisons at
  100 cases (1.5% at 45, 0% at 200); for a candidate that crashes on 10 / 20 / 50% of its calls in 11 / 89 / 100% (clean
  baseline: 55 / 100 / 100%), cycle4/loopback1/eval-scientist-gate/retry_share_warn_measure.py.
* Any difference between the evalsprint version of the baseline and of the candidate (patch included) is a WARNING;
  ``require_same_kit_version`` makes it a regression reason.
* ``tag_drop_warning_pp`` / ``tag_drop_min_cases`` only control a WARNING about a big drop in a non-critical tag
  (critical tags have their own strict rule); they never trip the gate.  The minimum is 20 paired cases: with 10 tags of
  10 cases the chance of at least one spurious tag warning in a comparison with no real change was 43% (5% noise),
  with 5 tags of 20 it is 3.3% (cycle-3 methodology critic, 03d_tag_warn.log).
"""

from __future__ import annotations

DEFAULT_GATE: dict = {
    "max_pass_rate_drop": "auto",  # "auto" | fraction | None; auto: net loss >= max(3, ceil(n/50)) cases AND one-sided McNemar p < alpha(n), see compare.auto_alpha
    "alpha": 0.05,  # one-sided significance level of a FIXED limit combined with require_significance (auto ignores it: alpha shrinks with n)
    "require_significance": False,  # only used when max_pass_rate_drop is a number
    "min_pass_rate": None,  # absolute floor on the candidate run, e.g. 0.8
    "critical_tags": [],  # any paired case in these tags flipping pass->fail is a regression
    "max_cost_increase_pct": None,  # e.g. 25 -> fail if mean cost per case grows > 25%
    "max_p95_latency_increase_pct": None,
    "max_error_rate": 0.05,  # candidate error_rate (per case AND per attempt) above this = unhealthy run (not a quality signal)
    "max_removed_cases": 0,  # baseline case ids missing from the candidate
    "max_changed_cases": 0,  # same id, different case_sha
    "min_judge_kappa": 0.6,  # compared with the LOWER bound of the calibration kappa CI
    "require_calibration": False,  # True: an uncalibrated / low-kappa LLM judge becomes a regression reason
    "allow_judge_changes": False,  # True: a different judges_sha (judge config, rubric, pass_rule, custom judge code) is a warning, not a regression
    "allow_repeat_change": False,  # True: a different repeat protocol (summary.repeat) between baseline and candidate is a warning, not a regression
    "allow_target_retries_change": False,  # True: a different target_retries between baseline and candidate is a warning, not a regression
    "retry_share_warn": 0.10,  # warn when the candidate retried this much more of its target calls (a share, 0-1) than the baseline (None: no warning)
    "require_same_kit_version": False,  # True: ANY different evalsprint version (patch included) between baseline and candidate is a regression reason (else a warning)
    "tag_drop_warning_pp": 20,  # warn when a non-critical tag's pass rate drops by at least this many p.p. (None: no warning)
    "tag_drop_min_cases": 20,  # ... and the tag has at least this many paired cases (fewer: a few flipped cases are chance)
}

GATE_KEYS: tuple[str, ...] = tuple(DEFAULT_GATE)
