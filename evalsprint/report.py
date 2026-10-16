"""Self-contained HTML quality report.

One file, inline CSS, no JavaScript and no external requests, so it can be
emailed, attached to a ticket or archived as a CI artifact. The page is ordered
for an executive reader: verdict and KPI tiles first, then the evidence
(quality by tag, stability, judges, errors, regressions, failing cases,
calibration, cost).

    render_report(run, comparison=None, calibration=None, baseline=None, monthly_requests=None) -> str
    write_report(path, run, comparison=None, calibration=None, baseline=None, monthly_requests=None) -> Path
    render_history(runs) -> str
    write_history(path, runs) -> Path

`render_history` is the page for the client's CTO: quality and cost over time, one row per run, with the points
where the judges changed marked (a pass rate before and after such a point is not comparable).

`comparison` is the dict returned by `compare.compare_runs`, `calibration` the
one returned by `calibration.calibrate` and `baseline` the baseline RunResult
(used to show the baseline output next to the candidate output for newly failing
cases); all are optional. `monthly_requests` turns on a monthly cost projection.

Reading rules that matter for a client:

* A case whose outcome is `error` (target crashed or a judge produced no
  verdict) is never shown as a quality failure. Errors get their own section,
  split by source, and are excluded from every pass rate.
* Runs written before the outcome field existed still render: the outcome is
  derived from `output.error`, the judge errors and `passed`.
* A run the summary flags `placeholder` (the scaffold's offline stub as target, or an LLM judge on the
  mock provider) says so in a banner above every number, and so does a placeholder baseline: its verdicts
  say nothing about the real product.
* A judge verdict decided without a model call because the output was over `max_output_chars`
  (`n_oversize`), target retries after a crash (`n_target_retries`) and a calibration made for other judges
  (`judges_sha` differs) are each said where the number they qualify is shown.
* The repeat protocol is `{"n", "per_tag"}` (a case runs the largest `per_tag` count of its tags, else `n`
  times); the older `{"n": 5, "tags": [...]}` is read as the same protocol. `tnr_reliable` is worded as what
  it is, a screening flag, not a guarantee. Any difference of evalsprint version between two runs is shown,
  a patch included.
* The header names the source file of the target (`target.source_sha`, 12 characters) when the run recorded
  one, and a resumed run whose fingerprint differed from the earlier one says which part differed and that
  nothing was reused (`resumed_from.fingerprint_mismatch`).
* The retry row of the header gives the configured `target_retries` (C22) next to the retry calls that were made
  (`n_target_retries`), and a run over part of the suite (`selection.limited`, C26) says so in its own row:
  it is no baseline for the whole suite.
* Every key added after the first version is read defensively (`dict.get`), so a
  report of an older run or comparison renders without those parts.

Every string that comes from the client's data (inputs, outputs, tags, model
reasoning, error messages) is HTML-escaped.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Sequence
from urllib.parse import quote

if TYPE_CHECKING:
    from .runner import RunResult

DASH = "—"
MINUS = "−"
UNTAGGED = "(untagged)"
OTHER_GROUP = "other"

MAX_FAILURES = 200  # failing cases rendered in full
MAX_ERRORS = 100  # errored cases listed in the errors section
MAX_UNSTABLE = 50  # unstable repeated cases listed in the stability section
MAX_DISAGREEMENTS = 50  # judge-vs-human disagreements rendered in full
MAX_ID_CHIPS = 100  # case ids listed in the regressions section
MAX_HISTORY_CHART_RUNS = 60  # runs drawn in the history charts (the table lists more)
MAX_HISTORY_ROWS = 500  # runs listed in the history table
TEXT_LIMIT = 3000  # characters of any single input/output shown
ERROR_TEXT_LIMIT = 400  # characters of an error message shown in the errors table
LOW_N = 10  # tags with fewer cases get a "small sample" badge
MIN_SEGMENT_N = 3  # tags with fewer cases are never called out as the weakest segment
SMALL_LABEL_SET = 30  # fewer human labels than this are flagged as a small calibration sample
DEFAULT_MAX_ERROR_RATE = 0.05  # used to flag an unhealthy run when the summary carries no gate
STALE_CALIBRATION_TEXT = "calibration was made for a different judge configuration"  # calibration.STALE_CALIBRATION_TEXT (a test keeps them equal)


# --------------------------------------------------------------------- formatting
def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Fraction -> percentage string ("83.3%"); None -> em dash."""
    return DASH if value is None else f"{value * 100:.{digits}f}%"


def fmt_ci(ci: Iterable[float] | None, sep: str = "–") -> str:
    """Confidence interval of fractions -> "72.1–91.4%"."""
    if not ci:
        return DASH
    lo, hi = ci
    if lo is None or hi is None:
        return DASH
    return f"{lo * 100:.1f}{sep}{hi * 100:.1f}%"


def fmt_money(value: float | None) -> str:
    """USD: $0.0123 below $1, $12.34 above. None and exactly zero render as a dash."""
    if not value:
        return DASH
    if abs(value) < 0.00005:
        return "<$0.0001"
    return f"${value:.4f}" if abs(value) < 1 else f"${value:,.2f}"


def fmt_latency(seconds: float | None) -> str:
    if seconds is None:
        return DASH
    if seconds < 0.001:
        return "<1 ms"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def fmt_int(value: int | float | None) -> str:
    return DASH if value is None else f"{value:,.0f}"


def _fmt_range(ci: Iterable[float] | None, digits: int = 2) -> str:
    if not ci:
        return DASH
    lo, hi = ci
    return DASH if lo is None or hi is None else f"{lo:.{digits}f} to {hi:.{digits}f}"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return DASH
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if moment.tzinfo is None:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n\n".join(value)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _verdict_word(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value).strip().lower() or DASH


def _pass_rule_text(rule: Any) -> str:
    if rule == "all" or rule is None:
        return "every judge must pass"
    if rule == "any":
        return "at least one judge must pass"
    names = [rule] if isinstance(rule, str) else list(rule)
    return f"{', '.join(names)} must pass"


def _num(value: Any) -> float | None:
    """A finite number or None (bools, strings, NaN and inf are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt_p(p: float | None) -> str:
    p = _num(p)
    if p is None:
        return DASH
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------ outcome bookkeeping
def _outcome(record: dict) -> str:
    """`pass`, `fail` or `error` for one result record.

    Records written by the current runner carry `outcome`. Older records do not:
    a target error or a judge error is an `error`, otherwise `passed` decides.
    """
    outcome = record.get("outcome")
    if outcome in ("pass", "fail", "error"):
        return outcome
    if (record.get("output") or {}).get("error"):
        return "error"
    if any(j.get("error") for j in record.get("judges") or []):
        return "error"
    return "pass" if record.get("passed") else "fail"


def _error_source(record: dict) -> str:
    """`target` or `judge` for an errored record."""
    source = record.get("error_source")
    if source in ("target", "judge"):
        return source
    return "target" if (record.get("output") or {}).get("error") else "judge"


def _error_message(record: dict) -> str:
    if _error_source(record) == "target":
        return str((record.get("output") or {}).get("error") or "unknown error")
    parts = [f"{j.get('judge')}: {j.get('error')}" for j in record.get("judges") or [] if j.get("error")]
    return "; ".join(parts) or "unknown error"


def _counts(run: RunResult) -> dict:
    """Case counts by outcome. The summary wins when it has the key, the records fill the gaps."""
    s = run.summary
    outcomes = Counter(_outcome(r) for r in run.results)
    sources = Counter(_error_source(r) for r in run.results if _outcome(r) == "error")

    def pick(key: str, fallback: int) -> int:
        value = _num(s.get(key))
        return int(value) if value is not None else fallback

    n_cases = pick("n_cases", len(run.results))
    n_passed = pick("n_passed", outcomes["pass"])
    n_failed = pick("n_failed", outcomes["fail"])
    n_target = pick("n_target_errors", sources["target"])
    n_judge = pick("n_judge_errors", sources["judge"])
    n_errors = pick("n_errors", n_target + n_judge)
    n_scored = pick("n_scored", n_passed + n_failed)
    return {
        "cases": n_cases,
        "passed": n_passed,
        "failed": n_failed,
        "scored": n_scored,
        "errors": n_errors,
        "target_errors": n_target,
        "judge_errors": n_judge,
        "error_rate": _num(s.get("error_rate")) if _num(s.get("error_rate")) is not None else (n_errors / n_cases if n_cases else 0.0),
    }


def _stability(run: RunResult) -> dict | None:
    """`{"n_repeated", "flip_rate"}` when the run carries the stability block, else None."""
    block = run.summary.get("stability")
    if not isinstance(block, dict):
        return None
    return {"n_repeated": int(_num(block.get("n_repeated")) or 0), "flip_rate": _num(block.get("flip_rate"))}


def _repeat_protocol(spec: Any) -> tuple[int, dict[str, int]] | None:
    """`(n, per_tag)` of a `repeat` setting as `summary["repeat"]` writes it (C17), read defensively; None when absent or malformed.

    A case runs `max(per_tag[t] for its tags)` times, else `n` times. The cycle-3 spelling `{"n": 5, "tags": ["safety"]}` is the same
    protocol as `{"n": 1, "per_tag": {"safety": 5}}` and is read as that; a `per_tag` entry equal to `n` changes nothing and is dropped,
    so two spellings of one protocol compare equal (the way compare reads them).
    """
    if isinstance(spec, int) and not isinstance(spec, bool):
        spec = {"n": spec}
    if not isinstance(spec, dict):
        return None
    n = _num(spec.get("n", 1))
    if n is None or n < 1:
        return None
    n = int(n)
    raw = spec.get("per_tag")
    per_tag: dict[str, int] = {}
    if isinstance(raw, dict):
        for tag, attempts in raw.items():
            count = _num(attempts)
            if count is None or count < 1:
                return None
            per_tag[str(tag)] = int(count)
    elif raw is None:
        tags = spec.get("tags")  # legacy: n attempts on the listed tags only, every other case once
        if n > 1 and isinstance(tags, (list, tuple)) and tags:
            per_tag = {str(t): n for t in tags}
            n = 1
    else:
        return None
    return n, {t: k for t, k in sorted(per_tag.items()) if k != n}


def _protocol_phrase(protocol: tuple[int, dict[str, int]] | None) -> str:
    """The repeat protocol in words: `safety 5 attempts, every other case 3 attempts`."""
    if protocol is None:
        return "unknown"
    n, per_tag = protocol
    rest = "once" if n <= 1 else f"{n} attempts"
    if not per_tag:
        return f"every case {rest}"
    tagged = ", ".join(f"{t} {k} attempts" for t, k in per_tag.items())
    return f"{tagged}, every other case {rest}"


def _run_protocol(run: RunResult) -> tuple[int, dict[str, int]] | None:
    return _repeat_protocol(run.summary.get("repeat"))


def _attempts_per_case(run: RunResult) -> Counter:
    """How many repeated cases ran each number of attempts: `Counter({5: 10, 3: 70})`, from the records."""
    return Counter(len(_attempts(r)) for r in run.results if len(_attempts(r)) > 1)


def _attempts(record: dict) -> list[dict]:
    attempts = record.get("attempts")
    return [a for a in attempts if isinstance(a, dict)] if isinstance(attempts, list) else []


def _attempt_counts(run: RunResult) -> dict | None:
    """Errors counted per attempt (C10), not per case: `{"attempts", "errors", "target", "judge", "rate"}` or None.

    The summary wins when it has the counters; a run written before they existed is counted from its records
    (a case without `attempts` ran once). A crash in one attempt of a repeated case leaves the majority verdict
    untouched, so it is invisible in the case-level error count: this is where it shows.
    """
    s = run.summary
    total, errors = _num(s.get("n_attempts")), _num(s.get("n_attempt_errors"))
    target, judge = _num(s.get("n_attempt_target_errors")), _num(s.get("n_attempt_judge_errors"))
    if total is None or errors is None:
        total = errors = target = judge = 0
        for record in run.results:
            attempts = _attempts(record)
            if attempts:
                total += len(attempts)
                for a in attempts:
                    if a.get("outcome") == "error":
                        errors += 1
                        target += (a.get("error_source") or "judge") == "target"
                        judge += (a.get("error_source") or "judge") != "target"
            else:
                total += 1
                if _outcome(record) == "error":
                    errors += 1
                    target += _error_source(record) == "target"
                    judge += _error_source(record) != "target"
        rate = errors / total if total else None
    else:
        rate = _num(s.get("attempt_error_rate"))
        if rate is None and total:
            rate = errors / total
    if not total:
        return None
    return {"attempts": int(total), "errors": int(errors), "target": int(target or 0), "judge": int(judge or 0), "rate": rate}


def _errored_attempts(record: dict) -> list[dict]:
    return [a for a in _attempts(record) if a.get("outcome") == "error"]


def _is_simulated(run: RunResult) -> bool:
    return any(((r.get("output") or {}).get("meta") or {}).get("simulated") for r in run.results)


def _is_placeholder(summary: Any) -> bool:
    """`summary.placeholder` (C18): the scaffold's offline stub as target, or an LLM judge on the mock provider. Only a literal true counts."""
    return isinstance(summary, dict) and summary.get("placeholder") is True


def _target_retries(run: RunResult) -> int | None:
    """Target calls that were retried after a crash (C18): `summary.n_target_retries`, else the records' `target_retries_used`; None when neither exists."""
    value = _num(run.summary.get("n_target_retries"))
    if value is not None:
        return max(int(value), 0)
    used = [r for r in run.results if _num(r.get("target_retries_used")) is not None]
    if not used:
        return None
    return sum(max(int(_num(r.get("target_retries_used")) or 0), 0) for r in used)


def _configured_retries(summary: dict) -> int | None:
    """`summary.target_retries` (C22): the configured number of times a failed target call is repeated inside the run; None when it is not a number (an older run)."""
    value = _num(summary.get("target_retries"))
    return int(value) if value is not None and value >= 0 and value == int(value) else None


MAX_SELECTION_NAMES = 5


def _selection_html(summary: dict) -> str | None:
    """The "Selection" row of the header (C26): said only when `summary.selection.limited` is literally true, i.e. the run covers fewer cases than the config selects.

    Every string in it (tags, case ids) comes from the run's data and is escaped."""
    selection = summary.get("selection")
    if not isinstance(selection, dict) or selection.get("limited") is not True:
        return None
    chosen, config = _num(selection.get("n_selected")), _num(selection.get("n_config"))
    covers = f"{fmt_int(chosen)} of the {fmt_int(config)} cases of the suite" if chosen is not None and config is not None else "only part of the cases of the suite"
    how = []
    limit = _num(selection.get("limit"))
    if limit is not None and limit >= 1:
        how.append(f"limit {int(limit)}")
    for key, label in (("tags", "tags"), ("ids", "case ids")):
        names = [str(v) for v in selection[key] if isinstance(v, (str, int)) and not isinstance(v, bool)] if isinstance(selection.get(key), list) else []
        if names:
            shown = ", ".join(names[:MAX_SELECTION_NAMES]) + (f" and {len(names) - MAX_SELECTION_NAMES} more" if len(names) > MAX_SELECTION_NAMES else "")
            how.append(f"{label}: {shown}")
    detail = f" ({_e('; '.join(how))})" if how else ""
    return f'{_badge("limited run", "warn")} covers {covers}{detail}: it is not a baseline for the whole suite'


def _judge_results(record: dict) -> list[dict]:
    judges = record.get("judges")
    return [j for j in judges if isinstance(j, dict)] if isinstance(judges, list) else []


def _n_oversize(run: RunResult) -> int:
    """Judge results decided without calling the model because the output was longer than `max_output_chars` (C18).

    `summary.n_oversize` wins; a run written before it existed is counted from the records' `details.oversize`.
    """
    value = _num(run.summary.get("n_oversize"))
    if value is not None:
        return max(int(value), 0)
    return sum(1 for r in run.results for j in _judge_results(r) if isinstance(j.get("details"), dict) and j["details"].get("oversize"))


def _stale_calibration_text(calibration: Any, run: RunResult | None) -> str | None:
    """The warning to show when `calibration.judges_sha` differs from the run's (C20), else None.

    The text is the one `calibration.stale_calibration` writes, so the report, compare and `calibrate` say the same thing; a missing
    `judges_sha` on either side says nothing (a calibration or a run written by an older evalsprint).
    """
    if run is None or not isinstance(calibration, dict):
        return None
    made_for, current = calibration.get("judges_sha"), run.summary.get("judges_sha")
    if not (isinstance(made_for, str) and made_for and isinstance(current, str) and current) or made_for == current:
        return None
    try:
        from .calibration import stale_calibration

        text = stale_calibration(calibration, run)
        if text:
            return text.replace(made_for, made_for[:12]).replace(current, current[:12])  # 12 characters, like the fingerprint in the header
    except Exception:  # noqa: BLE001 - the report must render even when the calibration module cannot be used
        pass
    return (
        f"This {STALE_CALIBRATION_TEXT} (judges_sha {made_for[:12]} in the calibration, {current[:12]} in this run): the agreement, kappa and "
        "TPR/TNR it reports describe the earlier judges. Calibrate this run again."
    )


def _max_error_rate(run: RunResult, comparison: dict | None) -> float:
    for gate in ((comparison or {}).get("gate"), run.summary.get("gate")):
        value = _num((gate or {}).get("max_error_rate")) if isinstance(gate, dict) else None
        if value is not None:
            return value
    return DEFAULT_MAX_ERROR_RATE


# ------------------------------------------------------------------ html building blocks
def _delta(value: float | None, text: str, higher_is_better: bool, compact: bool = False) -> str:
    """Sign-coloured delta that is also worded: arrow, signed number and a verdict word."""
    if value is None:
        return ""
    if round(value, 1) == 0:
        arrow, tone, word = "●", "flat", "no change"
    else:
        good = (value > 0) == higher_is_better
        arrow, tone, word = ("▲" if value > 0 else "▼"), ("good" if good else "bad"), ("better" if good else "worse")
    note = word if compact else f"vs baseline · {word}"
    return f'<span class="delta {tone}"><span aria-hidden="true">{arrow}</span> {_e(text)} <span class="delta-note">{note}</span></span>'


def _signed(value: float, suffix: str) -> str:
    rounded = round(value, 1)
    if rounded == 0:
        return f"0.0{suffix}"
    return f"{'+' if rounded > 0 else MINUS}{abs(rounded):.1f}{suffix}"


def _delta_pp(fraction: float | None, higher_is_better: bool, compact: bool = False) -> str:
    """Delta between two rates given as a fraction, shown in percentage points."""
    if fraction is None:
        return ""
    return _delta(fraction * 100, _signed(fraction * 100, " p.p."), higher_is_better, compact)


def _delta_pct(percent: float | None, higher_is_better: bool) -> str:
    """Relative change given in percent units (12.5 means +12.5%)."""
    if percent is None:
        return ""
    return _delta(percent, _signed(percent, "%"), higher_is_better)


def _tile(label: str, value: str, subs: Iterable[str] = (), delta: str = "") -> str:
    sub_html = "".join(f'<div class="tile-sub">{_e(s)}</div>' for s in subs if s)
    return (
        f'<div class="tile"><div class="tile-label">{_e(label)}</div>'
        f'<div class="tile-value">{_e(value)}</div>{sub_html}{delta}</div>'
    )


def _badge(text: str, tone: str = "") -> str:
    return f'<span class="badge {tone}">{_e(text)}</span>'


def _chips(values: Iterable[str]) -> str:
    return "".join(f'<span class="chip">{_e(v)}</span>' for v in values)


def _table(label: str, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
    """`headers` are (text, css class); `rows` hold ready-made (already escaped) cell HTML."""
    def attr(cls: str) -> str:
        return f' class="{cls}"' if cls else ""

    head = "".join(f'<th scope="col"{attr(cls)}>{_e(text)}</th>' for text, cls in headers)
    body = "".join(
        "<tr>" + "".join(f"<td{attr(cls)}>{cell}</td>" for cell, (_, cls) in zip(row, headers)) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-wrap" role="region" aria-label="{_e(label)}" tabindex="0">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _bar(rate: float | None, ci: Sequence[float] | None = None) -> str:
    """Horizontal bar drawn with div widths; the whisker marks the confidence interval."""
    if rate is None:
        return ""
    whisker = ""
    if ci and None not in ci:
        lo, hi = ci
        whisker = f'<span class="bar-ci" style="left:{lo * 100:.1f}%;width:{(hi - lo) * 100:.1f}%"></span>'
    return f'<div class="bar" aria-hidden="true"><span class="bar-fill" style="width:{rate * 100:.1f}%"></span>{whisker}</div>'


def _clip(text: str) -> tuple[str, int]:
    return (text, 0) if len(text) <= TEXT_LIMIT else (text[:TEXT_LIMIT], len(text) - TEXT_LIMIT)


def _pre(value: Any, empty: str = "(empty)") -> str:
    text = _as_text(value)
    if not text.strip():
        return f'<p class="muted">{_e(empty)}</p>'
    shown, hidden = _clip(text)
    more = f'<p class="muted">… {hidden:,} more characters not shown (full text in results.jsonl)</p>' if hidden else ""
    return f"<pre>{_e(shown)}</pre>{more}"


def _case_dom_id(case_id: str) -> str:
    return "case-" + "_".join(str(case_id).split())


def _case_link(case_id: str) -> str:
    return f'<a href="#{_e(quote(_case_dom_id(case_id), safe="-._~:"))}">{_e(case_id)}</a>'


def _section(anchor: str, title: str, body: str) -> tuple[str, str, str]:
    return anchor, title, f'<section id="{anchor}"><h2>{_e(title)}</h2>{body}</section>'


# -------------------------------------------------------------------------- header
def _param_chips(params: dict, limit: int = 8) -> str:
    def rank(key: str) -> int:
        return 0 if key == "model" else 1 if key.startswith("prompt") else 2

    items = sorted(params.items(), key=lambda kv: rank(str(kv[0]))) if isinstance(params, dict) else []
    chips = []
    for key, value in items[:limit]:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        text = text if len(text) <= 60 else text[:57] + "..."
        chips.append(f'<span class="chip"><span class="chip-key">{_e(key)}</span> {_e(text)}</span>')
    if len(items) > limit:
        chips.append(f'<span class="muted">+{len(items) - limit} more</span>')
    return "".join(chips) or DASH


# The parts of a run fingerprint that `summary.resumed_from.fingerprint_mismatch` can name (C20), in words a reader can act on.
_MISMATCH_LABELS = {
    "judges_sha": "judges (judges_sha)",
    "target": "target callable or parameters (target)",
    "target_source": "target source file (target_source)",
    "repeat": "repeat protocol (repeat)",
}


def _target_source(summary: dict) -> str | None:
    """`summary.target.source_sha` (C20): the first characters of the sha256 of the file behind the target callable; None when absent."""
    target = summary.get("target")
    sha = target.get("source_sha") if isinstance(target, dict) else None
    return sha[:12] if isinstance(sha, str) and sha.strip() else None


def _resumed_text(run: RunResult) -> str | None:
    """HTML for `summary["resumed_from"]` ({"run_id", "reused_records", "reused_cost_usd", ...}); a bare run id also works."""
    resumed = run.summary.get("resumed_from")
    if isinstance(resumed, str) and resumed:
        return f"<code>{_e(resumed)}</code>"
    if not isinstance(resumed, dict):
        return None
    run_id = resumed.get("run_id")
    text = f"<code>{_e(run_id)}</code>" if run_id else "an earlier run"
    mismatch = resumed.get("fingerprint_mismatch")
    if mismatch:
        parts = [str(m) for m in mismatch] if isinstance(mismatch, (list, tuple)) else [str(mismatch)]
        keys = ", ".join(_MISMATCH_LABELS.get(part, part) for part in parts)
        edited = " (the source file of the target callable was edited)" if "target_source" in parts else ""
        verb = "differs" if len(parts) == 1 else "differ"
        return f"{text}: nothing was reused, because its {_e(keys)} {verb} from this run's{_e(edited)}, so every case ran again"
    reused = _num(resumed.get("reused_records"))
    if reused is not None:
        text += f": {int(reused)} record(s) reused without calling the target or the judges again"
    cost = _num(resumed.get("reused_cost_usd"))
    if cost:
        text += f" ({fmt_money(cost)} of their cost counted as already spent)"
    retried = _num(resumed.get("retried_errors"))
    if retried:
        text += f"; {int(retried)} errored record(s) were run again"
    return text


def _header(run: RunResult) -> str:
    s = run.summary
    target = s.get("target") if isinstance(s.get("target"), dict) else {}
    label = s.get("label")
    run_line = (f"<strong>{_e(label)}</strong> · " if label else "") + f"<code>{_e(s.get('run_id'))}</code> · {_e(_fmt_date(s.get('created_at')))}"
    judges = s.get("judges") or []
    dataset = Path(s["dataset"]).name if s.get("dataset") else None
    counts = _counts(run)
    test_set = f"{fmt_int(counts['cases'])} cases"
    if counts["errors"]:
        test_set += f" ({fmt_int(counts['scored'])} scored)"
    if dataset:
        test_set += f" · <code>{_e(dataset)}</code>"
    rows = [
        ("Target", f"<code>{_e(target.get('callable'))}</code>"),
        ("Parameters", _param_chips(target.get("params") or {})),
        ("Test set", test_set),
        ("Judges", _chips(judges) or DASH),
        ("Pass rule", _e(_pass_rule_text(s.get("pass_rule")))),
    ]
    selection = _selection_html(s)
    if selection:
        rows.insert(3, ("Selection", selection))  # right under the test set it qualifies
    stability = _stability(run)
    protocol = _run_protocol(run)
    if stability and stability["n_repeated"]:
        per_case = _attempts_per_case(run)
        if protocol is not None and protocol[1]:  # a protocol with per-tag counts: attempts differ from case to case
            how = f" ({_e(_protocol_phrase(protocol))})"
        elif len(per_case) > 1:  # no protocol recorded but the records show cases with different counts
            how = " (" + ", ".join(f"{k} attempts: {v} case(s)" for k, v in sorted(per_case.items(), reverse=True)) + ")"
        else:
            attempts = protocol[0] if protocol is not None and protocol[0] > 1 else next(iter(per_case), 0)
            how = f" × {attempts} attempts" if attempts > 1 else ""
        rows.append(("Repeats", f"{fmt_int(stability['n_repeated'])} repeated cases{how}, majority outcome"))
    elif protocol is not None and (protocol[0] > 1 or protocol[1]):
        rows.append(("Repeats", f"no case ran more than once · protocol: {_e(_protocol_phrase(protocol))}"))
    retries, configured = _target_retries(run), _configured_retries(s)
    if configured is not None:
        made = (
            f"{fmt_int(retries)} retry call(s) made after a target crash (paid, and counted in the cost)" if retries
            else "no retry call was made" if retries == 0
            else "retry calls made: not recorded"
        )
        rows.append(("Target retries", f"configured <code>target_retries: {fmt_int(configured)}</code> · {made}"))
    elif retries:
        rows.append(("Target retries", f"{fmt_int(retries)} retry call(s) after a target crash, made within the run (paid, and counted in the cost)"))
    version = s.get("kit_version") or s.get("evalsprint_version")
    if version:
        rows.append(("Kit version", f"evalsprint <code>{_e(version)}</code>"))
    if isinstance(s.get("judges_sha"), str) and s["judges_sha"]:
        rows.append(("Judges fingerprint", f"<code>{_e(s['judges_sha'][:12])}</code>"))
    source = _target_source(s)
    if source:
        rows.append(("Target source", f'<code>{_e(source)}</code> <span class="muted">(sha256 of the file behind the callable)</span>'))
    resumed = _resumed_text(run)
    if resumed:
        rows.append(("Resumed from", resumed))
    meta = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in rows)
    return (
        '<header><p class="eyebrow">Eval Sprint · Quality report</p>'
        f"<h1>{_e(s.get('suite') or 'Evaluation run')}</h1><p class=\"run-line\">{run_line}</p>"
        f'<dl class="meta">{meta}</dl></header>'
    )


def _simulated_banner(run: RunResult) -> str:
    """Shown when the target says its outputs come from a simulator, so a screenshot cannot pass for a benchmark."""
    if not _is_simulated(run):
        return ""
    return (
        '<div class="sim-banner" role="note"><strong>Simulated run.</strong> The outputs in this report come from a '
        "simulator (fixed rules and hash-drawn dice), not from a language model. The numbers show that the evaluation pipeline works; they say nothing about how any real model performs.</div>"
    )


def _placeholder_banner(run: RunResult) -> str:
    """`summary.placeholder` (C18): the verdicts of this run say nothing about the real product, whatever the numbers look like."""
    if not _is_placeholder(run.summary):
        return ""
    return (
        '<div class="sim-banner placeholder-banner" role="note"><strong>Placeholder run.</strong> Its target is the scaffold\'s offline stub, or one of its '
        "LLM judges runs on the mock provider, so its verdicts say nothing about the real product. Do not record it as a baseline, and run it again with "
        "the real target and judge before you read any number here as quality.</div>"
    )


_PLACEHOLDER_BASELINE_TEXT = (
    "the baseline run is a placeholder (simulated target or mock judge: its verdicts say nothing about the real product), "
    "so this comparison is meaningless; record a baseline from a real run and commit that one"
)


def _placeholder_baseline_warning(comparison: dict | None) -> str | None:
    """compare's warning that the baseline is a placeholder (C18), when it wrote one."""
    for w in (comparison or {}).get("warnings") or []:
        if isinstance(w, str) and "placeholder" in w.lower() and "baseline" in w.lower():
            return w
    return None


def _baseline_placeholder_banner(comparison: dict | None, baseline: RunResult | None) -> str:
    """Shown above the verdict when the baseline is a placeholder: compare's own warning wins, else the baseline run's `placeholder` flag."""
    warning = _placeholder_baseline_warning(comparison)
    if warning is None and not (baseline is not None and _is_placeholder(baseline.summary)):
        return ""
    text = warning or _PLACEHOLDER_BASELINE_TEXT
    return f'<div class="health bad placeholder-banner" role="alert"><strong>Placeholder baseline.</strong> {_e(text[:1].upper() + text[1:])}</div>'


def _oversize_banner(run: RunResult) -> str:
    """Judge results that failed without a model call because the output was longer than `max_output_chars` (C18)."""
    n = _n_oversize(run)
    if not n:
        return ""
    return (
        f'<div class="health oversize-note" role="note"><strong>{n} judge verdict(s) are automatic fails</strong> because the output was longer than the judge\'s '
        "<code>max_output_chars</code>: the judge was not called (cost $0) and never read it. A runaway output may be a real defect, but this verdict is a size rule, not a "
        "reading of the output; raise the limit if such outputs are legitimate.</div>"
    )


# ------------------------------------------------------------------ verdict banner
def _pp(v: Any) -> str:
    n = _num(v)
    return DASH if n is None else f"{n * 100:.1f} p.p."


def _gate_drop(v: Any) -> str:
    return "automatic, by test-set size" if v == "auto" else _pp(v)


def _pct_or_dash(v: Any) -> str:
    return fmt_pct(_num(v))


def _g(v: Any) -> str:
    n = _num(v)
    return str(v) if n is None else f"{n:g}"


_GATE_LABELS = {
    "max_pass_rate_drop": ("Max pass-rate drop", _gate_drop),
    "require_significance": ("Drop must be significant", lambda v: "yes" if v else "no"),
    "alpha": ("Significance level (alpha, one-sided)", _g),
    "min_pass_rate": ("Min pass rate", _pct_or_dash),
    "critical_tags": ("Critical tags", lambda v: ", ".join(map(str, v)) or "none"),
    "max_cost_increase_pct": ("Max cost increase", lambda v: f"{_g(v)}%"),
    "max_p95_latency_increase_pct": ("Max p95 latency increase", lambda v: f"{_g(v)}%"),
    "max_error_rate": ("Max error rate", _pct_or_dash),
    "max_removed_cases": ("Max removed cases", _g),
    "max_changed_cases": ("Max changed cases", _g),
    "min_judge_kappa": ("Min judge kappa (interval lower bound)", _g),
    "require_calibration": ("Calibration required", lambda v: "yes" if v else "no"),
    "allow_judge_changes": ("Judge changes allowed", lambda v: "yes" if v else "no"),
    "allow_repeat_change": ("Repeat-protocol change allowed", lambda v: "yes" if v else "no"),
    "require_same_kit_version": ("Same evalsprint version required", lambda v: "yes" if v else "no"),
    "tag_drop_warning_pp": ("Warn when a tag drops by (p.p.)", _g),
    "tag_drop_min_cases": ("Cases a tag needs for that warning", _g),
}

_GATE_MODES = {  # comparison files written by evalsprint 0.1 say "auto-small" / "auto-large": both still render
    "auto-small": "automatic, small test set: a drop above {eff} fails, no significance test (too few cases to have the power)",
    "auto-large": "automatic, large test set: a drop above {eff} fails when a one-sided McNemar test also says it is unlikely to be noise",
    "fixed": "fixed limit: a drop above {eff} fails",
    "off": "off: the pass-rate drop is not checked",
}


def _alpha_text(value: Any) -> str | None:
    alpha = _num(value)
    return None if alpha is None else f"{alpha:.3g}"


def _gate_mode_text(comparison: dict) -> str:
    mode = comparison.get("gate_mode")
    if not mode:
        return ""
    eff = _pp(comparison.get("effective_max_pass_rate_drop"))
    if mode == "auto":  # one rule for every n: a net-loss floor AND a one-sided McNemar test at an alpha that depends on n
        k = _num(comparison.get("effective_min_net_loss"))
        alpha = _alpha_text(comparison.get("effective_alpha"))
        floor = f"a net loss of at least {int(k)} case{'' if int(k) == 1 else 's'} (a drop above {eff})" if k is not None and k >= 1 else f"a drop above {eff}"
        test = f" and a one-sided McNemar test at alpha {alpha} says it is unlikely to be noise" if alpha else ""
        return f"automatic, one rule for every test-set size: fails on {floor}{test}"
    template = _GATE_MODES.get(str(mode))
    return template.format(eff=eff) if template else f"{mode} (limit {eff})"


def _gate_details(comparison: dict) -> str:
    gate = comparison.get("gate") or {}
    rows = []
    for key, (label, fmt) in _GATE_LABELS.items():
        if key in gate:
            value = gate[key]
            try:
                shown = "not set" if value is None else fmt(value)
            except (TypeError, ValueError):
                shown = str(value)
            rows.append(f"<div><dt>{_e(label)}</dt><dd>{_e(shown)}</dd></div>")
    mode_text = _gate_mode_text(comparison)
    if mode_text:
        rows.insert(0, f"<div><dt>Mode for this comparison</dt><dd>{_e(mode_text)}</dd></div>")
    if not rows:
        return ""
    return f'<details class="gate"><summary>Gate thresholds</summary><dl class="meta">{"".join(rows)}</dl></details>'


def _error_fact(comparison: dict) -> str | None:
    errors = comparison.get("errors")
    if not isinstance(errors, dict):
        return None
    base, cand = errors.get("baseline") or {}, errors.get("candidate") or {}
    n_base, n_cand = int(_num(base.get("n_errors")) or 0), int(_num(cand.get("n_errors")) or 0)
    if not (n_base or n_cand):
        return None
    detail = f"{int(_num(cand.get('n_target_errors')) or 0)} target, {int(_num(cand.get('n_judge_errors')) or 0)} judge"
    return f"Errors (not counted as failures): baseline {n_base}, candidate {n_cand} ({detail})"


def _attempt_error_fact(comparison: dict) -> str | None:
    """Errors counted per attempt (C10): a crash in one attempt of a repeated case is invisible in the case-level count."""
    errors = comparison.get("errors")
    if not isinstance(errors, dict):
        return None
    base, cand = errors.get("baseline") or {}, errors.get("candidate") or {}
    n_base, n_cand = _num(base.get("n_attempt_errors")), _num(cand.get("n_attempt_errors"))
    if not ((n_base or 0) or (n_cand or 0)):
        return None
    parts = []
    for side, block, n_err in (("baseline", base, n_base), ("candidate", cand, n_cand)):
        total, rate = _num(block.get("n_attempts")), _num(block.get("attempt_error_rate"))
        if n_err is None:
            continue
        parts.append(f"{side} {int(n_err)}" + (f" of {int(total)} attempts" if total is not None else "") + (f" ({fmt_pct(rate)})" if rate is not None else ""))
    return "Attempt errors (not counted as failures): " + ", ".join(parts) if parts else None


def _repeat_text(protocol: Any) -> str:
    """A protocol as compare writes it (`{"n", "per_tag"}`, or the cycle-3 `{"n", "tags"}`) or as a summary's `repeat` holds it, in words."""
    return _protocol_phrase(_repeat_protocol(protocol)) if isinstance(protocol, dict) else "unknown"


def _kit_release(version: Any) -> tuple | str | None:
    """A comparable key for a version: `v0.2.0`, `0.2.0` and `0.2` are one release, `0.2.1` is another; anything else by its text."""
    if not isinstance(version, str) or not version.strip():
        return None
    text = re.sub(r"^v(?=\d)", "", version.strip(), flags=re.IGNORECASE)
    if not re.fullmatch(r"\d+(\.\d+)*", text):
        return text
    parts = [int(x) for x in text.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _protocol_facts(comparison: dict) -> list[str]:
    """Facts (HTML) about the evaluation protocol of the two runs: `repeat_changed` and `kit_version`."""
    facts = []
    repeat_changed = comparison.get("repeat_changed")
    protocol = comparison.get("repeat_protocol") if isinstance(comparison.get("repeat_protocol"), dict) else {}
    if repeat_changed is True:
        detail = f" (baseline: {_e(_repeat_text(protocol.get('baseline')))}; candidate: {_e(_repeat_text(protocol.get('candidate')))})" if protocol else ""
        facts.append(f"<strong>Evaluation protocol changed</strong>{detail}: pass rates and the gate's false-alarm rate are only comparable under the same repeat setting")
    elif repeat_changed is False:
        facts.append(f"Same repeat protocol as the baseline ({_e(_repeat_text(protocol.get('candidate')))})" if protocol.get("candidate") else "Same repeat protocol as the baseline")
    kit = comparison.get("kit_version")
    if isinstance(kit, dict):
        base, cand = kit.get("baseline"), kit.get("candidate")
        rel_base, rel_cand = _kit_release(base), _kit_release(cand)
        # the versions themselves are the evidence: a comparison written before C20 says "unchanged" for a patch release
        different = kit.get("changed") is True if rel_base is None or rel_cand is None else rel_base != rel_cand
        if different:
            if _release_series(base) is not None and _release_series(base) == _release_series(cand):
                said = any(isinstance(w, str) and "patch version" in w for w in comparison.get("warnings") or [])  # compare's own warning explains it below
                facts.append(
                    f"<strong>evalsprint patch version differs</strong> (baseline {_e(base)}, candidate {_e(cand)})"
                    + ("" if said else ": scoring is expected to be the same, but a patch release can fix how a case is scored, so check the pass rates if they look odd")
                )
            else:
                facts.append(f"<strong>evalsprint version differs</strong> (baseline {_e(base)}, candidate {_e(cand)}): scoring may differ, so the pass rates may not be comparable")
        elif cand and base and (kit.get("changed") is False or rel_base is not None):
            facts.append(f"evalsprint {_e(cand)} on both runs" if base == cand else f"evalsprint {_e(base)} and {_e(cand)} (the same release)")
    return facts


def _calibration_status_fact(comparison: dict) -> str | None:
    status = comparison.get("calibration_status")
    if not isinstance(status, dict):
        return None
    kappa = _num(status.get("kappa"))
    ci = status.get("kappa_ci")
    basis = {"weighted": "weighted ", "legacy": "unweighted "}.get(status.get("basis"), "")
    text = f"Judge {_e(status.get('judge') or 'quality')} calibration: {basis}κ " + (DASH if kappa is None else f"{kappa:.2f}")
    if ci and None not in ci:
        text += f" (95% CI {_fmt_range(ci)})"
    ok = status.get("ok")
    return text + (" meets the gate minimum" if ok else " is below the gate minimum" if ok is False else "")


def _verdict_banner(comparison: dict, placeholder_banner_shown: bool = False) -> str:
    regression = comparison.get("verdict") == "regression"
    pr = comparison.get("pass_rate") or {}
    tone, icon, title = ("bad", "✖", "Regression detected") if regression else ("ok", "✔", "No regression")
    reasons = comparison.get("reasons") or []
    reasons_html = f'<ul>{"".join(f"<li>{_e(r)}</li>" for r in reasons)}</ul>' if reasons else ""

    facts = [
        f"Candidate <code>{_e(comparison.get('candidate_run_id'))}</code> vs baseline <code>{_e(comparison.get('baseline_run_id'))}</code>",
        f"{fmt_int(comparison.get('n_paired'))} paired cases",
        f"{fmt_int(len(comparison.get('newly_failing') or []))} newly failing",
        f"{fmt_int(len(comparison.get('newly_passing') or []))} newly passing",
    ]
    p_value = _num(pr.get("p_value"))
    if p_value is not None:
        one_sided = "p_value_two_sided" in pr  # from the second contract on, p_value is the one-sided test that the candidate is worse
        facts.append("McNemar " + ("one-sided " if one_sided else "") + f"p {'' if p_value < 0.001 else '= '}{_fmt_p(p_value)}")
    mode_text = _gate_mode_text(comparison)
    if mode_text:
        alpha = _alpha_text(comparison.get("effective_alpha"))
        facts.append(f"Gate mode: {_e(comparison.get('gate_mode'))}" + (f" (alpha {_e(alpha)})" if alpha and comparison.get("gate_mode") == "auto" else ""))
    error_fact = _error_fact(comparison)
    if error_fact:
        facts.append(_e(error_fact))
    attempt_fact = _attempt_error_fact(comparison)
    if attempt_fact:
        facts.append(_e(attempt_fact))
    calibration_fact = _calibration_status_fact(comparison)
    if calibration_fact:
        facts.append(calibration_fact)
    judges_changed = comparison.get("judges_changed")
    if judges_changed is True:
        facts.append("<strong>Judge configuration or rubric changed</strong> since the baseline: pass rates may not be comparable")
    elif judges_changed is False:
        facts.append("Judges unchanged since the baseline")
    facts.extend(_protocol_facts(comparison))
    if placeholder_banner_shown:
        facts.append("<strong>Baseline is a placeholder run</strong>: this verdict says nothing about the real product")
    unpaired = comparison.get("unpaired") or {}
    n_unpaired = len(unpaired.get("only_baseline") or []) + len(unpaired.get("only_candidate") or [])
    notes = list(comparison.get("warnings") or [])
    if placeholder_banner_shown:  # the banner above the verdict already carries this warning, word for word
        notes = [w for w in notes if w != _placeholder_baseline_warning(comparison)]
    if n_unpaired:
        notes.append(f"{n_unpaired} case(s) exist in only one of the two runs and were not compared.")
    removed, changed = comparison.get("removed_cases") or [], comparison.get("changed_cases") or []
    if removed:
        notes.append(f"{len(removed)} baseline case(s) are missing from this run (see Regressions).")
    if changed:
        notes.append(f"{len(changed)} case(s) were edited since the baseline and were not compared (see Regressions).")
    warning = f'<ul class="verdict-notes">{"".join(f"<li>{_e(w)}</li>" for w in notes)}</ul>' if notes else ""
    return (
        f'<div class="verdict {tone}" role="status"><div class="verdict-title"><span aria-hidden="true">{icon}</span> {title}</div>'
        f'{reasons_html}<p class="verdict-note">{" · ".join(facts)}</p>{warning}{_gate_details(comparison)}</div>'
    )


def _health_banner(run: RunResult, comparison: dict | None) -> str:
    """An unhealthy run is a measurement problem, not a quality result: say it before any number."""
    counts = _counts(run)
    attempts = _attempt_counts(run)
    hidden = attempts["errors"] - counts["errors"] if attempts else 0  # errored attempts inside cases that still got a verdict
    limit = _max_error_rate(run, comparison)
    if not counts["errors"]:
        if hidden <= 0:
            return ""
        what = f"{attempts['errors']} of {attempts['attempts']} attempts errored ({fmt_pct(attempts['rate'])} of attempts, limit {fmt_pct(limit)})"
        if attempts["rate"] is not None and attempts["rate"] > limit:
            return (
                f'<div class="health bad" role="alert"><strong>Unhealthy run.</strong> {what}. Every case still has a verdict (the majority of its attempts), '
                "so no case is listed as an error, but a run where this many attempts crash is not a reliable measurement. See Errors.</div>"
            )
        return (
            f'<div class="health" role="note"><strong>{attempts["errors"]} attempt(s) errored</strong> inside repeated cases that still got a verdict. '
            "The majority outcome hides them: they are not quality failures. See Errors.</div>"
        )
    rate = counts["error_rate"]
    what = f"{counts['judge_errors']} judge error(s) and {counts['target_errors']} target error(s)"
    if attempts and hidden > 0:
        what += f"; {attempts['errors']} of {attempts['attempts']} attempts errored in all"
    if rate > limit:
        return (
            f'<div class="health bad" role="alert"><strong>Unhealthy run.</strong> {what} ({fmt_pct(rate)} of requests, limit {fmt_pct(limit)}). '
            "The pass rate below covers only the cases that could be scored, so it is not a quality signal until the errors are fixed and the run is repeated. See Errors.</div>"
        )
    return (
        f'<div class="health" role="note"><strong>{counts["errors"]} case(s) errored</strong> ({what}). '
        "They are excluded from the pass rate and are not quality failures. See Errors.</div>"
    )


# ------------------------------------------------------------------------ KPI tiles
def _baseline_mean_cost(baseline: RunResult | None, comparison: dict | None) -> float | None:
    if baseline is not None:
        value = _num((baseline.summary.get("cost") or {}).get("system_mean_usd"))
        if value is not None:
            return value
    return _num(((comparison or {}).get("cost") or {}).get("baseline_mean_usd"))


def _valid_monthly(monthly_requests: Any) -> int | None:
    value = _num(monthly_requests)
    return int(value) if value is not None and value > 0 else None


def _resumed_cost_note(summary: dict) -> str:
    resumed = summary.get("resumed_from")
    cost = _num(resumed.get("reused_cost_usd")) if isinstance(resumed, dict) else None
    return f"{fmt_money(cost)} of the run's cost was spent before it was resumed" if cost else ""


def _kpis(run: RunResult, comparison: dict | None, calibration: dict | None, monthly_requests: int | None = None) -> str:
    s = run.summary
    counts = _counts(run)
    n = counts["cases"]
    cost = s.get("cost") or {}
    latency = s.get("latency") or {}
    diff = comparison or {}
    cmp_pass = diff.get("pass_rate") or {}
    cmp_cost = diff.get("cost") or {}
    cmp_p95 = diff.get("latency_p95") or {}
    cmp_err = diff.get("error_rate") or {}

    err_delta = None
    if _num(cmp_err.get("baseline")) is not None and _num(cmp_err.get("candidate")) is not None:
        err_delta = cmp_err["candidate"] - cmp_err["baseline"]

    pass_subs = [f"95% CI {fmt_ci(s.get('pass_rate_ci'))}" if counts["scored"] else "", f"{fmt_int(counts['passed'])} of {fmt_int(counts['scored'])} scored cases"]
    if counts["errors"]:
        pass_subs.append(f"{fmt_int(counts['errors'])} errored, not counted")
    oversize = _n_oversize(run)
    if oversize:
        pass_subs.append(f"{fmt_int(oversize)} judge verdict(s) are automatic fails (output over max_output_chars)")
    if _is_placeholder(s):
        pass_subs.append("placeholder run: not a measure of the real product")
    quality = [_tile("Pass rate", fmt_pct(s.get("pass_rate")), pass_subs, _delta_pp(cmp_pass.get("delta"), True))]
    if calibration:
        corrected = calibration.get("corrected_pass_rate")
        view = _kappa_view(calibration)
        kappa = view["kappa"]
        decision = calibration.get("kappa_decision")
        kappa_word = "weighted κ" if view["weighted"] else "κ"
        agreement = f"{kappa_word} {kappa:.2f} ({_decision_word(decision) if decision else calibration.get('kappa_label')})" if kappa is not None else f"{kappa_word} n/a"
        judge = calibration.get("judge")
        stale = _stale_calibration_text(calibration, run)
        quality.append(
            _tile(
                "Calibrated pass rate",
                fmt_pct(corrected),
                [
                    f"95% CI {fmt_ci(calibration.get('corrected_pass_rate_ci'))}" if corrected is not None else "not estimable",
                    f"{agreement} · {fmt_int(calibration.get('n'))} labeled",
                    f"judge: {judge}" if judge not in (None, "overall") else "",
                    f"stale: {STALE_CALIBRATION_TEXT}" if stale else "",
                ],
            )
        )
    error_subs = [f"target {fmt_int(counts['target_errors'])} · judge {fmt_int(counts['judge_errors'])}", f"{fmt_pct(counts['error_rate'])} of requests, not quality failures"]
    attempts = _attempt_counts(run)
    if attempts and (attempts["attempts"] != counts["cases"] or attempts["errors"] != counts["errors"]):
        error_subs.append(f"attempts: {fmt_int(attempts['errors'])} of {fmt_int(attempts['attempts'])} errored ({fmt_pct(attempts['rate'])})")
    retries = _target_retries(run)
    if retries:
        error_subs.append(f"{fmt_int(retries)} target retry call(s) after a crash, not counted as errors")
    quality.append(_tile("Errors", fmt_int(counts["errors"]), error_subs, _delta_pp(err_delta, False)))
    stability = _stability(run)
    if stability and stability["n_repeated"]:
        quality.append(
            _tile(
                "Flip rate",
                fmt_pct(stability["flip_rate"]),
                [f"{fmt_int(stability['n_repeated'])} repeated cases", "share whose attempts disagree (noise floor)"],
            )
        )

    n_priced = cost.get("n_priced") or 0
    priced_note = ""
    if n_priced == 0:
        priced_note = "no cost data reported"
    elif n_priced < n:
        priced_note = f"priced {n_priced} of {n} requests"
    cost_delta = _delta_pct(cmp_cost.get("delta_pct"), False)
    economics = [
        _tile("Cost / 1k requests", fmt_money(cost.get("system_per_1k_usd")), [priced_note], cost_delta),
        _tile("Mean cost / request", fmt_money(cost.get("system_mean_usd")), [], cost_delta),
    ]
    if monthly_requests:
        per_request = _num(cost.get("system_mean_usd"))
        monthly = per_request * monthly_requests if per_request is not None and n_priced else None
        economics.append(_tile("Est. cost / month", fmt_money(monthly), [f"at {monthly_requests:,} requests per month"], cost_delta))
    economics += [
        _tile("Latency p50", fmt_latency(latency.get("p50_s"))),
        _tile("Latency p95", fmt_latency(latency.get("p95_s")), [], _delta_pct(cmp_p95.get("delta_pct"), False)),
        _tile("Eval run cost", fmt_money(cost.get("judge_total_usd")), ["cost of grading this run", _resumed_cost_note(s)]),
    ]
    return (
        '<div class="kpis"><h2 class="group">Quality</h2>'
        f'<div class="tiles">{"".join(quality)}</div>'
        '<h2 class="group">Cost and speed</h2>'
        f'<div class="tiles">{"".join(economics)}</div></div>'
    )


# ---------------------------------------------------------------------- key findings
def _failed_criteria(record: dict) -> set[str]:
    """Labels of what failed in one record: `judge / criterion` or the judge name."""
    failed: set[str] = set()
    for judge in record.get("judges") or []:
        if judge.get("passed"):
            continue
        criteria = (judge.get("details") or {}).get("criteria") or {}
        bad = [cid for cid, c in criteria.items() if not c.get("pass")]
        if bad:
            failed.update(f"{judge['judge']} / {cid}" for cid in bad)
        else:
            failed.add(judge["judge"])
    return failed


def _kappa_view(cal: dict) -> dict:
    """The kappa the judge verdict is decided on, and the secondary one.

    ``calibrate`` decides on the WEIGHTED kappa of the whole run (``kappa_weighted`` / ``kappa_weighted_ci``); the unweighted
    ``kappa`` / ``kappa_ci`` describe the stratified sample of labels, not the run, and are secondary.  A calibration dict
    without the weighted keys (written by an older evalsprint) falls back to the unweighted pair as its primary number.
    """
    weighted = "kappa_weighted" in cal or "kappa_weighted_ci" in cal
    if weighted:
        return {
            "weighted": True,
            "kappa": _num(cal.get("kappa_weighted")),
            "ci": cal.get("kappa_weighted_ci"),
            "sample": _num(cal.get("kappa")),
            "sample_ci": cal.get("kappa_ci"),
        }
    return {"weighted": False, "kappa": _num(cal.get("kappa")), "ci": cal.get("kappa_ci"), "sample": None, "sample_ci": None}


def _valid_range(ci: Any) -> bool:
    return isinstance(ci, (list, tuple)) and len(ci) == 2 and all(_num(x) is not None for x in ci)


_DECISION_WORDS = {
    "trustworthy": "trustworthy",
    "usable": "usable",
    "inconclusive": "inconclusive",
    "not_trustworthy": "not trustworthy",
    "insufficient_data": "insufficient data",
}
_DECISION_TEXT = {
    "trustworthy": "Trustworthy: even the low end of the kappa interval is at or above 0.8 and the true-negative screening check passed (a screen, not a guarantee).",
    "usable": "Usable: the low end of the kappa interval is at or above 0.6. Keep checking it on new labels.",
    "inconclusive": "Inconclusive: the kappa interval is too wide to say whether the judge can be trusted. More labels are needed.",
    "not_trustworthy": "Not trustworthy: even the high end of the kappa interval is below 0.6. Do not gate on this judge yet.",
    "insufficient_data": "Insufficient data: fewer than 30 labeled cases, too few to decide.",
}
_DECISION_TONE = {"trustworthy": "pass", "usable": "pass", "inconclusive": "warn", "not_trustworthy": "bad", "insufficient_data": "warn"}


def _decision_word(decision: Any) -> str:
    return _DECISION_WORDS.get(str(decision), str(decision).replace("_", " "))


def _findings(run: RunResult, comparison: dict | None, calibration: dict | None, failures: list[dict]) -> list[str]:
    """A few plain-language bullets (HTML) computed from the data, for readers who stop after the tiles."""
    s = run.summary
    counts = _counts(run)
    n = counts["cases"]
    if not n:
        return []
    bullets = []
    if _is_placeholder(s):
        bullets.append(
            "<strong>This is a placeholder run</strong> (the scaffold's offline stub as target, or an LLM judge on the mock provider): "
            "its verdicts say nothing about the real product."
        )
    baseline_warning = _placeholder_baseline_warning(comparison)
    if baseline_warning is not None:
        bullets.append("<strong>The baseline is a placeholder run</strong>: the comparison against it is meaningless until a real baseline is recorded.")
    if counts["errors"]:
        bullets.append(
            f"<strong>{counts['errors']}</strong> of {n} request(s) errored ({counts['target_errors']} in the target, {counts['judge_errors']} in a judge). "
            "They are excluded from the pass rate: they say the run is unhealthy, not that quality dropped."
        )
    if counts["scored"] and not failures:
        bullets.append(f"All <strong>{counts['scored']}</strong> scored cases passed.")
    elif failures:
        found = Counter(label for r in failures for label in _failed_criteria(r))
        if found:
            label, count = found.most_common(1)[0]
            if count >= 2:
                bullets.append(f"Most frequent failure: <code>{_e(label)}</code> in <strong>{count}</strong> of {len(failures)} failing cases.")
    segments = {t: v for t, v in (s.get("by_tag") or {}).items() if t != UNTAGGED and (v.get("n") or 0) >= MIN_SEGMENT_N and v.get("pass_rate") is not None}
    if len(segments) >= 2 and s.get("pass_rate") is not None:
        tag, stats = min(segments.items(), key=lambda kv: (kv[1]["pass_rate"], kv[0]))
        if stats["pass_rate"] < s["pass_rate"]:
            small = ", small sample" if stats["n"] < LOW_N else ""
            bullets.append(
                f"Weakest segment: <strong>{_e(tag)}</strong> at {fmt_pct(stats['pass_rate'])} (n = {stats['n']}{small}) "
                f"against {fmt_pct(s['pass_rate'])} overall."
            )
    attempts = _attempt_counts(run)
    if attempts and attempts["errors"] > counts["errors"]:
        bullets.append(
            f"<strong>{attempts['errors'] - counts['errors']}</strong> attempt(s) errored inside repeated cases that still got a verdict "
            f"({attempts['errors']} of {attempts['attempts']} attempts in all): the majority outcome hides them, they are not quality failures."
        )
    retries = _target_retries(run)
    if retries:
        bullets.append(
            f"<strong>{retries}</strong> target call(s) crashed and were retried within the run (paid, and counted in the cost): a flaky target, "
            "even where the retry recovered the case."
        )
    oversize = _n_oversize(run)
    if oversize:
        bullets.append(
            f"<strong>{oversize}</strong> judge verdict(s) are automatic fails: the output was longer than <code>max_output_chars</code>, so the judge was not called."
        )
    flaky = (comparison or {}).get("flaky_crashes")
    if isinstance(flaky, list) and flaky:
        bullets.append(
            f"<strong>{len(flaky)}</strong> case(s) that passed in the baseline crashed in some attempts of this run although the majority outcome is pass "
            "(flaky crash): the pass rate does not show it, check the target."
        )
    tag_warnings = _valid_tag_warnings(comparison)
    if tag_warnings:
        worst = tag_warnings[0]
        more = f" and {len(tag_warnings) - 1} more" if len(tag_warnings) > 1 else ""
        bullets.append(
            f"Tag <strong>{_e(worst['tag'])}</strong>{more} dropped a lot against the baseline "
            f"({fmt_pct(worst['baseline'])} → {fmt_pct(worst['candidate'])}, {int(worst['n'])} paired cases): a warning, not a gate reason."
        )
    stability = _stability(run)
    if stability and stability["n_repeated"] and stability["flip_rate"] is not None:
        bullets.append(
            f"Noise floor: <strong>{fmt_pct(stability['flip_rate'])}</strong> of the {stability['n_repeated']} repeated cases did not give the same verdict on every attempt. "
            "A single run of those cases can flip without any change to the system."
        )
    elif stability and not stability["n_repeated"]:
        bullets.append(
            "No case was repeated, so this run carries no measure of its own noise. Set <code>repeat</code> in the config "
            "(for example <code>{n: 3, per_tag: {safety: 5}}</code>: every case 3 times, the safety cases 5 times)."
        )
    stale = _stale_calibration_text(calibration, run)
    if stale:
        bullets.append(f"<strong>Stale calibration:</strong> this {_e(STALE_CALIBRATION_TEXT)}, so its agreement and kappa do not describe the judges that scored this run.")
    view = _kappa_view(calibration) if calibration else None
    if calibration and calibration.get("n") and view["kappa"] is not None:
        kappa, n_labeled = view["kappa"], calibration["n"]
        kappa_word = "weighted κ" if view["weighted"] else "κ"
        decision = calibration.get("kappa_decision")
        if decision:
            ci = f", 95% CI {_fmt_range(view['ci'])}" if _valid_range(view["ci"]) else ""
            bullets.append(
                f"Judge agreement with human labels is <strong>{_e(_decision_word(decision))}</strong> "
                f"({kappa_word} {kappa:.2f}{ci}, {n_labeled} labeled cases)."
            )
        else:
            caveats = []
            if kappa < 0.6:
                caveats.append("treat the automated pass rate with caution")
            if n_labeled < SMALL_LABEL_SET:
                caveats.append("the labeled sample is small")
            bullets.append(
                f"Judge agreement with human labels is <strong>{_e(calibration.get('kappa_label'))}</strong> "
                f"({kappa_word} {kappa:.2f}, {n_labeled} labeled cases)" + (f": {'; '.join(caveats)}." if caveats else ".")
            )
    return bullets


# ------------------------------------------------------------------ quality by tag
def _valid_tag_warnings(comparison: dict | None) -> list[dict]:
    """`comparison["tag_warnings"]` entries that are usable ({"tag", "n", "baseline", "candidate", "delta"}), worst first."""
    raw = (comparison or {}).get("tag_warnings")
    if not isinstance(raw, list):
        return []
    valid = []
    for w in raw:
        if not isinstance(w, dict) or w.get("tag") is None:
            continue
        n, base, cand, delta = _num(w.get("n")), _num(w.get("baseline")), _num(w.get("candidate")), _num(w.get("delta"))
        if None in (n, base, cand):
            continue
        valid.append({"tag": str(w["tag"]), "n": n, "baseline": base, "candidate": cand, "delta": delta if delta is not None else cand - base})
    return sorted(valid, key=lambda w: (w["delta"], w["tag"]))


def _tag_table(label: str, by_tag: dict, tags: Iterable[str], cmp_tags: dict, comparison: dict | None) -> str:
    stats = [(t, by_tag[t]) for t in tags if t in by_tag]
    ordered = sorted(stats, key=lambda kv: (kv[1].get("pass_rate") is None, kv[1].get("pass_rate") or 0, kv[0]))
    # Pass rate comes right after the name so it stays visible when the table scrolls on a phone.
    headers = [("Tag", "wrap"), ("Pass rate", "num"), ("Cases", "num"), ("Bar", "bar-cell"), ("95% CI", "num")]
    if comparison:
        headers.append(("Change vs baseline", ""))
    warned = {w["tag"] for w in _valid_tag_warnings(comparison)}
    rows = []
    for tag, st in ordered:
        n = st.get("n") or 0
        small = f" {_badge('small sample', 'warn')}" if n < LOW_N else ""
        small += f" {_badge('big drop', 'warn')}" if tag in warned else ""
        row = [f"<strong>{_e(tag)}</strong>{small}", fmt_pct(st.get("pass_rate")), fmt_int(n), _bar(st.get("pass_rate"), st.get("ci")), fmt_ci(st.get("ci"))]
        if comparison:
            row.append(_delta_pp((cmp_tags.get(tag) or {}).get("delta"), True, compact=True) or DASH)
        rows.append(row)
    return _table(label, headers, rows)


def _tag_groups(summary: dict, by_tag: dict) -> list[tuple[str, list[str]]]:
    """`[(group, tags present in by_tag)]` from summary["tag_groups"], plus "other" for the rest."""
    groups = summary.get("tag_groups")
    if not isinstance(groups, dict) or not groups:
        return []
    result: list[tuple[str, list[str]]] = []
    used: set[str] = set()
    for name, tags in groups.items():
        present = [t for t in (tags if isinstance(tags, (list, tuple)) else []) if t in by_tag]
        if present:
            result.append((str(name), present))
            used.update(present)
    if not result:  # nothing usable in tag_groups: the caller draws the single table
        return []
    rest = [t for t in by_tag if t not in used]
    if rest:
        result.append((OTHER_GROUP, rest))
    return result


def _tags_section(run: RunResult, comparison: dict | None) -> tuple[str, str, str] | None:
    by_tag = run.summary.get("by_tag") or {}
    if not by_tag:
        return None
    if set(by_tag) == {UNTAGGED}:
        note = '<p class="muted">No cases carry tags, so there is no per-segment breakdown. Add <code>"tags": [...]</code> to the cases in the test set to see quality by topic or risk area.</p>'
        return _section("tags", "Quality by tag", note)

    cmp_tags = (comparison or {}).get("by_tag") or {}
    groups = _tag_groups(run.summary, by_tag)
    warnings = _valid_tag_warnings(comparison)
    warning_html = ""
    if warnings:
        rows = [
            [f"<strong>{_e(w['tag'])}</strong>", fmt_int(w["n"]), fmt_pct(w["baseline"]), fmt_pct(w["candidate"]), _delta_pp(w["delta"], True, compact=True)]
            for w in warnings
        ]
        warning_html = (
            '<div class="alert"><strong>Big drop in a tag (a warning, not a gate reason).</strong> A tag is a slice of the suite, so with many tags a few '
            "move by chance; critical tags have their own rule. Check these before you merge.</div>"
            + _table("Tags whose pass rate fell a lot", [("Tag", "wrap"), ("Paired cases", "num"), ("Baseline", "num"), ("Candidate", "num"), ("Change", "")], rows)
        )
    if groups:
        body = warning_html + (
            '<p class="muted small">Each table slices the same cases along one dimension, so a case is counted once per table. '
            "A tag can be strong in one slicing and weak in another: read the small tables, not only the average.</p>"
        )
        for name, tags in groups:
            body += f"<h3>{_e(name)}</h3>" + _tag_table(f"Quality by tag: {name}", by_tag, tags, cmp_tags, comparison)
    else:
        body = warning_html + _tag_table("Quality by tag", by_tag, by_tag, cmp_tags, comparison)
    legend = '<p class="muted small">Sorted weakest first. Bar = pass rate; whisker = 95% Wilson confidence interval. Scored cases only: errors are excluded.</p>'
    return _section("tags", "Quality by tag", body + legend)


# ------------------------------------------------------------------------ stability
def _stability_section(run: RunResult, shown_ids: set[str]) -> tuple[str, str, str] | None:
    stability = _stability(run)
    if not stability or not stability["n_repeated"]:
        return None
    repeated = [r for r in run.results if len(_attempts(r)) > 1]
    unstable = [r for r in repeated if len({a.get("outcome") for a in _attempts(r)}) > 1]
    varied = [r for r in repeated if len({a.get("output_sha") for a in _attempts(r) if a.get("output_sha")}) > 1]
    per_case = _attempts_per_case(run)
    if len(per_case) > 1:
        ran = "ran " + ", ".join(f"{k} times ({v})" for k, v in sorted(per_case.items(), reverse=True)) + " (the protocol gives each tag its own number of attempts)"
    elif per_case:
        ran = f"ran {next(iter(per_case))} times each"
    else:  # no per-attempt records in this run: the protocol is all there is to say
        protocol = _run_protocol(run)
        ran = f"ran {protocol[0]} times each" if protocol and protocol[0] > 1 else "were repeated"
    flip = stability["flip_rate"]
    intro = (
        f"<p><strong>{stability['n_repeated']}</strong> case(s) {ran}. "
        f"<strong>Flip rate {fmt_pct(flip)}</strong>: the share of repeated cases whose attempts did not all agree. "
        "The case verdict is the majority of its attempts (an even split counts as a fail).</p>"
    )
    if repeated:
        intro += f'<p class="muted small">The output text changed between attempts in {len(varied)} of {len(repeated)} repeated cases.</p>'
    if not unstable:
        body = intro + '<p class="ok-note"><span aria-hidden="true">✔</span> Every repeated case gave the same verdict on every attempt.</p>'
    else:
        rows = []
        for r in unstable[:MAX_UNSTABLE]:
            tally = Counter(a.get("outcome") for a in _attempts(r))
            case_id = r["case_id"]
            link = _case_link(case_id) if case_id in shown_ids else _e(case_id)
            stab = _num(r.get("stability"))
            rows.append(
                [
                    f'<span class="case-id">{link}</span>',
                    f"{tally.get('pass', 0)} / {tally.get('fail', 0)} / {tally.get('error', 0)}",
                    fmt_pct(stab, 0),
                    _badge(_outcome(r), "pass" if _outcome(r) == "pass" else "bad" if _outcome(r) == "fail" else "warn"),
                ]
            )
        headers = [("Case", ""), ("Pass / fail / error", "num"), ("Agreement", "num"), ("Verdict", "")]
        body = intro + f"<p>{len(unstable)} case(s) flipped:</p>" + _table("Unstable repeated cases", headers, rows)
        if len(unstable) > MAX_UNSTABLE:
            body += f'<p class="muted small">+ {len(unstable) - MAX_UNSTABLE} more not listed.</p>'
    body += (
        '<p class="muted small">This is the noise the system shows with nothing changed. A gate that reacts to a single flip '
        "will raise false alarms; repeating the cases that gate a release and taking the majority removes most of them.</p>"
    )
    return _section("stability", "Stability", body)


# --------------------------------------------------------------------------- judges
def _judges_section(run: RunResult) -> tuple[str, str, str] | None:
    by_judge = run.summary.get("by_judge") or {}
    if not by_judge:
        return None
    headers = [("Judge", ""), ("Pass rate", "num"), ("Cases", "num"), ("95% CI", "num"), ("Mean score", "num"), ("Errors", "num"), ("Cost", "num")]
    rows = []
    for name, st in by_judge.items():
        score = st.get("mean_score")
        errors = st.get("n_errors") or 0
        rows.append(
            [
                f"<strong>{_e(name)}</strong>",
                fmt_pct(st.get("pass_rate")),
                fmt_int(st.get("n")),
                fmt_ci(st.get("ci")),
                DASH if score is None else f"{score:.2f}",
                f"{errors} {_badge('errors', 'warn')}" if errors else "0",
                fmt_money(st.get("cost_usd")),
            ]
        )
    note = (
        '<p class="muted small">Errors are cases where this judge produced no verdict (API or parsing failure). They are left out of the judge\'s pass rate '
        "and count as run errors, not as quality failures. Cases where the target failed first are not graded. "
        "Cost is the price of running the judge, not of the feature under test.</p>"
    )
    return _section("judges", "Judges", _table("Judges", headers, rows) + note)


# ---------------------------------------------------------------------------- errors
def _errors_section(run: RunResult, comparison: dict | None) -> tuple[str, str, str] | None:
    errored = [r for r in run.results if _outcome(r) == "error"]
    counts = _counts(run)
    flaky = [r for r in run.results if _outcome(r) != "error" and _errored_attempts(r)]
    if not errored and not counts["errors"] and not flaky:
        return None
    rows = []
    for r in errored[:MAX_ERRORS]:
        source = _error_source(r)
        message = _error_message(r)
        message = message if len(message) <= ERROR_TEXT_LIMIT else message[:ERROR_TEXT_LIMIT] + "…"
        used = _num(r.get("target_retries_used"))
        if used:
            message += f" (still failing after {int(used)} retry call(s))"
        rows.append([f'<span class="case-id">{_e(r["case_id"])}</span>', _badge(source, "warn"), _e(message)])
    n_err = counts["errors"] or len(errored)
    intro = (
        f"<p><strong>{n_err}</strong> of {counts['cases']} case(s) errored: <strong>{counts['target_errors']}</strong> in the target (the system under test failed or returned nothing usable), "
        f"<strong>{counts['judge_errors']}</strong> in a judge (no verdict after its own retries). "
        "An errored case is not a quality failure: it is left out of every pass rate and out of paired comparisons. Fix the cause and run again.</p>"
    ) if n_err else ""
    body = intro
    retries = _target_retries(run)
    if retries:
        body += (
            f'<p>{fmt_int(retries)} target retry call(s) were made within the run after a crash (<code>target_retries</code>): they are paid and counted in the cost, '
            "and a case whose retry succeeded is not an error, so this count is the only trace of a flaky target.</p>"
        )
    if rows:
        body += _table("Errored cases", [("Case", ""), ("Source", ""), ("Error", "wrap")], rows)
    if len(errored) > MAX_ERRORS:
        body += f'<p class="muted small">+ {len(errored) - MAX_ERRORS} more not listed; results.jsonl has all of them.</p>'
    if flaky:
        rows = []
        for r in flaky[:MAX_ERRORS]:
            errored_attempts = _errored_attempts(r)
            target = sum(1 for a in errored_attempts if (a.get("error_source") or "judge") == "target")
            rows.append(
                [
                    f'<span class="case-id">{_e(r["case_id"])}</span>',
                    fmt_int(len(_attempts(r))),
                    f"{len(errored_attempts)} (target {target} · judge {len(errored_attempts) - target})",
                    _badge(_outcome(r), "pass" if _outcome(r) == "pass" else "bad"),
                ]
            )
        body += (
            f"<h3>Errors inside repeated cases ({len(flaky)})</h3>"
            "<p>These cases still got a verdict, the majority of their attempts that did not error, so the case-level error count does not include them. "
            "A target crash in one attempt is a flaky crash: the pass rate does not show it.</p>"
            + _table("Errored attempts inside repeated cases", [("Case", ""), ("Attempts", "num"), ("Errored attempts", "num"), ("Verdict", "")], rows)
        )
        if len(flaky) > MAX_ERRORS:
            body += f'<p class="muted small">+ {len(flaky) - MAX_ERRORS} more not listed; results.jsonl has all of them.</p>'
    limit = _max_error_rate(run, comparison)
    if counts["error_rate"] > limit:
        body += f'<p class="alert">Error rate {fmt_pct(counts["error_rate"])} is above the {fmt_pct(limit)} limit: this run is unhealthy and should not be used as a baseline.</p>'
    attempts = _attempt_counts(run)
    if attempts and attempts["rate"] is not None and attempts["rate"] > limit and counts["error_rate"] <= limit:
        body += f'<p class="alert">Attempt error rate {fmt_pct(attempts["rate"])} ({attempts["errors"]} of {attempts["attempts"]} attempts) is above the {fmt_pct(limit)} limit: this run is unhealthy and should not be used as a baseline.</p>'
    return _section("errors", "Errors", body)


# ------------------------------------------------------------------- regressions
def _id_list(ids: list[str], by_case: dict, shown_ids: set[str], critical: set[str], link: bool = True) -> str:
    def item(case_id: str) -> str:
        tags = (by_case.get(case_id, {}).get("case") or {}).get("tags") or []
        text = _case_link(case_id) if link and case_id in shown_ids else _e(case_id)
        flag = f" {_badge('critical', 'bad')}" if case_id in critical else ""
        tag_html = f' <span class="muted small">{_e(", ".join(tags))}</span>' if tags else ""
        return f"<li>{text}{flag}{tag_html}</li>"

    shown = ids[:MAX_ID_CHIPS]
    more = f'<p class="muted small">+ {len(ids) - len(shown)} more not listed.</p>' if len(ids) > len(shown) else ""
    return f'<ul class="idlist">{"".join(item(c) for c in shown)}</ul>{more}'


def _regressions_section(run: RunResult, comparison: dict, shown_ids: set[str], baseline: RunResult | None = None) -> tuple[str, str, str]:
    by_case = run.by_case()
    base_by_case = baseline.by_case() if baseline is not None else {}
    critical = set(comparison.get("critical_regressions") or [])
    newly_failing = comparison.get("newly_failing") or []
    newly_passing = comparison.get("newly_passing") or []
    removed = comparison.get("removed_cases") or []
    changed = comparison.get("changed_cases") or []
    flaky = [c for c in comparison.get("flaky_crashes") or [] if isinstance(c, str)] if isinstance(comparison.get("flaky_crashes"), list) else []

    if newly_failing:
        body = (
            f"<p><strong>{len(newly_failing)}</strong> case(s) passed in the baseline and fail now.</p>"
            + _id_list(list(newly_failing), by_case, shown_ids, critical)
        )
    else:
        body = '<p class="ok-note"><span aria-hidden="true">✔</span> No case went from pass to fail.</p>'
    if newly_passing:
        body += (
            f'<details><summary>{len(newly_passing)} case(s) fixed (fail → pass)</summary>'
            f'<p>{_chips(newly_passing[:MAX_ID_CHIPS])}</p></details>'
        )
    if flaky:
        body += (
            f"<h3>Flaky crashes ({len(flaky)})</h3>"
            '<p class="alert">These cases passed in the baseline. Their majority outcome is still pass, but the target crashed in some of their attempts, '
            "so the pass rate does not show it. Check the target before you rely on this run.</p>"
            + _id_list(flaky, by_case, shown_ids, critical, link=False)
        )
    if removed:
        body += (
            f"<h3>Removed cases ({len(removed)})</h3>"
            '<p class="alert">These baseline cases are missing from this run, so nothing was compared for them. A shrinking test set can hide a regression. '
            "If the removal is intended, update the baseline in a reviewed change.</p>"
            + _id_list(list(removed), base_by_case, shown_ids, critical, link=False)
        )
    if changed:
        body += (
            f"<h3>Changed cases ({len(changed)})</h3>"
            '<p class="alert">These cases keep their id but their input, reference, expected behavior or checks were edited since the baseline. '
            "They were not compared: update the baseline in a reviewed change.</p>"
            + _id_list(list(changed), by_case, shown_ids, critical, link=False)
        )
    return _section("regressions", "Regressions", body)


# --------------------------------------------------------------------------- failures
def _newly_failing_ids(run: RunResult, comparison: dict | None, baseline: RunResult | None) -> set[str]:
    if comparison is not None:
        return set(comparison.get("newly_failing") or [])
    if baseline is None:
        return set()
    base = baseline.by_case()
    return {r["case_id"] for r in run.results if _outcome(r) == "fail" and r["case_id"] in base and _outcome(base[r["case_id"]]) == "pass"}


def _failing_records(run: RunResult, comparison: dict | None, newly: set[str] | None = None) -> list[dict]:
    """Failing cases (errors are not failures), regressions first (critical, then newly failing), then dataset order."""
    critical = set((comparison or {}).get("critical_regressions") or [])
    newly = set((comparison or {}).get("newly_failing") or []) if newly is None else newly
    failing = [r for r in run.results if _outcome(r) == "fail"]
    return sorted(failing, key=lambda r: 0 if r["case_id"] in critical else 1 if r["case_id"] in newly else 2)


def _judge_block(judge: dict) -> str:
    score = judge.get("score")
    head = f'<strong>{_e(judge["judge"])}</strong> {_badge("fail", "bad")}' + (f' <span class="muted small">score {score:.2f}</span>' if score is not None else "")
    parts = [f'<div class="judge-head">{head}</div>']
    if judge.get("error"):
        parts.append(f'<p class="alert">Judge error: {_e(judge["error"])}</p>')
    criteria = (judge.get("details") or {}).get("criteria") or {}
    if criteria:
        items = []
        for cid, c in criteria.items():
            ok = bool(c.get("pass"))
            mark = "✔ pass" if ok else "✘ fail"
            items.append(
                f'<li class="{"crit-pass" if ok else "crit-fail"}"><span class="mark">{mark}</span> '
                f'<code>{_e(cid)}</code> {_e(c.get("reasoning"))}</li>'
            )
        parts.append(f'<ul class="criteria">{"".join(items)}</ul>')
    elif judge.get("reasoning"):
        parts.append(f'<p class="reasoning">{_e(judge["reasoning"])}</p>')
    return f'<div class="judge-block">{"".join(parts)}</div>'


def _side_by_side(record: dict, base_record: dict | None) -> str:
    """Baseline output next to the candidate output, for a case that got worse."""
    if base_record is None:
        return '<p class="muted small">This case is not in the baseline run, so there is no baseline output to show.</p>'
    base_out, cand_out = (base_record.get("output") or {}), (record.get("output") or {})
    base_outcome = _outcome(base_record)
    base_badge = _badge(base_outcome, "pass" if base_outcome == "pass" else "bad" if base_outcome == "fail" else "warn")
    return (
        '<h4>Baseline vs candidate output</h4><div class="sbs">'
        f'<div class="pane"><div class="pane-head">Baseline {base_badge}</div>{_pre(base_out.get("text"), "(empty output)")}</div>'
        f'<div class="pane"><div class="pane-head">Candidate {_badge("fail", "bad")}</div>{_pre(cand_out.get("text"), "(empty output)")}</div>'
        "</div>"
    )


def _attempts_line(record: dict) -> str:
    attempts = _attempts(record)
    if len(attempts) < 2:
        return ""
    tally = Counter(a.get("outcome") for a in attempts)
    parts = ", ".join(f"{tally[k]} {k}" for k in ("pass", "fail", "error") if tally.get(k))
    stab = _num(record.get("stability"))
    stab_text = f" (agreement {fmt_pct(stab, 0)})" if stab is not None else ""
    return f'<p><span class="muted">Attempts:</span> {len(attempts)} ({_e(parts)}){stab_text}. The output shown is from the first attempt that matches the majority.</p>'


def _failure_entry(record: dict, critical: set[str], newly: set[str], base_by_case: dict | None = None) -> str:
    case = record.get("case") or {}
    output = record.get("output") or {}
    case_id = record["case_id"]
    judges = record.get("judges") or []
    failed = [j for j in judges if not j.get("passed")]
    passed_names = [j["judge"] for j in judges if j.get("passed")]

    flags = ""
    if case_id in critical:
        flags += " " + _badge("critical regression", "bad")
    elif case_id in newly:
        flags += " " + _badge("new failure", "bad")
    failed_names = ", ".join(j["judge"] for j in failed)
    summary = (
        f'<span class="case-id">{_e(case_id)}</span>{flags} {_chips(case.get("tags") or [])}'
        + (f' <span class="muted small">failed: {_e(failed_names)}</span>' if failed_names else "")
    )

    body = [f"<h4>Input</h4>{_pre(case.get('input'))}"]
    body.append(f"<h4>Output</h4>{_pre(output.get('text'), '(empty output)')}")
    if base_by_case is not None and case_id in newly:
        body.append(_side_by_side(record, base_by_case.get(case_id)))
    attempts = _attempts_line(record)
    if attempts:
        body.append(attempts)
    if case.get("reference"):
        body.append(f"<h4>Reference answer</h4>{_pre(case['reference'])}")
    if case.get("expected_behavior"):
        body.append(f'<p><span class="muted">Expected behavior:</span> {_chips([case["expected_behavior"]])}</p>')
    context = (output.get("meta") or {}).get("context")
    if context:
        body.append(f"<details><summary>Retrieved context</summary>{_pre(context)}</details>")
    if failed:
        body.append("<h4>Failed judges</h4>" + "".join(_judge_block(j) for j in failed))
    else:
        note = "No judge produced a verdict for this case." if not judges else "No judge failed on its own; the case failed under the pass rule."
        body.append(f'<p class="muted">{note}</p>')
    if passed_names:
        body.append(f'<p class="muted small">Passed: {_e(", ".join(passed_names))}</p>')
    return f'<details class="case" id="{_e(_case_dom_id(case_id))}"><summary>{summary}</summary><div class="case-body">{"".join(body)}</div></details>'


def _failures_section(run: RunResult, comparison: dict | None, failures: list[dict], newly: set[str], baseline: RunResult | None) -> tuple[str, str, str]:
    counts = _counts(run)
    if not failures:
        message = "No failing cases." if not counts["errors"] else "No failing cases among the scored ones."
        return _section("failures", "Failures", f'<p class="ok-note"><span aria-hidden="true">✔</span> {message}</p>')
    critical = set((comparison or {}).get("critical_regressions") or [])
    shown = failures[:MAX_FAILURES]
    base_by_case = baseline.by_case() if baseline is not None else None
    scored = counts["scored"] or counts["cases"]
    intro = f"<p><strong>{len(failures)}</strong> of {scored} scored cases failed" + (" (regressions listed first)." if newly else ".")
    intro += " Errored cases are not listed here; see Errors.</p>" if counts["errors"] else "</p>"
    if len(failures) > len(shown):
        intro += (
            f'<p class="alert">Showing {len(shown)} of {len(failures)} failing cases; '
            f"{len(failures) - len(shown)} hidden. results.jsonl in the run directory has all of them.</p>"
        )
    return _section("failures", "Failures", intro + "".join(_failure_entry(r, critical, newly, base_by_case) for r in shown))


# ------------------------------------------------------------------------ calibration
def _stat(label: str, value: str, hint: str = "") -> str:
    hint_html = f"<small>{_e(hint)}</small>" if hint else ""
    return f"<div><dt>{_e(label)}</dt><dd>{_e(value)}{hint_html}</dd></div>"


def _rate_with_ci(rate: float | None, ci: Sequence[float] | None) -> str:
    return fmt_pct(rate) if rate is None or not ci else f"{fmt_pct(rate)} (95% CI {fmt_ci(ci)})"


def _confusion_table(conf: dict) -> str:
    hp_jp, hp_jf = conf.get("human_pass_judge_pass", 0), conf.get("human_pass_judge_fail", 0)
    hf_jp, hf_jf = conf.get("human_fail_judge_pass", 0), conf.get("human_fail_judge_fail", 0)
    return (
        '<div class="table-wrap" role="region" aria-label="Confusion matrix" tabindex="0"><table class="confusion">'
        '<thead><tr><th scope="col"><span class="sr-only">Human verdict</span></th><th scope="col" class="num">Judge: pass</th><th scope="col" class="num">Judge: fail</th><th scope="col" class="num">Total</th></tr></thead><tbody>'
        f'<tr><th scope="row">Human: pass</th><td class="num diag">{hp_jp}</td><td class="num off">{hp_jf}<small>judge too strict</small></td><td class="num">{hp_jp + hp_jf}</td></tr>'
        f'<tr><th scope="row">Human: fail</th><td class="num off">{hf_jp}<small>judge too lenient</small></td><td class="num diag">{hf_jf}</td><td class="num">{hf_jp + hf_jf}</td></tr>'
        "</tbody></table></div>"
    )


def _disagreement(d: dict) -> str:
    human, judge = _verdict_word(d.get("human")), _verdict_word(d.get("judge"))
    parts = [f"<h4>Input</h4>{_pre(d.get('input'))}", f"<h4>Output</h4>{_pre(d.get('output'), '(empty output)')}"]
    if d.get("judge_reasoning"):
        parts.append(f'<h4>Judge reasoning</h4><p class="reasoning">{_e(d["judge_reasoning"])}</p>')
    if d.get("note"):
        parts.append(f'<h4>Human note</h4><p class="reasoning">{_e(d["note"])}</p>')
    return (
        f'<details class="case"><summary><span class="case-id">{_e(d.get("case_id"))}</span> '
        f'{_badge("human: " + human, "pass" if human == "pass" else "bad")} {_badge("judge: " + judge, "pass" if judge == "pass" else "bad")}</summary>'
        f'<div class="case-body">{"".join(parts)}</div></details>'
    )


def _tnr_flag_text(cal: dict) -> str:
    """Why the true-negative screening check did not pass (`tnr_reliable_reason`), with the thresholds the calibration recorded."""
    rule = cal.get("tnr_reliable_rule") if isinstance(cal.get("tnr_reliable_rule"), dict) else {}
    fails, half, floor = _num(rule.get("min_human_fails")), _num(rule.get("max_half_width")), _num(rule.get("min_tnr"))
    reason = cal.get("tnr_reliable_reason")
    if reason == "few_human_fails":
        return "screening check not passed: " + (f"fewer than {int(fails)} bad outputs labeled" if fails else "too few bad outputs labeled")
    if reason == "wide_ci":
        return "screening check not passed: " + (f"interval wider than ±{half * 100:.0f} points" if half else "interval too wide")
    if reason == "low_tnr":
        return "low: " + (f"the judge misses more than {1 - floor:.0%} of the bad outputs (below {floor:.0%})" if floor else "the judge misses too many of the bad outputs")
    if reason == "undefined":
        return "screening check not passed: interval undefined"
    return "screening check not passed"


def _tnr_screen_meaning(cal: dict) -> str:
    """What the screening flag does and does not say: the calibration's own wording when it recorded one."""
    rule = cal.get("tnr_reliable_rule") if isinstance(cal.get("tnr_reliable_rule"), dict) else {}
    meaning = rule.get("meaning")
    if isinstance(meaning, str) and meaning.strip():
        return meaning
    return (
        "The true-negative screening flag is a screen, not a guarantee: it says the rate is measured narrowly and its estimate is high enough. "
        "It does not show that the true rate is that high."
    )


def _calibration_section(cal: dict, run: RunResult | None = None) -> tuple[str, str, str]:
    n = cal.get("n") or 0
    judge = cal.get("judge") or "overall"
    scope = "the overall case verdict" if judge == "overall" else f"judge {judge}"
    notes = []
    if cal.get("n_excluded_sha_mismatch"):
        notes.append(f"{cal['n_excluded_sha_mismatch']} label(s) were excluded because they describe a different output than this run produced (output_sha mismatch).")
    if cal.get("n_unmatched"):
        notes.append(f"{cal['n_unmatched']} label(s) refer to case ids that are not in this run.")
    notes_html = "".join(f'<p class="muted small">{_e(t)}</p>' for t in notes)
    warnings = [w for w in (cal.get("warnings") or []) if w]
    stale = _stale_calibration_text(cal, run)
    if stale and not any(STALE_CALIBRATION_TEXT in str(w) for w in warnings):
        warnings.insert(0, stale)
    warnings_html = f'<ul class="cal-warnings">{"".join(f"<li>{_e(w)}</li>" for w in warnings)}</ul>' if warnings else ""
    if not n:
        return _section(
            "calibration",
            "Calibration",
            f'<p class="alert">No labeled cases could be matched to this run, so {_e(scope)} is not calibrated.</p>{notes_html}{warnings_html}',
        )

    view = _kappa_view(cal)
    kappa = view["kappa"]
    decision = cal.get("kappa_decision")
    label_word = _decision_word(decision) if decision else cal.get("kappa_label")
    kappa_text = DASH if kappa is None else f"{kappa:.2f} ({label_word})"
    if _valid_range(view["ci"]):
        kappa_text += f", 95% CI {_fmt_range(view['ci'])}"
    if view["weighted"]:
        sample = view["sample"]
        sample_text = DASH if sample is None else f"{sample:.2f}" + (f", 95% CI {_fmt_range(view['sample_ci'])}" if _valid_range(view["sample_ci"]) else "")
        kappa_items = [
            _stat("Cohen's kappa (whole run, weighted)", kappa_text, "the verdict is decided on this one: each labeled stratum counts by its share of the run"),
            _stat("Cohen's kappa (labeled sample, unweighted)", sample_text, "secondary: describes the labels that were drawn, not the run"),
        ]
    else:
        kappa_items = [_stat("Cohen's kappa", kappa_text, "agreement beyond chance")]
    corrected = cal.get("corrected_pass_rate")
    corrected_text = (
        "not estimable"
        if corrected is None
        else f"{fmt_pct(corrected)} (95% CI {fmt_ci(cal.get('corrected_pass_rate_ci'))})"
    )
    tnr_hint = "human-rejected outputs the judge also fails"
    if cal.get("tnr_reliable") is False:
        tnr_hint += "; " + _tnr_flag_text(cal)
    elif cal.get("tnr_reliable") is True:
        tnr_hint += "; screening check passed (a screen, not a guarantee)"
    labelable = _num(cal.get("n_labelable"))
    labelable_hint = f"of {int(labelable)} labelable in this run (cases with an error cannot be labeled)" if labelable is not None and labelable >= 0 else ""
    agreement_items = [_stat("Agreement", fmt_pct(cal.get("agreement")), "cases where judge and human agree (labeled sample)")]
    if _num(cal.get("agreement_weighted")) is not None:
        agreement_items.append(_stat("Agreement, weighted to the run", fmt_pct(cal.get("agreement_weighted")), "labeled strata re-weighted to their share of the run"))
    stats = "".join(
        [
            _stat("Calibrated", scope),
            _stat("Labeled cases", fmt_int(n), labelable_hint),
            *agreement_items,
            *kappa_items,
            _stat("True-positive rate", _rate_with_ci(cal.get("tpr"), cal.get("tpr_ci")), "human-approved outputs the judge also passes"),
            _stat("True-negative rate", _rate_with_ci(cal.get("tnr"), cal.get("tnr_ci")), tnr_hint),
            _stat("Pass rate, human labels", fmt_pct(cal.get("human_pass_rate"))),
            _stat("Pass rate, judge (same cases)", fmt_pct(cal.get("judge_pass_rate_labeled"))),
            _stat("Observed pass rate (run)", fmt_pct(cal.get("run_observed_pass_rate"))),
            _stat("Calibrated pass rate", corrected_text, "observed rate corrected for judge error (Rogan–Gladen)"),
        ]
    )
    decision_html = ""
    if decision:
        extra = ""
        needed = _num(cal.get("labels_needed_estimate"))
        labelable_n = _num(cal.get("n_labelable"))
        min_release = _num(cal.get("min_labels_release"))
        if decision != "trustworthy":
            if cal.get("labels_needed_capped") is True and labelable_n:
                if n >= labelable_n:
                    extra = f" All {int(labelable_n)} labelable cases are already labeled: add cases or repeats for more precision."
                else:
                    extra = (
                        f" This run has only {int(labelable_n)} labelable cases, fewer than the labels needed for a ±0.1 interval: "
                        f"label all {int(labelable_n)}; add cases or repeats for more precision."
                    )
            elif needed is not None:
                extra = f" About {int(needed)} labeled cases would narrow the kappa interval to ±0.1."
            if min_release and n < min_release:
                extra += f" At least {int(min_release)} labeled cases are needed before a judge is released (usable or trustworthy)."
        decision_html = (
            f'<div class="callout decision {_DECISION_TONE.get(str(decision), "warn")}"><strong>Judge verdict: {_badge(_decision_word(decision), _DECISION_TONE.get(str(decision), "warn"))}</strong> '
            f"{_e(_DECISION_TEXT.get(str(decision), ''))}{_e(extra)}</div>"
        )
    body = f"{decision_html}{warnings_html}<dl class=\"stats\">{stats}</dl>{_confusion_table(cal.get('confusion') or {})}{notes_html}"
    if cal.get("tnr_reliable") in (True, False):
        body += f'<p class="muted small">{_e(_tnr_screen_meaning(cal))}</p>'
    if corrected is not None:
        body += (
            '<p class="muted small">The interval on the calibrated pass rate describes the population this test set was sampled from, '
            "not only these exact cases, so it stays wide even when every case is labeled.</p>"
        )
    if view["weighted"]:
        body += (
            '<p class="muted small">The verdict rests on the weighted kappa, which counts each labeled stratum by its share of the run. '
            "The unweighted kappa only describes the sample of labels that was drawn.</p>"
        )
    if cal.get("recommendation"):
        body += f'<div class="callout"><strong>Recommendation.</strong> {_e(cal["recommendation"])}</div>'
    disagreements = cal.get("disagreements") or []
    if disagreements:
        shown = disagreements[:MAX_DISAGREEMENTS]
        hidden = len(disagreements) - len(shown)
        body += f"<h3>Disagreements ({len(disagreements)})</h3>" + "".join(_disagreement(d) for d in shown)
        if hidden:
            body += f'<p class="alert">{hidden} more disagreement(s) not shown.</p>'
    else:
        body += '<p class="ok-note"><span aria-hidden="true">✔</span> Judge and humans agree on every labeled case.</p>'
    return _section("calibration", "Calibration", body)


# ---------------------------------------------------------------------- cost and tokens
def _monthly_projection(run: RunResult, comparison: dict | None, baseline: RunResult | None, monthly_requests: int) -> str:
    cost = run.summary.get("cost") or {}
    per_request = _num(cost.get("system_mean_usd"))
    n_priced = cost.get("n_priced") or 0
    if per_request is None or not n_priced:
        return f'<h3>Monthly projection</h3><p class="alert">No cost data, so no projection at {monthly_requests:,} requests per month.</p>'
    monthly = per_request * monthly_requests
    base_mean = _baseline_mean_cost(baseline, comparison)
    headers = [("Run", ""), ("Per request", "num"), (f"Per month ({monthly_requests:,} requests)", "num")]
    rows = [["<strong>This run</strong>", fmt_money(per_request), fmt_money(monthly)]]
    if base_mean is not None:
        base_monthly = base_mean * monthly_requests
        headers.append(("Change per month", "num"))
        rows[0].append(_signed_money(monthly - base_monthly))
        rows.insert(0, ["Baseline", fmt_money(base_mean), fmt_money(base_monthly), DASH])
    body = f"<h3>Monthly projection</h3>{_table('Monthly cost projection', headers, rows)}"
    body += (
        '<p class="muted small">Projection = mean system cost per request measured on this test set x monthly requests. '
        "It assumes production traffic looks like the test set and leaves out retrieval, hosting and evaluation costs, so treat it as an estimate.</p>"
    )
    if n_priced < (run.summary.get("n_cases") or n_priced):
        body += '<p class="alert">Not every request could be priced, so the projection is understated.</p>'
    return body


def _signed_money(value: float) -> str:
    if abs(value) < 0.00005:
        return "no change"
    sign = "+" if value > 0 else MINUS
    return f"{sign}{fmt_money(abs(value))}"


def _cost_section(run: RunResult, comparison: dict | None, baseline: RunResult | None = None, monthly_requests: int | None = None) -> tuple[str, str, str]:
    s = run.summary
    n = s.get("n_cases") or 0
    cost = s.get("cost") or {}
    n_priced = cost.get("n_priced") or 0
    judge_total = cost.get("judge_total_usd")
    system_total = cost.get("system_total_usd") if n_priced else None

    def per(total: float | None, count: int, scale: int = 1) -> float | None:
        return total / count * scale if total and count else None

    cost_rows = [
        ["<strong>System</strong> (feature under test)", fmt_money(system_total), fmt_money(cost.get("system_mean_usd")), fmt_money(cost.get("system_per_1k_usd"))],
        ["<strong>Evaluation</strong> (judges)", fmt_money(judge_total), fmt_money(per(judge_total, n)), fmt_money(per(judge_total, n, 1000))],
    ]
    body = _table("Cost", [("Cost", ""), ("Total", "num"), ("Per request", "num"), ("Per 1k requests", "num")], cost_rows)
    if n_priced == 0:
        body += '<p class="alert">No cost data: the target reported neither <code>cost_usd</code> nor token usage for a priced model.</p>'
    elif n_priced < n:
        body += f'<p class="alert">Only {n_priced} of {n} requests could be priced; system cost is understated.</p>'
    body += '<p class="muted small">System cost is what the feature costs in production. Evaluation cost is spent once to grade this run and is not part of production spend.</p>'

    if monthly_requests:
        body += _monthly_projection(run, comparison, baseline, monthly_requests)

    cmp_cost = (comparison or {}).get("cost") or {}
    if cmp_cost.get("baseline_mean_usd") is not None and cmp_cost.get("candidate_mean_usd") is not None:
        body += (
            f'<p>Mean cost per request: {fmt_money(cmp_cost["candidate_mean_usd"])} vs {fmt_money(cmp_cost["baseline_mean_usd"])} '
            f'in the baseline {_delta_pct(cmp_cost.get("delta_pct"), False)}</p>'
        )

    tokens = s.get("tokens") or {}
    token_rows = [
        [label, fmt_int(tokens.get(key)), fmt_int((tokens.get(key) or 0) / n) if n else DASH]
        for label, key in (("Input (uncached)", "input"), ("Output", "output"), ("Cache read", "cache_read"), ("Cache write", "cache_write"))
    ]
    body += "<h3>Tokens</h3>" + _table("Tokens", [("Type", ""), ("Total", "num"), ("Per request", "num")], token_rows)

    models = sorted((s.get("models") or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    if models:
        model_rows = [[f"<code>{_e(m)}</code>", fmt_int(c), fmt_pct(c / n) if n else DASH] for m, c in models]
        body += "<h3>Models used by the target</h3>" + _table("Models", [("Model", ""), ("Requests", "num"), ("Share", "num")], model_rows)
    return _section("cost", "Cost and tokens", body)


# ------------------------------------------------------------------------------- CSS
_DARK_VARS = (
    "--bg:#0e1116;--surface:#171b22;--surface-2:#1e232b;--text:#e7eaef;--muted:#9ba5b3;--border:#2c333d;"
    "--accent:#8db1ea;"
    "--pass:#6ddba0;--pass-bg:#11271b;--fail:#ff9d92;--fail-bg:#331816;--warn:#f2c46c;--warn-bg:#2c2410;"
    "--bar:#3fb676;--track:#2c333d;--shadow:none;"
    "--viz-pass:#3987e5;--viz-cost:#d95926;"
)

_CSS_TEMPLATE = """
:root{
  color-scheme:light dark;
  --bg:#f4f5f7;--surface:#fff;--surface-2:#eef0f3;--text:#161a20;--muted:#57606d;--border:#d5dae1;
  --accent:#2f4f86;
  --pass:#0d6437;--pass-bg:#e2f3e9;--fail:#a3271a;--fail-bg:#fbe9e7;--warn:#7a4a00;--warn-bg:#fff2cf;
  --bar:#25874c;--track:#e3e6eb;
  --viz-pass:#2a78d6;--viz-cost:#eb6834;
  --radius:10px;--shadow:0 1px 2px rgba(16,24,40,.06);
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
@media screen and (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){@@DARK@@}
}
:root[data-theme="dark"]{color-scheme:dark;@@DARK@@}
:root[data-theme="light"]{color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 var(--font)}
.page{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.65rem;line-height:1.2;margin:.1rem 0 .4rem;overflow-wrap:anywhere}
h2{font-size:1.15rem;margin:0 0 .75rem}
h3{font-size:1rem;margin:1.25rem 0 .5rem}
h4{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:.9rem 0 .25rem}
p{margin:.5rem 0}
a{color:var(--accent);text-decoration:underline;text-underline-offset:2px}
a:focus-visible,summary:focus-visible,.table-wrap:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
code{font-family:var(--mono);font-size:.86em;overflow-wrap:anywhere}
small{display:block;color:var(--muted);font-size:.75rem;font-weight:400}
.muted{color:var(--muted)}.small{font-size:.82rem}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.eyebrow{margin:0;font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.run-line{margin:0 0 .75rem;color:var(--muted)}
dl{margin:0}dd{margin:0}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px 24px}
.meta dt{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.meta dd{overflow-wrap:anywhere}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;margin:16px 0 0;font-size:.9rem}
.chip{display:inline-block;margin:1px 4px 1px 0;padding:0 7px;background:var(--surface-2);border:1px solid var(--border);border-radius:6px;font-family:var(--mono);font-size:.8rem;overflow-wrap:anywhere}
.chip-key{color:var(--muted)}
.verdict{border:1px solid;border-radius:var(--radius);padding:14px 16px;margin:20px 0 0}
.verdict.ok{background:var(--pass-bg);border-color:var(--pass)}
.verdict.bad{background:var(--fail-bg);border-color:var(--fail)}
.verdict-title{font-size:1.2rem;font-weight:700}
.verdict.ok .verdict-title{color:var(--pass)}.verdict.bad .verdict-title{color:var(--fail)}
.verdict ul{margin:.4rem 0 .2rem 1.1rem;padding:0}
.verdict-notes{font-size:.88rem}
.verdict-note{margin:.4rem 0 0;font-size:.88rem}
.gate{margin-top:.6rem;font-size:.9rem}.gate summary{cursor:pointer}.gate .meta{margin-top:.5rem}
.kpis{margin-top:20px}
h2.group{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;min-width:0;box-shadow:var(--shadow)}
.tile-label{font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.tile-value{font-size:1.65rem;font-weight:650;line-height:1.25;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.tile-sub{font-size:.8rem;color:var(--muted)}
.delta{font-size:.82rem;font-weight:650}
.tile .delta{display:block;margin-top:4px}
.delta.good{color:var(--pass)}.delta.bad{color:var(--fail)}.delta.flat{color:var(--muted)}
.delta-note{font-weight:400;color:var(--muted)}
.findings{margin:0;padding:12px 16px 12px 32px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)}
.findings li{margin:2px 0}
section{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin:16px 0;box-shadow:var(--shadow)}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:middle;white-space:nowrap}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
th[scope=row]{font-weight:600}
.num{text-align:right}
td.wrap{white-space:normal}
.bar-cell{width:100%;min-width:100px}
.bar{position:relative;height:10px;background:var(--track);border-radius:5px}
.bar-fill{position:absolute;top:0;bottom:0;left:0;background:var(--bar);border-radius:5px}
.bar-ci{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--text);opacity:.6}
.bar-ci::before,.bar-ci::after{content:"";position:absolute;top:-3px;width:2px;height:8px;background:inherit}
.bar-ci::before{left:0}.bar-ci::after{right:0}
.badge{display:inline-block;padding:0 8px;border-radius:999px;border:1px solid currentColor;font-size:.72rem;font-weight:650;white-space:nowrap}
.badge.bad{background:var(--fail-bg);color:var(--fail)}
.badge.warn{background:var(--warn-bg);color:var(--warn)}
.badge.pass{background:var(--pass-bg);color:var(--pass)}
.badge.neutral{background:var(--surface-2);color:var(--muted)}
.ok-note{color:var(--pass);font-weight:600}
.sim-banner,.health{margin:16px 0 0;padding:10px 14px;border:1px solid var(--warn);border-left-width:6px;border-radius:var(--radius);background:var(--warn-bg);color:var(--text)}
.health.bad{border-color:var(--fail);background:var(--fail-bg)}
.sbs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:10px;margin:.25rem 0 .5rem}
.pane{min-width:0}.pane-head{font-size:.8rem;font-weight:650;color:var(--muted);margin-bottom:2px}
.cal-warnings{margin:.5rem 0 .75rem 1.1rem;padding:0;color:var(--warn);overflow-wrap:anywhere}
.callout.decision.pass{border-left-color:var(--pass)}.callout.decision.warn{border-left-color:var(--warn)}.callout.decision.bad{border-left-color:var(--fail)}
.alert{padding:8px 12px;background:var(--warn-bg);color:var(--text);border-left:4px solid var(--warn);border-radius:4px}
.callout{margin:12px 0;padding:10px 14px;background:var(--surface-2);border-left:4px solid var(--accent);border-radius:4px}
.idlist{list-style:none;margin:.25rem 0;padding:0}.idlist li{padding:4px 0;border-bottom:1px solid var(--border)}
details.case{border:1px solid var(--border);border-radius:8px;margin:8px 0;background:var(--surface)}
details.case>summary{cursor:pointer;padding:10px 12px}
.case-id{font-family:var(--mono);font-weight:650;overflow-wrap:anywhere}
.case-body{padding:0 12px 12px;border-top:1px solid var(--border)}
details summary{cursor:pointer}
pre{margin:.25rem 0 .5rem;padding:10px;background:var(--surface-2);border-radius:6px;font:.82rem/1.45 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto}
.judge-block{margin:6px 0 10px;padding:8px 12px;border-left:4px solid var(--fail);background:var(--surface-2);border-radius:4px}
.criteria{list-style:none;margin:.4rem 0 0;padding:0}.criteria li{margin:3px 0}
.mark{display:inline-block;min-width:4.4em;font-weight:650}
.crit-fail .mark{color:var(--fail)}.crit-pass .mark{color:var(--pass)}.crit-pass{color:var(--muted)}
.reasoning{margin:.25rem 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px 20px;margin-bottom:16px}
.stats dt{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.stats dd{font-weight:600;overflow-wrap:anywhere}
.confusion{width:auto}
.confusion .diag{font-weight:700}.confusion .off small{white-space:normal;max-width:8em;margin-left:auto}
footer{margin-top:24px;font-size:.8rem;color:var(--muted)}
@media(max-width:480px){.page{padding:16px 16px 32px}section{padding:12px}.tile-value{font-size:1.4rem}}
@@HISTORY_CSS@@
@media print{
  :root{--shadow:none}
  body{font-size:11pt}
  .toc{display:none}
  section,.tile,.verdict,details.case{break-inside:avoid}
  pre{max-height:none;overflow:visible}
  .table-wrap{overflow:visible}
  a{color:inherit;text-decoration:none}
}
@media print{details::details-content{content-visibility:visible;display:block}}
"""

_HISTORY_CSS = """
.chart{max-width:760px}
.trend{display:block;width:100%;height:auto;overflow:visible}
.chart-narrow{display:none}
.trend-fig{margin:0 0 20px}
.trend-fig figcaption{font-size:.82rem;color:var(--muted);margin-top:2px}
.trend-title{margin:0 0 4px;font-size:.95rem;font-weight:650}
.trend text{font:11px var(--font);fill:var(--muted)}
.trend .tip text{fill:var(--text)}
.trend .tip .tip-strong{font-weight:650}
.trend .grid{stroke:var(--border);stroke-width:1}
.trend .axis{stroke:var(--muted);stroke-width:1;opacity:.55}
.trend .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.trend .line.pass,.trend .whisker.pass{stroke:var(--viz-pass)}
.trend .line.cost{stroke:var(--viz-cost)}
.trend .band{stroke:none;opacity:.14}
.trend .band.pass{fill:var(--viz-pass)}
.trend .whisker{stroke-width:2;opacity:.55}
.trend .dot{stroke:var(--surface);stroke-width:2}
.trend .dot.pass{fill:var(--viz-pass)}
.trend .dot.cost{fill:var(--viz-cost)}
.trend .chg{stroke:var(--text);stroke-width:1;opacity:.55}
.trend .chg-mark{fill:var(--text)}
.trend text.chg-label{fill:var(--text);font-weight:600}
.trend .pt{outline:none}
.trend .hit{fill:transparent;pointer-events:all}
.trend .xh,.trend .ring,.trend .tip{opacity:0;pointer-events:none}
.trend .xh{stroke:var(--muted);stroke-width:1}
.trend .ring{fill:none;stroke-width:2}
.trend .ring.pass{stroke:var(--viz-pass)}
.trend .ring.cost{stroke:var(--viz-cost)}
.trend .tip-box{fill:var(--surface);stroke:var(--border);stroke-width:1}
.trend .pt:hover .xh,.trend .pt:focus .xh,.trend .pt:hover .ring,.trend .pt:focus .ring,.trend .pt:hover .tip,.trend .pt:focus .tip{opacity:1}
.trend .pt:focus-visible .hit{stroke:var(--accent);stroke-width:2}
.hist tbody td:first-child{white-space:normal;min-width:8.5rem;max-width:12rem;overflow-wrap:anywhere}
.hist td small{font-size:.72rem}
@media(min-width:561px){.hist thead th:first-child,.hist tbody td:first-child{position:sticky;left:0;z-index:1;background:var(--surface)}}
@media(max-width:560px){.chart-wide{display:none}.chart-narrow{display:block}}
"""

_CSS = _CSS_TEMPLATE.replace("@@DARK@@", _DARK_VARS).replace("@@HISTORY_CSS@@", _HISTORY_CSS)


# ------------------------------------------------------------------------- public API
def render_report(
    run: RunResult,
    comparison: dict | None = None,
    calibration: dict | None = None,
    baseline: RunResult | None = None,
    monthly_requests: int | None = None,
) -> str:
    """Render one run as a single self-contained HTML document.

    `baseline` (a RunResult) adds the baseline output next to the candidate output for every
    newly failing case; `monthly_requests` adds a monthly cost projection.
    """
    s = run.summary
    monthly = _valid_monthly(monthly_requests)
    newly = _newly_failing_ids(run, comparison, baseline)
    failures = _failing_records(run, comparison, newly)
    shown_ids = {r["case_id"] for r in failures[:MAX_FAILURES]}
    sections = [
        sec
        for sec in (
            _tags_section(run, comparison),
            _stability_section(run, shown_ids),
            _judges_section(run),
            _errors_section(run, comparison),
            _regressions_section(run, comparison, shown_ids, baseline) if comparison else None,
            _failures_section(run, comparison, failures, newly, baseline),
            _calibration_section(calibration, run) if calibration else None,
            _cost_section(run, comparison, baseline, monthly),
        )
        if sec
    ]
    toc = "".join(f'<a href="#{anchor}">{_e(title)}</a>' for anchor, title, _ in sections)
    baseline_banner = _baseline_placeholder_banner(comparison, baseline) if comparison else ""
    bullets = _findings(run, comparison, calibration, failures)
    findings = (
        f'<div class="kpis"><h2 class="group">Key findings</h2><ul class="findings">{"".join(f"<li>{b}</li>" for b in bullets)}</ul></div>'
        if bullets
        else ""
    )
    title = " · ".join(["Eval report", str(s.get("suite") or "run"), str(s.get("label") or s.get("run_id") or "")]).rstrip(" ·")
    kit = s.get("kit_version") or s.get("evalsprint_version")
    version = f" {kit}" if kit else ""
    footer = f"Generated by the Eval Sprint kit{_e(version)}. Per-case data: results.jsonl in the run directory."
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class=\"page\">"
        f'{_header(run)}{_placeholder_banner(run)}{_simulated_banner(run)}<nav class="toc" aria-label="Sections">{toc}</nav>'
        f"{_health_banner(run, comparison)}{_oversize_banner(run)}"
        f"{baseline_banner}"
        f"{_verdict_banner(comparison, bool(baseline_banner)) if comparison else ''}"
        f"{_kpis(run, comparison, calibration, monthly)}{findings}"
        f'{"".join(html_ for _, _, html_ in sections)}'
        f"<footer>{footer}</footer></div></body></html>"
    )


def write_report(
    path: str | Path,
    run: RunResult,
    comparison: dict | None = None,
    calibration: dict | None = None,
    baseline: RunResult | None = None,
    monthly_requests: int | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run, comparison, calibration, baseline, monthly_requests), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ history (C15)
def _hist_moment(iso: Any) -> datetime | None:
    if not isinstance(iso, str) or not iso:
        return None
    try:
        moment = datetime.fromisoformat(iso[:-1] + "+00:00" if iso.endswith("Z") else iso)
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _release_series(version: Any) -> tuple[int, int] | None:
    match = re.match(r"\s*v?(\d+)\.(\d+)", version) if isinstance(version, str) else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def _repeat_key(summary: dict) -> tuple | None:
    """The repeat protocol as a comparable key: `(n, (("safety", 5), ...))`, cycle-3 `tags` folded into `per_tag`; None when the run does not say."""
    protocol = _repeat_protocol(summary.get("repeat"))
    return None if protocol is None else (protocol[0], tuple(protocol[1].items()))


def _hist_cost(value: Any) -> float | None:
    cost = _num(value)
    return cost if cost is not None and cost >= 0 else None


def _hist_money(value: float | None) -> str:
    return "$0.00" if value == 0 else fmt_money(value)


def _hist_point(run: Any) -> dict | None:
    """One row of the history, read defensively from a run's summary; None when the object has no summary at all."""
    summary = getattr(run, "summary", None)
    if not isinstance(summary, dict):
        return None
    if not isinstance(getattr(run, "results", None), list):
        run = SimpleNamespace(summary=summary, results=[])
    counts = _counts(run)
    ci = summary.get("pass_rate_ci")
    lo = hi = None
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and _num(ci[0]) is not None and _num(ci[1]) is not None:
        lo, hi = float(ci[0]), float(ci[1])
    rate = _num(summary.get("pass_rate"))
    if rate is None and counts["scored"]:
        rate = counts["passed"] / counts["scored"]
    if rate is not None:
        rate = min(max(rate, 0.0), 1.0)  # a garbage value must not draw outside the axis
    if lo is not None:
        lo, hi = sorted((min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0)))
    stability = _stability(run)
    attempts = _attempt_counts(run)
    cost = summary.get("cost") if isinstance(summary.get("cost"), dict) else {}
    latency = summary.get("latency") if isinstance(summary.get("latency"), dict) else {}
    sha = summary.get("judges_sha")
    kit = summary.get("kit_version") or summary.get("evalsprint_version")
    run_id = str(summary.get("run_id") or "")
    protocol = _repeat_protocol(summary.get("repeat"))
    return {
        "run_id": run_id,
        "label": str(summary.get("label") or run_id or "(unnamed run)"),
        "moment": _hist_moment(summary.get("created_at")),
        "suite": str(summary.get("suite") or ""),
        "rate": rate,
        "ci": (lo, hi) if lo is not None else None,
        "scored": counts["scored"],
        "cases": counts["cases"],
        "errors": counts["errors"],
        "attempt_rate": attempts["rate"] if attempts else None,
        "attempt_errors": attempts["errors"] if attempts else None,
        "flip": stability["flip_rate"] if stability and stability["n_repeated"] else None,
        "cost": _hist_cost(cost.get("system_per_1k_usd")),
        "p95": _num(latency.get("p95_s")),
        "sha": sha if isinstance(sha, str) and sha else None,
        "kit": str(kit) if kit else None,
        "repeat": _repeat_key(summary),
        "repeat_text": _protocol_phrase(protocol) if protocol is not None else None,
        "simulated": _is_simulated(run),
        "placeholder": _is_placeholder(summary),
        "retries": _target_retries(run),
        "oversize": _n_oversize(run),
        "resumed": bool(summary.get("resumed_from")),
    }


def _hist_order(points: list[dict]) -> list[dict]:
    """Oldest first by created_at (a run without a readable date goes last), then by run id."""
    far = datetime.max.replace(tzinfo=timezone.utc)
    ordered = sorted(points, key=lambda p: (p["moment"] is None, p["moment"] or far, p["run_id"]))
    for i, p in enumerate(ordered):
        prev = ordered[i - 1] if i else None
        p["judges_changed"] = bool(prev and prev["sha"] and p["sha"] and prev["sha"] != p["sha"])
        p["protocol_changed"] = bool(prev and prev["repeat"] is not None and p["repeat"] is not None and prev["repeat"] != p["repeat"])
        a, b = (_release_series(prev["kit"]), _release_series(p["kit"])) if prev else (None, None)
        p["kit_changed"] = bool(a and b and a != b)
        ra, rb = (_kit_release(prev["kit"]), _kit_release(p["kit"])) if prev else (None, None)
        p["kit_patch_changed"] = bool(a and b and a == b and ra != rb)  # same release series, another patch
        p["previous"] = prev
    return ordered


def _nice_ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    span = hi - lo if hi > lo else (abs(hi) or 1.0)
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if raw <= m * magnitude + 1e-12)
    first = math.floor(lo / step + 1e-9) * step
    ticks, value = [], first
    while value <= hi + step * 1e-6 or not ticks:
        ticks.append(round(value, 10))
        value += step
    if ticks[-1] < hi - 1e-9:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _hist_money_tick(value: float, step: float) -> str:
    decimals = 0 if step >= 1 else 1 if step >= 0.1 else 2 if step >= 0.01 else 4
    return f"${value:.{decimals}f}"


def _chart_label(label: str) -> str:
    """The run label for an axis tick: the `-SIMULATED` suffix is in the banner and the table, not on every tick."""
    return label[: -len("-SIMULATED")] if label.endswith("-SIMULATED") and len(label) > len("-SIMULATED") else label


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _trend_svg(points: list[dict], kind: str, wide: bool) -> str | None:
    """Inline SVG of one measure over the runs (`kind` is "pass" or "cost"), or None when no run has that measure.

    One series, one axis, hairline grid, a 2px line, 9px markers with a 2px surface ring. The pass-rate line is drawn in
    separate pieces on either side of a judge change (the numbers are not comparable across it); the cost line continues.
    A vertical rule with a diamond marks each point where `judges_sha` changed. Hover and keyboard focus show the readout
    (pure CSS: the page has no JavaScript); every value is also in the table below.
    """
    n = len(points)
    if kind == "pass":
        values = [None if p["rate"] is None else p["rate"] * 100 for p in points]
        los = [None if p["ci"] is None else p["ci"][0] * 100 for p in points]
        his = [None if p["ci"] is None else p["ci"][1] * 100 for p in points]
        known = [v for v in values if v is not None]
        if not known:
            return None
        data_lo = min([v for v in los if v is not None] + known)
        data_hi = max([v for v in his if v is not None] + known)
        lo, hi = max(0.0, data_lo - 5), min(100.0, data_hi + 5)
        if hi - lo < 20:  # a flat history still gets a readable axis
            mid = (hi + lo) / 2
            lo, hi = mid - 10, mid + 10
            lo, hi = (0.0, 20.0) if lo < 0 else (80.0, 100.0) if hi > 100 else (lo, hi)
        ticks = [t for t in _nice_ticks(lo, hi) if -1e-9 <= t <= 100 + 1e-9]
    else:
        values = [None if p["cost"] is None else p["cost"] for p in points]
        los = his = [None] * n
        known = [v for v in values if v is not None]
        if not known:
            return None
        ticks = _nice_ticks(0.0, max(known) * 1.1 if max(known) > 0 else 1.0)
    y0, y1 = ticks[0], ticks[-1]
    if y1 <= y0:
        y1 = y0 + 1
    step_value = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0

    width, height = (720, 250) if wide else (320, 232)
    ml, mr, mt, mb = (50, 16, 26, 32) if wide else (42, 12, 26, 32)
    plot_w, plot_h = width - ml - mr, height - mt - mb
    step = plot_w / n

    def x_of(i: int) -> float:
        return ml + (i + 0.5) * step

    def y_of(v: float) -> float:
        return mt + plot_h * (1 - (v - y0) / (y1 - y0))

    def tick_text(v: float) -> str:
        return f"{v:g}%" if kind == "pass" else _hist_money_tick(v, step_value)

    parts = []
    for t in ticks:
        y = y_of(t)
        parts.append(f'<line class="grid" x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{_e(tick_text(t))}</text>')
    parts.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{mt + plot_h}" y2="{mt + plot_h}"/>')

    # x labels: the run labels, a few of them
    max_labels = 8 if wide else 4
    shown = list(range(n)) if n <= max_labels else sorted({round(j * (n - 1) / (max_labels - 1)) for j in range(max_labels)})
    # the room one label has: the distance to the next labelled run (about 6.8 px a character); a label pushed inwards by the
    # edge of the chart is shortened until it clears its neighbour, and dropped if it cannot (the table below has every run)
    spacing = step * min([b - a for a, b in zip(shown, shown[1:])] or [n])
    limit = max(4, min(14, int(spacing / 6.8)))
    prev_end = -1e9
    for i in shown:
        label, x = _chart_label(points[i]["label"]), x_of(i)
        for room in range(limit, 2, -1):
            text = _short(label, room)
            half = 3.4 * len(text)
            if x - half < 2:
                anchor, tx, left, right = "start", max(2.0, x - 4), max(2.0, x - 4), max(2.0, x - 4) + 2 * half
            elif x + half > width - 2:
                anchor, tx, left, right = "end", min(width - 2.0, x + 4), min(width - 2.0, x + 4) - 2 * half, min(width - 2.0, x + 4)
            else:
                anchor, tx, left, right = "middle", x, x - half, x + half
            if left >= prev_end + 4:
                parts.append(f'<text class="tick" x="{tx:.1f}" y="{height - 10}" text-anchor="{anchor}">{_e(text)}</text>')
                prev_end = right
                break

    # pieces of the line: a missing value, and (pass rate only) a judge change, end a piece
    pieces: list[list[int]] = []
    current: list[int] = []
    for i, p in enumerate(points):
        if values[i] is None or (kind == "pass" and p["judges_changed"] and current):
            if current:
                pieces.append(current)
                current = []
        if values[i] is not None:
            current.append(i)
    if current:
        pieces.append(current)
    for piece in pieces:
        if kind == "pass" and all(los[i] is not None and his[i] is not None for i in piece):
            if len(piece) > 1:
                upper = " ".join(f"{x_of(i):.1f},{y_of(his[i]):.1f}" for i in piece)
                lower = " ".join(f"{x_of(i):.1f},{y_of(los[i]):.1f}" for i in reversed(piece))
                parts.append(f'<polygon class="band {kind}" points="{upper} {lower}"/>')
            else:
                i = piece[0]
                parts.append(f'<line class="whisker {kind}" x1="{x_of(i):.1f}" x2="{x_of(i):.1f}" y1="{y_of(his[i]):.1f}" y2="{y_of(los[i]):.1f}"/>')
        if len(piece) > 1:
            parts.append(f'<polyline class="line {kind}" points="{" ".join(f"{x_of(i):.1f},{y_of(values[i]):.1f}" for i in piece)}"/>')

    # where the judges changed
    last_label_end = -1e9
    for i, p in enumerate(points):
        if not p["judges_changed"]:
            continue
        x = ml + i * step
        parts.append(f'<line class="chg" x1="{x:.1f}" x2="{x:.1f}" y1="{mt - 6}" y2="{mt + plot_h}"/>')
        parts.append(f'<path class="chg-mark" d="M{x:.1f},{mt - 15} l4.5,4.5 l-4.5,4.5 l-4.5,-4.5 z"/>')
        label = "judges changed"
        room_right = width - mr - (x + 8)
        start_x, anchor = (x + 8, "start") if room_right >= 6.2 * len(label) else (x - 8, "end")
        span_lo, span_hi = (start_x, start_x + 6.2 * len(label)) if anchor == "start" else (start_x - 6.2 * len(label), start_x)
        if span_lo > last_label_end + 6 and span_lo >= 0:
            parts.append(f'<text class="chg-label" x="{start_x:.1f}" y="{mt - 7}" text-anchor="{anchor}">{label}</text>')
            last_label_end = span_hi

    # dots first, then the hover layer, so a readout is always painted over every dot
    dots, layer = [], []
    for i, p in enumerate(points):
        v = values[i]
        if v is None:
            continue
        x, y = x_of(i), y_of(v)
        dots.append(f'<circle class="dot {kind}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>')
        lines = [p["label"]]
        if kind == "pass":
            ci = f" (95% CI {fmt_pct(p['ci'][0], 0)} to {fmt_pct(p['ci'][1], 0)})" if p["ci"] else ""
            lines += [f"Pass rate {fmt_pct(p['rate'])}{ci}", f"{p['scored']} scored cases · {p['errors']} errors"]
        else:
            lines += [f"{_hist_money(p['cost'])} per 1k requests", f"p95 latency {fmt_latency(p['p95'])}" if p["p95"] is not None else "p95 latency not recorded"]
        lines.append(_fmt_date(None if p["moment"] is None else p["moment"].isoformat()))
        if p["judges_changed"]:
            lines.append("◆ judges changed at this run")
        lines = [_short(line, 44) for line in lines]
        box_w = max(len(line) for line in lines) * 6.0 + 18
        box_h = 15 * len(lines) + 10
        bx = min(max(x - box_w / 2, 2.0), width - 2 - box_w)
        by = mt + 6 if y > mt + plot_h / 2 else mt + plot_h - box_h - 6
        texts = "".join(
            f'<text class="{"tip-strong" if k == 0 else ""}" x="{bx + 9:.1f}" y="{by + 18 + 15 * k:.1f}">{_e(line)}</text>' for k, line in enumerate(lines)
        )
        aria = "; ".join(lines)
        layer.append(
            f'<g class="pt" tabindex="0" role="img" aria-label="{_e(aria)}">'
            f'<rect class="hit" x="{ml + i * step:.1f}" y="{mt}" width="{step:.1f}" height="{plot_h}"/>'
            f'<line class="xh" x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{mt + plot_h}"/>'
            f'<circle class="ring {kind}" cx="{x:.1f}" cy="{y:.1f}" r="8"/>'
            f'<g class="tip"><rect class="tip-box" x="{bx:.1f}" y="{by:.1f}" width="{box_w:.1f}" height="{box_h}" rx="4"/>{texts}</g></g>'
        )
    title = "Pass rate by run" if kind == "pass" else "Cost per 1,000 requests by run"
    return (
        f'<svg class="trend {"chart-wide" if wide else "chart-narrow"}" viewBox="0 0 {width} {height}" role="group" aria-label="{_e(title)}" '
        f'focusable="false">{"".join(parts)}{"".join(dots)}{"".join(layer)}</svg>'
    )


def _hist_delta_pp(p: dict) -> str:
    prev = p["previous"]
    if prev is None:
        return DASH
    if p["judges_changed"]:
        return _badge("not comparable", "warn")
    if p["rate"] is None or prev["rate"] is None:
        return DASH
    return _delta_pp(p["rate"] - prev["rate"], True, compact=True) or DASH


def _hist_notes(p: dict) -> str:
    notes = []
    if p["judges_changed"]:
        notes.append(_badge("judges changed", "warn"))
    if p["protocol_changed"]:
        notes.append(_badge("repeat protocol changed", "warn"))
    if p["kit_changed"]:
        notes.append(_badge("kit version changed", "warn"))
    elif p["kit_patch_changed"]:
        notes.append(_badge("kit patch changed", "warn"))
    if p["placeholder"]:
        notes.append(_badge("placeholder", "warn"))
    if p["simulated"]:
        notes.append(_badge("simulated", "neutral"))
    if p["retries"]:
        notes.append(_badge(f"{fmt_int(p['retries'])} target retries", "neutral"))
    if p["oversize"]:
        notes.append(_badge(f"{fmt_int(p['oversize'])} oversize", "neutral"))
    if p["resumed"]:
        notes.append(_badge("resumed", "neutral"))
    return " ".join(notes) or DASH


def _hist_rows(points: list[dict]) -> list[list[str]]:
    rows = []
    for p in points:
        ci = DASH if p["ci"] is None else fmt_ci(p["ci"])
        rows.append(
            [
                f"<strong>{_e(p['label'])}</strong>" + (f'<small><code>{_e(p["run_id"])}</code></small>' if p["run_id"] and p["run_id"] != p["label"] else ""),
                _e(_fmt_date(None if p["moment"] is None else p["moment"].isoformat())),
                fmt_pct(p["rate"]),
                _hist_delta_pp(p),
                ci,
                fmt_int(p["errors"]),
                DASH if p["attempt_rate"] is None else fmt_pct(p["attempt_rate"]),
                DASH if p["flip"] is None else fmt_pct(p["flip"]),
                _hist_money(p["cost"]),
                fmt_latency(p["p95"]),
                f"<code>{_e(p['sha'][:12])}</code>" if p["sha"] else DASH,
                _e(p["kit"]) if p["kit"] else DASH,
                _e(p["repeat_text"]) if p["repeat_text"] else DASH,
                _hist_notes(p),
            ]
        )
    return rows


_HISTORY_HEADERS = [
    ("Run", ""),
    ("Date", ""),
    ("Pass rate", "num"),
    ("Change vs previous run", ""),
    ("95% CI", "num"),
    ("Errors", "num"),
    ("Attempt error rate", "num"),
    ("Flip rate", "num"),
    ("Cost / 1k requests", "num"),
    ("p95 latency", "num"),
    ("Judges fingerprint", ""),
    ("Kit version", ""),
    ("Repeat", ""),
    ("Notes", "wrap"),
]


def _hist_tiles(points: list[dict]) -> str:
    latest = points[-1]
    prev = latest["previous"]
    dates = [p["moment"] for p in points if p["moment"]]
    span = f"{_fmt_date(min(dates).isoformat())} to {_fmt_date(max(dates).isoformat())}" if dates else ""
    changes = sum(1 for p in points if p["judges_changed"])
    pass_delta = ""
    pass_sub = ["95% CI " + fmt_ci(latest["ci"])] if latest["ci"] else []
    if prev is not None and latest["judges_changed"]:
        pass_sub.append("the judges changed since the previous run: not comparable")
    elif prev is not None and latest["rate"] is not None and prev["rate"] is not None:
        pass_delta = _delta_pp(latest["rate"] - prev["rate"], True, compact=True)
        pass_sub.append("vs the previous run")
    cost_delta, cost_sub = "", []
    if prev is not None and latest["cost"] is not None and prev["cost"]:
        pct = (latest["cost"] - prev["cost"]) / prev["cost"] * 100
        cost_delta = _delta(pct, _signed(pct, "%"), False, compact=True)
        cost_sub.append("vs the previous run")
    tiles = [
        _tile("Runs", fmt_int(len(points)), [span]),
        _tile("Latest pass rate", fmt_pct(latest["rate"]), pass_sub, pass_delta),
        _tile("Latest cost / 1k requests", _hist_money(latest["cost"]), cost_sub, cost_delta),
        _tile("Judge changes", fmt_int(changes), ["each one breaks the comparison of the runs around it" if changes else "no run changed the judges"]),
    ]
    return f'<div class="kpis"><div class="tiles">{"".join(tiles)}</div></div>'


def _hist_figure(points: list[dict], kind: str) -> str:
    wide, narrow = _trend_svg(points, kind, True), _trend_svg(points, kind, False)
    title, unit = (
        ("Pass rate", "Line: pass rate on the scored cases. Band: 95% Wilson confidence interval.")
        if kind == "pass"
        else ("Cost per 1,000 requests", "Cost of the feature under test, judging excluded. It does not depend on the judges, so the line continues across a judge change.")
    )
    if wide is None or narrow is None:
        return f'<figure class="trend-fig"><p class="trend-title">{title}</p><p class="muted">No run recorded {"a pass rate" if kind == "pass" else "a cost"}.</p></figure>'
    return f'<figure class="trend-fig"><p class="trend-title">{title}</p><div class="chart">{wide}{narrow}</div><figcaption>{unit}</figcaption></figure>'


def render_history(runs: Iterable[RunResult]) -> str:
    """One self-contained HTML page with the trend of a list of runs: quality and cost over time.

    The page is ordered by each run's `created_at`. Per run it lists the label, date, pass rate with its interval, errors,
    the attempt error rate, the flip rate, cost per 1k requests, p95 latency, the judges fingerprint (`judges_sha`) and the
    kit version. Two small inline SVG charts (pass rate with its confidence band, and cost per 1k requests) mark the runs
    where `judges_sha` changed, because pass rates on either side of that point come from different judges and are not
    comparable. No JavaScript, no external request; every string that comes from the runs is escaped.
    """
    items = list(runs)
    points = [pt for pt in (_hist_point(r) for r in items) if pt is not None]
    skipped = len(items) - len(points)
    ordered = _hist_order(points)
    suites = sorted({p["suite"] for p in ordered if p["suite"]})
    head_suite = suites[0] if len(suites) == 1 else ("several suites" if suites else "")
    title = "Eval history" + (f" · {head_suite}" if head_suite else "")

    if not ordered:
        body = (
            f'<header><p class="eyebrow">Eval Sprint · History</p><h1>{_e(title)}</h1></header>'
            '<section><p class="alert">No run to show. Point <code>evalsprint history</code> at a directory with finished runs.</p>'
            + (f'<p class="muted small">{skipped} item(s) without a summary were skipped.</p>' if skipped else "")
            + "</section>"
        )
        return _history_document(title, body)

    notes = []
    simulated = sum(1 for p in ordered if p["simulated"])
    if simulated:
        notes.append(
            f'<div class="sim-banner" role="note"><strong>{simulated} of {len(ordered)} run(s) here are simulated.</strong> Their outputs come from a simulator, '
            "not from a language model: the trend shows that the pipeline works, not how a real model performs.</div>"
        )
    if len(suites) > 1:
        notes.append(
            f'<div class="health" role="note"><strong>This page mixes {len(suites)} suites</strong> ({_e(", ".join(suites))}). Their test sets differ, '
            "so pass rates of different suites are not comparable; filter with <code>evalsprint history --suite NAME</code>.</div>"
        )
    placeholders = sum(1 for p in ordered if p["placeholder"])
    if placeholders:
        notes.append(
            f'<div class="sim-banner placeholder-banner" role="note"><strong>{placeholders} of {len(ordered)} run(s) here are placeholders</strong> (the scaffold\'s offline stub as target, '
            "or an LLM judge on the mock provider): their verdicts say nothing about the real product, so they are not a trend of quality.</div>"
        )
    unknown = sum(1 for p in ordered if not p["sha"])
    if unknown:
        notes.append(
            f'<div class="health" role="note"><strong>{unknown} run(s) carry no judges fingerprint</strong> (written before evalsprint 0.2), so a change of judges '
            "cannot be detected around them.</div>"
        )
    if skipped:
        notes.append(f'<div class="health" role="note">{skipped} item(s) without a summary were skipped.</div>')

    chart_points = ordered[-MAX_HISTORY_CHART_RUNS:]
    if len(chart_points) and chart_points[0] is not ordered[0]:
        chart_points[0] = {**chart_points[0], "judges_changed": False}  # its predecessor is not drawn: no rule at the left edge
    table_points = ordered[-MAX_HISTORY_ROWS:]
    trend_intro = (
        '<p class="muted small"><span aria-hidden="true">◆</span> A vertical rule marks a run whose judges differ from the run before it (<code>judges_sha</code> changed): '
        "pass rates on either side are not comparable, so the pass-rate line is cut there. Hover or focus a point to read it; every value is in the table below.</p>"
    )
    if len(ordered) > len(chart_points):
        trend_intro += f'<p class="muted small">The charts show the latest {len(chart_points)} of {len(ordered)} runs.</p>'
    trend = _section("trend", "Trend", trend_intro + _hist_figure(chart_points, "pass") + _hist_figure(chart_points, "cost"))
    table_note = (
        '<p class="muted small">Oldest first. Errors are cases without a verdict and never count as failures; the attempt error rate counts every attempt of a repeated case. '
        "The flip rate is the share of repeated cases whose attempts disagreed. Cost is the feature under test per 1,000 requests.</p>"
    )
    if len(ordered) > len(table_points):
        table_note += f'<p class="muted small">The table lists the latest {len(table_points)} of {len(ordered)} runs.</p>'
    runs_section = _section("runs", "Runs", _table("Runs", _HISTORY_HEADERS, _hist_rows(table_points)).replace('class="table-wrap"', 'class="table-wrap hist"', 1) + table_note)
    sections = [trend, runs_section]
    toc = "".join(f'<a href="#{anchor}">{_e(name)}</a>' for anchor, name, _ in sections)
    first, last = ordered[0], ordered[-1]
    run_line = f"{len(ordered)} run(s)"
    if first["moment"] and last["moment"]:
        run_line += f" · {_e(_fmt_date(first['moment'].isoformat()))} to {_e(_fmt_date(last['moment'].isoformat()))}"
    body = (
        f'<header><p class="eyebrow">Eval Sprint · History</p><h1>{_e(title)}</h1><p class="run-line">{run_line}</p></header>'
        f'{"".join(notes)}<nav class="toc" aria-label="Sections">{toc}</nav>'
        f"{_hist_tiles(ordered)}{''.join(html_ for _, _, html_ in sections)}"
        "<footer>Generated by the Eval Sprint kit. One row per run; per-case data is in each run's results.jsonl.</footer>"
    )
    return _history_document(title, body)


def _history_document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f'<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class="page">{body}</div></body></html>'
    )


def write_history(path: str | Path, runs: Iterable[RunResult]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_history(runs), encoding="utf-8")
    return path
