"""Run a suite: cases -> target -> judges -> results.jsonl + summary.json.

Run directory layout (the contract every other module reads):

    <runs_dir>/<run_id>/
        results.jsonl          one CaseResult.to_dict() per line, dataset order
        summary.json           output of `summarize()` plus run metadata
        results.partial.jsonl  only while a run is in progress (or after it was interrupted):
                               one line per finished case attempt, so a crash never loses paid work
        run_meta.json          written when the run starts: the run-level fingerprint (judges_sha, target callable, params and
                               source file sha, repeat protocol) that `run --resume` compares before reusing any record
        .resume.lock           only while a `run --resume DIR` reads DIR: a second concurrent resume of DIR is refused

Both final files are written atomically (temp file + rename, retried when Windows holds the file), so a reader
never sees half a run. `run --resume DIR` continues an interrupted run: records of DIR whose case_sha and fingerprint
match are reused, everything else runs.

Every case ends as one of three outcomes:

    pass / fail   the case was scored
    error         the target or a judge failed (outage, timeout, unparsable judge reply)

Errors say nothing about quality, so they are kept out of every pass rate and counted separately. A target attempt that
errors is first called again up to `target_retries` times (default 1) inside the run: a transient 429 or timeout does not
turn into a failure of the gate, and the retried calls are paid, counted in the cost and reported (`n_target_retries`).

Run-level keys of summary.json that describe the protocol of the run (a gate compares them between baseline and candidate):

    repeat          {"n": int, "per_tag": {tag: int}}  the effective repeat protocol
    target_retries  int  the configured `target_retries` (0..5), always written, also when no call was retried;
                    `n_target_retries` is how many calls were actually repeated and `n_attempts` how many attempts ran
    selection       {"limited": bool, "n_selected": int, "n_config": int, "limit": int | None, "tags": [str] | None, "ids": [str] | None}
                    the part of the suite this run covers: `limited` is True when it ran fewer cases than the config selects
                    (`--limit`, `--tags`, `ids`); `n_config` counts every case of the dataset the config points at
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import MAX_REPEAT, SuiteConfig, parse_repeat, redact_secrets
from .dataset import filter_cases, load_cases
from .judges import build_judges, run_judges, validate_pass_rule
from .pricing import PriceTable
from .stats import mean, percentile, wilson_ci
from .targets import Target
from .types import OUTCOMES, Case, CaseResult, JudgeResult, TargetOutput, Usage, derive_outcome, sanitize_obj

PARTIAL_NAME = "results.partial.jsonl"
RUN_META_NAME = "run_meta.json"
LOCK_NAME = ".resume.lock"
_SLEEP = time.sleep  # patched by tests: the retry backoff below would otherwise cost seconds


@dataclass
class RunResult:
    run_dir: Path | None
    summary: dict
    results: list[dict]

    @property
    def run_id(self) -> str:
        return self.summary.get("run_id", "")

    def by_case(self) -> dict[str, dict]:
        return {r["case_id"]: r for r in self.results}


def _pass_rule(rule, judge_results) -> bool:
    if not judge_results:
        return False
    if rule == "all":
        return all(j.passed for j in judge_results)
    if rule == "any":
        return any(j.passed for j in judge_results)
    names = [rule] if isinstance(rule, str) else list(rule)
    selected = [j for j in judge_results if j.judge in names]
    return bool(selected) and all(j.passed for j in selected)


# ------------------------------------------------------------------ outcomes and repeats
def outcome_of(record: dict) -> tuple[str, str | None]:
    """(outcome, error_source) of a result record; records written before outcomes existed are derived
    from `passed`, the target error and the judge errors."""
    outcome = record.get("outcome")
    if outcome in OUTCOMES:
        source = record.get("error_source")
        if outcome == "error" and source not in ("target", "judge"):
            source = derive_outcome((record.get("output") or {}).get("error"), record.get("judges") or [], False)[1] or "judge"
        return outcome, (source if outcome == "error" else None)
    return derive_outcome((record.get("output") or {}).get("error"), record.get("judges") or [], bool(record.get("passed")))


def majority_outcome(outcomes: list[str]) -> str:
    """Verdict of a repeated case. Errors at least half of the attempts -> `error` (no reliable signal);
    otherwise the majority of pass/fail among the scored attempts, and a tie is a `fail` (conservative)."""
    n = len(outcomes)
    if outcomes.count("error") * 2 >= n:
        return "error"
    return "pass" if outcomes.count("pass") > outcomes.count("fail") else "fail"


def _judge_cost(judges) -> float:
    return sum((j.cost_usd if isinstance(j, JudgeResult) else (j.get("cost_usd") or 0.0)) or 0.0 for j in judges)


def _count_oversize(judges) -> int:
    """Judge results that refused an output above `max_output_chars` without calling the model (`details.oversize`, C18)."""
    total = 0
    for j in judges or []:
        details = j.details if isinstance(j, JudgeResult) else (j.get("details") if isinstance(j, dict) else None)
        total += bool(isinstance(details, dict) and details.get("oversize"))
    return total


def _non_negative(value) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and value > 0 else 0


def _attempt_entry(res: CaseResult) -> dict:
    outcome, source = res.resolved()
    return {
        "outcome": outcome,
        "passed": outcome == "pass",
        "output_sha": res.output.sha,
        "error_source": source,
        "cost_usd": res.output.cost_usd,
        "judge_cost_usd": _judge_cost(res.judges),
        "target_retries_used": int(res.target_retries_used or 0),
        "n_oversize": _count_oversize(res.judges),
    }


def _finish_repeated(rep: CaseResult, entries: list[dict], verdict: str) -> CaseResult:
    """`rep` (an attempt whose outcome is the majority) as the record of the repeated case whose attempts are `entries`."""
    outcomes = [e.get("outcome") for e in entries]
    rep.outcome, rep.error_source = rep.resolved()
    rep.passed = verdict == "pass"
    rep.attempts = entries
    rep.stability = outcomes.count(verdict) / len(outcomes)
    rep.target_retries_used = sum(_non_negative(e.get("target_retries_used")) for e in entries)  # every attempt's retries, not only the representative's
    return rep


def _combine_attempts(results: list[CaseResult]) -> CaseResult:
    """One CaseResult for a repeated case: the first attempt whose outcome equals the majority represents it."""
    if len(results) == 1:
        return results[0]
    outcomes = [r.resolved()[0] for r in results]
    verdict = majority_outcome(outcomes)
    rep = next(r for r in results if r.resolved()[0] == verdict)
    return _finish_repeated(rep, [_attempt_entry(r) for r in results], verdict)


# ------------------------------------------------------------------------- summary
def _rate(n_pass: int, n: int) -> dict:
    lo, hi = wilson_ci(n_pass, n)
    return {"n": n, "n_passed": n_pass, "pass_rate": (n_pass / n) if n else None, "ci": [lo, hi]}


def summarize(results: list[dict]) -> dict:
    """Pure function: aggregate metrics from result records (dicts).

    `pass_rate` is passed / scored, where scored = passed + failed: cases whose target or judge errored
    are excluded (they are reported in n_errors / error_rate instead of being counted as quality failures).
    """
    n = len(results)
    outcomes = [outcome_of(r) for r in results]
    n_pass = sum(1 for o, _ in outcomes if o == "pass")
    n_fail = sum(1 for o, _ in outcomes if o == "fail")
    n_err = n - n_pass - n_fail
    n_target = sum(1 for o, s in outcomes if o == "error" and s == "target")
    n_judge = n_err - n_target
    n_scored = n_pass + n_fail

    by_tag_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r, (o, _) in zip(results, outcomes):
        if o == "error":
            continue
        for tag in r["case"].get("tags") or ["(untagged)"]:
            by_tag_counts[tag][0] += 1
            by_tag_counts[tag][1] += int(o == "pass")
    by_tag = {t: _rate(p, k) for t, (k, p) in sorted(by_tag_counts.items())}

    judge_rows: dict[str, list[dict]] = defaultdict(list)
    for r, (o, source) in zip(results, outcomes):
        if o == "error" and source == "target":
            continue  # the judges were skipped: their placeholder rows are not judge results
        for j in r["judges"]:
            judge_rows[j["judge"]].append(j)
    by_judge = {}
    for name, rows in judge_rows.items():
        scored = [j for j in rows if not j.get("error")]
        entry = _rate(sum(1 for j in scored if j["passed"]), len(scored))
        entry["mean_score"] = mean([j["score"] for j in scored if j.get("score") is not None])
        entry["n_errors"] = len(rows) - len(scored)
        entry["cost_usd"] = sum(j.get("cost_usd") or 0.0 for j in rows)  # errored calls are still billed
        by_judge[name] = entry

    sys_costs = [r["output"].get("cost_usd") for r in results if r["output"].get("cost_usd") is not None]
    judge_total = 0.0
    for r in results:
        attempts = r.get("attempts")
        judge_total += sum(a.get("judge_cost_usd") or 0.0 for a in attempts) if attempts else _judge_cost(r["judges"])
    sys_total = sum(sys_costs)
    sys_mean = (sys_total / len(sys_costs)) if sys_costs else None
    latencies = [r["output"].get("latency_s") for r in results if r["output"].get("latency_s") is not None]
    tokens = Counter()
    for r in results:
        for k, v in (r["output"].get("usage") or {}).items():
            tokens[k] += v or 0
    models = Counter(r["output"].get("model") or "unknown" for r in results)

    repeated = [r for r in results if r.get("attempts")]
    flips = sum(1 for r in repeated if len({a.get("outcome") for a in r["attempts"]}) > 1)
    overall = _rate(n_pass, n_scored)

    # attempt level (C10): a crash in ONE attempt of a repeated case does not change the majority outcome, so it is
    # invisible in n_errors / error_rate; these counters see every attempt (a case that ran once is one attempt)
    n_attempts = n_attempt_target = n_attempt_judge = 0
    for r, (o, source) in zip(results, outcomes):
        attempts = r.get("attempts")
        if attempts:
            n_attempts += len(attempts)
            for a in attempts:
                if a.get("outcome") == "error":
                    if (a.get("error_source") or "judge") == "target":
                        n_attempt_target += 1
                    else:
                        n_attempt_judge += 1
        else:
            n_attempts += 1
            if o == "error":
                if source == "target":
                    n_attempt_target += 1
                else:
                    n_attempt_judge += 1
    n_attempt_errors = n_attempt_target + n_attempt_judge

    # C18: target calls repeated after a crash, and judge results that refused an oversize output. A repeated case's record keeps the
    # totals of ALL its attempts (`target_retries_used`) and its attempt entries the per-attempt counts (`n_oversize`).
    n_retries = n_oversize = 0
    for r in results:
        attempts = r.get("attempts") if isinstance(r.get("attempts"), list) else None
        if r.get("target_retries_used") is not None:
            n_retries += _non_negative(r["target_retries_used"])
        elif attempts:
            n_retries += sum(_non_negative(a.get("target_retries_used")) for a in attempts if isinstance(a, dict))
        if attempts and any(isinstance(a, dict) and "n_oversize" in a for a in attempts):
            n_oversize += sum(_non_negative(a.get("n_oversize")) for a in attempts if isinstance(a, dict))
        else:
            n_oversize += _count_oversize(r.get("judges"))
    return {
        "n_cases": n,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "n_scored": n_scored,
        "pass_rate": overall["pass_rate"],
        "pass_rate_ci": overall["ci"],
        "n_errors": n_err,
        "n_target_errors": n_target,
        "n_judge_errors": n_judge,
        "error_rate": (n_err / n) if n else 0.0,
        "n_attempts": n_attempts,
        "n_attempt_errors": n_attempt_errors,
        "n_attempt_target_errors": n_attempt_target,
        "n_attempt_judge_errors": n_attempt_judge,
        "attempt_error_rate": (n_attempt_errors / n_attempts) if n_attempts else 0.0,
        "n_target_retries": n_retries,
        "n_oversize": n_oversize,
        "stability": {"n_repeated": len(repeated), "flip_rate": (flips / len(repeated)) if repeated else None},
        "by_tag": by_tag,
        "by_judge": by_judge,
        "cost": {
            "system_total_usd": sys_total,
            "system_mean_usd": sys_mean,
            "system_per_1k_usd": sys_mean * 1000 if sys_mean is not None else None,
            "judge_total_usd": judge_total,
            "n_priced": len(sys_costs),
        },
        "latency": {
            "mean_s": mean(latencies),
            "p50_s": percentile(latencies, 50),
            "p95_s": percentile(latencies, 95),
            "max_s": max(latencies) if latencies else None,
        },
        "tokens": {
            "input": tokens["input_tokens"],
            "output": tokens["output_tokens"],
            "cache_read": tokens["cache_read_tokens"],
            "cache_write": tokens["cache_write_tokens"],
        },
        "models": dict(models),
    }


_redact = redact_secrets  # kept under its old name; the rules live in config.redact_secrets


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:40]


def make_run_id(label: str | None = None, now: datetime | None = None) -> str:
    """`YYYYMMDD-HHMMSS-ffffff[-label]` (UTC, microseconds so two runs started in the same second differ)."""
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S-%f")
    slug = _slug(label) if label else ""
    return f"{ts}-{slug}" if slug else ts


def _claim_run_dir(runs_dir: Path, run_id: str) -> tuple[str, Path]:
    """Create `<runs_dir>/<run_id>`; an existing directory is never reused (a numeric suffix is added)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, 1000):
        rid = run_id if k == 1 else f"{run_id}-{k}"
        path = runs_dir / rid
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return rid, path
    raise RuntimeError(f"could not create a fresh run directory under {runs_dir}")


def _dataset_ref(config: SuiteConfig) -> str:
    """The dataset path as stored in summary.json: relative to the eval.yaml directory, never absolute
    (an absolute path would put the user's home directory into the committed baseline)."""
    try:
        return Path(os.path.relpath(config.dataset, config.base_dir)).as_posix()
    except ValueError:  # another drive on Windows
        return config.dataset.name


def _is_run_summary(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "run_id" in data


def _is_run_results(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            first = next((line for line in fh if line.strip()), "")
        record = json.loads(first) if first else {}
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and "case_id" in record


def _clear_previous_run(run_dir: Path) -> None:
    """`run --out DIR` onto a directory that holds an earlier run: drop that run's summary.json and results.jsonl
    BEFORE the new run starts. Otherwise an interrupted run leaves the old finished results next to the new
    results.partial.jsonl, and every consumer reads the old run as if it were this one.

    Only files that are evalsprint's own are deleted: a summary.json / results.jsonl that is not a run file
    means --out points at somebody else's directory, and that is an error, raised before anything is spent.
    summary.json goes first, so a crash in between leaves a directory that reads as unfinished, never as complete.
    """
    _check_previous_run(run_dir)
    for name in ("summary.json", "results.jsonl"):
        unlink_with_retry(run_dir / name)


def _check_previous_run(run_dir: Path) -> None:
    """The refusal half of `_clear_previous_run`: raise when summary.json / results.jsonl in `run_dir` are not evalsprint's."""
    for name, is_ours in (("summary.json", _is_run_summary), ("results.jsonl", _is_run_results)):
        path = run_dir / name
        if path.exists() and not is_ours(path):
            raise ValueError(f"--out {run_dir} already contains a {name} that is not an evalsprint run file; refusing to overwrite it (use an empty directory)")


def check_run_files(run_dir: str | Path) -> None:
    """Refuse a directory whose results.partial.jsonl is newer than its summary.json: a newer run started in it
    (or is still running), so summary.json and results.jsonl describe an older run."""
    run_dir = Path(run_dir)
    partial, summary = run_dir / PARTIAL_NAME, run_dir / "summary.json"
    try:
        newer = partial.stat().st_mtime_ns > summary.stat().st_mtime_ns
    except OSError:  # one of them is missing: the caller reports that
        return
    if newer:
        raise ValueError(
            f"{run_dir} is being overwritten by a newer run or that run was interrupted: {PARTIAL_NAME} is newer than summary.json, "
            "so summary.json and results.jsonl belong to an older run; run it again (into a clean directory)"
        )


def _source_text(path: Path) -> str | None:
    """The text of a source or rubric file for hashing: CRLF and LF checkouts of the same file must hash alike, and so
    must a copy an editor saved with a UTF-8 BOM (the bytes changed, the file did not). None when it cannot be read."""
    try:
        return path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    except OSError:
        return None


def _rubric_text(path: Path) -> str:
    text = _source_text(path)
    return "<unreadable>" if text is None else text  # build_judges already loaded it; only a race with a deletion gets here


def _file_digest(path: Path) -> str:
    """sha256 (16 hex) of a judge's data file: read as text (BOM dropped, CRLF normalised, like the sources) when it is UTF-8,
    as raw bytes otherwise. `<unreadable>` when it cannot be read (build_judges has already checked that it exists)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return "<unreadable>"
    try:
        raw = raw.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()[:16]


def _callable_source(spec: str, base_dir: str | Path) -> str | None:
    """The source text of the file behind a `file.py:function` or `package.module:function` spec, or None when it
    cannot be resolved (a compiled module, a namespace package, a spec that names no file)."""
    if not isinstance(spec, str) or ":" not in spec:
        return None
    module_ref = spec.rsplit(":", 1)[0]
    if module_ref.lower().endswith(".py"):
        return _source_text(Path(base_dir) / module_ref)
    module = sys.modules.get(module_ref)  # build_judges imported it already: this is the file that is really used
    origin = getattr(module, "__file__", None)
    if not origin:
        try:
            found = importlib.util.find_spec(module_ref)
        except (ImportError, ValueError, AttributeError):
            found = None
        origin = getattr(found, "origin", None)
    if origin and str(origin).lower().endswith(".py"):
        return _source_text(Path(origin))
    return None


def judges_fingerprint(judge_cfgs: list, pass_rule, base_dir: str | Path, warnings: list[str] | None = None) -> tuple[str, list]:
    """(judges_sha, judges_config) of a suite.

    `judges_sha` is the first 12 hex chars of the sha256 of the canonical JSON (sort_keys, default=str) of
    {"judges": <the judges list after env expansion and --set overrides, secrets masked>,
     "rubrics": {<rubric path as written in the config, or "inline:<judge name>">: <the file text, or the inline dict>,
                 "custom:<judge name>": <the source text of that custom judge's callable file>},
     "pass_rule": <pass_rule>}.
    Files are read relative to the config directory (CRLF normalised, BOM dropped). For a `module:function` spec the
    module's source file is hashed when it can be resolved; otherwise the spec string is hashed and a warning is
    appended to `warnings` (a change of that code then goes unseen). Secrets are masked BEFORE hashing, so rotating an
    API key does not read as a changed judge. `judges_config` is that masked list, stored in summary.json so a reviewer
    can see what changed when the sha differs. A custom judge's `data_files: [path, ...]` (relative to `base_dir`) are hashed too,
    under "data_files:<judge name>" (path -> digest of the file). What is NOT covered: files a custom judge imports but does not
    list, the target's code, and the judge model's own behaviour behind an unchanged model name.
    """
    masked = redact_secrets(list(judge_cfgs or []))
    rubrics: dict = {}
    for cfg in judge_cfgs or []:
        if not isinstance(cfg, dict):
            continue
        if cfg.get("type") == "llm":
            ref = cfg.get("rubric")
            if isinstance(ref, dict):
                rubrics[f"inline:{cfg.get('name') or 'llm'}"] = ref
            elif isinstance(ref, str) and ref:
                rubrics[ref] = _rubric_text(Path(base_dir) / ref)
        elif cfg.get("type") == "custom":
            name, spec = cfg.get("name") or "custom", cfg.get("callable")
            text = _callable_source(spec, base_dir)
            if text is None:
                text = f"<unresolved:{spec}>"  # only the spec string is hashed
                if warnings is not None:
                    warnings.append(f"custom judge {name!r}: the source file of {spec!r} could not be found, so editing that code will NOT change judges_sha; use a 'file.py:function' spec")
            rubrics[f"custom:{name}"] = text
            files = cfg.get("data_files")
            if files:  # C20: what the function reads counts as part of the judge (only added when there are any: no data_files, no change)
                files = [files] if isinstance(files, str) else list(files)
                rubrics[f"data_files:{name}"] = {str(item): _file_digest(Path(base_dir) / str(item)) for item in files}
    blob = json.dumps({"judges": masked, "rubrics": rubrics, "pass_rule": pass_rule}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12], masked


def _attempts_of(case: Case, n: int, per_tag: dict[str, int]) -> int:
    """How many times a case runs: the LARGEST `per_tag` count among its tags, else `n` (C17)."""
    counts = [per_tag[t] for t in case.tags if t in per_tag]
    return max(counts) if counts else n


def _repeat_plan(config: SuiteConfig, cases: list[Case], repeat) -> tuple[dict[str, int], dict]:
    """({case_id: attempts} for the repeated cases (2 or more), the effective repeat protocol).

    The protocol is what `summary["repeat"]` records: {"n": int, "per_tag": {tag: attempts}} (sorted keys, `{}` when no tag has a
    count of its own; `n: 1` = every case without a `per_tag` tag runs once). `run --repeat N` (an int) sets `n` and KEEPS the
    config's `per_tag`; a dict ({"n", "per_tag"} or the old {"n", "tags"}) replaces the whole protocol. It is the DECLARED
    protocol: a run limited to cases that no `per_tag` tag matches still declares it, so two runs of the same suite compare as
    the same protocol."""
    if repeat is None:
        spec = config.repeat
    elif isinstance(repeat, dict):
        spec = parse_repeat({"n": 1, **repeat}, "repeat")
    else:
        if isinstance(repeat, bool) or int(repeat) != repeat or repeat < 1:
            raise ValueError("repeat must be at least 1")
        if repeat > MAX_REPEAT:
            raise ValueError(f"repeat is limited to {MAX_REPEAT} attempts per case, got {repeat}")
        spec = {"n": int(repeat), "per_tag": dict((config.repeat or {}).get("per_tag") or {})}
    n, per_tag = (spec["n"], dict(spec.get("per_tag") or {})) if spec else (1, {})
    if n < 1:
        raise ValueError("repeat must be at least 1")
    if n > MAX_REPEAT or any(k > MAX_REPEAT for k in per_tag.values()):
        raise ValueError(f"repeat is limited to {MAX_REPEAT} attempts per case, got {max([n, *per_tag.values()])}")
    plan = {c.id: k for c in cases if (k := _attempts_of(c, n, per_tag)) > 1}
    return plan, {"n": n, "per_tag": dict(sorted(per_tag.items()))}


# ------------------------------------------------------------------------------ resume
def target_fingerprint(config: SuiteConfig) -> dict:
    """The target as `summary["target"]` and the run fingerprint record it: {callable, params (secrets masked), source_sha}.

    `source_sha` (C20) is the first 12 hex chars of the sha256 of the SOURCE FILE behind `callable` (CRLF normalised, BOM dropped),
    None when it cannot be resolved. A changed adapter therefore changes the fingerprint, and `run --resume` reuses nothing. Not
    covered: the files that adapter imports."""
    text = _callable_source(config.target["callable"], config.base_dir)
    return {
        "callable": config.target["callable"],
        "params": redact_secrets(config.target.get("params", {})),
        "source_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text is not None else None,
    }


def run_fingerprint(config: SuiteConfig, judges_sha: str, protocol: dict, target: dict | None = None) -> dict:
    """What must be identical for records of an earlier run to be reusable: the judges (judges_sha: config, rubrics,
    custom-judge code and data files, pass_rule), the target (callable, params with secrets masked, and the sha of the
    callable's source file) and the repeat protocol."""
    fingerprint = {
        "judges_sha": judges_sha,
        "target": target if target is not None else target_fingerprint(config),
        "repeat": protocol,
    }
    return json.loads(json.dumps(fingerprint, sort_keys=True, default=str))  # plain JSON types: comparable with what a file holds


def placeholder_reason(config: SuiteConfig) -> str | None:
    """Why this run proves the pipeline and measures nothing (C18), or None. True when the target callable is the scaffold's
    `offline_stub` (recognised by the function name: `init` writes it in `target_adapter.py`) or an LLM judge is configured with the
    provider `mock` (`JUDGE_PROVIDER=mock`). Both are read from the configuration after `${VAR}` expansion and `--set` overrides."""
    spec = str(config.target.get("callable") or "")
    if spec.rsplit(":", 1)[-1].strip() == "offline_stub":
        return f"the target is the scaffold's offline_stub ({spec}), which answers with canned text"
    for judge in config.judges:
        if isinstance(judge, dict) and judge.get("type") == "llm" and str(judge.get("provider", "anthropic")).strip().lower() == "mock":
            return f"the LLM judge {judge.get('name') or 'llm'!r} runs on the mock provider (JUDGE_PROVIDER=mock), which does not judge"
    return None


def _read_jsonl_lenient(path: Path) -> tuple[list[dict], int]:
    """(records, number of lines that could not be read). The last line of a partial file may be cut short by the crash."""
    records, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    return records, bad


def _entry_cost(entry: dict) -> float:
    """USD spent on one attempt entry of a repeated case (target plus judges)."""
    return float((entry.get("cost_usd") or 0.0) + (entry.get("judge_cost_usd") or 0.0)) if isinstance(entry, dict) else 0.0


def _record_cost(record: dict) -> float:
    """USD spent on one result record: every attempt of a repeated case, or the target plus the judges of a single attempt."""
    attempts = record.get("attempts")
    if isinstance(attempts, list) and attempts:
        return float(sum(_entry_cost(a) for a in attempts))
    spent = (record.get("output") or {}).get("cost_usd") or 0.0
    return float(spent) + float(_judge_cost(record.get("judges") or []))


@dataclass
class _Patched:
    """A finished repeated case whose CRASHED attempts run again (`--retry-errors`, C20): the attempts that worked are kept (their
    entries in `record["attempts"]`; `record` itself holds the complete result of ONE of them, the representative), `fresh` are
    attempts that already ran again in an interrupted resume, and `rerun` are the attempt indexes that still have to run."""

    record: dict
    rerun: list[int]
    fresh: dict  # attempt index -> record of one attempt (from a loose line of results.partial.jsonl)


@dataclass
class _Resume:
    """Records of an earlier run that this run reuses instead of paying for again."""

    source_run_id: str
    attempts: dict  # (case_id, attempt index | None) -> record of ONE attempt
    combined: dict  # case_id -> record of a finished repeated case (all of its attempts in one record)
    patched: dict = field(default_factory=dict)  # case_id -> _Patched
    retried_errors: int = 0  # errored records (and crashed attempts of finished repeated cases) that run again
    discarded_cost_usd: float = 0.0
    mismatch: list[str] | None = None  # fingerprint parts that differ: nothing was reused

    @property
    def n_reused(self) -> int:
        return len(self.attempts) + len(self.combined) + len(self.patched)


_MISMATCH_PARTS = ("judges_sha", "target", "target_source", "repeat")


def _fingerprint_parts(fp: dict) -> dict:
    """The comparable parts of a fingerprint: the target split into what it calls (callable, params) and the source file it runs."""
    target = fp.get("target") if isinstance(fp.get("target"), dict) else {}
    return {
        "judges_sha": fp.get("judges_sha"),
        "target": {k: v for k, v in target.items() if k != "source_sha"},
        "target_source": target.get("source_sha"),
        "repeat": fp.get("repeat"),
    }


def _stored_fingerprint(run_dir: Path) -> tuple[dict | None, str]:
    """(fingerprint, run_id) recorded by an earlier run: run_meta.json, else derived from a finished summary.json."""
    meta_path, summary_path = run_dir / RUN_META_NAME, run_dir / "summary.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            meta = None
        if isinstance(meta, dict) and isinstance(meta.get("fingerprint"), dict):
            return json.loads(json.dumps(meta["fingerprint"], sort_keys=True, default=str)), str(meta.get("run_id") or run_dir.name)
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            summary = None
        if isinstance(summary, dict) and summary.get("judges_sha") and isinstance(summary.get("target"), dict):
            repeat = summary.get("repeat")
            if isinstance(repeat, dict) and isinstance(repeat.get("per_tag"), dict):
                protocol = {"n": repeat.get("n", 1), "per_tag": dict(sorted(repeat["per_tag"].items()))}
            else:  # evalsprint 0.1 / cycle 3: {"n": 5, "tags": [...]} is {"n": 1, "per_tag": {tag: 5}}
                n = repeat.get("n", 1) if isinstance(repeat, dict) else 1
                tags = repeat.get("tags") if isinstance(repeat, dict) and isinstance(n, int) and n > 1 else None
                protocol = {"n": 1, "per_tag": {t: n for t in sorted(tags)}} if tags else {"n": n if isinstance(n, int) else 1, "per_tag": {}}
            fingerprint = {
                "judges_sha": summary["judges_sha"],
                "target": {
                    "callable": summary["target"].get("callable"),
                    "params": summary["target"].get("params", {}),
                    "source_sha": summary["target"].get("source_sha"),
                },
                "repeat": protocol,
            }
            return json.loads(json.dumps(fingerprint, sort_keys=True, default=str)), str(summary.get("run_id") or run_dir.name)
    return None, run_dir.name


def _load_resume(resume_from: str | Path, cases: list[Case], plan: dict[str, int], fingerprint: dict, retry_errors: bool = False) -> _Resume:
    """Read what `run --resume DIR` can reuse: records whose case_sha equals the current case's AND whose run
    fingerprint equals this run's. Reads only (the new run directory is claimed afterwards, so `--out DIR --resume DIR` works).

    With `retry_errors` the records whose outcome is `error` are not reused, and neither are the CRASHED ATTEMPTS inside a finished
    repeated case whose majority verdict still stands (C20): only those attempts run again, the others are kept."""
    run_dir = Path(resume_from)
    if not run_dir.is_dir():
        raise ValueError(f"--resume {run_dir} is not a directory")
    partial, results = run_dir / PARTIAL_NAME, run_dir / "results.jsonl"
    source = partial if partial.is_file() else results if results.is_file() else None
    if source is None:
        raise ValueError(f"nothing to resume in {run_dir}: it has neither {PARTIAL_NAME} nor results.jsonl")
    stored, source_run_id = _stored_fingerprint(run_dir)
    if stored is None:
        raise ValueError(
            f"cannot resume {run_dir}: it has no {RUN_META_NAME} (written by evalsprint 0.1), so nothing proves its records were "
            "produced by the same judges, target and repeat protocol; run again without --resume"
        )
    if stored != fingerprint:
        old, new = _fingerprint_parts(stored), _fingerprint_parts(fingerprint)
        differ = [key for key in _MISMATCH_PARTS if old.get(key) != new.get(key)] or ["fingerprint"]
        return _Resume(source_run_id, {}, {}, mismatch=differ)

    records, bad = _read_jsonl_lenient(source)
    by_id = {c.id: c for c in cases}
    resume = _Resume(source_run_id, {}, {})
    whole: dict[str, dict] = {}  # finished repeated cases: case_id -> the record with all of its attempts
    for record in records:
        case = by_id.get(record.get("case_id"))
        if case is None or record.get("case_sha") != case.sha:
            continue  # not selected in this run, or the case was edited: it runs again
        n_plan = plan.get(case.id)
        if isinstance(record.get("attempts"), list):  # a finished repeated case: every attempt in one record
            if n_plan and len(record["attempts"]) == n_plan:
                whole[case.id] = record
            continue
        if retry_errors and outcome_of(record)[0] == "error":
            resume.retried_errors += 1
            resume.discarded_cost_usd += _record_cost(record)
            continue
        attempt = record.get("attempt")
        if attempt is None and not n_plan:
            resume.attempts[(case.id, None)] = record
        elif isinstance(attempt, int) and not isinstance(attempt, bool) and n_plan and 0 <= attempt < n_plan:
            resume.attempts[(case.id, attempt)] = record

    for cid, record in whole.items():
        entries = record["attempts"]
        errored = [k for k, a in enumerate(entries) if isinstance(a, dict) and a.get("outcome") == "error"]
        # Loose attempt lines of a case that also has a combined record come from an interrupted resume that was running its crashed
        # attempts again: they replace those entries. Any other loose line of the case is stale (the combined record decided it).
        loose = {key[1]: resume.attempts.pop(key) for key in [k for k in resume.attempts if k[0] == cid]}
        fresh = {k: loose[k] for k in errored if k in loose}
        if outcome_of(record)[0] == "error":  # the majority of attempts crashed
            if retry_errors:
                resume.retried_errors += 1  # every attempt runs again (no attempt of it has a complete result to keep)
                resume.discarded_cost_usd += _record_cost(record)
            else:
                resume.combined[cid] = record
            continue
        rerun: list[int] = []
        if retry_errors:
            for k in errored:
                if k in fresh and outcome_of(fresh[k])[0] == "error":  # crashed again in the interrupted retry: one more try
                    resume.discarded_cost_usd += _record_cost(fresh.pop(k))
                if k not in fresh:
                    rerun.append(k)
                    resume.discarded_cost_usd += _entry_cost(entries[k])
            resume.retried_errors += len(rerun)
        if fresh or rerun:
            resume.patched[cid] = _Patched(record, rerun, fresh)
        else:
            resume.combined[cid] = record
    if bad:
        print(f"[evalsprint] warning: {bad} unreadable line(s) in {source.name} were ignored (a crash cuts the last line short); those attempts run again", file=sys.stderr)
    return resume


def _merge_patched(old_rep: CaseResult, old_entries: list[dict], fresh: dict[int, CaseResult]) -> CaseResult:
    """The record of a finished repeated case after its crashed attempts ran again: the old entries of the attempts that worked plus
    the fresh ones, the majority recomputed over all of them. The representative is the first attempt with the verdict outcome whose
    COMPLETE result is known (the old representative, or a fresh attempt). One always exists: the old verdict was pass or fail, the
    entries that worked keep their counts, so a new verdict can only differ from it through a fresh attempt of that outcome."""
    entries = [_attempt_entry(fresh[k]) if k in fresh else dict(old_entries[k]) for k in range(len(old_entries))]
    verdict = majority_outcome([e.get("outcome") for e in entries])
    rep_idx = next((k for k, e in enumerate(old_entries) if e.get("outcome") == old_rep.outcome), 0)
    known = {rep_idx: old_rep, **fresh}
    rep = next((known[k] for k in sorted(known) if known[k].resolved()[0] == verdict), old_rep)
    return _finish_repeated(rep, entries, verdict)


def _partial_line(res: CaseResult, attempt: int | None) -> str:
    line = sanitize_obj(res.to_dict())
    line["attempt"] = attempt
    return json.dumps(line, ensure_ascii=False, default=str) + "\n"


# ------------------------------------------------------------------------------ resume lock
class ResumeLocked(ValueError):
    """Another `run --resume` of the same directory is running (exit code 2)."""


class _Busy(Exception):
    pass


def _lock_fd(fd: int) -> None:
    """Take an exclusive, non-blocking lock on an open file; `_Busy` when another process holds it. The OS drops it with the process,
    so a killed resume never leaves a lock behind."""
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLK", -1)):
            raise _Busy from exc
        raise


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextlib.contextmanager
def _resume_lock(resume_from: str | Path | None):
    """Hold `<resume_from>/.resume.lock` while `run --resume` reads and continues that directory (C20): two resumes of the same
    folder would pay for every missing case twice. A second one raises `ResumeLocked`. No-op without `resume_from`. A directory that
    cannot be written (a read-only checkout) or a file system without locks runs unprotected, with a warning: the lock guards against
    a double payment, it is not needed for the records to be right."""
    if not resume_from or not Path(resume_from).is_dir():  # `_load_resume` explains what is wrong with a missing directory
        yield
        return
    path = Path(resume_from) / LOCK_NAME
    fd = None
    try:
        for _ in range(20):
            candidate = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                _lock_fd(candidate)
            except _Busy:
                holder = ""
                try:
                    holder = os.pread(candidate, 32, 0).decode("ascii", "replace").strip() if hasattr(os, "pread") else ""
                except OSError:
                    pass
                os.close(candidate)
                raise ResumeLocked(
                    f"another `evalsprint run --resume` is already working on {resume_from} (lock file {path}"
                    + (f", held by process {holder}" if holder.isdigit() else "")
                    + "): two resumes of one folder would pay for every missing case twice. Wait for it to finish, or resume a copy of the folder"
                ) from None
            except OSError:
                os.close(candidate)
                raise
            try:  # the holder before us may have removed the file between our open and our lock: then we hold a dead file, try again
                same = os.path.samestat(os.fstat(candidate), os.stat(path))
            except OSError:
                same = False
            if same:
                fd = candidate
                break
            os.close(candidate)
    except ResumeLocked:
        raise
    except OSError as exc:
        print(f"[evalsprint] warning: could not take the resume lock {path} ({exc}); continuing without it", file=sys.stderr)
        fd = None
    if fd is not None:
        try:
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("ascii"))
        except OSError:
            pass
    try:
        yield
    finally:
        if fd is not None:
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    os.unlink(path)  # while we still hold the lock: nobody can have locked the file we remove
            _unlock_fd(fd)
            os.close(fd)
            if os.name == "nt":
                with contextlib.suppress(OSError):
                    os.unlink(path)


def _call_target(target: Target, case: Case, attempt: int | None, retries: int, stop: threading.Event) -> tuple[TargetOutput, int]:
    """(output, retries used): the target called again, up to `retries` times, while the attempt ends in an error (C18).

    The calls that failed were paid: their cost and tokens are added to the output that is kept, and their error messages go to
    `output.meta["target_retry_errors"]`. No new call starts once a stop was requested (Ctrl-C, SIGTERM)."""
    failed_cost, failed_usage, errors, used = 0.0, Usage(), [], 0
    while True:
        output = target(case, attempt)
        if not output.error or used >= retries or stop.is_set():
            break
        used += 1
        errors.append(output.error)
        failed_cost += output.cost_usd or 0.0
        failed_usage = failed_usage + output.usage
    if used:
        if failed_cost > 0 or output.cost_usd is not None:
            output.cost_usd = (output.cost_usd or 0.0) + failed_cost
        output.usage = output.usage + failed_usage
        output.meta = {**output.meta, "target_retry_errors": errors}
    return output, used


def run_suite(
    config: SuiteConfig,
    label: str | None = None,
    out_dir: str | Path | None = None,
    tags: list[str] | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    progress: bool = True,
    repeat: int | dict | None = None,
    resume_from: str | Path | None = None,
    retry_errors: bool = False,
) -> RunResult:
    """Run the suite. `repeat` (int, or {"n", "per_tag"}) overrides the config's `repeat:` (`run --repeat N` sets `n` and keeps `per_tag`).

    The run id is fixed at the start; results stream to `results.partial.jsonl` as attempts finish and
    the final files replace it atomically at the end. `run_meta.json` records the run fingerprint at the start.

    `resume_from` (a run directory) continues an earlier run: its records whose case_sha matches the current case
    AND whose run fingerprint (judges_sha, target callable, params and source file, repeat protocol) matches this run's are reused
    without calling the target or the judges again; every other case or attempt runs. The new run has its own run_id
    and `summary["resumed_from"]`. `retry_errors` also re-runs the reusable records whose outcome is `error` and the crashed attempts
    of finished repeated cases. A resume holds `<resume_from>/.resume.lock` (`ResumeLocked` when another one does).

    A target attempt that errors is called again up to `config.target_retries` times before it counts as an error (C18). Anything
    that ends the run midway (Ctrl-C, SIGTERM, a full disk, a failed final save) leaves `results.partial.jsonl` and sets `exc.run_dir`.
    """
    with _resume_lock(resume_from):
        return _run_suite(config, label, out_dir, tags, ids, limit, progress, repeat, resume_from, retry_errors)


_MISMATCH_TEXT = {
    "judges_sha": "the judges (judges_sha)",
    "target": "the target callable or its params",
    "target_source": "the source file of the target callable",
    "repeat": "the repeat protocol",
}


def _run_suite(config, label, out_dir, tags, ids, limit, progress, repeat, resume_from, retry_errors) -> RunResult:
    t_start = time.perf_counter()
    started = datetime.now(timezone.utc)
    run_id = make_run_id(label, started)
    prices = PriceTable(config.pricing)
    config_cases = load_cases(config.dataset, behaviors=config.behaviors)  # what the config selects, before --tags / --ids / --limit
    cases: list[Case] = filter_cases(config_cases, tags=tags, ids=ids, limit=limit)
    if not cases:
        # An empty run has no pass rate and would crash compare/report/gate later.
        raise ValueError("no cases selected: check --tags / --ids / --limit against the dataset")
    target = Target.from_config(config.target, config.base_dir, prices)
    judges = build_judges(config.judges, config.base_dir, prices, cases)
    if not judges:
        raise ValueError("no judges configured: add at least one under 'judges:' (or per-case 'checks' in the test set)")
    validate_pass_rule(config.pass_rule, [j.name for j in judges])
    plan, protocol = _repeat_plan(config, cases, repeat)
    if not plan and any(k > 1 for k in protocol["per_tag"].values()):
        print(f"[evalsprint] warning: repeat per_tag {sorted(protocol['per_tag'])} match no selected case; every case runs once", file=sys.stderr)
    if retry_errors and not resume_from:
        raise ValueError("retry_errors only applies together with resume_from")

    # Fingerprint at the START (after build_judges imported the custom judges): what the run really uses, and what
    # `run --resume` compares before it trusts a record of this run.
    warnings: list[str] = []
    judges_sha, judges_config = judges_fingerprint(config.judges, config.pass_rule, config.base_dir, warnings)
    for text in dict.fromkeys(warnings):
        print(f"[evalsprint] warning: {text}", file=sys.stderr)
    target_info = target_fingerprint(config)
    fingerprint = run_fingerprint(config, judges_sha, protocol, target_info)
    resume = _load_resume(resume_from, cases, plan, fingerprint, retry_errors) if resume_from else None
    if resume is not None and resume.mismatch:
        print(
            f"[evalsprint] warning: --resume {resume_from}: {', '.join(resume.mismatch)} differ from that run, so none of its records "
            "can be trusted for this configuration; every case runs again"
            + (" (target_source: the source file of the target callable was edited)" if "target_source" in resume.mismatch else ""),
            file=sys.stderr,
        )

    if out_dir:
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _check_previous_run(run_dir)
    else:
        run_id, run_dir = _claim_run_dir(config.runs_dir or Path("runs"), run_id)

    # Reused records become CaseResults again; a finished repeated case comes back as one combined record.
    attempts_by_id: dict[str, dict[int, CaseResult]] = defaultdict(dict)
    finished: dict[str, CaseResult] = {}
    patches: dict[str, tuple[CaseResult, list[dict], set[int]]] = {}  # case_id -> (old representative, old attempt entries, indexes to run)
    seed: list[str] = []
    reused_cost = 0.0
    if resume is not None:
        for case in cases:
            if case.id in resume.combined:
                record = resume.combined[case.id]
                finished[case.id] = CaseResult.from_record(record, case)
                seed.append(_partial_line(finished[case.id], None))
                reused_cost += _record_cost(record)
                continue
            if case.id in resume.patched:
                patch = resume.patched[case.id]
                old_rep = CaseResult.from_record(patch.record, case)
                patches[case.id] = (old_rep, list(old_rep.attempts or []), set(patch.rerun))
                seed.append(_partial_line(old_rep, None))  # as the earlier run left it: a further interruption resumes from the same state
                reused_cost += sum(_entry_cost(e) for k, e in enumerate(old_rep.attempts or []) if k not in patch.rerun and k not in patch.fresh)
                for k, record in sorted(patch.fresh.items()):
                    attempts_by_id[case.id][k] = CaseResult.from_record(record, case)
                    seed.append(_partial_line(attempts_by_id[case.id][k], k))
                    reused_cost += _record_cost(record)
                continue
            for k in range(plan.get(case.id, 1)):
                key = (case.id, k if plan.get(case.id) else None)
                record = resume.attempts.get(key)
                if record is not None:
                    attempts_by_id[case.id][k] = CaseResult.from_record(record, case)
                    seed.append(_partial_line(attempts_by_id[case.id][k], key[1]))
                    reused_cost += _record_cost(record)

    def available(case: Case, k: int | None) -> bool:
        if case.id in finished or (k or 0) in attempts_by_id[case.id]:
            return True
        return case.id in patches and (k or 0) not in patches[case.id][2]  # an attempt of a patched case that worked: its entry is kept

    units = [(c, k if plan.get(c.id) else None) for c in cases for k in range(plan.get(c.id, 1))]
    pending = [(c, k) for c, k in units if not available(c, k)]

    partial = run_dir / PARTIAL_NAME
    # The new partial file starts as the reused lines (none for a fresh run). It is replaced atomically, since
    # --out DIR --resume DIR overwrites the very file those lines were read from, and it comes BEFORE the old
    # summary.json / results.jsonl are dropped: a crash in between cannot lose a reused record.
    _atomic_write(partial, "".join(seed))
    if out_dir:
        _clear_previous_run(run_dir)
    # run_meta.json goes last: whatever a crash leaves behind, it never pairs the fingerprint of THIS run with the
    # records of an older one (the previous run's meta stays until the old records are gone or replaced).
    meta = {
        "run_id": run_id,
        "suite": config.suite,
        "created_at": started.isoformat(timespec="seconds"),
        "evalsprint_version": __version__,
        "fingerprint": fingerprint,
    }
    if resume is not None:
        meta["resumed_from"] = resume.source_run_id
    _atomic_write(run_dir / RUN_META_NAME, json.dumps(sanitize_obj(meta), ensure_ascii=False, indent=2, default=str))
    if resume is not None and progress:
        print(f"[evalsprint] resuming {resume.source_run_id}: {resume.n_reused} record(s) reused (${reused_cost:.4f} already spent), {len(pending)} attempt(s) to run", file=sys.stderr)

    stop = threading.Event()  # set by the first Ctrl-C / fatal error: queued cases must not start (and spend) after it
    write_lock = threading.Lock()
    running = [0]  # attempts a worker has started and not yet saved

    def work(case: Case, attempt: int | None) -> CaseResult | None:
        """Run one attempt AND save its line, from the worker thread: an attempt that is still in flight when Ctrl-C
        arrives has been paid for, so it is saved too instead of being dropped with the main loop."""
        if stop.is_set():
            return None
        with write_lock:
            running[0] += 1
        try:
            try:
                output, retries_used = _call_target(target, case, attempt, config.target_retries, stop)
                judge_results = run_judges(judges, case, output)
                passed = (not output.error) and _pass_rule(config.pass_rule, judge_results)
                res = CaseResult(case=case, output=output, judges=judge_results, passed=passed, target_retries_used=retries_used)
            except Exception as exc:  # noqa: BLE001 - a bug in one case must not lose the other paid results
                failed = JudgeResult("(runner)", False, error=f"internal error: {type(exc).__name__}: {exc}")
                res = CaseResult(case=case, output=TargetOutput(text=""), judges=[failed], passed=False)
            except BaseException:  # KeyboardInterrupt / SystemExit raised inside user code
                stop.set()
                raise
            with write_lock:
                partial_fh.write(_partial_line(res, attempt))
                partial_fh.flush()
            return res
        finally:
            with write_lock:
                running[0] -= 1

    pool = ThreadPoolExecutor(max_workers=max(1, config.concurrency))
    partial_fh = open(partial, "a", encoding="utf-8", newline="\n")
    try:
        try:
            futures = {pool.submit(work, c, k): (c, k) for c, k in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if res is None:  # skipped after a stop request
                    continue
                case, attempt = futures[fut]
                attempts_by_id[case.id][attempt or 0] = res
                if progress:
                    outcome = res.resolved()[0]
                    mark = {"pass": "PASS", "error": "ERR "}.get(outcome, "FAIL")
                    suffix = f" [attempt {attempt + 1}/{plan[case.id]}]" if attempt is not None else ""
                    retried = f" (after {res.target_retries_used} retry)" if res.target_retries_used == 1 else f" (after {res.target_retries_used} retries)" if res.target_retries_used else ""
                    print(f"[{i:>4}/{len(pending)}] {mark} {case.id}{suffix}{retried}", file=sys.stderr)
        except BaseException as exc:
            stop.set()  # Ctrl-C / disk full: no new attempt starts (nothing more is spent on the remaining cases)...
            pool.shutdown(wait=False, cancel_futures=True)
            with write_lock:
                waiting = running[0]
            if waiting and progress:
                print(f"[evalsprint] stopping: waiting for {waiting} running attempt(s) to finish so their results are saved (Ctrl-C again to abandon them)", file=sys.stderr)
            try:
                pool.shutdown(wait=True)  # ...but those already running are paid for: let them land in results.partial.jsonl
            except BaseException:  # noqa: BLE001 - a second Ctrl-C gives up on them
                pass
            _tag_run_dir(exc, run_dir)  # so the CLI can tell where `--resume` should point
            raise
        pool.shutdown()
    finally:
        with write_lock:
            partial_fh.close()

    try:
        records = []
        for c in cases:  # keep dataset order
            if c.id in finished:
                res = finished[c.id]
            elif c.id in patches:
                old_rep, old_entries, _ = patches[c.id]
                res = _merge_patched(old_rep, old_entries, attempts_by_id[c.id])
            else:
                res = _combine_attempts([attempts_by_id[c.id][k] for k in sorted(attempts_by_id[c.id])])
            records.append(sanitize_obj(res.to_dict()))
        reason = placeholder_reason(config)
        summary = {
            "run_id": run_id,
            "suite": config.suite,
            "label": label,
            "created_at": started.isoformat(timespec="seconds"),
            "evalsprint_version": __version__,
            "target": target_info,
            "judges": [j.name for j in judges],
            "judges_sha": judges_sha,
            "judges_config": judges_config,
            "pass_rule": config.pass_rule,
            "gate": config.gate,
            "dataset": _dataset_ref(config),
            **summarize(records),
            "placeholder": reason is not None,  # C18: the scaffold's offline stub as target, or an LLM judge on the mock provider
            "duration_s": time.perf_counter() - t_start,
            "repeat": protocol,  # the effective protocol ({"n": 1, "per_tag": {}} = every case ran once): a gate must compare like with like
            "target_retries": config.target_retries,  # C22: the configured value (int, always written); `n_target_retries` is what was actually used
            "selection": _selection(len(config_cases), len(cases), tags, ids, limit),  # C26: a run over a subset is no baseline for the whole suite
        }
        if resume is not None:
            summary["resumed_from"] = {"run_id": resume.source_run_id, "reused_records": resume.n_reused, "reused_cost_usd": reused_cost}
            if resume.mismatch:
                summary["resumed_from"]["fingerprint_mismatch"] = resume.mismatch
            if retry_errors:
                summary["resumed_from"].update({"retried_errors": resume.retried_errors, "discarded_cost_usd": resume.discarded_cost_usd})
        if config.tag_groups:
            summary["tag_groups"] = config.tag_groups

        save_run(run_dir, summary, records)
    except BaseException as exc:  # a failed final save (full disk, a locked file, SIGTERM): the partial file is still there, say where
        _tag_run_dir(exc, run_dir)
        raise
    try:
        unlink_with_retry(partial)
    except OSError as exc:  # the run is saved and complete: a scanner holding the leftover file must not turn that into a failure
        print(f"[evalsprint] warning: the run is complete but {partial} could not be removed ({exc}); it is older than summary.json, so nothing reads it as unfinished", file=sys.stderr)
    return RunResult(run_dir=run_dir, summary=sanitize_obj(summary), results=records)


def _selection(n_config: int, n_selected: int, tags: list[str] | None, ids: list[str] | None, limit: int | None) -> dict:
    """`summary.selection` (C26): which part of the suite this run covers, so `baseline` can warn about a run over a subset.

    `limited` is True when the run covers fewer cases than the config selects (`n_selected < n_config`), whatever the cause
    (`--limit`, `--tags`, `ids`); a `--limit` or `--tags` that still selects every case is not limited. `n_config` counts every case
    the config's dataset holds, before `--tags` / `--ids` / `--limit`. `limit`, `tags` and `ids` are the selectors that were given
    (None when not given; a `limit` of 0 means no limit, as in `filter_cases`).
    """
    return {
        "limited": n_selected < n_config,
        "n_selected": n_selected,
        "n_config": n_config,
        "limit": int(limit) if limit else None,
        "tags": [str(t) for t in tags] if tags else None,
        "ids": [str(i) for i in ids] if ids else None,
    }


def _tag_run_dir(exc: BaseException, run_dir: Path) -> None:
    """Remember on the exception which directory holds the partial results (`cli` turns it into the `--resume` hint)."""
    try:
        exc.run_dir = run_dir
    except AttributeError:
        pass


REPLACE_RETRIES = 5
REPLACE_BACKOFF_S = 0.2

# Conditions no amount of waiting fixes: retrying them would only delay the real error message by seconds.
_PERMANENT_ERRNOS = {errno.ENOENT, errno.ENOSPC, errno.EROFS, errno.ENAMETOOLONG, errno.EISDIR, errno.ENOTDIR}
if hasattr(errno, "EDQUOT"):
    _PERMANENT_ERRNOS.add(errno.EDQUOT)


def _retryable(exc: OSError) -> bool:
    """Windows answers a replace or delete of a file that an antivirus scanner, an editor or another process holds
    open with PermissionError / WinError 5, 32 or 33, and lets go a moment later."""
    if isinstance(exc, (FileNotFoundError, NotADirectoryError, IsADirectoryError)):
        return False
    return exc.errno not in _PERMANENT_ERRNOS


def _with_retry(action, retries: int = REPLACE_RETRIES, delay: float = REPLACE_BACKOFF_S):
    """Run `action()`; on PermissionError / OSError wait 0.2 s, 0.4 s, 0.8 s ... and try again, `retries` times, then raise."""
    for attempt in range(retries + 1):
        try:
            return action()
        except OSError as exc:
            if attempt == retries or not _retryable(exc):
                raise
            _SLEEP(delay * (2**attempt))


def replace_with_retry(src: str | Path, dst: str | Path, retries: int = REPLACE_RETRIES, delay: float = REPLACE_BACKOFF_S) -> None:
    """`os.replace` that survives a destination another Windows process holds open for a moment."""
    _with_retry(lambda: os.replace(src, dst), retries, delay)


def unlink_with_retry(path: str | Path, retries: int = REPLACE_RETRIES, delay: float = REPLACE_BACKOFF_S) -> None:
    """`Path.unlink(missing_ok=True)` with the same retry."""
    _with_retry(lambda: Path(path).unlink(missing_ok=True), retries, delay)


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` so a reader (or a crash) never sees a half-written file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:  # \n on every OS: baselines diff cleanly across machines
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_run(run_dir: str | Path, summary: dict, records: list[dict]) -> Path:
    """Write results.jsonl, then summary.json (its presence marks the run as complete), both atomically."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # default=str: a client adapter may put non-JSON values (datetime, Path, objects) in output.meta;
    # never lose a finished run over that. sanitize_obj: nor over a lone surrogate anywhere in the text.
    lines = "".join(json.dumps(sanitize_obj(rec), ensure_ascii=False, default=str) + "\n" for rec in records)
    _atomic_write(run_dir / "results.jsonl", lines)
    _atomic_write(run_dir / "summary.json", json.dumps(sanitize_obj(summary), ensure_ascii=False, indent=2, default=str))
    return run_dir


def load_run(path: str | Path) -> RunResult:
    """Load a run from its directory (or from a path to its summary.json)."""
    path = Path(path)
    run_dir = path.parent if path.name == "summary.json" else path
    check_run_files(run_dir)
    with open(run_dir / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    results = []
    with open(run_dir / "results.jsonl", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{run_dir / 'results.jsonl'}:{lineno}: invalid JSON ({exc.msg}); the file is truncated or was edited by hand") from exc
    return RunResult(run_dir=run_dir, summary=summary, results=results)
