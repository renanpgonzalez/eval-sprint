"""`evalsprint` command line (also reachable as `python -m evalsprint`).

Exit codes: 0 ok, 1 regression detected (`compare` only), 2 usage or configuration
error, 3 unhealthy run (`run` or `baseline`: too many target/judge errors, counted per
case AND per attempt, to say anything about quality; `run` still writes its files; `baseline`
also refuses a placeholder run), 130 interrupted (Ctrl-C) and 143 terminated (SIGTERM), both
after the attempts that were running have been saved and with the `--resume` command on
stderr. Any other failure, expected or not, ends as a one-line message on stderr with exit
code 2, so that 1 always means "the gate failed" in CI. Pass `--debug` to get the traceback instead.

Files are written before anything is printed, and stdout/stderr never raise on
characters the console cannot show (a redirected legacy Windows console is cp1252):
they are replaced, so a verdict never turns into a crash.

`compare`, `calibration`, `labeling` and `report.write_history` are imported inside their commands so
that `run` and `describe` do not depend on them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import sys
import threading
from pathlib import Path

import yaml

from . import __version__
from .config import DEFAULT_GATE, SuiteConfig, load_gate, redact_secrets
from .dataset import describe, filter_cases, load_cases
from .report import DASH, fmt_ci, fmt_latency, fmt_money, fmt_pct, write_report
from .runner import (
    PARTIAL_NAME,
    RunResult,
    _attempts_of,
    check_run_files,
    judges_fingerprint,
    load_run,
    placeholder_reason,
    replace_with_retry,
    run_suite,
    unlink_with_retry,
)
from .types import sanitize_text, suggest

EXIT_OK, EXIT_REGRESSION, EXIT_USAGE, EXIT_UNHEALTHY = 0, 1, 2, 3
EXIT_INTERRUPTED, EXIT_TERMINATED = 130, 143

_EPILOG = """\
typical flow:
  evalsprint init      --ci                          scaffold evals/ (and the GitHub workflows) in this repo
  evalsprint describe  -c eval.yaml                  validate the test set, config and judges
  evalsprint run       -c eval.yaml --label v2       run, write runs/<id>/report.html
  evalsprint run       -c eval.yaml --resume runs/<id>   continue an interrupted run (paid cases are not paid twice)
  evalsprint label     runs/<id>                     sheet for human labeling
  evalsprint calibrate runs/<id> --labels labels.jsonl --json runs/<id>/calibration.json
  evalsprint label     runs/<id> --previous-labels labels.jsonl   skip the cases that already have a label for their current output
  evalsprint noise     runs/<id>                     false alarm and detection of the whole gate for this suite's measured noise
  evalsprint baseline  runs/<id> --to evals/baseline commit the reference run
  evalsprint compare   evals/baseline runs/<new>     CI gate (exit 1 on regression)
  evalsprint history   runs                          one page with the trend of every run

exit codes: 0 ok, 1 regression detected (compare), 2 usage or configuration error,
            3 unhealthy run (run/baseline: too many target or judge errors, per case or per attempt, to trust the numbers;
            baseline also refuses a placeholder run), 130 interrupted, 143 terminated
"""


class CliError(Exception):
    """A usage or configuration problem, reported as one line on stderr (exit code 2)."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        raise CliError(f"{self.prog}: {message}")


# ------------------------------------------------------------------------ output
_stdout_broken = False


def _say(text: str = "") -> None:
    """print() that survives a closed pipe (`evalsprint compare ... | head`): the verdict still sets the exit code."""
    global _stdout_broken
    if _stdout_broken:
        return
    try:
        print(text)
    except BrokenPipeError:
        _stdout_broken = True


def _warn(text: str) -> None:
    try:
        print(text, file=sys.stderr)
    except BrokenPipeError:
        pass


def _configure_streams() -> None:
    """Never let a character the console cannot encode crash the command (redirected legacy Windows
    consoles use cp1252). Files are always UTF-8 and are written before anything is printed."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _finish_stdout() -> None:
    global _stdout_broken
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        _stdout_broken = True
    if _stdout_broken:
        try:  # the interpreter flushes stdout again at exit: point it at /dev/null so that does not fail too
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
    _stdout_broken = False


# ------------------------------------------------------------------------ helpers
def _plain(text: str) -> str:
    """Terminal-safe text: the report's dash for missing values becomes `n/a`."""
    return text.replace(DASH, "n/a")


def _split_csv(value: str | None) -> list[str] | None:
    items = [item.strip() for item in (value or "").split(",") if item.strip()]
    return items or None


def _run_dir(path: str | Path) -> Path:
    """Resolve a run directory (or its summary.json) and check it holds a finished run."""
    given = Path(path)
    run_dir = given.parent if given.name == "summary.json" else given
    for name in ("summary.json", "results.jsonl"):
        if not (run_dir / name).is_file():
            if (run_dir / PARTIAL_NAME).is_file():
                raise CliError(f"{path} is an unfinished run ({PARTIAL_NAME} exists but no {name}): the run was interrupted, run it again (`evalsprint run -c <eval.yaml> --resume {run_dir}` keeps the cases that finished)")
            raise CliError(f"{path} is not a run directory (no {name})")
    try:
        check_run_files(run_dir)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    return run_dir


def _load_run(path: str | Path) -> RunResult:
    return load_run(_run_dir(path))


def _gate(config_path: str | None, gate_config: str | None = None) -> dict | None:
    """Gate thresholds for `compare`.

    `--gate-config PATH`: ONLY the `gate:` of that YAML (defaults for everything it does not set), so CI can pass the
    base branch's copy and a pull request cannot loosen its own gate. `-c eval.yaml`: that file's gate. Neither:
    None, and `compare_runs` uses the candidate run's own gate.
    """
    if gate_config:
        return load_gate(gate_config)
    return SuiteConfig.load(config_path).gate if config_path else None


def _check_gate(gate: dict) -> None:
    """Fail on a malformed gate (wrong type, out-of-range fraction) now instead of at compare time."""
    try:
        from . import compare
    except ImportError:
        return
    resolve = getattr(compare, "resolve_gate", None)
    if resolve is not None:
        resolve({}, gate)


def _baseline_calibration(baseline_dir: Path) -> dict | None:
    """`<baseline>/calibration.json` (written by `calibrate --json`), when there is one."""
    path = baseline_dir / "calibration.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CliError(f"{path} is not valid JSON ({exc}); recreate it with `evalsprint calibrate ... --json {path}`") from exc
    if not isinstance(data, dict):
        raise CliError(f"{path} must hold the JSON object written by `evalsprint calibrate --json`")
    return data


def _write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(sanitize_text(text))


def _write_json(path: str | Path, data: dict) -> None:
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def _calibrate(run: RunResult, labels: str | list[str], judge: str | None, exclude_holdout: str | None = None) -> dict:
    """Calibrate the run against human labels: one labels file, or several (`--labels a --labels b`, read as one list, C20)."""
    from . import calibration

    extra = {}
    if exclude_holdout:
        load = getattr(calibration, "load_holdout_ids", None)
        if load is None:
            raise CliError("--exclude-holdout needs calibration.load_holdout_ids, which this install does not have")
        extra["exclude_case_ids"] = load(exclude_holdout)
    return calibration.calibrate(run, calibration.load_labels(labels), judge=judge, **extra)


def _stale_calibration_note(cal, run) -> str | None:
    """`calibration.stale_calibration`: the warning when `cal` (a calibration.json) was made for other judges than `run` has (C20)."""
    if not isinstance(cal, dict):
        return None
    try:
        from .calibration import stale_calibration
    except ImportError:
        return None
    return stale_calibration(cal, run)


def _print_rows(rows: list[tuple[str, str]]) -> None:
    width = max(len(key) for key, _ in rows) + 1
    for key, value in rows:
        _say(f"{key + ':':<{width}} {value}")


def _health(summary: dict, gate: dict | None = None) -> tuple[bool, str]:
    """(unhealthy, explanation). A run is unhealthy when its error rate is above the gate's `max_error_rate`, counted per CASE and per
    ATTEMPT (a repeated case whose majority verdict stands can hide crashed attempts), or when nothing could be scored: such a run
    says nothing about quality."""
    n = summary.get("n_cases") or 0
    n_errors = summary.get("n_errors") or 0
    rate = summary.get("error_rate")
    if rate is None:
        rate = (n_errors / n) if n else 0.0
    n_attempts = summary.get("n_attempts") or 0
    n_attempt_errors = summary.get("n_attempt_errors") or 0
    attempt_rate = summary.get("attempt_error_rate")
    if attempt_rate is None and n_attempts:  # a summary written before attempts were counted has no attempt counters: only cases
        attempt_rate = n_attempt_errors / n_attempts
    limit = (gate if gate is not None else summary.get("gate") or {}).get("max_error_rate", DEFAULT_GATE["max_error_rate"])
    nothing_scored = summary.get("n_scored") == 0 and n > 0
    case_bad = limit is not None and rate > limit
    attempt_bad = limit is not None and attempt_rate is not None and attempt_rate > limit
    parts = f"{summary.get('n_judge_errors', 0)} judge error(s), {summary.get('n_target_errors', 0)} target error(s)"
    text = f"{n_errors} of {n} cases errored ({parts}; error rate {fmt_pct(rate)}"
    text += f", limit {fmt_pct(limit)})" if limit is not None else ")"
    if attempt_bad and n_attempts != n:
        text += f"; {n_attempt_errors} of {n_attempts} attempts errored ({fmt_pct(attempt_rate)} of the attempts, limit {fmt_pct(limit)})"
        if not case_bad:
            text += ", which the majority vote of the repeated cases hides: a target or judge that crashes at random"
    if nothing_scored:
        text += "; no case could be scored"
    return (case_bad or attempt_bad or nothing_scored), text


def _placeholder_why(summary: dict) -> str | None:
    """Why a run summary is a placeholder (`summary.placeholder`, C18), in words; None when it is not one."""
    if summary.get("placeholder") is not True:
        return None
    target = summary.get("target") if isinstance(summary.get("target"), dict) else {}
    if str(target.get("callable") or "").rsplit(":", 1)[-1] == "offline_stub":
        return f"its target is the scaffold's offline_stub ({target.get('callable')}), which answers with canned text"
    for judge in summary.get("judges_config") or []:
        if isinstance(judge, dict) and judge.get("type") == "llm" and str(judge.get("provider", "")).lower() == "mock":
            return f"its LLM judge {judge.get('name') or 'llm'!r} runs on the mock provider (JUDGE_PROVIDER=mock), which does not judge"
    return "its target is the scaffold's offline stub or one of its judges ran on the mock provider"


# ------------------------------------------------------------- forged or hand-edited summary.json (C26)
_NUM, _PAIR = "number", "pair"  # a number (or null where the value does not exist), a [low, high] pair of them
_SUMMARY_NUMBERS: dict = {
    **{k: _NUM for k in (
        "n_cases", "n_passed", "n_failed", "n_scored", "pass_rate", "n_errors", "n_target_errors", "n_judge_errors", "error_rate", "n_attempts",
        "n_attempt_errors", "n_attempt_target_errors", "n_attempt_judge_errors", "attempt_error_rate", "n_target_retries", "n_oversize", "target_retries", "duration_s",
    )},
    "pass_rate_ci": _PAIR,
    "stability": {"n_repeated": _NUM, "flip_rate": _NUM},
    "by_tag": {"*": {"n": _NUM, "n_passed": _NUM, "pass_rate": _NUM, "ci": _PAIR}},
    "by_judge": {"*": {"n": _NUM, "n_passed": _NUM, "pass_rate": _NUM, "ci": _PAIR, "mean_score": _NUM, "n_errors": _NUM, "cost_usd": _NUM}},
    "cost": {k: _NUM for k in ("system_total_usd", "system_mean_usd", "system_per_1k_usd", "judge_total_usd", "n_priced")},
    "latency": {k: _NUM for k in ("mean_s", "p50_s", "p95_s", "max_s")},
    "tokens": {k: _NUM for k in ("input", "output", "cache_read", "cache_write")},
    "models": {"*": _NUM},
    "repeat": {"n": _NUM, "per_tag": {"*": _NUM}},
    "selection": {"n_selected": _NUM, "n_config": _NUM, "limit": _NUM},
    "resumed_from": {"reused_records": _NUM, "reused_cost_usd": _NUM, "discarded_cost_usd": _NUM},
}


def _is_number(value) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _non_numeric_field(node, spec=None, prefix: str = "") -> tuple[str, object] | None:
    """The first field of a summary that should hold a number (or null) and holds something else: (its dotted name, its value); None when
    every field is as the runner writes it. A missing field is fine (an older summary); only a present field of the wrong type counts."""
    spec = _SUMMARY_NUMBERS if spec is None else spec
    for key, kind in spec.items():
        if key == "*":
            continue
        if not isinstance(node, dict) or key not in node:
            continue
        found = _check_field(node[key], kind, f"{prefix}{key}")
        if found:
            return found
    wildcard = spec.get("*")
    if wildcard is not None and isinstance(node, dict):
        for key, value in node.items():
            found = _check_field(value, wildcard, f"{prefix}{key}")
            if found:
                return found
    return None


def _check_field(value, kind, name: str) -> tuple[str, object] | None:
    if kind == _NUM:
        return None if _is_number(value) else (name, value)
    if kind == _PAIR:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return (name, value)
        for i, item in enumerate(value):
            if not _is_number(item):
                return (f"{name}[{i}]", item)
        return None
    if not isinstance(value, dict):
        return (name, value)  # a section (by_tag, cost, ...) that is not an object
    return _non_numeric_field(value, kind, name + ".")


@contextlib.contextmanager
def _forged_summary_as_error(*run_paths):
    """Turn the crash a summary.json with text where a number belongs would cause (`TypeError: bad operand type for abs()`, `Unknown format code 'f'`)
    into the one-line error that names the field (exit code 2, as before). Only when a command FAILED and one of the runs holds such a field: a
    summary the command copes with stays fine, and any other failure keeps its own message."""
    try:
        yield
    except (TypeError, ValueError, AttributeError, KeyError, ArithmeticError) as exc:
        for given in run_paths:
            if not given:
                continue
            path = Path(given)
            file = path if path.name == "summary.json" else path / "summary.json"
            try:
                summary = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found = _non_numeric_field(summary) if isinstance(summary, dict) else None
            if found:
                name, value = found
                raise CliError(
                    f"{file}: the field '{name}' should be a number but is {_short(repr(value), 40)}: this summary.json was edited or does not come from evalsprint "
                    "(run the suite again, or restore the file)"
                ) from exc
        raise


# ----------------------------------------------------------------------- commands
def _quote(text: str) -> str:
    """One shell word: shlex quoting on POSIX, double quotes on Windows (cmd.exe and PowerShell)."""
    if os.name != "nt":
        return shlex.quote(text)
    return f'"{text.replace(chr(34), chr(92) + chr(34))}"' if not text or re.search(r"[\s\"'$&|;<>()*?\[\]{}!^`]", text) else text


def _resume_command(args: argparse.Namespace, run_dir: Path) -> str:
    """The command that continues an interrupted run. It repeats every option that is part of the run fingerprint
    (--set, --repeat) and of the case selection: resuming with another judge or target config would reuse nothing."""
    parts = ["evalsprint", "run", "-c", _quote(str(args.config))]
    for item in args.set or []:
        parts += ["--set", _quote(item)]
    for flag, value in (("--repeat", args.repeat), ("--tags", args.tags), ("--limit", args.limit), ("--label", args.label)):
        if value is not None:
            parts += [flag, _quote(str(value))]
    return " ".join(parts + ["--resume", _quote(str(run_dir))])


def _attach_resume_hint(args: argparse.Namespace, exc: BaseException) -> None:
    """Put the way to continue on an exception that ended a run mid-way; `main` prints it after the error line."""
    run_dir = getattr(exc, "run_dir", None)
    if run_dir is None or not (Path(run_dir) / PARTIAL_NAME).is_file():
        return
    try:
        exc.resume_hint = (
            f"the attempts that finished are saved in {Path(run_dir) / PARTIAL_NAME}; continue without paying for them again with:\n  {_resume_command(args, Path(run_dir))}"
        )
    except AttributeError:
        pass


class _Terminated(KeyboardInterrupt):
    """SIGTERM (a CI cancel, `kill`, a container stop), raised in the main thread like Ctrl-C so the run drains, keeps
    `results.partial.jsonl` and prints the `--resume` command (C20)."""


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """While a run is going, SIGTERM is handled like SIGINT. Only possible in the main thread; anywhere else (or where the platform has no
    SIGTERM handler) the run is simply not protected and dies as it did before."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        yield
        return

    def handler(signum, frame):
        raise _Terminated()

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def cmd_run(args: argparse.Namespace) -> int:
    config = SuiteConfig.load(args.config, overrides=args.set)
    _check_gate(config.gate)
    tags = _split_csv(args.tags)
    if args.limit is not None and args.limit < 1:
        raise CliError("--limit must be a positive integer")
    if args.repeat is not None and args.repeat < 1:
        raise CliError("--repeat must be a positive integer")
    if args.retry_errors and not args.resume:
        raise CliError("--retry-errors only applies together with --resume")
    if not filter_cases(load_cases(config.dataset, behaviors=config.behaviors), tags=tags, limit=args.limit):
        raise CliError(f"no cases match --tags {','.join(tags)}" if tags else "the test set has no cases to run")

    try:
        with _sigterm_as_interrupt():
            result = run_suite(
                config, label=args.label, out_dir=args.out, tags=tags, limit=args.limit, progress=not args.quiet, repeat=args.repeat,
                resume_from=args.resume, retry_errors=args.retry_errors,
            )
    except (KeyboardInterrupt, Exception) as exc:
        _attach_resume_hint(args, exc)  # a crash mid-run (Ctrl-C, disk full, ...) never loses what was paid for: say how to continue
        raise
    report_path, report_problem = None, None
    if not args.no_report:
        try:
            report_path = write_report(result.run_dir / "report.html", result)
        except Exception as exc:  # noqa: BLE001 - the results are saved; a report problem must not hide the exit code
            if args.debug:
                raise
            report_problem = f"{type(exc).__name__}: {exc}"

    s, cost = result.summary, result.summary["cost"]
    unhealthy, why = _health(s, config.gate)
    scored = s.get("n_scored", s["n_cases"])
    rate = s["pass_rate"]
    resumed = s.get("resumed_from")
    hidden_crashes = sum(r.get("n_attempt_errors") or 0 for r in result.results if r.get("outcome") != "error")  # crashed attempts of cases that still passed or failed
    _print_rows(
        [
            ("Suite", f"{s['suite']}  (run {s['run_id']})"),
            ("Cases", f"{s['n_cases']}" + (f" ({scored} scored)" if scored != s["n_cases"] else "")),
            (
                "Pass rate",
                _plain(f"{fmt_pct(rate)}  (95% CI {fmt_ci(s['pass_rate_ci'], sep='-')})") if rate is not None else "n/a  (no case could be scored)",
            ),
            ("Errors", f"{s['n_errors']} ({fmt_pct(s['error_rate'])}; {s.get('n_target_errors', 0)} target, {s.get('n_judge_errors', 0)} judge)"),
            *(
                [("Attempt errors", f"{s['n_attempt_errors']} of {s['n_attempts']} attempts ({fmt_pct(s['attempt_error_rate'])}; {s.get('n_attempt_target_errors', 0)} target, {s.get('n_attempt_judge_errors', 0)} judge)")]
                if s.get("n_attempts") not in (None, s["n_cases"]) and s.get("n_attempt_errors")
                else []
            ),
            *([("Stability", _stability_line(s))] if s.get("stability", {}).get("n_repeated") else []),
            *(
                [("Target retries", f"{s['n_target_retries']} call(s) repeated after a crash (target_retries: {s.get('target_retries', config.target_retries)}); paid and counted in the cost")]
                if s.get("n_target_retries")
                else []
            ),
            ("Cost / 1k requests", _plain(fmt_money(cost["system_per_1k_usd"]))),
            ("Latency p95", _plain(fmt_latency(s["latency"]["p95_s"]))),
            ("Judge cost", _plain(fmt_money(cost["judge_total_usd"]))),
            *([("Resumed", f"{resumed['reused_records']} record(s) of run {resumed['run_id']} reused, {_plain(fmt_money(resumed.get('reused_cost_usd')))} not spent again")] if resumed else []),
            ("Run dir", str(result.run_dir)),
            ("Report", str(report_path) if report_path else ("not written (--no-report)" if args.no_report else "not written (see warning)")),
        ]
    )
    if hidden_crashes > 0:
        _warn(
            f"warning: {hidden_crashes} attempt(s) of repeated cases errored but did not change their case's majority verdict, so the error rate above does not show them "
            f"(attempt error rate {fmt_pct(s['attempt_error_rate'])}); that is a flaky target or judge, not a quality signal"
        )
    if report_problem:
        _warn(f"warning: report.html was not written: {report_problem}")
    if s.get("placeholder"):
        _warn(
            f"warning: placeholder run: {placeholder_reason(config) or 'the target or a judge is simulated'}. It proves the pipeline runs here and measures "
            "nothing: `evalsprint baseline` refuses it without --force, and a baseline made from it makes every later comparison meaningless."
        )
    if unhealthy:
        _warn(f"error: unhealthy run: {why}. Errors are not quality failures and are left out of the pass rate, but this run is not a quality signal: fix the cause and run again (exit 3).")
        return EXIT_UNHEALTHY
    return EXIT_OK


def _stability_line(summary: dict) -> str:
    st = summary["stability"]
    flip = st.get("flip_rate")
    return f"{st['n_repeated']} repeated case(s), flip rate {fmt_pct(flip)} (share whose attempts disagree)" if flip is not None else f"{st['n_repeated']} repeated case(s)"


def _compare(baseline_path: str, candidate: RunResult, gate: dict | None, baseline: RunResult | None = None) -> tuple[dict, RunResult]:
    from . import compare

    baseline_dir = _run_dir(baseline_path)
    baseline = baseline or load_run(baseline_dir)
    calibration = _baseline_calibration(baseline_dir)
    extra = {"calibration": calibration} if calibration is not None else {}
    comparison = compare.compare_runs(baseline, candidate, gate=gate, **extra)
    stale = _stale_calibration_note(calibration, candidate)  # C20: the auto-loaded calibration.json may describe other judges than the run's
    if stale and isinstance(comparison.get("warnings"), list) and stale not in comparison["warnings"]:
        comparison["warnings"].append(stale)
    return comparison, baseline


def cmd_compare(args: argparse.Namespace) -> int:
    from . import compare

    candidate = _load_run(args.candidate)
    with _forged_summary_as_error(args.candidate, args.baseline):
        comparison, _ = _compare(args.baseline, candidate, _gate(args.config, args.gate_config))
        markdown = compare.to_markdown(comparison)
    if args.md:
        _write_text(args.md, markdown + "\n")
    if args.json:
        _write_json(args.json, comparison)
    _say(markdown)
    return EXIT_REGRESSION if comparison["verdict"] == "regression" else EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    if args.judge and not args.labels:
        raise CliError("--judge only applies together with --labels")
    if (args.config or args.gate_config) and not args.baseline:
        raise CliError("--config / --gate-config only apply together with --baseline")
    if args.monthly_requests is not None and args.monthly_requests < 1:
        raise CliError("--monthly-requests must be a positive integer")
    run = _load_run(args.run)

    comparison, baseline = None, None
    with _forged_summary_as_error(args.run, args.baseline):
        if args.baseline:
            comparison, baseline = _compare(args.baseline, run, _gate(args.config, args.gate_config))
        calibration = _calibrate(run, args.labels, args.judge) if args.labels else None

        extra = {}
        if baseline is not None:
            extra["baseline"] = baseline
        if args.monthly_requests is not None:
            extra["monthly_requests"] = args.monthly_requests
        out = write_report(args.output or run.run_dir / "report.html", run, comparison, calibration, **extra)
    _say(f"Report written to {out}")
    if comparison:
        _say(f"Verdict vs baseline: {comparison['verdict']}")
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibration

    if not args.export_fewshot and (args.fewshot_max is not None or args.holdout_frac is not None):
        raise CliError("--fewshot-max / --holdout-frac only apply together with --export-fewshot")
    fewshot_max = 6 if args.fewshot_max is None else args.fewshot_max
    holdout_frac = 0.3 if args.holdout_frac is None else args.holdout_frac
    if fewshot_max < 0:
        raise CliError("--fewshot-max must be zero or a positive integer")
    if not 0 < holdout_frac < 1:
        raise CliError("--holdout-frac must be strictly between 0 and 1 (the few-shot examples come from the cases that are held out)")
    run = _load_run(args.run)
    cal = _calibrate(run, args.labels, args.judge, args.exclude_holdout)
    markdown = calibration.to_markdown(cal)
    if args.md:
        _write_text(args.md, markdown + "\n")
    if args.json:
        _write_json(args.json, cal)
    fewshot = None
    if args.export_fewshot:
        export = getattr(calibration, "export_fewshot", None)
        if export is None:
            raise CliError("--export-fewshot needs calibration.export_fewshot, which this install does not have")
        fewshot = export(cal, run, args.export_fewshot, max_examples=fewshot_max, holdout_frac=holdout_frac, seed=0)
    _say(markdown)
    if fewshot is not None:
        _say(f"\nFew-shot examples written to {fewshot}")
        _say("Next: paste its `examples:` under the rubric's `examples:`, then calibrate again leaving its holdout_case_ids out (--exclude-holdout), so the judge is not graded on cases it was shown.")
    return EXIT_OK


def cmd_history(args: argparse.Namespace) -> int:
    from . import report

    write_history = getattr(report, "write_history", None)
    if write_history is None:
        raise CliError("history needs report.write_history, which this install does not have")
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise CliError(f"{runs_dir} is not a directory")
    runs, skipped = [], []
    for child in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (child / "summary.json").is_file() and not (child / "results.jsonl").is_file():
            continue  # not a run (a baseline copy, a scratch folder)
        try:
            run = _load_run(child)
        except (CliError, ValueError, OSError) as exc:
            skipped.append(f"{child.name}: {_describe_error(exc)}")
            continue
        if args.suite and run.summary.get("suite") != args.suite:
            continue
        runs.append(run)
    for line in skipped:
        _warn(f"warning: skipped run {line}")
    if not runs:
        suites = f" of suite {args.suite!r}" if args.suite else ""
        raise CliError(f"no finished run{suites} under {runs_dir}" + (f" ({len(skipped)} unreadable or unfinished run(s) were skipped)" if skipped else ""))
    runs.sort(key=lambda r: (str(r.summary.get("created_at") or ""), r.run_id))
    out = write_history(args.output or runs_dir / "history.html", runs)
    _say(f"History of {len(runs)} run(s) written to {out}")
    return EXIT_OK


def cmd_label(args: argparse.Namespace) -> int:
    from . import labeling

    if args.sample is not None and args.sample < 1:
        raise CliError("--sample must be a positive integer")
    run = _load_run(args.run)
    extra = {}
    if args.judge:
        known = run.summary.get("judges") or []
        if args.judge not in known:
            raise CliError(f"--judge {args.judge!r} is not a judge of this run{suggest(args.judge, known)} (judges: {', '.join(known) or 'none'})")
        extra["judge"] = args.judge
    previous = list(args.previous_labels) if args.previous_labels else None
    calibrate_labels = "<labels file>"
    if previous:
        # Incremental relabel (C20): only the cases whose CURRENT output has no label in the previous files are exported.
        plan = labeling.plan_relabel(run, previous, sample=args.sample, seed=args.seed, stratify=not args.no_stratify, judge=args.judge)
        _say(
            f"Previous labels: {plan['covered']} of {plan['eligible']} labelable case(s) already have a label for their current output, "
            f"{plan['uncovered']} do not; {plan['to_label']} to label now."
        )
        extra["previous_labels"] = previous
        calibrate_labels = " --labels ".join([*(_quote(p) for p in previous), "<the new labels file>"])
        if plan["to_label"] == 0 and plan["eligible"] > 0:
            _say("Nothing to label: every labelable case of this run already has a label for its current output, so no page was written.")
            _say(f"Next: evalsprint calibrate {args.run} --labels {' --labels '.join(_quote(p) for p in previous)}")
            return EXIT_OK
    export = labeling.export_labeling_csv if args.csv else labeling.export_labeling_html
    default_name = "labeling.csv" if args.csv else "labeling.html"
    try:
        out = export(run, args.output or run.run_dir / default_name, sample=args.sample, seed=args.seed, stratify=not args.no_stratify, **extra)
    except labeling.NothingToLabel as exc:  # the plan said there was something; the labels changed under us: still not an error
        _say(f"Nothing to label: {exc}")
        return EXIT_OK
    _say(f"Labeling {'sheet' if args.csv else 'page'} written to {out}")
    _say(f"Next: evalsprint calibrate {args.run} --labels {calibrate_labels}")
    return EXIT_OK


def _install_files(source: Path, dest: Path, names: list[str]) -> list[tuple[str, Path]]:
    """Copy `names` from `source` to `dest` so the destination is never left torn (new results.jsonl next to the
    previous summary.json). Everything is staged under temporary names first: a failure there changes nothing.
    Then the old summary.json (the marker of a complete run) is removed, the other files are swapped in, and the
    new summary.json goes last: a failure while committing leaves a directory without summary.json, which every
    command reports as "not a run directory" instead of reading half-old, half-new data."""
    staged: dict[str, Path] = {}
    try:
        for name in names:
            tmp = dest / f"{name}.tmp{os.getpid()}"
            shutil.copyfile(source / name, tmp)
            staged[name] = tmp
        unlink_with_retry(dest / "summary.json")
        for name in [n for n in names if n != "summary.json"] + ["summary.json"]:
            replace_with_retry(staged[name], dest / name)  # Windows: a scanner or an editor may hold the file for a moment
            staged.pop(name)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
    return [(name, dest / name) for name in names]


def _selection_warning(summary: dict, run_id: str) -> str | None:
    """C26: a baseline made from a subset of the suite is copied (never refused), but with a warning. `summary.selection` (written by the runner) says
    whether the run was limited (`--limit`, `--tags`, or any selection that covers fewer cases than the config selects); a summary without it, or with
    one that cannot be read, is an older run: one line says that the run does not record what it covers. None for a full run."""
    run_id = _short(run_id, 80)
    selection = summary.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("limited"), bool):
        return (
            f"warning: run {run_id} does not record which cases it covers (its summary.json has no readable `selection`: it was written by an older evalsprint), "
            "so evalsprint cannot tell whether it is a full run; a baseline made from `--limit` or `--tags` is compared on those cases only"
        )
    if not selection["limited"]:
        return None
    n_selected, n_config = selection.get("n_selected"), selection.get("n_config")
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (n_selected, n_config))
    parts = []
    if selection.get("limit"):
        parts.append(f"--limit {_short(selection['limit'], 12)}")
    if isinstance(selection.get("tags"), list) and selection["tags"]:
        parts.append("--tags " + ",".join(_short(t, 30) for t in selection["tags"][:8]) + (" ..." if len(selection["tags"]) > 8 else ""))
    if isinstance(selection.get("ids"), list) and selection["ids"]:
        shown = ",".join(_short(i, 30) for i in selection["ids"][:5])
        parts.append(f"ids {shown}" + (f" (+{len(selection['ids']) - 5} more)" if len(selection["ids"]) > 5 else ""))
    how = ", ".join(parts) if parts else "no selector recorded"
    covers = f"{_short(n_selected, 12)} of the {_short(n_config, 12)} cases the config selects"
    return (
        f"warning: run {run_id} was limited ({how}): it covers only {covers}. "
        + (f"Against a full candidate run only those {n_selected} cases are paired, the other {n_config - n_selected} show up as new cases in every comparison and the gate decides on {n_selected}. " if ints and n_selected <= n_config else "")
        + "Build the baseline from a run without --limit, --tags or ids (`evalsprint run -c <eval.yaml>`), or ignore this if gating that subset is what you want."
    )


def cmd_baseline(args: argparse.Namespace) -> int:
    source, dest = _run_dir(args.run), Path(args.to)
    if dest.resolve() == source.resolve():
        raise CliError("--to must be a different directory than the run")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    run_id = summary.get("run_id", source.name)
    with _forged_summary_as_error(source):
        unhealthy, why = _health(summary)
    placeholder = _placeholder_why(summary)
    if (unhealthy or placeholder) and not args.force:
        if unhealthy:
            _warn(
                f"error: refusing to use run {run_id} as the baseline: {why}. A baseline built on errors makes the gate blind "
                "(every candidate looks fine against it). Fix the cause and run again, or pass --force to copy it anyway (exit 3)."
            )
        if placeholder:
            _warn(
                f"error: refusing to use run {run_id} as the baseline: it is a placeholder run ({placeholder}). Its verdicts say nothing about your "
                "product, so every later comparison against it would be meaningless. Point evals/target_adapter.py at your real system, "
                "unset JUDGE_PROVIDER, run again, or pass --force to copy it anyway (exit 3)."
            )
        return EXIT_UNHEALTHY
    dest.mkdir(parents=True, exist_ok=True)
    notes = []
    if unhealthy:
        notes.append(f"warning: --force: this run is unhealthy ({why})")
    if placeholder:
        notes.append(f"warning: --force: this run is a placeholder ({placeholder}); the baseline says nothing about the real product")
    selection = _selection_warning(summary, run_id)
    if selection:
        notes.append(selection)
    names = ["results.jsonl", "summary.json"]
    calibration = source / "calibration.json"
    if calibration.is_file():
        names.append("calibration.json")
        try:
            stale = _stale_calibration_note(json.loads(calibration.read_text(encoding="utf-8")), summary)
        except (ValueError, OSError):
            stale = None  # an unreadable calibration.json is copied as it is; `compare` reports it when it reads it
        if stale:
            notes.append(f"warning: {calibration}: {stale}")
    elif (dest / "calibration.json").is_file():
        unlink_with_retry(dest / "calibration.json")  # it described the previous baseline's labels, not this run
        notes.append("removed the previous baseline's calibration.json (this run has none: run `evalsprint calibrate ... --json` on it)")
    copied = _install_files(source, dest, names)
    _say(f"Baseline {run_id} copied to {dest}:")
    for name, target in copied:
        _say(f"  {target}  ({target.stat().st_size / 1024:.1f} KB)")
    for note in notes:
        _warn(note)
    _say("Commit these files; CI compares new runs against them with `evalsprint compare`.")
    return EXIT_OK


# ---------------------------------------------------------------------------- init
# (path under DIR, path of its template under the packaged `evalsprint/templates/` directory)
_SCAFFOLD = [
    ("evals/eval.yaml", "eval.yaml"),
    ("evals/rubric.yaml", "rubric.yaml"),
    ("evals/cases.jsonl", "cases.example.jsonl"),
    ("evals/target_adapter.py", "target_adapter.py"),
    ("evals/requirements.txt", "evals/requirements.txt"),
]
_SCAFFOLD_CI = [
    (".github/workflows/evals.yml", "github/evals.yml"),
    (".github/workflows/evals-baseline.yml", "github/evals-baseline.yml"),
    (".github/CODEOWNERS.evals.example", "github/CODEOWNERS.example"),
]


def _templates_root():
    """The templates directory shipped inside the package (`evalsprint/templates/`), read through importlib.resources so
    an installed or zipped copy works too. None when this install ships without it."""
    from importlib import resources

    try:
        packaged = resources.files("evalsprint").joinpath("templates")
        return packaged if packaged.is_dir() else None
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return None


def _template_text(root, rel: str) -> str | None:
    node = root.joinpath(*rel.split("/"))
    if not node.is_file():
        return None
    text = node.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    return _pin_kit_version(text) if rel.endswith("requirements.txt") else text


def _pin_kit_version(text: str) -> str:
    """The scaffold pins the kit to the version that wrote it (the template names the release it was cut for)."""
    text = re.sub(r"(evalsprint @ git\+\S*?@)v?\d[\w.\-]*", lambda m: f"{m.group(1)}v{__version__}", text)
    return re.sub(r"(evalsprint==)\d[\w.\-]*", lambda m: f"{m.group(1)}{__version__}", text)


# What `init` keeps out of git (C18), and the other spellings of the same ignore that already count: run folders hold the model's
# answers, and `.label_secret` (written next to a run by `label`) must never be committed. `evals/baseline/` is NOT ignored.
_GITIGNORE_COMMENT = "# evalsprint: run outputs hold your model's answers, and the label secret must stay private (the committed baseline is not ignored)"
_GITIGNORE_LINES = {"runs/": {"runs/", "runs", "**/runs/", "**/runs"}, ".label_secret": {".label_secret", "**/.label_secret"}}


def _extend_gitignore(base: Path) -> tuple[str, list[str]]:
    """Create or extend `base/.gitignore` with `runs/` and `.label_secret`; ("created" | "updated" | "kept", the lines added).

    An existing file is only appended to (its bytes, line endings and encoding stay as they are) and a line it already has, in any
    spelling that ignores the same thing, is not added again, so a second `init` changes nothing."""
    path = base / ".gitignore"
    if path.is_dir():
        raise CliError(f"{path} is a directory; cannot write the scaffold file there")
    raw = path.read_bytes() if path.exists() else b""
    present = {line.strip() for line in raw.decode("utf-8", errors="replace").lstrip("\ufeff").splitlines()}
    missing = [line for line, spellings in _GITIGNORE_LINES.items() if not spellings & present]
    if not missing:
        return "kept", []
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    addition = eol.join(line.encode("utf-8") for line in [_GITIGNORE_COMMENT, *missing]) + eol
    if raw and not raw.endswith(b"\n"):
        addition = eol + addition
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(raw + addition)
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return ("updated" if raw else "created"), missing


def _display(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def cmd_init(args: argparse.Namespace) -> int:
    root = _templates_root()
    if root is None:
        raise CliError("the templates are missing from this evalsprint install (evalsprint/templates/ was not packaged); reinstall the kit")
    base = Path(args.dir)
    if base.exists() and not base.is_dir():
        raise CliError(f"{base} is not a directory")
    if (base / ".gitignore").is_dir():
        raise CliError(f"{base / '.gitignore'} is a directory; cannot write the scaffold file there")  # before anything is written: no half scaffold
    plan = []  # (destination, text)
    for dest_rel, source_rel in _SCAFFOLD + (_SCAFFOLD_CI if args.ci else []):
        text = _template_text(root, source_rel)
        if text is None:
            raise CliError(f"the template for {dest_rel} (evalsprint/templates/{source_rel}) is missing from this evalsprint install; reinstall the kit")
        plan.append((base / dest_rel, text))

    written, kept = [], []
    for dest, text in plan:
        if dest.exists() and not args.force:
            kept.append(dest)  # never overwrite the client's own edits without --force
            continue
        if dest.is_dir():
            raise CliError(f"{dest} is a directory; cannot write the scaffold file there")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            replace_with_retry(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        written.append(dest)

    ignore_state, ignore_added = _extend_gitignore(base)
    _say(f"Scaffold in {base.resolve()}:")
    for dest in written:
        _say(f"  wrote  {_display(dest, base)}")
    for dest in kept:
        _say(f"  kept   {_display(dest, base)}  (already exists; --force replaces it)")
    if ignore_state == "created":
        _say(f"  wrote  .gitignore  ({', '.join(ignore_added)})")
    elif ignore_state == "updated":
        _say(f"  added  {', '.join(ignore_added)} to .gitignore")
    else:
        _say("  kept   .gitignore  (it already ignores runs/ and .label_secret)")
    if kept and not written and ignore_state == "kept":
        _say("Nothing to do: every file exists already.")
    step = iter(range(1, 20))
    where = f"cd {_quote(str(base))} && " if str(base) != "." else ""
    _say("\nNext steps:")
    _say(f"  {next(step)}. Install the kit where the evals run:  {where}pip install -r evals/requirements.txt")
    _say("     (evals/requirements.txt pins the kit to a git tag of github.com/renanpgonzalez/eval-sprint; only a private fork needs the EVALSPRINT_READ_TOKEN secret)")
    _say(f"  {next(step)}. Check the scaffold (offline, no API key):  evalsprint describe -c evals/eval.yaml")
    _say(f"  {next(step)}. Smoke test, offline (no API key, nothing is measured):  JUDGE_PROVIDER=mock evalsprint run -c evals/eval.yaml --label smoke")
    _say("     (PowerShell: $env:JUDGE_PROVIDER = 'mock'; evalsprint run ...)  The target answers with canned text and the LLM judge is a placeholder: this proves the")
    _say("     pipeline runs here, and `evalsprint baseline` refuses such a run (summary.placeholder) until the real target and judge are wired.")
    _say(f"  {next(step)}. Wire the feature: edit evals/target_adapter.py (offline_stub -> call_function / call_http / call_llm), write the rubric in evals/rubric.yaml, replace evals/cases.jsonl with real cases")
    _say(f"  {next(step)}. The real run (the LLM judge is a real call: export its API key first, e.g. ANTHROPIC_API_KEY), then commit the baseline (.gitignore already keeps evals/runs/ out of git):")
    _say("       evalsprint run -c evals/eval.yaml --label baseline && evalsprint baseline evals/runs/<run_id> --to evals/baseline")
    if args.ci:
        _say(f"  {next(step)}. CI: set EVAL_PATHS in .github/workflows/evals.yml, add the API key secrets and EVALSPRINT_READ_TOKEN, and copy .github/CODEOWNERS.evals.example to .github/CODEOWNERS with your team (protect evals/ with code-owner review)")
    else:
        _say(f"  {next(step)}. For the pull-request gate run `evalsprint init --ci` (adds .github/workflows/evals.yml, evals-baseline.yml and CODEOWNERS.evals.example; files that exist are kept)")
    return EXIT_OK


# ------------------------------------------------------------------------ describe
def _offline_judge_cfgs(judge_cfgs: list[dict]) -> list[dict]:
    """LLM judges with the provider swapped for `mock`, so `describe` validates every setting without
    needing an API key or network. The real provider name is still checked."""
    from .providers import _PROVIDERS

    out = []
    for cfg in judge_cfgs:
        if isinstance(cfg, dict) and cfg.get("type") == "llm":
            provider = cfg.get("provider", "anthropic")
            if provider not in _PROVIDERS:
                raise ValueError(f"judge {cfg.get('name') or 'llm'!r}: unknown provider {provider!r}{suggest(provider, _PROVIDERS)} (known: {', '.join(sorted(_PROVIDERS))})")
            cfg = {**cfg, "provider": "mock", "provider_options": {}}
        out.append(cfg)
    return out


def _repeat_text(repeat: dict) -> str:
    per_tag = repeat.get("per_tag") or {}
    return f"n={repeat['n']}" + (f", per_tag={{{', '.join(f'{t}: {k}' for t, k in per_tag.items())}}}" if per_tag else ", all cases")


def _config_warnings(config: SuiteConfig, cases, judges) -> list[str]:
    """Things that are valid but almost certainly not what the author meant."""
    tags = {t for c in cases for t in c.tags}
    warnings = []
    for tag in config.gate.get("critical_tags") or []:
        if tag not in tags:
            warnings.append(f"gate.critical_tags: no case has the tag {tag!r}{suggest(tag, tags)}; the critical-case rule would never fire")
    critical = [t for t in (config.gate.get("critical_tags") or []) if t in tags]
    repeat = config.repeat
    n_default = repeat["n"] if repeat else 1
    per_tag = (repeat or {}).get("per_tag") or {}
    for tag in per_tag:
        if tag not in tags:
            warnings.append(f"repeat.per_tag: no case has the tag {tag!r}{suggest(tag, tags)}")
    attempts = {c.id: _attempts_of(c, n_default, per_tag) for c in cases}
    if repeat:
        repeated = sum(1 for k in attempts.values() if k > 1)
        extra = sum(k - 1 for k in attempts.values())
        warnings.append(f"info: repeat {_repeat_text(repeat)} runs {repeated} case(s) more than once: {extra} extra target call(s) and judge run(s) per run")
    if critical:
        try:
            from .compare import RECOMMENDED_CRITICAL_REPEATS as recommended
        except ImportError:  # compare is optional for describe
            recommended = 5
        short = [t for t in critical if any(attempts[c.id] < recommended for c in cases if t in c.tags)]
        if short:
            spell = ", ".join(f"{t}: {recommended}" for t in short)
            never = all(attempts[c.id] == 1 for c in cases if set(c.tags) & set(critical))
            how = f"adding per_tag: {{{spell}}} under repeat (n: {n_default} stays as it is)" if repeat else f"adding repeat: {{n: 1, per_tag: {{{spell}}}}}"
            warnings.append(
                f"critical tags {critical} are not repeated: one noisy pass->fail blocks a merge. Consider {how}"
                if never
                else f"critical tags {short} have cases that run fewer than {recommended} times: one noisy pass->fail can still block a merge. Consider {how}"
            )
    for group, group_tags in config.tag_groups.items():
        for tag in group_tags:
            if tag not in tags:
                warnings.append(f"tag_groups.{group}: no case has the tag {tag!r}{suggest(tag, tags)}")
    for judge in judges:
        applicable = sum(1 for c in cases if judge.applies(c))
        if applicable == 0:
            warnings.append(f"judge {judge.name!r} applies to no case (check its when_tags / when_behavior)")
    return warnings


def cmd_describe(args: argparse.Namespace) -> int:
    from .judges import build_judges, validate_pass_rule
    from .pricing import PriceTable

    config = SuiteConfig.load(args.config)
    _check_gate(config.gate)
    cases = load_cases(config.dataset, behaviors=config.behaviors)
    judges = build_judges(_offline_judge_cfgs(config.judges), config.base_dir, PriceTable(config.pricing), cases)
    validate_pass_rule(config.pass_rule, [j.name for j in judges])
    info = describe(cases)
    warnings = _config_warnings(config, cases, judges)
    fingerprint_warnings: list[str] = []
    judges_sha, _ = judges_fingerprint(config.judges, config.pass_rule, config.base_dir, fingerprint_warnings)
    warnings.extend(fingerprint_warnings)
    if not judges:
        warnings.append("no judges configured: every case would fail; add judges or per-case checks")

    def table(title: str, counts: dict[str, int]) -> None:
        _say(f"\n{title}")
        for name, count in counts.items():
            _say(f"  {name:<24} {count}")
        if not counts:
            _say("  (none)")

    params = redact_secrets(config.target.get("params") or {})
    try:
        dataset = os.path.relpath(config.dataset, config.base_dir)
    except ValueError:
        dataset = str(config.dataset)
    rows = [
        ("Suite", config.suite),
        ("Dataset", f"{Path(dataset).as_posix()}  (validated)"),
        ("Cases", f"{info['n_cases']} ({info['with_reference']} with a reference answer)"),
        ("Target", config.target["callable"]),
        ("Target params", json.dumps(params, ensure_ascii=False, default=str)),
        ("Pass rule", config.pass_rule if isinstance(config.pass_rule, str) else ", ".join(config.pass_rule)),
        ("Judges sha", f"{judges_sha}  (judge settings, rubrics and custom-judge code: a change of any of them changes it)"),
    ]
    if config.repeat:
        rows.append(("Repeat", _repeat_text(config.repeat)))
    rows.append(("Target retries", f"{config.target_retries}" + (" (a target call that errors is repeated up to that many times inside the run)" if config.target_retries else " (off)")))
    if config.behaviors:
        rows.append(("Behaviors", ", ".join(config.behaviors)))
    _print_rows(rows)
    table("Tags (cases per tag)", info["tags"])
    table("Expected behavior", info["expected_behavior"])
    if config.tag_groups:
        _say("\nTag groups")
        for group, group_tags in config.tag_groups.items():
            _say(f"  {group:<24} {', '.join(group_tags)}")

    _say("\nJudges")
    for judge_cfg, judge in zip(config.judges, judges):
        kind = judge_cfg.get("type", "?")
        model = f"  {judge_cfg.get('provider', 'anthropic')}/{judge_cfg['model']}" if kind == "llm" and judge_cfg.get("model") else ""
        applicable = sum(1 for c in cases if judge.applies(c))
        _say(f"  {judge.name:<24} {kind}{model}  ({applicable}/{len(cases)} cases)")
    for judge in judges[len(config.judges):]:
        _say(f"  {judge.name:<24} case_checks  ({sum(1 for c in cases if judge.applies(c))} cases, per-case checks from the test set, added automatically)")
    if not judges:
        _say("  (none configured)")

    gate = {key: value for key, value in config.gate.items() if value is not None and value != []}
    _say("\nRegression gate\n  " + ", ".join(f"{k}={v}" for k, v in gate.items()))
    if warnings:
        _say("\nWarnings")
        for line in warnings:
            _say(f"  - {line}")
    _say("\nConfig and test set are valid.")
    return EXIT_OK


# ------------------------------------------------------------------------------ noise
_GATE_PARAGRAPH = (
    "How the gate decides (the methodology review's wording, shortened). The gate compares each change with the committed baseline on the "
    "same cases. It blocks the pass-rate rule only when the cases that went from pass to fail outnumber those that went from fail to pass "
    "by at least max(3, n/50), and a one-sided exact sign test (McNemar) says that such an imbalance would be unusual by chance; the test "
    "is lenient on small suites and tightens as the suite grows. It is a noise-aware threshold, not a guarantee: for one committed "
    "baseline the false-alarm rate can be far higher or lower than the average over possible baselines. Cases tagged critical sit outside "
    "this statistical rule: any single pass-to-fail flip blocks, so they should be repeated 5 times (repeat: {n: 3, per_tag: {<critical tag>: 5}})."
)


def _pct(x: float | None) -> str:
    """A rate as a percentage, with the decimals it needs (2 below 1%, else 1); `n/a` when it does not exist."""
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%" if 0 < x < 0.01 else f"{x * 100:.1f}%"


def _wrap(text: str, indent: str = "  ", width: int = 100) -> str:
    import textwrap

    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _rate_pct(x: float | None) -> str:
    """A rate of the gate as a percentage with two decimals (the exact figures of the documentation: 0.21%, 1.11%, 5.79%); `n/a` when it does not exist."""
    return "n/a" if x is None else f"{x * 100:.2f}%"


_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def _short(value, limit: int = 60) -> str:
    """One terminal-safe line out of a value that comes from a run artifact (a tag, a path): control characters (an ANSI escape, a line break) and
    runs of whitespace become one space, and the text is cut at `limit` characters."""
    text = " ".join(_CONTROL.sub(" ", sanitize_text(str(value))).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


NOISE_MIN_REPEATED = 20  # fewer repeated cases than this give no stable estimate: one case more or less that flips moves the flip rate by 5 points


def _noise_gate(args: argparse.Namespace, summary: dict):
    """(the gate layer to price, the effective gate, where it came from, a note or None): the gate `compare` would use.

    `--gate PATH` (CI passes the base branch's copy to `compare --gate-config`) or `-c eval.yaml` are read the way `compare` reads them; without
    either, the gate stored in the run's summary (the `gate:` of the eval.yaml the run used). A stored gate that cannot be read falls back to the
    defaults with a note; one the user named must be valid (a one-line error otherwise)."""
    from . import compare

    explicit = _gate(args.config, args.gate)
    if explicit is not None:
        where = f"the `gate:` of {_short(args.gate, 90)} (--gate)" if args.gate else f"the `gate:` of {_short(args.config, 90)} (-c)"
        return explicit, compare.resolve_gate({}, explicit), where, None
    stored = summary.get("gate") if isinstance(summary.get("gate"), dict) else None
    where = "the gate stored in this run (its summary.json, from the eval.yaml it ran with); `compare` uses it unless it is given -c or --gate-config"
    try:
        return stored, compare.resolve_gate(summary, None), where, None
    except (ValueError, TypeError) as exc:
        note = f"the gate stored in this run ({_describe_error(exc)}) could not be read: the default gate is used"
        return None, compare.resolve_gate({}, {}), "the default gate (the one stored in this run could not be read)", note


def _is_scored(record: dict) -> bool:
    outcome = record.get("outcome")
    return outcome in ("pass", "fail") or (outcome is None and record.get("passed") is not None)


def _attempt_outcomes(record: dict) -> list:
    attempts = record.get("attempts")
    return [a.get("outcome") for a in attempts if isinstance(a, dict)] if isinstance(attempts, list) else []


def cmd_noise(args: argparse.Namespace) -> int:
    """What the gate will do on THIS suite (C19, C21): the false alarm and the detection of the WHOLE gate (pass-rate rule and critical-case rule)
    that `compare.expected_gate_rates_protocol` enumerates for the run's own protocol (the attempts of every scored case, which cases carry a critical
    tag) and the noise its repeated cases measured (`stats.attempt_noise_estimate`)."""
    from . import compare, stats

    run = _load_run(args.run)
    s = run.summary
    scored = [r for r in run.results if _is_scored(r)]
    if not scored or s.get("n_scored") == 0:
        raise CliError(f"no case of run {run.run_id} could be scored, so there is nothing to say about its noise or about a gate on it")
    n = len(scored)
    attempts = [len(_attempt_outcomes(r)) if len(_attempt_outcomes(r)) >= 2 else 1 for r in scored]
    repeated = [r for r, k in zip(scored, attempts) if k >= 2]
    n_flipped = sum(1 for r in repeated if len(set(_attempt_outcomes(r))) > 1)
    n_repeated = len(repeated)
    gate_layer, gate, gate_source, gate_note = _noise_gate(args, s)
    critical_tags = [str(t) for t in (gate.get("critical_tags") or [])]
    critical = [bool(set(str(t) for t in ((r.get("case") or {}).get("tags") or [])) & set(critical_tags)) for r in scored]
    n_critical = sum(critical)

    _say(f"Suite {s.get('suite')}  (run {run.run_id})")
    _say(_wrap(f"Gate used: {gate_source}. Critical tags: " + (", ".join(_short(t) for t in critical_tags) if critical_tags else "none (the critical-case rule does not apply)") + ".", indent=""))
    if n_repeated == 0:
        _say("")
        _say(
            _wrap(
                "Nothing was repeated in this run (every case ran once), so the noise of your target and judge is unknown: there is no flip rate to turn "
                "into the false-alarm and detection rates of the gate. Measure it by running every case 3 times:"
            )
        )
        _say("    evalsprint run -c <eval.yaml> --repeat 3")
        _say("(`--repeat 3` keeps the `per_tag` counts of your eval.yaml, e.g. 5 attempts for the critical cases.) Then run `evalsprint noise` on that run (it costs 3 times the target and judge calls of one run).")
        return EXIT_OK

    pass_rate = s.get("pass_rate")
    baseline_pass = float(pass_rate) if isinstance(pass_rate, (int, float)) and not isinstance(pass_rate, bool) and 0 <= pass_rate <= 1 else 0.8
    ks = [k for k in attempts if k >= 2]
    estimate = stats.attempt_noise_estimate(n_flipped, ks)
    flip_rate = estimate["flip_rate"]
    shown_attempts = f"{ks[0]} attempts each" if len(set(ks)) == 1 else f"{min(ks)} to {max(ks)} attempts"
    repeated_row = ("  Repeated cases", f"{n_repeated} of {n}  ({shown_attempts})")
    rows = [("  Scored cases (n)", f"{n}  (the size of the comparison: the cases scored in both runs)")]
    rows.append(("  Critical cases", f"{n_critical}  (any pass-to-fail flip of one of them blocks the gate)" if n_critical else "0  (the critical-case rule does not apply to this suite)"))
    rows.append(repeated_row)
    if n_repeated < NOISE_MIN_REPEATED:
        rows += [
            ("  Flip rate", f"{_pct(flip_rate)} of the repeated cases had attempts that disagree  ({n_flipped} of {n_repeated}: too few cases to rest a number on)"),
            ("  Noise per attempt", "not estimated"),
        ]
        _say("")
        _say("Measured on this run")
        _print_rows(rows)
        _say("")
        if n_repeated == n:
            # every scored case is already repeated: repeating them again cannot help, the suite is what is too small (the 10 cases of `evalsprint init`)
            all_repeated = f"All {n} scored cases were" if n > 1 else "The one scored case was"
            _say(
                _wrap(
                    f"Noise: not estimated. {all_repeated} repeated, but the suite itself is below {NOISE_MIN_REPEATED} scored cases, and fewer than "
                    f"{NOISE_MIN_REPEATED} repeated cases give no stable estimate: one case that flips or not moves the flip rate by {100 / n_repeated:.0f} points. Repeating "
                    "these cases more often does not change that. The false-alarm and detection rates of the gate are not printed, because they would be priced on a noise "
                    f"nobody measured. Grow the suite to {NOISE_MIN_REPEATED} or more cases ({NOISE_MIN_REPEATED - n} more than now), every one repeated, and run `evalsprint noise` on that run:"
                )
            )
            _say("    evalsprint run -c <eval.yaml> --repeat 3")
            _say(
                _wrap(
                    "Or read the exact tables of the README: they give the false alarm and the detection of the default gate by test-set size, for an assumed noise "
                    "per case (2%, 5%, 10%) instead of a measured one."
                )
            )
        else:
            _say(
                _wrap(
                    f"Noise: not estimated. Only {n_repeated} of the {n} scored cases were repeated, and fewer than {NOISE_MIN_REPEATED} repeated cases give no stable estimate: one case that "
                    f"flips or not moves the flip rate by {100 / n_repeated:.0f} points. The false-alarm and detection rates of the gate are not printed, because they "
                    "would be priced on a noise nobody measured. Repeat every case, and the critical ones more often:"
                )
            )
            _say("    evalsprint run -c <eval.yaml> --repeat 3")
            _say("    # or in eval.yaml:  repeat: {n: 3, per_tag: {<critical tag>: 5}}")
            _say(_wrap(f"then run `evalsprint noise` on that run (every case then counts, {n} of them here)."))
        if gate_note:
            _say(f"note: {gate_note}")
        return EXIT_OK

    bound = n_flipped == 0
    attempt_noise = estimate["noise_upper95"] if bound else estimate["noise"]
    rates = compare.expected_gate_rates_protocol(attempts, critical, attempt_noise, baseline_pass, (0.05, 0.10), gate_layer if gate_layer is not None else {})
    verdict_noise = rates["verdict_noise_mean"]
    if bound:
        rows += [
            ("  Flip rate", f"{_pct(flip_rate)} of the repeated cases had attempts that disagree  (0 of {n_repeated}; upper 95% bound {_pct(estimate['flip_rate_upper95'])})"),
            ("  Noise per attempt", f"about {_pct(estimate['noise'])} measured; the figures below use the upper 95% bound, about {_pct(attempt_noise)}  (the chance that one attempt lands on the wrong side of its case)"),
        ]
    else:
        rows += [
            ("  Flip rate", f"{_pct(flip_rate)} of the repeated cases had attempts that disagree  ({n_flipped} of {n_repeated})"),
            ("  Noise per attempt", f"about {_pct(attempt_noise)}  (the chance that one attempt lands on the wrong side of its case)"),
        ]
    per_case = "mean over the scored cases, each priced at the majority of its own attempts" if len(set(attempts)) > 1 else f"majority of {attempts[0]} attempts"
    rows.append(("  Noise per case verdict", f"about {_pct(verdict_noise)}  ({per_case})"))
    rows.append(("  Pass rate", f"{_pct(baseline_pass)}  (taken as the suite's real pass rate)" if pass_rate is not None else f"{_pct(baseline_pass)}  (no pass rate in the run: the 80% of the documentation)"))
    _say("")
    _say("Measured on this run")
    _print_rows(rows)
    if bound:
        _say("")
        _say(
            _wrap(
                f"No repeated case disagreed with itself, which does not make the suite noiseless: with {n_repeated} repeated cases the true flip rate may still be as high as "
                f"{_pct(estimate['flip_rate_upper95'])} (95% upper bound). Every figure below is priced at that upper 95% bound of the noise, so read it as a ceiling, not as an estimate."
            )
        )
    _say("")
    mode = rates["gate_mode"]
    critical_clause = (
        f"critical-case rule: any pass-to-fail flip of one of the {n_critical} critical cases blocks" if n_critical else "critical-case rule: no scored case carries a critical tag, so it never applies"
    )
    if mode == "off":
        _say(_wrap(f"What the gate does on a suite like this one: its pass-rate rule is off (max_pass_rate_drop: null), so it never blocks on the pass rate; {critical_clause}.", indent=""))
    else:
        rule = (
            f"auto: a net loss of at least {rates['effective_min_net_loss']} case(s) and a one-sided exact McNemar p < {rates['effective_alpha']:.3f}"
            if mode == "auto"
            else f"fixed: a pass-rate drop above {rates['effective_max_pass_rate_drop']:.3f}" + (f" and p < {rates['effective_alpha']:.3f}" if rates.get("effective_alpha") else "")
        )
        _say(_wrap(f"What the gate does on a suite like this one (pass-rate rule, {rule}; {critical_clause}):", indent=""))
    fa, detection, lottery = rates["false_alarm"], rates["detection"], rates["baseline_lottery"]
    priced = "priced at the upper 95% bound of the noise" if bound else ""

    def detect(drop: float) -> str:
        found = detection.get(drop)
        if found is None:
            return "n/a (a drop larger than the pass rate)"
        notes = [n for n in ((f"whole gate; the pass-rate rule alone: {_rate_pct(found['pass_rate_rule'])}" if n_critical else ""), priced) if n]
        return f"about {_rate_pct(found['whole_gate'])}" + (f"  ({'; '.join(notes)})" if notes else "")

    alarm_notes = "; ".join(n for n in ("the whole gate: either rule trips", priced) if n)
    _print_rows(
        [
            ("  False alarm", f"about {_rate_pct(fa['whole_gate'])} of the pull requests that change nothing are blocked  ({alarm_notes})"),
            ("    pass-rate rule alone", f"about {_rate_pct(fa['pass_rate_rule'])}"),
            ("    critical-case rule alone", f"about {_rate_pct(fa['critical_rule'])}" + ("" if n_critical else "  (no critical case: the rule does not apply)")),
            ("  Detection, 5-point drop", detect(0.05)),
            ("  Detection, 10-point drop", detect(0.1)),
        ]
    )
    _say(
        _wrap(
            f"Baseline lottery (pass-rate rule): the committed baseline is one noisy draw, so how often later pull requests are blocked depends on how lucky it was. The chance "
            f"that it leaves more than 15% of the unchanged pull requests blocked is about {_rate_pct(lottery['p_fa_over_15'])} (the median baseline blocks about "
            f"{_rate_pct(lottery['median_fa'])} of them, the 95th percentile baseline about {_rate_pct(lottery['p95_fa'])})."
        )
    )
    if gate_note:
        _say(f"note: {gate_note}")
    _say("")
    even = any(k % 2 == 0 for k in ks)
    _say(
        _wrap(
            "Read every figure as \"about\": they are exact for a model in which the cases are independent and each verdict is wrong with the same probability, "
            "independently in every run. The noise measured on the repeated cases is assumed to be the noise of every case (a case that ran once is priced at "
            "the per-attempt noise). Real suites have a few flaky cases that carry most of the flips, the attempts of a repeated case are not "
            "really independent, and a true drop of d points is modelled as every case losing d in probability (an exact number of broken cases "
            "gives detection from 2.4 points lower to 7.9 points higher). A target that crashes on a case counts as a failure and makes the false alarm worse."
            + (" An even number of attempts counts a tie as a fail: the figures price that exactly, and an odd number (3, 5) is the better choice." if even else "")
        )
    )
    _say("")
    _say(_wrap(_GATE_PARAGRAPH))
    return EXIT_OK


# ------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="evalsprint",
        description="Measure the quality, cost and regressions of an LLM feature.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"evalsprint {__version__}")
    parser.add_argument("--debug", action="store_true", help="show the full traceback on errors instead of a one-line message")
    # Also accepted after the subcommand; SUPPRESS keeps a value given before it from being overwritten.
    common = _Parser(add_help=False)
    common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def command(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text, parents=[common])
        p.set_defaults(func=func)
        return p

    p = command("init", cmd_init, "Scaffold evals/ (eval.yaml, rubric.yaml, cases.jsonl, target_adapter.py, requirements.txt) in a repository; --ci adds the GitHub workflows. Never overwrites without --force.")
    p.add_argument("dir", nargs="?", default=".", metavar="DIR", help="repository root that gets evals/ (default: the current directory)")
    p.add_argument("--ci", action="store_true", help="also write .github/workflows/evals.yml, evals-baseline.yml and .github/CODEOWNERS.evals.example")
    p.add_argument("--force", action="store_true", help="overwrite files that already exist (default: keep them)")

    p = command("describe", cmd_describe, "Validate the test set, the config and the judges an eval.yaml uses, and show their statistics.")
    p.add_argument("-c", "--config", required=True, help="path to eval.yaml")

    p = command("run", cmd_run, "Run the suite and write results, summary and an HTML report. Exit 3 when too many cases errored.")
    p.add_argument("-c", "--config", required=True, help="path to eval.yaml")
    p.add_argument("--label", help="short name for this run, added to the run id (e.g. prompt-v2)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. --set target.params.prompt=v2 (repeatable)")
    p.add_argument("--tags", help="only run cases with at least one of these comma-separated tags")
    p.add_argument("--limit", type=int, metavar="N", help="only run the first N selected cases")
    p.add_argument("--repeat", type=int, metavar="N", help="run every case N times and take the majority (measures the noise floor; overrides `repeat:` in eval.yaml)")
    p.add_argument("--out", metavar="DIR", help="run directory (default: <runs_dir>/<run_id>, never an existing one)")
    p.add_argument(
        "--resume", metavar="DIR",
        help="continue an interrupted run: reuse the finished attempts in DIR/results.partial.jsonl (or a finished DIR/results.jsonl) whose case and run "
        "fingerprint (judges, rubrics, custom-judge code, target callable and params, repeat protocol) are unchanged, and run only the rest; writes a NEW run",
    )
    p.add_argument("--retry-errors", action="store_true", help="with --resume: run again the reusable records whose outcome is `error` (an outage) instead of keeping them")
    p.add_argument("--quiet", action="store_true", help="do not print per-case progress")
    p.add_argument("--no-report", action="store_true", help="do not write report.html")

    p = command("compare", cmd_compare, "Compare a candidate run with a baseline; exit 1 on regression. Loads <baseline>/calibration.json when present.")
    p.add_argument("baseline", metavar="BASELINE_RUN", help="baseline run directory")
    p.add_argument("candidate", metavar="CANDIDATE_RUN", help="candidate run directory")
    p.add_argument("-c", "--config", help="eval.yaml whose gate thresholds apply (default: the candidate run's own)")
    p.add_argument("--gate-config", metavar="PATH", help="take the gate ONLY from the gate: block of this YAML (CI: the base branch's copy, so a PR cannot loosen its own gate); wins over -c")
    p.add_argument("--md", metavar="FILE", help="also write the markdown summary here")
    p.add_argument("--json", metavar="FILE", help="also write the full comparison as JSON here")

    p = command("noise", cmd_noise, "Turn the noise a repeated run measured (its flip rate) into the false-alarm and detection rates of the whole gate (pass-rate rule and critical-case rule) for that suite and its repeat protocol. Needs a run with at least 20 repeated cases (`run --repeat 3`); no labels, no key.")
    p.add_argument("run", metavar="RUN", help="run directory of a run whose cases ran more than once")
    p.add_argument("--gate", "--gate-config", dest="gate", metavar="PATH", help="price the gate defined by the gate: block of this YAML (CI: the base branch's copy, the one `compare --gate-config` uses); wins over -c")
    p.add_argument("-c", "--config", help="eval.yaml whose gate applies (default: the gate stored in the run, which is what `compare` uses without -c or --gate-config)")

    p = command("report", cmd_report, "Write the HTML report for a run, optionally with a baseline comparison and judge calibration.")
    p.add_argument("run", metavar="RUN", help="run directory")
    p.add_argument("--baseline", metavar="RUN", help="compare against this baseline run (also shows its outputs next to the candidate's)")
    p.add_argument("-c", "--config", help="eval.yaml whose gate thresholds apply to --baseline (default: the run's own)")
    p.add_argument("--gate-config", metavar="PATH", help="take the gate ONLY from the gate: block of this YAML (needs --baseline)")
    p.add_argument("--labels", action="append", metavar="FILE", help="human labels (.jsonl or .csv) to calibrate the judge against; repeat the option to read several files as one")
    p.add_argument("--judge", help="calibrate this judge instead of the overall verdict (needs --labels)")
    p.add_argument("--monthly-requests", type=int, metavar="N", help="project the monthly cost of the feature at N requests per month")
    p.add_argument("-o", "--output", metavar="FILE", help="report path (default: <run>/report.html)")

    p = command("calibrate", cmd_calibrate, "Measure how well the judge agrees with human labels.")
    p.add_argument("run", metavar="RUN", help="run directory")
    p.add_argument("--labels", required=True, action="append", metavar="FILE", help="human labels (.jsonl or .csv); repeat the option to read several files as one (a later label for the same case replaces an earlier one)")
    p.add_argument("--judge", help="calibrate this judge instead of the overall verdict")
    p.add_argument("--md", metavar="FILE", help="also write the markdown summary here")
    p.add_argument("--json", metavar="FILE", help="also write the full result as JSON here (use <run or baseline>/calibration.json so compare can find it)")
    p.add_argument("--export-fewshot", metavar="PATH", help="write the clearest human-vs-judge disagreements as rubric `examples:` (YAML), plus `holdout_case_ids:` to leave out of the next calibration")
    p.add_argument("--fewshot-max", type=int, metavar="N", help="with --export-fewshot: at most N examples (default: 6)")
    p.add_argument("--holdout-frac", type=float, metavar="F", help="with --export-fewshot: share of the labeled cases set aside for the examples, strictly between 0 and 1 (default: 0.3)")
    p.add_argument("--exclude-holdout", metavar="FILE", help="leave out the cases listed in the holdout_case_ids of this --export-fewshot file (calibrate the judge only on labels it was not shown)")

    p = command("label", cmd_label, "Export a blind labeling page (or CSV) for human review of a run's outputs.")
    p.add_argument("run", metavar="RUN", help="run directory")
    p.add_argument("-o", "--output", metavar="FILE", help="output path (default: <run>/labeling.html or labeling.csv)")
    p.add_argument("--csv", action="store_true", help="write a CSV sheet instead of the HTML page")
    p.add_argument("--sample", type=int, metavar="N", help="label only N cases (default: all)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument("--judge", metavar="NAME", help="stratify the sample on this judge's verdict instead of the overall one (label the cases you will calibrate that judge with)")
    p.add_argument("--no-stratify", action="store_true", help="sample at random instead of balancing passing and failing cases")
    p.add_argument("--previous-labels", action="append", metavar="FILE", help="labels you already have (.jsonl or .csv; repeat the option for several files). Skip every case that already has a label for its current output. Only the changed outputs need a label when the earlier labels covered every unchanged case; otherwise the export also draws never-labeled unchanged cases.")

    p = command("history", cmd_history, "Write one HTML page with the trend of every finished run under a directory (pass rate, errors, cost, latency, judge changes).")
    p.add_argument("runs_dir", metavar="RUNS_DIR", help="directory that holds the run directories (the runs_dir of eval.yaml)")
    p.add_argument("-o", "--output", metavar="FILE", help="output path (default: <RUNS_DIR>/history.html)")
    p.add_argument("--suite", metavar="NAME", help="only the runs of this suite")

    p = command("baseline", cmd_baseline, "Copy a run's results.jsonl, summary.json (and calibration.json) to a directory you commit as the baseline.")
    p.add_argument("run", metavar="RUN", help="run directory")
    p.add_argument("--to", required=True, metavar="DIR", help="destination, e.g. evals/baseline")
    p.add_argument("--force", action="store_true", help="copy even if the run is unhealthy (too many target/judge errors)")
    return parser


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        message = f"file not found: {exc.filename or exc}"
    elif isinstance(exc, yaml.YAMLError):
        message = f"invalid YAML: {exc}"
    elif isinstance(exc, (CliError, ValueError, OSError)):
        message = str(exc)
    else:
        message = f"{type(exc).__name__}: {exc} (run with --debug for the traceback)"
    return " ".join(message.split())


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    debug = False
    try:
        args = build_parser().parse_args(argv)
        debug = args.debug
        code = args.func(args)
        _finish_stdout()
        return code
    except SystemExit as exc:  # --help and --version
        return int(exc.code or 0)
    except KeyboardInterrupt as exc:
        terminated = isinstance(exc, _Terminated)
        _warn("terminated" if terminated else "interrupted")
        if getattr(exc, "resume_hint", None):
            _warn(exc.resume_hint)
        return EXIT_TERMINATED if terminated else EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - every failure becomes exit code 2, never a bare traceback (exit 1 = regression)
        if debug:
            raise
        _warn(f"error: {_describe_error(exc)}")
        if getattr(exc, "resume_hint", None):
            _warn(exc.resume_hint)
        return EXIT_USAGE
