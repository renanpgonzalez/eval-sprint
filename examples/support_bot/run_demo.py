#!/usr/bin/env python3
"""Run the Ledgerly support-bot demo end to end.

    python run_demo.py                     # offline: bot and judge run on the simulator in sim.py
    python run_demo.py --real              # real models (needs ANTHROPIC_API_KEY); shows a cost estimate first
    python run_demo.py --real --yes        # ... and does not ask for confirmation
    python run_demo.py --out my_dir        # where to write the results (default ./demo_out)
    python run_demo.py --monthly-requests 250000   # volume for the monthly cost projection (default 100000)

The story it tells:

    v1  baseline   generic prompt, top_k 3, Sonnet
    v2  grounded   production prompt (answer only from context, abstain, refuse, cite)
    v3  cost cut   v2 prompt + top_k 1 + Haiku, to save money

Each variant runs through the same suite (eval.yaml; the critical safety cases repeated, majority verdict), v1 -> v2 and
v2 -> v3 are compared with the regression gate, the v2 run is calibrated against labels (offline: synthetic
labels), and every run gets a self-contained report.html. SUMMARY.md puts it all in one table. Offline, a fourth run (v2 with a
stricter length judge) is added so that history.html, the quality and cost trend over the runs, has a judge change to mark, and a fifth
(v2 with every case repeated, kept in noise_check/ and out of the history) gives `evalsprint noise` a run it can price: with only the 8 safety cases
repeated the command prints "not estimated" and no rate (fewer than 20 repeated cases). Every offline run is a placeholder run (the judge is the mock provider): the demo never records a baseline.

OFFLINE NUMBERS COME FROM A SIMULATOR (sim.py), NOT FROM A LANGUAGE MODEL. Every claim printed below is
tagged [MECHANISM] (code that really runs), [SCRIPTED] (true by construction of the simulator) or
[INVENTED] (a rate picked for the story). They prove the pipeline works; they are not a benchmark of any model.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

try:
    import evalsprint  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from evalsprint import calibration, compare, report  # noqa: E402
from evalsprint.config import SuiteConfig  # noqa: E402
from evalsprint.loader import load_callable  # noqa: E402
from evalsprint.pricing import PriceTable  # noqa: E402
from evalsprint.runner import RunResult, run_suite  # noqa: E402
from evalsprint.types import Usage  # noqa: E402

# The three variants: (key, title, --set overrides applied to eval.yaml)
VARIANTS = [
    ("v1-baseline", "v1 baseline (generic prompt)", []),
    ("v2-grounded", "v2 grounded prompt", ["target.params.prompt=v2"]),
    (
        "v3-cost-cut",
        "v3 cost cut (v2 prompt, top_k 1, Haiku)",
        ["target.params.prompt=v2", "target.params.top_k=1", "target.params.model=claude-haiku-4-5"],
    ),
]
# Offline only: v2 again, with the `length` judge (judges.2 in eval.yaml) tightened from 1,200 to 300 characters. The bot did not change, the
# judges did, so history.html marks the run ("judges changed") and the pass-rate drop is not read as a worse bot.
HISTORY_VARIANT = ("v4-stricter-judge", "v2 grounded prompt, length judge tightened to 300 characters", ["target.params.prompt=v2", "judges.2.value=300"])
# Offline only: v2 again with every case repeated (what `evalsprint run --repeat 3` does: n = 3 and the safety cases keep their 5). It is the run
# `evalsprint noise` can price, which SUMMARY.md sets beside the "not estimated" of the 8 safety cases; it is not part of the story, the history or the gate.
NOISE_CHECK = ("v2-noise-check", "v2 grounded prompt, every case 3 times and the safety cases 5", ["target.params.prompt=v2", "repeat={n: 3, per_tag: {safety: 5}}"])
TYPES = ["answerable", "multi_part", "out_of_scope", "safety"]  # the type tag of each case
CALIBRATED_VARIANT = "v2-grounded"
DEFAULT_MONTHLY_REQUESTS = 100_000  # an assumption for the projection, not a measurement: pass --monthly-requests

# Assumed sizes of what a real model WRITES, for the pre-run cost estimate (input sizes are measured on the real prompts).
EST_BOT_OUTPUT_TOKENS = 180  # v2 asks for at most 120 words
EST_JUDGE_OUTPUT_TOKENS = 350  # per judge sample: five criteria, each with a short reasoning and a verdict

UNCALIBRATED_NOTICE = (
    "The verdicts below come from an UNCALIBRATED LLM judge: nobody has compared it with human labels yet.\n"
    "Treat every pass rate as provisional until you label a sample and run `evalsprint calibrate` (steps are\n"
    "printed at the end). The regression gate is computed without a calibration and its report says so."
)


# ------------------------------------------------------------------------------ formatting
def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def pct_ci(summary: dict) -> str:
    lo, hi = summary["pass_rate_ci"]
    return f"{pct(summary['pass_rate'])} ({lo * 100:.0f}-{hi * 100:.0f}%)"


def usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def by_type(summary: dict, tag: str) -> str:
    t = summary.get("by_tag", {}).get(tag)
    return "n/a" if not t else f"{t['pass_rate'] * 100:.0f}% ({t['n_passed']}/{t['n']})"


def errors_text(summary: dict) -> str:
    n, target, judge = summary.get("n_errors", 0), summary.get("n_target_errors", 0), summary.get("n_judge_errors", 0)
    return "0" if not n else f"{n} (target {target}, judge {judge})"


def flip_text(summary: dict) -> str:
    st = summary.get("stability") or {}
    if not st.get("n_repeated"):
        return "n/a"
    return f"{st['flip_rate'] * 100:.0f}% of {st['n_repeated']} repeated cases"


def say(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:  # legacy Windows consoles cannot print the gate's arrows
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def count_cases() -> int:
    lines = (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.startswith("//"))


def repeat_attempts(tag: str = "safety") -> int:
    """How many times eval.yaml runs a case tagged `tag` (C17): the `per_tag` count of that tag, else `n`; 1 when it has no `repeat:`.

    The config normalises every spelling (`{n: 5, tags: [safety]}` is `{n: 1, per_tag: {safety: 5}}`), so `n` alone is NOT the answer:
    with `repeat: {n: 1, per_tag: {safety: 5}}` the safety cases run 5 times and `n` is 1."""
    repeat = SuiteConfig.load(HERE / "eval.yaml").repeat
    if not repeat:
        return 1
    per_tag = repeat.get("per_tag") or {}
    return int(per_tag.get(tag, repeat["n"]))


def sim_constants() -> dict:
    """The simulator's invented rates, read from sim.py so the text can never drift from the code."""
    return {
        "tiers": load_callable("sim.py:TIERS", HERE),
        "noise": load_callable("sim.py:DEFAULT_NOISE", HERE),
        "handoff": load_callable("sim.py:HANDOFF_ON_INNATE_ABSTAIN", HERE),
    }


# ------------------------------------------------------------------------------ the cost estimate for --real
def estimate_real_cost(extra_overrides: list[str] | tuple[str, ...] = ()) -> dict:
    """Estimate what `--real` will spend: cases x variants x (bot + judge x samples), priced with pricing.yaml.

    A dry run on the simulator measures the size of every real prompt (tokens = characters / 4) with the real
    model ids from the configuration. What a real model writes is not known in advance, so the output size is an
    assumption (EST_BOT_OUTPUT_TOKENS, EST_JUDGE_OUTPUT_TOKENS). Repeated cases count once per attempt.
    Treat the total as +-50%: real answers and retries differ from the simulator's.
    `extra_overrides` (`--set` strings, e.g. `["repeat={n: 3}"]`) prices another repeat protocol than eval.yaml's.
    """
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for key, title, overrides in VARIANTS:
            config = SuiteConfig.load(HERE / "eval.yaml", overrides=overrides + list(extra_overrides) + ["target.params.provider=mock", "judges.0.provider=mock"])
            prices = PriceTable(config.pricing)
            judge_cfg = config.judges[0]
            samples = max(1, int(judge_cfg.get("samples", 1)))
            judge_model = judge_cfg["model"]
            bot_model = config.target["params"].get("model")
            run = run_suite(config, label=f"estimate-{key}", out_dir=Path(tmp) / key, progress=False)
            n_calls, bot_usd, judge_usd = 0, 0.0, 0.0
            for rec in run.results:
                attempts = len(rec.get("attempts") or []) or 1
                n_calls += attempts
                bot_in = (rec["output"].get("usage") or {}).get("input_tokens", 0)
                bot_usd += attempts * (prices.cost(bot_model, Usage(input_tokens=bot_in, output_tokens=EST_BOT_OUTPUT_TOKENS)) or 0.0)
                judge = next((j for j in rec["judges"] if j["judge"] == judge_cfg.get("name", "llm")), None)
                if judge is not None:
                    judge_in = (judge.get("usage") or {}).get("input_tokens", 0)  # already summed over the samples
                    judge_usd += attempts * (
                        prices.cost(judge_model, Usage(input_tokens=judge_in, output_tokens=samples * EST_JUDGE_OUTPUT_TOKENS)) or 0.0
                    )
            rows.append(
                {"variant": key, "title": title, "attempts": n_calls, "bot_model": bot_model, "judge_model": judge_model,
                 "judge_samples": samples, "bot_usd": bot_usd, "judge_usd": judge_usd}
            )
    total = sum(r["bot_usd"] + r["judge_usd"] for r in rows)
    return {"rows": rows, "total_usd": total, "low_usd": total * 0.5, "high_usd": total * 1.5}


def print_estimate(est: dict) -> None:
    say("Estimated API cost of this run (cases x variants x (bot + judge x samples), prices from evalsprint/pricing.yaml):")
    for r in est["rows"]:
        say(
            f"  {r['title']:<42} {r['attempts']:>3} case runs   bot {r['bot_model']:<18} {usd(r['bot_usd']):>8}   "
            f"judge {r['judge_model']} x{r['judge_samples']} samples {usd(r['judge_usd']):>8}"
        )
    say(f"  TOTAL about {usd(est['total_usd'])} (plausible range {usd(est['low_usd'])} to {usd(est['high_usd'])}).")
    say(
        f"  Basis: prompt sizes measured on the real prompts; assumed output {EST_BOT_OUTPUT_TOKENS} tokens per bot answer and "
        f"{EST_JUDGE_OUTPUT_TOKENS} per judge sample. Judge retries and longer real answers are not included."
    )


def confirm(question: str) -> bool:
    """Ask on stdin. No terminal, an empty answer or anything but y/yes is a no."""
    try:
        answer = input(question)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


# ------------------------------------------------------------------------------ main steps
def run_variants(out: Path, real: bool) -> dict[str, RunResult]:
    runs: dict[str, RunResult] = {}
    for key, title, overrides in VARIANTS:
        t0 = time.perf_counter()
        config = SuiteConfig.load(HERE / "eval.yaml", overrides=overrides)
        label = key if real else f"{key}-SIMULATED"  # the label shows up in every report header
        run = run_suite(config, label=label, out_dir=out / "runs" / key, progress=False)
        s = run.summary
        say(
            f"  {title:<42} pass {pct(s['pass_rate']):>6} of {s['n_scored']} scored   errors {errors_text(s):<3}   "
            f"flip rate {flip_text(s):<28} cost/1k {usd(s['cost']['system_per_1k_usd']):>8}   ({time.perf_counter() - t0:.1f}s)"
        )
        runs[key] = run
    return runs


def compare_steps(runs: dict[str, RunResult], out: Path) -> dict[str, dict]:
    comparisons: dict[str, dict] = {}
    keys = [v[0] for v in VARIANTS]
    for base_key, cand_key in zip(keys, keys[1:]):
        result = compare.compare_runs(runs[base_key], runs[cand_key])  # no calibration: the judge is not calibrated for this gate
        comparisons[cand_key] = result
        stem = f"compare_{base_key.split('-')[0]}_to_{cand_key.split('-')[0]}"
        (out / f"{stem}.md").write_text(compare.to_markdown(result), encoding="utf-8")
        (out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        say(f"  gate {base_key} -> {cand_key}: {result['verdict'].upper()}   (mode: {gate_rule(result)})")
        for reason in result.get("reasons", []):
            say(f"      - {reason}")
    return comparisons


def _limit(comparison: dict) -> str:
    limit = comparison.get("effective_max_pass_rate_drop")
    return "none" if limit is None else f"{limit * 100:.1f} p.p."


def gate_rule(comparison: dict) -> str:
    """The rule the gate applied, in a few words: `auto: net loss >= 3 cases and McNemar alpha 0.24` (old comparison files: the limit)."""
    mode = compare.normalize_gate_mode(comparison.get("gate_mode"))
    if mode == "auto" and comparison.get("effective_min_net_loss") is not None and comparison.get("effective_alpha") is not None:
        return f"auto: net loss >= {comparison['effective_min_net_loss']} cases and McNemar alpha {comparison['effective_alpha']:.2f}"
    return f"{mode}, limit {_limit(comparison)}"


PLACEHOLDER_NOTICE = (
    "Every run here is a PLACEHOLDER run (`placeholder: true` in its `summary.json`): its judge runs on the mock provider, so its verdicts say nothing "
    "about a real product. `evalsprint baseline` refuses to record such a run without `--force`, `evalsprint compare` warns about it on every gate below, "
    "and each report opens with a banner. This demo never records a baseline (it compares two runs directly), so it needs no `--force`; "
    "the warning is the kit telling the truth about these runs, not a fault of the demo."
)


def noise_output(run: RunResult) -> str | None:
    """What `evalsprint noise <run dir>` prints for this run, exactly as the command prints it (None when it fails)."""
    from evalsprint import cli

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["noise", str(run.run_dir)])
    except SystemExit as exc:
        code = exc.code
    return buffer.getvalue().rstrip() if code in (0, None) and buffer.getvalue().strip() else None


_NOISE_LINES = (  # key, pattern: one line of `evalsprint noise` each; the groups are the figures
    ("repeated", r"Repeated cases:\s+(\d+) of (\d+)"),
    ("flip", r"Flip rate:\s+([\d.]+%)[^\n]*?\((\d+) of (\d+)"),
    ("noise_per_attempt", r"Noise per attempt:\s+([^\n]*)"),
    ("false_alarm", r"False alarm:\s+about ([\d.]+%)"),
    ("false_alarm_pass_rate", r"pass-rate rule alone:\s+about ([\d.]+%)\s*\n"),
    ("false_alarm_critical", r"critical-case rule alone:\s+about ([\d.]+%)"),
    ("detect_5", r"Detection, 5-point drop:\s+about ([\d.]+%)"),
    ("detect_10", r"Detection, 10-point drop:\s+about ([\d.]+%)"),
)


def noise_numbers(text: str | None) -> dict | None:
    """The figures of `evalsprint noise` that the summary and the README quote, read from what the command printed (C21).

    A run with fewer than 20 repeated cases prints `Noise per attempt: not estimated` and no rate: then `estimated` is False and every rate is None.
    Keys: estimated, n_repeated, n_scored, flip_rate ("8.9%"), n_flipped, upper_bound (True when nothing flipped and the command priced the upper 95% bound),
    noise_per_attempt ("2.7%", the figure the rates are priced at), false_alarm (the whole gate), false_alarm_pass_rate, false_alarm_critical, detect_5, detect_10
    (percent strings as printed). None when the text is not the output of the command."""
    if not text:
        return None
    found: dict = {}
    for key, pattern in _NOISE_LINES:
        match = re.search(pattern, text)
        found[key] = match.groups() if match else None
    if found["repeated"] is None or found["flip"] is None or found["noise_per_attempt"] is None:
        return None
    line = found["noise_per_attempt"][0]
    estimated = not line.startswith("not estimated")
    percents = re.findall(r"about ([\d.]+%)", line)
    if estimated and not percents:
        return None
    figure = lambda key: found[key][0] if estimated and found[key] else None  # noqa: E731
    if estimated and any(found[k] is None for k in ("false_alarm", "false_alarm_pass_rate", "false_alarm_critical", "detect_5", "detect_10")):
        return None
    return {
        "estimated": estimated,
        "n_repeated": int(found["repeated"][0]),
        "n_scored": int(found["repeated"][1]),
        "flip_rate": found["flip"][0],
        "n_flipped": int(found["flip"][1]),
        "upper_bound": estimated and "upper 95% bound" in line,  # no case flipped: every figure is priced at the upper 95% bound of the noise
        "noise_per_attempt": percents[-1] if estimated else None,  # the figure the rates are priced at
        "false_alarm": figure("false_alarm"),
        "false_alarm_pass_rate": figure("false_alarm_pass_rate"),
        "false_alarm_critical": figure("false_alarm_critical"),
        "detect_5": figure("detect_5"),
        "detect_10": figure("detect_10"),
    }


NOISE_VS_AA_FILE = "noise_vs_aa.json"  # next to this script, like salt_robustness.json
# The price of the recommended `repeat` line in this simulator (README, "How much noise the gate tolerates"; tests pin these to salt_robustness.json):
# false alarms of the whole gate in the pooled A/A pairs with `{n: 1, per_tag: {safety: 5}}` and with `{n: 3, per_tag: {safety: 5}}`, and the extra case runs.
REPEAT_GAIN = {"alarms_safety5": 87, "alarms_pertag": 74, "pairs": 600, "more_runs_pct": 96}
_AA_KEYS = ("salts", "aa_alarms", "aa_ci95", "aa_pass_rule", "aa_critical", "printed_whole_mean", "printed_whole", "expected_alarms", "flipped_safety_pct", "flipped_other_pct", "by_printed_whole")


def noise_vs_aa() -> dict | None:
    """The kept comparison of what `evalsprint noise` prints with what the A/A measures, per variant, under the per-tag protocol `repeat: {n: 3, per_tag: {safety: 5}}`
    (`noise_vs_aa.py`, kept next to the README as noise_vs_aa.json): its `summary` dict, {"v1": {...}, "v2": {...}, "v3": {...}}; None when the file is not there or
    does not have every figure the summary quotes."""
    try:
        summary = json.loads((HERE / NOISE_VS_AA_FILE).read_text(encoding="utf-8"))["summary"]
        for variant in ("v1", "v2", "v3"):
            missing = [k for k in _AA_KEYS if k not in summary[variant]]
            if missing or summary[variant]["by_printed_whole"].get("under 1%") is None or set(summary[variant]["printed_whole"]) < {"median", "max"}:
                return None
        return summary
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def calibration_step(runs: dict[str, RunResult], out: Path, real: bool) -> dict | None:
    if real:
        say(
            "  [skip] synthetic labels are for the offline demo only. With real models, label a sample yourself:\n"
            "         evalsprint label <run_dir> --judge quality -o labeling.html   then   evalsprint calibrate <run_dir> --labels labels.jsonl --judge quality"
        )
        return None
    make_labels = load_callable("sim.py:make_synthetic_labels", HERE)
    run = runs[CALIBRATED_VARIANT]
    labels_path = make_labels(run.run_dir, out / "labels_SYNTHETIC_v2.jsonl", n=40, seed=0)
    labels = calibration.load_labels(labels_path)
    result = calibration.calibrate(run, labels, judge="quality")
    (out / "calibration_v2_SYNTHETIC.md").write_text(calibration.to_markdown(result), encoding="utf-8")
    (out / "calibration_v2_SYNTHETIC.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    say(
        f"  quality judge vs SYNTHETIC labels on v2: n={result['n']}, agreement {pct(result['agreement'])}, "
        f"{_kappa_word(result)}kappa {_kappa(result)}, decision: {result.get('kappa_decision')}"
    )
    for warning in result.get("warnings") or []:
        say(f"      warning: {warning}")
    return result


def _has_weighted(cal: dict) -> bool:
    return "kappa_weighted" in cal or "kappa_weighted_ci" in cal


def _fmt_kappa(kappa: float | None, ci) -> str:
    if kappa is None:
        return "n/a"
    return f"{kappa:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f})" if ci else f"{kappa:.2f}"


def _kappa_word(cal: dict) -> str:
    return "weighted " if _has_weighted(cal) else ""


def _kappa(cal: dict) -> str:
    """The kappa the judge decision is made on: the weighted one of the whole run (the unweighted pair for old calibration dicts)."""
    if _has_weighted(cal):
        return _fmt_kappa(cal.get("kappa_weighted"), cal.get("kappa_weighted_ci"))
    return _fmt_kappa(cal.get("kappa"), cal.get("kappa_ci"))


def report_step(runs: dict[str, RunResult], comparisons: dict[str, dict], calibration_result: dict | None, monthly_requests: int) -> list[Path]:
    keys = [v[0] for v in VARIANTS]
    written = []
    for i, key in enumerate(keys):
        cal = calibration_result if key == CALIBRATED_VARIANT else None
        previous = runs[keys[i - 1]] if i else None  # the run this one was compared with: shown side by side for new failures
        written.append(
            report.write_report(
                runs[key].run_dir / "report.html", runs[key], comparisons.get(key), cal, baseline=previous, monthly_requests=monthly_requests
            )
        )
    return written


def history_step(runs: dict[str, RunResult], out: Path, real: bool, monthly_requests: int) -> tuple[Path, RunResult | None, dict | None]:
    """history.html: quality and cost over the runs. Offline, a fourth run (v2 under a stricter length judge) is added first.

    Real mode does not add it: it would be one more paid run of the v2 configuration only to draw a marker.
    """
    ordered = list(runs.values())
    extra, gate = None, None
    if not real:
        key, title, overrides = HISTORY_VARIANT
        config = SuiteConfig.load(HERE / "eval.yaml", overrides=overrides)
        extra = run_suite(config, label=f"{key}-SIMULATED", out_dir=out / "runs" / key, progress=False)
        report.write_report(extra.run_dir / "report.html", extra, None, None, monthly_requests=monthly_requests)
        ordered.append(extra)
        gate = compare.compare_runs(runs["v2-grounded"], extra)  # the gate refuses it on purpose: the judges changed
        s = extra.summary
        say(f"  {title:<62} pass {pct(s['pass_rate']):>6}   judges fingerprint {str(s.get('judges_sha'))[:8]} (v2: {str(runs['v2-grounded'].summary.get('judges_sha'))[:8]})")
    path = report.write_history(out / "history.html", ordered)
    return path, extra, gate


def noise_check_step(out: Path, real: bool) -> RunResult | None:
    """Offline only (real mode would pay for another 151 case runs): v2 with every case repeated, a run `evalsprint noise` can price."""
    if real:
        return None
    key, _title, overrides = NOISE_CHECK
    config = SuiteConfig.load(HERE / "eval.yaml", overrides=overrides)
    run = run_suite(config, label=f"{key}-SIMULATED", out_dir=out / "noise_check", progress=False)  # not under runs/: `evalsprint history <out>/runs` must not pick it up
    say(f"  {NOISE_CHECK[1]:<70} flip rate {flip_text(run.summary)}")
    return run


# ------------------------------------------------------------------------------ SUMMARY.md
def _provenance_lines(runs: dict[str, RunResult]) -> list[str]:
    c = sim_constants()
    tiers = c["tiers"]
    slip = ", ".join(f"{name} {tiers[name]['slip'] * 100:.0f}%" for name in ("haiku", "sonnet", "opus") if name in tiers)
    innate = ", ".join(f"{name} {tiers[name]['innate'] * 100:.0f}%" for name in ("haiku", "sonnet", "opus") if name in tiers)
    v1, v2, v3 = (runs[v[0]].summary for v in VARIANTS)

    def rate(s: dict, tag: str) -> str:
        t = s["by_tag"].get(tag)
        return "n/a" if not t else f"{t['pass_rate'] * 100:.0f}%"

    return [
        "## What in this simulated run is real, scripted or invented",
        "",
        "Every sentence about *why* a number moved belongs to one of three kinds. Only the first is evidence about the kit.",
        "",
        "- **[MECHANISM] code that really runs.** The BM25 retrieval over the help center, the harness (repeats, majority vote, errors kept out of "
        "the pass rate), the deterministic checks, the statistics (Wilson intervals, McNemar, kappa), the gate and the cost arithmetic "
        "(list prices x tokens, with tokens = characters / 4).",
        f"  - Multi-part questions: {rate(v2, 'multi_part')} on v2, {rate(v3, 'multi_part')} on v3. v3 retrieves one section (`top_k` 1), and a "
        "question whose two facts live in two sections cannot be answered completely from one. The missing half comes from the retrieval; what the "
        "simulated bot does about it (answer half, or abstain) is scripted.",
        "- **[SCRIPTED] true by construction.** The simulated bot follows the prompt's rules the way a language model is supposed to: a prompt "
        "without the rule leaks, promises, invents or complies; a prompt with it does not. So a weak prompt failing safety and out-of-scope cases is a "
        f"property of the simulator, not a finding (out of scope {rate(v1, 'out_of_scope')} on v1 vs {rate(v2, 'out_of_scope')} on v2; "
        f"safety {rate(v1, 'safety')} vs {rate(v2, 'safety')}). The same goes for the simulated judge and the synthetic labels: two rule sets written for this demo.",
        f"- **[INVENTED] rates picked to look like a real run.** Chance that a model ignores a rule on one attempt ({slip}); that a model "
        f"declines or says \"I don't know\" without being told to ({innate}); general run-to-run noise ({c['noise'] * 100:.0f}% of samples); how "
        "often a menu path is written with an arrow or bold. None of them is measured, so **the flip rates, the safety rates of the smaller model "
        "and whether a safety case flips are properties of these numbers**, and they change when the sampling salt changes.",
        "",
    ]


def _noise_vs_aa_lines(noise_run: RunResult, ab: dict) -> list[str]:
    """What the command printed for the run it CAN price (v2, `repeat: {n: 3, per_tag: {safety: 5}}`), set beside the A/A of the 200 salts for the same protocol
    (noise_vs_aa.json, written by noise_vs_aa.py): where the two agree and where they do not, and why. [INVENTED]: the noise is the simulator's."""
    v1, v2, v3 = ab["v1"], ab["v2"], ab["v3"]
    pc = lambda x: f"{x:.2f}%"  # noqa: E731
    pc1 = lambda x: f"{x:.1f}%"  # noqa: E731
    pc0 = lambda x: f"{x:.0f}%"  # noqa: E731
    salts = v2["salts"]
    here = noise_numbers(noise_output(noise_run))
    lines = [
        f"**[INVENTED] Is that figure the false alarm of this demo? Set it beside the A/A.** The simulator draws the same suite again under {salts} salts "
        f"(`noise_vs_aa.py`, `noise_vs_aa.json`; README, \"How much noise the gate tolerates\"). For every salt and variant, under this same protocol: run, ask "
        "`evalsprint noise` about that run, run again under another salt and ask the gate. Every REGRESSION of the second step is a false alarm of the whole gate:",
        "",
        f"| Variant | `evalsprint noise`, whole gate: mean over the {salts} first runs (median) | A/A: false alarms in {salts} pairs | pass-rate rule | critical-case rule |",
        "|---|---|---|---|---|",
    ]
    for name, title in (("v1", "v1 baseline"), ("v2", "v2 grounded"), ("v3", "v3 cost cut")):
        a = ab[name]
        lines.append(
            f"| {title} | {pc(a['printed_whole_mean'])} ({pc(a['printed_whole']['median'])}) | {a['aa_alarms']} of {a['salts']} ({pc1(a['aa_alarms'] / a['salts'] * 100)}) | "
            f"{a['aa_pass_rule']} | {a['aa_critical']} |"
        )
    lines += [
        "",
        f"For v2, the variant the demo is about, the two agree that the false alarm is small: the command printed {pc(v2['printed_whole_mean'])} on average over the {salts} runs "
        f"(median {pc(v2['printed_whole']['median'])}, at most {pc(v2['printed_whole']['max'])}"
        + (f"; {here['false_alarm']} for the run above" if here and here["estimated"] else "")
        + f") and the A/A raised {v2['aa_alarms']} false alarms in {salts} pairs (95% interval {pc1(v2['aa_ci95'][0])} to {pc1(v2['aa_ci95'][1])}). That is weak evidence: "
        f"{salts} pairs cannot tell 0.1% from 1%, and the printed figures add up to {v2['expected_alarms']} expected alarms, so {v2['aa_alarms']} is what both predict.",
        "",
        f"For v1 and v3 they disagree. The command printed {pc(v1['printed_whole_mean'])} and {pc(v3['printed_whole_mean'])} on average where the A/A raised "
        f"{pc1(v1['aa_alarms'] / salts * 100)} and {pc1(v3['aa_alarms'] / salts * 100)}: {v1['aa_alarms']} alarms against the {v1['expected_alarms']} that v1's printed figures add up to, "
        f"{v3['aa_alarms']} against {v3['expected_alarms']} for v3. The command gives every case the noise of the average repeated case. In the simulator the noise is not the same on "
        f"every case: {pc1(v1['flipped_safety_pct'])} of v1's 8 safety cases flip between attempts against {pc1(v1['flipped_other_pct'])} of its other 37 "
        f"(v3: {pc1(v3['flipped_safety_pct'])} against {pc1(v3['flipped_other_pct'])}; v2: {pc1(v2['flipped_safety_pct'])} against {pc1(v2['flipped_other_pct'])}), "
        f"and the critical-case rule blocks the change when any one safety case goes from pass to fail ({v1['aa_critical']} of v1's {v1['aa_alarms']} alarms). "
        f"(The safety cases ran 5 times and the others 3, which alone makes a flip about 1.6 times as likely at equal noise; for v1 it is "
        f"{v1['flipped_safety_pct'] / v1['flipped_other_pct']:.1f} times.) "
        f"The printed figure is a weak guide even inside v1: its A/A raised an alarm in {v1['by_printed_whole']['under 1%']['alarms']} of the {v1['by_printed_whole']['under 1%']['salts']} salts "
        f"where the command printed under 1% ({pc0(v1['by_printed_whole']['under 1%']['alarms'] / v1['by_printed_whole']['under 1%']['salts'] * 100)}) and in "
        f"{v1['by_printed_whole']['1% to 10%']['alarms']} of the {v1['by_printed_whole']['1% to 10%']['salts']} where it printed 1% to 10% "
        f"({pc0(v1['by_printed_whole']['1% to 10%']['alarms'] / v1['by_printed_whole']['1% to 10%']['salts'] * 100)}).",
        "",
        "Read the command's figure as the answer of a model in which every case has the same noise (exact for a suite like that), and check it with an A/A of your own suite; "
        "it is not a measured false-alarm rate of this demo.",
        "",
    ]
    return lines


def _noise_section(runs: dict[str, RunResult], real: bool, noise_run: RunResult | None = None) -> list[str]:
    """`evalsprint noise` on the calibrated variant. With the demo's own protocol (8 repeated cases) the command declines to give a rate, and the section says why;
    offline, one more run (v2 with every case repeated) shows what it prints when it can price the suite."""
    key = CALIBRATED_VARIANT
    text = noise_output(runs[key])
    if text is None:
        return []
    numbers = {k: noise_numbers(noise_output(runs[k])) for k, _t, _o in VARIANTS}
    lines = [
        "## How noisy is this suite? (`evalsprint noise`)",
        "",
        f"`evalsprint noise <run dir>` turns the flip rate of a run's repeated cases into what the gate will do on a suite like this one: how often it blocks a change "
        f"that changes nothing (the false alarm of the whole gate, and of each of its two rules) and how often it catches a real 5 or 10 point drop. It prices the "
        f"protocol the run used. This is its output for the {key} run as `eval.yaml` makes it"
        + ("" if real else " (simulated: the noise is the simulator's invented noise, and the gate arithmetic is the real one)")
        + ":",
        "",
        "```text",
        text,
        "```",
        "",
    ]
    mine = numbers.get(key)
    if mine is not None and not mine["estimated"]:
        n_repeated, n_cases = mine["n_repeated"], mine["n_scored"]
        lines += [
            f"**It prints no false-alarm rate for this run, and that is its honest answer.** `eval.yaml` repeats only the {n_repeated} safety cases ({repeat_attempts()} times each) "
            f"and runs the other {n_cases - n_repeated} once, so the noise of the whole suite would have to be read from {n_repeated} cases: one case that flips or not moves "
            f"the flip rate by {100 / n_repeated:.1f} points, and below 20 repeated cases the command declines to turn that into a rate. The three variants show the same thing "
            "(their flip rates really differ, but each rests on a handful of cases):",
            "",
            "| Variant | Repeated cases | Flip rate | Noise per attempt |",
            "|---|---|---|---|",
        ]
        for k, title, _o in VARIANTS:
            n = numbers.get(k)
            if n is not None:
                lines.append(f"| {title} | {n['n_repeated']} of {n['n_scored']} | {n['flip_rate']} ({n['n_flipped']} of {n['n_repeated']}) | {n['noise_per_attempt'] or 'not estimated'} |")
        lines += [""]
    elif mine is not None:
        lines += [
            f"The command prices this run: {mine['n_repeated']} of {mine['n_scored']} cases were repeated, the flip rate is {mine['flip_rate']} ({mine['n_flipped']} of {mine['n_repeated']}) and "
            f"the whole gate would block about {mine['false_alarm']} of the changes that change nothing (the pass-rate rule alone {mine['false_alarm_pass_rate']}, the critical-case rule "
            f"alone {mine['false_alarm_critical']}). Read every figure as \"about\".",
            "",
        ]
    priced_text = noise_output(noise_run) if noise_run is not None else None
    priced = noise_numbers(priced_text)
    if priced is not None and priced["estimated"]:
        s = noise_run.summary
        lines += [
            f"**A run the command can price.** The same v2 with every case repeated, `repeat: {{n: 3, per_tag: {{safety: 5}}}}` (what `evalsprint run -c eval.yaml --repeat 3` makes of this "
            f"suite: {priced['n_repeated']} repeated cases, {s['n_attempts']} case runs, `noise_check/`; {s['n_attempts'] - runs[key].summary['n_attempts']} more case runs, and judge "
            "calls, than the run above):",
            "",
            "```text",
            priced_text,
            "```",
            "",
            f"Flip rate {priced['flip_rate']} ({priced['n_flipped']} of {priced['n_repeated']}), noise per attempt {priced['noise_per_attempt']}: the whole gate blocks about "
            f"{priced['false_alarm']} of the changes that change nothing, of which the pass-rate rule alone {priced['false_alarm_pass_rate']} and the critical-case rule alone "
            f"{priced['false_alarm_critical']}.",
            "",
        ]
        ab = noise_vs_aa()
        if ab is not None:
            lines += _noise_vs_aa_lines(noise_run, ab)
    lines += [
        "For a rate on your own suite run `evalsprint run -c eval.yaml --repeat 3` (every case 3 times, the safety cases keep their 5) and `evalsprint noise` on that run; "
        "it costs more target and judge calls, and the README puts a price on it. That is the protocol the kit's docs recommend below about 150 cases, "
        "`repeat: {n: 3, per_tag: {<critical tag>: 5}}`. This demo keeps `repeat: {n: 1, per_tag: {safety: 5}}` because here the extra attempts take the false alarm of the whole gate "
        f"from {REPEAT_GAIN['alarms_safety5']} to {REPEAT_GAIN['alarms_pertag']} of {REPEAT_GAIN['pairs']} A/A pairs for {REPEAT_GAIN['more_runs_pct']}% more case runs, a gain smaller than the spread between "
        "batches of salts (README, \"How much noise the gate tolerates\"); a suite copied from this one should switch that one line of `eval.yaml`.",
        "",
    ]
    return lines


def write_summary(
    out: Path,
    runs: dict[str, RunResult],
    comparisons: dict[str, dict],
    calibration_result: dict | None,
    real: bool,
    monthly_requests: int,
    history_run: RunResult | None = None,
    history_gate: dict | None = None,
    noise_run: RunResult | None = None,
) -> Path:
    lines = ["# Ledgerly support bot: eval summary", ""]
    if real:
        lines += [
            "Mode: **real models** (bot and judge call the provider APIs; costs are priced from `evalsprint/pricing.yaml`).",
            "",
            "> **UNCALIBRATED JUDGE.** " + UNCALIBRATED_NOTICE.replace("\n", " "),
            "",
        ]
    else:
        lines += [
            "> **SIMULATED RUN.** The bot and the judge in this run are a simulator (`sim.py`), not language models.",
            "> The numbers show that the pipeline works (run, repeat, judge, compare, calibrate, gate, report). They are **not** model benchmarks.",
            "> Run `python run_demo.py --real` for real numbers.",
            "",
            "> **PLACEHOLDER RUNS.** " + PLACEHOLDER_NOTICE,
            "",
        ]
    header = (
        ["Variant", "Pass rate on scored cases (95% CI)", "Errors", f"Flip rate (safety, {repeat_attempts()} attempts)"]
        + [t.replace("_", " ") for t in TYPES]
        + ["Cost / 1k requests", "p95 latency", "Gate vs previous (mode)"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for key, title, _ in VARIANTS:
        s = runs[key].summary
        comp = comparisons.get(key)
        if key == VARIANTS[0][0]:
            gate = "baseline"
        else:
            gate = "n/a" if comp is None else f"{comp['verdict'].upper()} ({gate_rule(comp)})"
        row = [title, pct_ci(s), errors_text(s), flip_text(s)] + [by_type(s, t) for t in TYPES]
        row += [usd(s["cost"]["system_per_1k_usd"]), f"{s['latency']['p95_s']:.2f}s", gate]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Errors (a target or judge call that failed) are never counted as quality failures: they are excluded from the pass rate. "
        "Flip rate is the share of repeated cases whose attempts did not all agree; it is the noise floor of one run.",
        "",
    ]

    lines += [f"## Monthly cost projection ({monthly_requests:,} requests a month, an assumption: change it with `--monthly-requests`)", ""]
    lines += ["| Variant | Bot cost per month | Change vs previous variant |", "|---|---|---|"]
    previous = None
    for key, title, _ in VARIANTS:
        per_1k = runs[key].summary["cost"]["system_per_1k_usd"]
        month = None if per_1k is None else per_1k * monthly_requests / 1000
        change = "baseline" if previous is None else ("n/a" if month is None or previous is None else f"{'+' if month >= previous else '-'}{money(abs(month - previous))}")
        lines.append(f"| {title} | {money(month)} | {change} |")
        previous = month
    lines += ["", "Bot cost only (the judge runs in CI, not per customer request).", ""]

    for key in [v[0] for v in VARIANTS][1:]:
        comp = comparisons.get(key)
        if not comp:
            continue
        lines.append(f"## Gate: {comp['baseline_run_id']} -> {comp['candidate_run_id']}: {comp['verdict'].upper()}")
        lines.append("")
        lines.append(f"- Mode: {gate_rule(comp)}")
        for reason in comp.get("reasons") or ["no gate rule was triggered"]:
            lines.append(f"- {reason}")
        for label, ids in (
            ("Newly failing", comp.get("newly_failing")),
            ("Newly passing", comp.get("newly_passing")),
            ("Critical regressions", comp.get("critical_regressions")),
            ("Removed cases", comp.get("removed_cases")),
            ("Changed cases", comp.get("changed_cases")),
        ):
            if ids:
                lines.append(f"- {label} ({len(ids)}): {', '.join(str(i) for i in ids)}")
        for warning in comp.get("warnings") or []:
            lines.append(f"- Warning: {warning}")
        lines.append("")

    lines += _noise_section(runs, real, noise_run)

    lines += ["## History", ""]
    if history_run is not None:
        v2, v4 = runs["v2-grounded"].summary, history_run.summary
        lines += [
            f"`history.html` plots the pass rate (with its interval) and the cost per 1,000 requests over the {len(runs) + 1} runs, "
            "and marks every run whose judges differ from the run before. The fourth run is v2 again with the `length` judge tightened from "
            f"1,200 to 300 characters: the bot did not change, the judges did, and the pass rate went from {pct(v2['pass_rate'])} to {pct(v4['pass_rate'])}. "
            "The marker is there so that a step like this is not read as a worse bot.",
            "",
        ]
        if history_gate is not None:
            lines += [
                f"Asked to compare that run with v2, the gate answers **{history_gate['verdict'].upper()}** and says why: "
                + "; ".join(str(r) for r in history_gate.get("reasons") or ["no reason listed"])
                + ". The bot did not change, so every reason above is the effect of the judges; the gate compares their fingerprint first and wants a reviewed baseline before it says anything about quality.",
                "",
            ]
    else:
        lines += [f"`history.html` plots the pass rate (with its interval) and the cost per 1,000 requests over the {len(runs)} runs; run the suite again after each change and `evalsprint history <runs dir>` adds the new point.", ""]
    lines += ["## Judge calibration", ""]
    if calibration_result:
        cal = calibration_result
        lines += [
            "**SYNTHETIC labels.** They come from `sim.make_synthetic_labels`, a rule-based stand-in for a human. This shows the workflow, "
            "not how good any judge is. Replace them with real labels for a real project.",
            "",
            f"- Labeled cases used: {cal['n']}",
            f"- Decision: **{cal.get('kappa_decision')}**. Agreement {pct(cal['agreement'])}, {_kappa_word(cal)}Cohen's kappa {_kappa(cal)}"
            + (
                f", which the decision uses; unweighted on the {cal['n']} labeled cases: {_fmt_kappa(cal.get('kappa'), cal.get('kappa_ci'))}"
                if _has_weighted(cal)
                else ""
            ),
            *(
                [
                    f"- Cases of the run that can be labeled: {cal['n_labelable']} ({cal['n']} labeled; at least "
                    f"{cal['min_labels_release']} labeled cases are needed before a judge is released)"
                ]
                if cal.get("n_labelable") is not None and cal.get("min_labels_release") is not None
                else []
            ),
            f"- Observed pass rate {pct(cal.get('run_observed_pass_rate'))}, corrected for judge errors {pct(cal.get('corrected_pass_rate'))}",
        ]
        for warning in cal.get("warnings") or []:
            lines.append(f"- Warning: {warning}")
        lines += [f"- {cal.get('recommendation', '')}", ""]
    else:
        lines += [
            "Not done: the judge has not been compared with human labels, so its verdicts are provisional and the gate says the judge is not calibrated.",
            "Label a sample (`evalsprint label`) and run `evalsprint calibrate`; the steps are in the README.",
            "",
        ]
    if not real:
        lines += _provenance_lines(runs)

    lines += [
        "## Files",
        "",
        "- `runs/<variant>/report.html`: self-contained report per run (open in a browser)",
        "- `history.html`: quality and cost over the runs, judge changes marked (self-contained)",
        "- `runs/<variant>/results.jsonl`, `summary.json`: raw results (the extra run `evalsprint noise` can price is in `noise_check/`, offline only)",
        "- `compare_v1_to_v2.md`, `compare_v2_to_v3.md`: regression-gate output (also `.json`)",
    ]
    if not real:
        lines.append("- `labels_SYNTHETIC_v2.jsonl`, `calibration_v2_SYNTHETIC.md`: offline calibration demo")
    lines.append("")
    path = out / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _next_steps_real() -> str:
    return (
        "Next: calibrate the judge before you trust these numbers.\n"
        "  evalsprint label <out>/runs/v2-grounded --judge quality --sample 80 -o labeling.html\n"
        "  evalsprint calibrate <out>/runs/v2-grounded --labels labels.jsonl --judge quality"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Ledgerly support-bot eval demo (v1 -> v2 -> v3).")
    parser.add_argument("--out", default="demo_out", help="output directory (default: ./demo_out)")
    parser.add_argument("--real", action="store_true", help="use real models instead of the offline simulator (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--yes", action="store_true", help="with --real: do not ask for confirmation after the cost estimate")
    parser.add_argument(
        "--monthly-requests", type=int, default=DEFAULT_MONTHLY_REQUESTS, help=f"requests per month for the cost projection (default {DEFAULT_MONTHLY_REQUESTS:,})"
    )
    args = parser.parse_args(argv)
    if args.monthly_requests < 1:
        parser.error("--monthly-requests must be at least 1")
    out = Path(args.out).resolve()

    if args.real:
        os.environ.setdefault("LLM_PROVIDER", "anthropic")
        os.environ.setdefault("JUDGE_PROVIDER", "anthropic")
        if "anthropic" in (os.environ["LLM_PROVIDER"], os.environ["JUDGE_PROVIDER"]) and not os.environ.get("ANTHROPIC_API_KEY"):
            say("error: --real needs ANTHROPIC_API_KEY (or set LLM_PROVIDER / JUDGE_PROVIDER to another provider).")
            return 2
        say("REAL MODE: bot and judge call the provider APIs.")
        say(f"  bot: {os.environ['LLM_PROVIDER']} / {os.environ.get('BOT_MODEL', 'claude-sonnet-5-5')}   judge: {os.environ['JUDGE_PROVIDER']} / {os.environ.get('JUDGE_MODEL', 'claude-opus-5-5')}")
        say()
        for line in UNCALIBRATED_NOTICE.splitlines():
            say(line)
        say()
        print_estimate(estimate_real_cost())
        say()
        if args.yes:
            say("--yes given: starting without asking.")
        elif not confirm("Start the run and spend about that much? [y/N] "):
            say("Cancelled: nothing was called. Pass --yes to skip this question.")
            return 1
        say()
    else:
        os.environ["LLM_PROVIDER"] = "mock"  # never touch the network in offline mode
        os.environ["JUDGE_PROVIDER"] = "mock"
        say("SIMULATED MODE: the bot and the judge are a simulator (sim.py), not language models.")
        say("Numbers below show the pipeline working. They are NOT model benchmarks. Use --real for real models.")
        say("Every run is a placeholder run (placeholder: true in its summary.json): its judge is the mock provider, so `evalsprint baseline` would refuse it without --force.")
        say("This demo never records a baseline, so it needs no --force; `compare` and the reports say the same thing.")
    out.mkdir(parents=True, exist_ok=True)
    say()

    say(f"1. Running the three variants ({count_cases()} cases each; safety cases repeated {repeat_attempts()} times)")
    runs = run_variants(out, args.real)
    say("\n2. Regression gate")
    comparisons = compare_steps(runs, out)
    say("\n3. Judge calibration")
    calibration_result = calibration_step(runs, out, args.real)
    say("\n4. Reports")
    reports = report_step(runs, comparisons, calibration_result, args.monthly_requests)
    for path in reports:
        say(f"  {path}")
    say("\n5. History over the runs")
    history_path, history_run, history_gate = history_step(runs, out, args.real, args.monthly_requests)
    say(f"  {history_path}")
    if not args.real:
        say("\n6. A run `evalsprint noise` can price (offline only: one more simulated run of v2, every case repeated)")
    noise_run = noise_check_step(out, args.real)
    summary_path = write_summary(out, runs, comparisons, calibration_result, args.real, args.monthly_requests, history_run, history_gate, noise_run)
    say(f"\nSummary: {summary_path}\n")
    say(summary_path.read_text(encoding="utf-8"))
    if args.real:
        say(_next_steps_real())
    return 0


if __name__ == "__main__":
    sys.exit(main())
