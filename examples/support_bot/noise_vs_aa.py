#!/usr/bin/env python3
"""What `evalsprint noise` prints for the demo's runs, set beside what the A/A measures for the same runs (the same two batches of salts as salt_check.py).

The question: the command prices the gate of a run from the flip rate of that run's repeated cases. For the protocol it can price on this suite,
`repeat: {n: 3, per_tag: {safety: 5}}` (45 repeated cases), is the false alarm it prints the false alarm the gate really has? The A/A answers it: the same
configuration run twice under two salts (salt, then salt + "-rerun"), compared by the gate; every REGRESSION is a false alarm. For each salt and each of v1, v2 and v3
this script runs the first run, calls `evalsprint noise` on it (the figure a user with that run as the baseline would read), runs the repeat and asks the gate,
exactly as salt_check.py's per-tag A/A does (it imports its functions: the A/A counts must be the ones of salt_robustness.json for this protocol, and the script
checks that and exits with 1 when a full run (100 salts per batch) does not reproduce them).

    python3 noise_vs_aa.py [--jobs 2] [--out noise_vs_aa.json] [--rows-out rows.json]   # 2 batches x 100 salts x 3 variants; about 6 minutes on 2 processes
    python3 noise_vs_aa.py --from-rows rows.json --out noise_vs_aa.json                  # rebuild the summary from the rows of an earlier run
    python3 noise_vs_aa.py --salts 3 --out /tmp/x.json                                   # a smoke run: the first 3 salts of each batch

`--out` gets the summary only (what the README quotes, about 15 KB: noise_vs_aa.json, kept next to the README like salt_robustness.json); `--rows-out` gets the same
plus one line per salt and variant (about 190 KB, what the summary is computed from), for whoever wants to check it.

OFFLINE ONLY: every provider is `mock`, the verdicts come from the simulator (sim.py), the gate arithmetic and the command are the real ones. Every count here is a
property of the simulator's invented noise ([INVENTED]); what it shows is where the command's model fits that noise and where it does not. No real API call.
The salts are fixed strings (salt00..salt99, critic000..critic099), so the result does not depend on the number of processes or on the order.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["LLM_PROVIDER"] = "mock"
os.environ["JUDGE_PROVIDER"] = "mock"

import run_demo  # noqa: E402
import salt_check as sc  # noqa: E402  (its A/A functions: same runs, same compare)
from evalsprint.stats import wilson_ci  # noqa: E402

PROTOCOL = "repeat: {n: 3, per_tag: {safety: 5}}"
BATCHES = (("salt", 2), ("critic", 3))  # salt prefix, digits of the number: salt00..salt99 and critic000..critic099, as in salt_robustness.json
VARIANTS = (("v1", sc.V1), ("v2", sc.V2), ("v3", sc.V3))
FIELDS = ("noise_per_attempt", "false_alarm", "false_alarm_pass_rate", "false_alarm_critical", "detect_5", "detect_10")


def _percent(text: str | None) -> float | None:
    return None if text is None else float(text.rstrip("%"))


def _flips_by_group(run) -> tuple[int, int, int]:
    """(repeated safety cases whose attempts disagree, repeated other cases whose attempts disagree, repeated safety cases): where the run's flips sit."""
    safety_flips = other_flips = n_safety = 0
    for record in run.results:
        attempts = record.get("attempts") or []
        if len(attempts) < 2:
            continue
        flipped = len({a.get("outcome") for a in attempts}) > 1
        if "safety" in (record.get("case") or {}).get("tags", []):
            n_safety += 1
            safety_flips += flipped
        else:
            other_flips += flipped
    return safety_flips, other_flips, n_safety


def one_salt(salt: str) -> dict:
    """For one salt: per variant, what the command printed for the first run and what the A/A of that run did. One process = one SIM_SEED."""
    row: dict = {"salt": salt}
    sc._set_seed(salt)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name, overrides in VARIANTS:
                first = sc.run_variant(overrides + sc.PERTAG, f"{name}-noise", tmp)
                again = sc.rerun(salt, tmp, overrides + sc.PERTAG, f"{name}-noise-again")
                cell = sc.aa_cell(first, again)
                noise = run_demo.noise_numbers(run_demo.noise_output(first))
                if noise is None or not noise["estimated"]:
                    raise SystemExit(f"{salt} {name}: `evalsprint noise` did not price the run: {noise}")
                safety_flips, other_flips, n_safety = _flips_by_group(first)
                row[name] = {
                    "repeated": noise["n_repeated"],
                    "flipped": noise["n_flipped"],
                    "n_safety": n_safety,
                    "flipped_safety": safety_flips,
                    "flipped_other": other_flips,
                    "upper_bound": noise["upper_bound"],
                    **{key: _percent(noise[key]) for key in FIELDS},
                    "aa_alarm": cell["verdict"] == "regression",
                    "aa_pass_rule": bool(cell["pass_rule"]),
                    "aa_critical": bool(cell["critical"]),
                }
    finally:
        sc._set_seed(None)
    return row


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def summarise(rows: list[dict]) -> dict:
    """Per variant: the A/A counts, what the command printed (mean and spread over the salts), and how many alarms those figures add up to."""
    out: dict = {}
    for name, _o in VARIANTS:
        cells = [r[name] for r in rows]
        n = len(cells)
        whole = [c["false_alarm"] for c in cells]
        by_bucket = {}
        for label, low, high in (("under 1%", 0.0, 1.0), ("1% to 10%", 1.0, 10.0), ("10% to 30%", 10.0, 30.0), ("30% or more", 30.0, 101.0)):
            group = [c for c in cells if low <= c["false_alarm"] < high]
            by_bucket[label] = {"salts": len(group), "alarms": sum(c["aa_alarm"] for c in group), "printed_mean": round(statistics.fmean(c["false_alarm"] for c in group), 2) if group else None}
        out[name] = {
            "salts": n,
            "aa_alarms": sum(c["aa_alarm"] for c in cells),
            "aa_ci95": [round(x * 100, 2) for x in wilson_ci(sum(c["aa_alarm"] for c in cells), n)],  # Wilson, in percent, as the kit does it everywhere else
            "aa_pass_rule": sum(c["aa_pass_rule"] for c in cells),
            "aa_critical": sum(c["aa_critical"] for c in cells),
            "printed_whole_mean": round(statistics.fmean(whole), 2),
            "printed_pass_rate_mean": round(statistics.fmean(c["false_alarm_pass_rate"] for c in cells), 2),
            "printed_critical_mean": round(statistics.fmean(c["false_alarm_critical"] for c in cells), 2),
            "printed_whole": {"min": min(whole), "p10": _quantile(whole, 0.10), "median": _quantile(whole, 0.50), "p90": _quantile(whole, 0.90), "max": max(whole)},
            "expected_alarms": round(sum(whole) / 100, 1),
            "expected_pass_rule": round(sum(c["false_alarm_pass_rate"] for c in cells) / 100, 1),
            "expected_critical": round(sum(c["false_alarm_critical"] for c in cells) / 100, 1),
            "flip_rate": {"min": round(min(c["flipped"] for c in cells) / cells[0]["repeated"] * 100, 1), "median": round(_quantile([c["flipped"] for c in cells], 0.5) / cells[0]["repeated"] * 100, 1),
                          "max": round(max(c["flipped"] for c in cells) / cells[0]["repeated"] * 100, 1)},
            "no_flip_salts": sum(c["flipped"] == 0 for c in cells),
            "flipped_safety_mean": round(statistics.fmean(c["flipped_safety"] for c in cells), 2),
            "flipped_other_mean": round(statistics.fmean(c["flipped_other"] for c in cells), 2),
            "n_safety": cells[0]["n_safety"],
            "n_other": cells[0]["repeated"] - cells[0]["n_safety"],
            "flipped_safety_pct": round(statistics.fmean(c["flipped_safety"] for c in cells) / cells[0]["n_safety"] * 100, 1),  # of the repeated safety cases (5 attempts each)
            "flipped_other_pct": round(statistics.fmean(c["flipped_other"] for c in cells) / (cells[0]["repeated"] - cells[0]["n_safety"]) * 100, 1),  # of the other repeated cases (3 attempts each)
            "printed_whole_over_50": sum(w >= 50.0 for w in whole),
            # what the command printed for a given number of flipped cases among the repeated ones: the figure is a function of that count (a run with no flip is priced at the upper 95% bound)
            "by_flips": {str(k): {"salts": len(group), "printed_whole": sorted({c["false_alarm"] for c in group}), "aa_alarms": sum(c["aa_alarm"] for c in group)}
                         for k in sorted({c["flipped"] for c in cells}) for group in [[c for c in cells if c["flipped"] == k]]},
            "by_printed_whole": by_bucket,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--salts", type=int, default=100, help="salts per batch (default 100)")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes (default 1); the result does not depend on it")
    parser.add_argument("--out", default="noise_vs_aa.json", help="where to write the summary (default ./noise_vs_aa.json)")
    parser.add_argument("--rows-out", default=None, help="also write the summary with every row (one line per salt) here")
    parser.add_argument("--from-rows", default=None, help="do not run anything: rebuild the summary from the rows file of an earlier run (its --rows-out)")
    args = parser.parse_args(argv)
    if args.salts < 1 or args.jobs < 1:
        parser.error("--salts and --jobs must be at least 1")
    if args.from_rows:
        try:
            earlier = json.loads(Path(args.from_rows).read_text(encoding="utf-8"))
            rows, batches, python = earlier["rows"], earlier["batches"], earlier["python"]
            if not isinstance(rows, list) or not rows or not isinstance(batches, list):
                raise TypeError("no rows")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"--from-rows {args.from_rows}: not the output of a run with --rows-out ({exc})")
    else:
        if not sc.per_tag_supported():
            parser.error("this checkout's config does not understand `repeat: {n, per_tag}`")
        salts = [f"{prefix}{i:0{width}d}" for prefix, width in BATCHES for i in range(args.salts)]
        if args.jobs == 1:
            rows = [one_salt(s) for s in salts]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                rows = list(pool.map(one_salt, salts))
        batches = [{"prefix": prefix, "width": width, "n": args.salts, "first": f"{prefix}{0:0{width}d}", "last": f"{prefix}{args.salts - 1:0{width}d}"} for prefix, width in BATCHES]
        python = sys.version.split()[0]
    by_batch = {b["first"] + ".." + b["last"]: summarise([r for r in rows if r["salt"].startswith(b["prefix"])]) for b in batches}
    summary = summarise(rows)
    match = matches_salt_robustness(summary)
    data = {
        "protocol": PROTOCOL,
        "what": "evalsprint noise on the first run of each A/A pair, and the A/A verdict of that pair, per salt and variant (noise_vs_aa.py); offline simulator, INVENTED noise",
        "python": python,
        "batches": batches,
        "matches_salt_robustness": match,
        "summary": summary,
        "summary_by_batch": by_batch,
    }
    Path(args.out).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    if args.rows_out:
        full = json.dumps({**data, "rows": "@@ROWS@@"}, indent=1).replace('"@@ROWS@@"', "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n ]")  # one line per salt
        Path(args.rows_out).write_text(full + "\n", encoding="utf-8")
    for name, s in data["summary"].items():
        print(f"{name}: A/A {s['aa_alarms']} of {s['salts']} (pass-rate rule {s['aa_pass_rule']}, critical rule {s['aa_critical']}); the command printed on average {s['printed_whole_mean']}% "
              f"(= {s['expected_alarms']} alarms; median {s['printed_whole']['median']}%, p90 {s['printed_whole']['p90']}%, max {s['printed_whole']['max']}%)")
    if match["checked"]:
        print("the A/A counts " + ("are the ones of salt_robustness.json (protocol `pertag`)" if match["ok"] else f"DO NOT match salt_robustness.json: {match}"))
    return 0 if match["ok"] else 1


def matches_salt_robustness(summary: dict) -> dict:
    """The A/A counts of this run against the ones salt_check.py kept in salt_robustness.json for the same protocol and the same salts. Only a full run (200 salts) is checked."""
    try:
        rr = json.loads((HERE / "salt_robustness.json").read_text(encoding="utf-8"))["repeat_ranges"]["pertag"]
    except (OSError, ValueError, KeyError, TypeError):
        return {"checked": False, "ok": True, "why": "salt_robustness.json is not there or has no `pertag` protocol"}
    if any(s["salts"] != 200 for s in summary.values()):
        return {"checked": False, "ok": True, "why": "not the full 200 salts"}
    kept = {name: rr[f"false_alarm_{name}"]["pooled"] for name, _o in VARIANTS}
    got = {name: summary[name]["aa_alarms"] for name, _o in VARIANTS}
    return {"checked": True, "ok": kept == got, "salt_robustness": kept, "this_run": got}


if __name__ == "__main__":
    sys.exit(main())
