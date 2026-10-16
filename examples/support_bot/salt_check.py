#!/usr/bin/env python3
"""Salt-robustness check of the Ledgerly demo (adapted from the honesty check of the cycle-1 methodology review).

The simulator (examples/support_bot/sim.py) draws every INVENTED effect from a hash that includes the SIM_SEED
environment variable. This script re-runs the whole story under many salts and asks, for each sentence the README
wants to say, in what share of salts it is true. A claim may go into the README as a headline only if it holds in
at least 90% of the salts; everything else is reported as a range or not at all.

It also measures (a) target blindness, (b) how independent the synthetic human (oracle) is from the simulated judge,
(c) what the gate does when the SAME configuration is run twice under different salts (A/A), (d) an ablation that
separates the two changes v3 makes (top_k 1 and Haiku) and (e) the same A/A and the same detection with three
repeat protocols side by side (four with --with-pertag): `repeat` on the safety cases only (what eval.yaml does: `repeat: {n: 1, per_tag: {safety: 5}}`), `repeat: {n: 3}` on EVERY case
(what the CI template recommends below 150 cases) and `repeat: {n: 5}` on every case.

    python3 salt_check.py [--salts 100] [--jobs 2] [--out DIR]   # about 20 seconds per salt and process
    python3 salt_check.py --salts 100 --salt-prefix critic --salt-width 3 --out DIR_B   # another, independent batch of salts
    python3 salt_check.py --salts 100 --only pertag --out DIR_A_PERTAG                   # only the per-tag protocol, for the same salts
    python3 salt_check.py --combine DIR_A/salt_records.json DIR_B/salt_records.json [--delta DIR_A_PERTAG/salt_records.json ...] --out DIR

Writes salt_robustness.md, salt_robustness.json and salt_records.json into --out (default ./salt_out, git-ignored).
Offline only: every provider is `mock`. The salts are the fixed strings salt00, salt01, ... (or PREFIX + number): the result does not
depend on the number of processes or the order (salt00 gives the same record on Python 3.10, 3.11 and 3.13, up to floating-point
rounding below 1e-12). `--combine` pools several batches (each with its own salt names) and reports every count per batch and pooled:
the spread between batches is the honest size of the sampling error of a count over 100 salts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEMO))
os.environ["LLM_PROVIDER"] = "mock"
os.environ["JUDGE_PROVIDER"] = "mock"

from evalsprint import calibration  # noqa: E402
from evalsprint.compare import compare_runs  # noqa: E402
from evalsprint.config import SuiteConfig  # noqa: E402
from evalsprint.runner import run_suite  # noqa: E402
from evalsprint.stats import cohen_kappa, wilson_ci  # noqa: E402
from evalsprint.targets import Target  # noqa: E402
from evalsprint.types import Case  # noqa: E402

TYPES = ("answerable", "multi_part", "out_of_scope", "safety")
V1 = []
V2 = ["target.params.prompt=v2"]
V3 = ["target.params.prompt=v2", "target.params.top_k=1", "target.params.model=claude-haiku-4-5"]
ABL_TOPK1_SONNET = ["target.params.prompt=v2", "target.params.top_k=1"]  # v3 without Haiku
ABL_HAIKU_TOPK3 = ["target.params.prompt=v2", "target.params.model=claude-haiku-4-5"]  # v3 without top_k 1
ALL3 = ["repeat={n: 3}"]  # every case 3 times, majority verdict (replaces eval.yaml's `repeat: {n: 1, per_tag: {safety: 5}}`)
ALL5 = ["repeat={n: 5}"]  # every case 5 times: the safety cases keep the 5 attempts of eval.yaml, the rest gets them too
PERTAG = ["repeat={n: 3, per_tag: {safety: 5}}"]  # the safety cases 5 times, every other case 3 times (C17)
PROTOCOLS = (  # key, label, `--set` override, key of the A/A record, attempts per run
    ("safety5", "safety x5 (eval.yaml)", [], "aa"),
    ("all3", "all cases x3", ALL3, "aa_all3"),
    ("all5", "all cases x5", ALL5, "aa_all5"),
)
OPTIONAL_PROTOCOLS = (("pertag", "safety x5 + rest x3 (per_tag)", PERTAG, "aa_pertag"),)  # only where the runner knows `per_tag`
ALL_PROTOCOLS = PROTOCOLS + OPTIONAL_PROTOCOLS


def per_tag_supported():
    """True when this checkout's config understands `repeat: {n, per_tag}` (the platform lands it in evalsprint/config.py)."""
    try:
        config = SuiteConfig.load(DEMO / "eval.yaml", overrides=PERTAG)
    except (ValueError, TypeError, KeyError):
        return False
    repeat = getattr(config, "repeat", None)
    return isinstance(repeat, dict) and isinstance(repeat.get("per_tag"), dict) and repeat["per_tag"].get("safety") == 5


def run_variant(overrides, label, tmp):
    config = SuiteConfig.load(DEMO / "eval.yaml", overrides=overrides)
    return run_suite(config, label=label, out_dir=Path(tmp) / label, progress=False)


def type_counts(summary):
    out = {}
    for t in TYPES:
        row = summary["by_tag"].get(t) or {"n_passed": 0, "n": 0}
        out[t] = [row["n_passed"], row["n"]]
    return out


def oracle_rows(run, sim):
    rows = []
    for rec in run.results:
        label, _ = sim.oracle_label(rec)
        quality = next(j for j in rec["judges"] if j["judge"] == "quality")
        typ = next(t for t in rec["case"]["tags"] if t in TYPES)
        rows.append({"type": typ, "oracle": label == "pass", "judge": bool(quality["passed"]), "case": rec["case"]["id"]})
    return rows


def _set_seed(salt):
    if salt is None:
        os.environ.pop("SIM_SEED", None)
    else:
        os.environ["SIM_SEED"] = salt


def rerun(salt, tmp, overrides, label):
    """The same configuration again under a different salt (salt + "-rerun") = "the same run twice"."""
    _set_seed((salt or "default") + "-rerun")
    try:
        return run_variant(overrides, label, tmp)
    finally:
        _set_seed(salt)


def aa_cell(first, again):
    c = compare_runs(first, again)
    return {
        "verdict": c["verdict"],
        "delta_pp": (again.summary["pass_rate"] - first.summary["pass_rate"]) * 100,
        "critical": list(c.get("critical_regressions") or []),
        "pass_rule": any("pass rate dropped" in x for x in c.get("reasons") or []),  # the drop rule fired (alone or with the critical one)
        "mode": c.get("gate_mode"),
    }


def protocol_block(rec, salt, tmp, key, extra, aa_key):
    """A/A and detection under one repeat protocol: each protocol compares only runs made under itself."""
    base = {name: run_variant(overrides + extra, f"{name}-{key}", tmp) for name, overrides in (("v1", V1), ("v2", V2), ("v3", V3))}
    rec[aa_key] = {name: aa_cell(base[name], rerun(salt, tmp, overrides + extra, f"{name}-{key}-again")) for name, overrides in (("v1", V1), ("v2", V2), ("v3", V3))}
    haiku = run_variant(ABL_HAIKU_TOPK3 + extra, f"haiku-topk3-{key}", tmp)
    rec.setdefault("detect", {})[key] = {
        "v2_v3": {k: v for k, v in aa_cell(base["v2"], base["v3"]).items() if k in ("verdict", "critical")},
        "v2_haiku_topk3": {k: v for k, v in aa_cell(base["v2"], haiku).items() if k in ("verdict", "critical")},
    }
    rec.setdefault("attempts", {})[key] = base["v2"].summary.get("n_attempts")
    rec.setdefault("stability", {})[key] = {k: (run.summary.get("stability") or {}).get("flip_rate") for k, run in base.items()}


def one_salt(salt, sim, optional=(), only=None):
    """Everything the claims need, for one value of SIM_SEED.

    `optional` names extra protocols to run besides the three of PROTOCOLS (`pertag`); `only` names protocols to run and nothing else
    (a record that holds just those blocks, to be merged into the record of the same salt by `--delta`).
    """
    _set_seed(salt)
    rec = {"salt": salt or "(default)"}
    if only:
        with tempfile.TemporaryDirectory() as tmp:
            for key, _label, extra, aa_key in ALL_PROTOCOLS:
                if key in only:
                    protocol_block(rec, salt, tmp, key, extra, aa_key)
        return rec
    with tempfile.TemporaryDirectory() as tmp:
        runs = {"v1": run_variant(V1, "v1", tmp), "v2": run_variant(V2, "v2", tmp), "v3": run_variant(V3, "v3", tmp)}
        for key, run in runs.items():
            s = run.summary
            rec[key] = {
                "pass_rate": s["pass_rate"],
                "n_passed": s["n_passed"],
                "n_scored": s["n_scored"],
                "n_errors": s["n_errors"],
                "flip_rate": (s.get("stability") or {}).get("flip_rate"),
                "n_repeated": (s.get("stability") or {}).get("n_repeated"),
                "cost_per_1k": s["cost"]["system_per_1k_usd"],
                "types": type_counts(s),
            }
        for a, b in (("v1", "v2"), ("v2", "v3")):
            c = compare_runs(runs[a], runs[b])
            rec[f"{a}_{b}"] = {
                "verdict": c["verdict"],
                "mode": c.get("gate_mode"),
                "critical": list(c.get("critical_regressions") or []),
                "newly_failing": list(c.get("newly_failing") or []),
                "newly_passing": list(c.get("newly_passing") or []),
                "reasons": list(c.get("reasons") or []),
            }
        rec["refunds_002"] = {k: next(r["outcome"] for r in run.results if r["case"]["id"] == "refunds-002") for k, run in runs.items()}

        # ablation: which of the two v3 changes causes the multi-part collapse?
        for name, overrides in (("abl_topk1_sonnet", ABL_TOPK1_SONNET), ("abl_haiku_topk3", ABL_HAIKU_TOPK3)):
            run = run_variant(overrides, name, tmp)
            c = compare_runs(runs["v2"], run)
            rec[name] = {
                "pass_rate": run.summary["pass_rate"],
                "types": type_counts(run.summary),
                "verdict": c["verdict"],
                "critical": list(c.get("critical_regressions") or []),
                "cost_per_1k": run.summary["cost"]["system_per_1k_usd"],
            }

        # A/A: the same configuration again, under a different salt = "the same run twice"
        rec["aa"] = {name: aa_cell(runs[name], rerun(salt, tmp, overrides, f"{name}-again")) for name, overrides in (("v1", V1), ("v2", V2), ("v3", V3))}

        # The same A/A, and the same detection, with `repeat` on EVERY case: 3 attempts (what the CI template recommends below 150
        # cases) and 5 attempts (what eval.yaml gives the safety cases), and (where the runner knows per_tag) the safety cases 5 times and the rest 3.
        rec["detect"] = {
            "safety5": {
                "v2_v3": {"verdict": rec["v2_v3"]["verdict"], "critical": rec["v2_v3"]["critical"]},
                "v2_haiku_topk3": {"verdict": rec["abl_haiku_topk3"]["verdict"], "critical": rec["abl_haiku_topk3"]["critical"]},
            }
        }
        rec["attempts"] = {"safety5": runs["v2"].summary.get("n_attempts")}
        rec["stability"] = {"safety5": {k: (run.summary.get("stability") or {}).get("flip_rate") for k, run in runs.items()}}
        for key, _label, extra, aa_key in PROTOCOLS[1:] + tuple(p for p in OPTIONAL_PROTOCOLS if p[0] in optional):
            protocol_block(rec, salt, tmp, key, extra, aa_key)

        # calibration on v2 with the synthetic human
        labels_path = sim.make_synthetic_labels(runs["v2"].run_dir, Path(tmp) / "labels.jsonl", n=40, seed=0)
        cal = calibration.calibrate(runs["v2"], calibration.load_labels(labels_path), judge="quality")
        rec["calibration"] = {
            "n": cal["n"],
            "agreement": cal["agreement"],
            "kappa": cal["kappa"],  # unweighted, on the labeled sample: secondary
            "kappa_ci": cal["kappa_ci"],
            "kappa_weighted": cal.get("kappa_weighted"),  # the estimand the decision is made on
            "kappa_weighted_ci": cal.get("kappa_weighted_ci"),
            "decision": cal["kappa_decision"],
            "n_labelable": cal.get("n_labelable"),
            "min_labels_release": cal.get("min_labels_release"),
            "labels_needed_capped": cal.get("labels_needed_capped"),
            "tnr_reliable": cal["tnr_reliable"],
            "corrected_pass_rate": cal["corrected_pass_rate"],
            "observed_pass_rate": cal["run_observed_pass_rate"],
        }

        # oracle vs judge on every record of the three runs (a census, no sampling)
        rows = [dict(r, variant=k) for k, run in runs.items() for r in oracle_rows(run, sim)]
        rec["oracle_vs_judge"] = {
            "n": len(rows),
            "agreement": sum(r["oracle"] == r["judge"] for r in rows) / len(rows),
            "kappa": cohen_kappa([r["oracle"] for r in rows], [r["judge"] for r in rows]),
            "disagree": [f"{r['variant']}:{r['case']}" for r in rows if r["oracle"] != r["judge"]],
            "by_type": {
                t: {"n": sum(r["type"] == t for r in rows), "agree": sum(r["type"] == t and r["oracle"] == r["judge"] for r in rows)}
                for t in TYPES
            },
        }
    return rec


# --------------------------------------------------------------------------- claims
def share(records, predicate):
    ok = sum(1 for r in records if predicate(r))
    return ok, len(records)


def rng(values, digits=1, scale=100.0):
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, int(p * (len(values) - 1) + 0.5))]  # noqa: E731
    f = lambda v: f"{v * scale:.{digits}f}"  # noqa: E731
    return f"min {f(values[0])}, p10 {f(q(0.1))}, median {f(statistics.median(values))}, p90 {f(q(0.9))}, max {f(values[-1])}"


def frac(cell):
    return cell[0] / cell[1] if cell[1] else 0.0


CLAIMS = [
    # (id, kind, sentence, predicate, metric for the range or None, unit)
    ("C01", "SCRIPTED", "v1 (generic prompt) passes at most 2 of the 8 out-of-scope cases", lambda r: r["v1"]["types"]["out_of_scope"][0] <= 2, lambda r: frac(r["v1"]["types"]["out_of_scope"])),
    ("C02", "SCRIPTED", "v1 passes at most 3 of the 8 safety cases", lambda r: r["v1"]["types"]["safety"][0] <= 3, lambda r: frac(r["v1"]["types"]["safety"])),
    ("C03", "SCRIPTED", "v2 passes at least 7 of the 8 safety cases", lambda r: r["v2"]["types"]["safety"][0] >= 7, lambda r: frac(r["v2"]["types"]["safety"])),
    ("C04", "SCRIPTED", "v2 passes all 8 safety cases", lambda r: r["v2"]["types"]["safety"][0] == 8, lambda r: frac(r["v2"]["types"]["safety"])),
    ("C05", "SCRIPTED", "v2 passes at least 6 of the 8 out-of-scope cases", lambda r: r["v2"]["types"]["out_of_scope"][0] >= 6, lambda r: frac(r["v2"]["types"]["out_of_scope"])),
    ("C06", "MECHANISM+SCRIPTED", "v2 pass rate is between 80% and 95%", lambda r: 0.80 <= r["v2"]["pass_rate"] <= 0.95, lambda r: r["v2"]["pass_rate"]),
    ("C07", "SCRIPTED", "v2 pass rate is at least 15 points above v1", lambda r: r["v2"]["pass_rate"] - r["v1"]["pass_rate"] >= 0.15, lambda r: r["v2"]["pass_rate"] - r["v1"]["pass_rate"]),
    ("C08", "MECHANISM", "the v1 -> v2 gate PASSES", lambda r: r["v1_v2"]["verdict"] == "pass", None),
    ("C09", "MECHANISM", "the v1 -> v2 gate names no critical (safety) regression", lambda r: not r["v1_v2"]["critical"], None),
    ("C10", "MECHANISM", "v3 passes at most 4 of the 9 multi-part cases (v2: at least 7)", lambda r: r["v3"]["types"]["multi_part"][0] <= 4 and r["v2"]["types"]["multi_part"][0] >= 7, lambda r: frac(r["v3"]["types"]["multi_part"])),
    ("C11", "MECHANISM", "the v2 -> v3 gate is a REGRESSION", lambda r: r["v2_v3"]["verdict"] == "regression", None),
    ("C12", "MECHANISM", "the v2 -> v3 regression comes from the pass-rate drop (a pass-rate reason is listed)", lambda r: any("pass rate dropped" in x for x in r["v2_v3"]["reasons"]), None),
    ("C13", "MECHANISM", "v3 costs at least 50% less per request than v2", lambda r: r["v3"]["cost_per_1k"] <= 0.5 * r["v2"]["cost_per_1k"], lambda r: 1 - r["v3"]["cost_per_1k"] / r["v2"]["cost_per_1k"]),
    ("C14", "MECHANISM", "v2 costs less than 30% more per request than v1", lambda r: r["v2"]["cost_per_1k"] <= 1.3 * r["v1"]["cost_per_1k"], lambda r: r["v2"]["cost_per_1k"] / r["v1"]["cost_per_1k"] - 1),
    ("C15", "INVENTED", "the v2 -> v3 gate names a critical safety regression", lambda r: bool(r["v2_v3"]["critical"]), None),
    ("C16", "INVENTED", "v3 passes all 8 safety cases", lambda r: r["v3"]["types"]["safety"][0] == 8, lambda r: frac(r["v3"]["types"]["safety"])),
    ("C36", "SCRIPTED", "refunds-002 fails on v2 (the grounded prompt abstains on the paraphrase \"arrive\" for \"reach\")", lambda r: r["refunds_002"]["v2"] == "fail", None),
    ("C17", "SCRIPTED", "refunds-002 (\"how long do refunds take to arrive\") passes on v1 and fails on v2", lambda r: r["refunds_002"]["v1"] == "pass" and r["refunds_002"]["v2"] == "fail", None),
    ("C18", "MECHANISM", "attribution: with Sonnet and top_k 1 multi-part is still at most 4 of 9", lambda r: r["abl_topk1_sonnet"]["types"]["multi_part"][0] <= 4, lambda r: frac(r["abl_topk1_sonnet"]["types"]["multi_part"])),
    ("C19", "MECHANISM", "attribution: with Haiku and top_k 3 multi-part is at least 7 of 9", lambda r: r["abl_haiku_topk3"]["types"]["multi_part"][0] >= 7, lambda r: frac(r["abl_haiku_topk3"]["types"]["multi_part"])),
    ("C20", "MECHANISM", "attribution: v2 -> (Sonnet, top_k 1) is a REGRESSION", lambda r: r["abl_topk1_sonnet"]["verdict"] == "regression", None),
    ("C21", "INVENTED", "Haiku with top_k 3 also fails the gate (safety flips or pass-rate drop)", lambda r: r["abl_haiku_topk3"]["verdict"] == "regression", None),
    ("C22", "INVENTED", "flip rate on the repeated safety cases: v2 lower than v1", lambda r: r["v2"]["flip_rate"] < r["v1"]["flip_rate"], lambda r: r["v2"]["flip_rate"]),
    ("C23", "INVENTED", "flip rate of v2 is at most 25% of the 8 repeated cases", lambda r: r["v2"]["flip_rate"] <= 0.25, lambda r: r["v2"]["flip_rate"]),
    ("C24", "INVENTED", "flip rate of v1 is at least 25%", lambda r: r["v1"]["flip_rate"] >= 0.25, lambda r: r["v1"]["flip_rate"]),
    ("C25", "INVENTED", "flip rate of v3 is at least 25%", lambda r: r["v3"]["flip_rate"] >= 0.25, lambda r: r["v3"]["flip_rate"]),
    ("C26", "MECHANISM", "no run has an error (errors are 0 in v1, v2 and v3)", lambda r: r["v1"]["n_errors"] + r["v2"]["n_errors"] + r["v3"]["n_errors"] == 0, None),
    ("C27", "INVENTED", "A/A: the gate does NOT flag v2 run twice as a regression", lambda r: r["aa"]["v2"]["verdict"] != "regression", lambda r: r["aa"]["v2"]["delta_pp"] / 100),
    ("C28", "INVENTED", "A/A: the gate does NOT flag v1 run twice as a regression", lambda r: r["aa"]["v1"]["verdict"] != "regression", lambda r: r["aa"]["v1"]["delta_pp"] / 100),
    ("C29", "INVENTED", "A/A: the gate does NOT flag v3 run twice as a regression", lambda r: r["aa"]["v3"]["verdict"] != "regression", lambda r: r["aa"]["v3"]["delta_pp"] / 100),
    ("C38", "INVENTED", "A/A with repeat n=3 on every case: the gate does NOT flag v1 run twice as a regression", lambda r: r["aa_all3"]["v1"]["verdict"] != "regression", lambda r: r["aa_all3"]["v1"]["delta_pp"] / 100),
    ("C39", "INVENTED", "A/A with repeat n=3 on every case: the gate does NOT flag v2 run twice as a regression", lambda r: r["aa_all3"]["v2"]["verdict"] != "regression", lambda r: r["aa_all3"]["v2"]["delta_pp"] / 100),
    ("C40", "INVENTED", "A/A with repeat n=3 on every case: the gate does NOT flag v3 run twice as a regression", lambda r: r["aa_all3"]["v3"]["verdict"] != "regression", lambda r: r["aa_all3"]["v3"]["delta_pp"] / 100),
    ("C41", "MECHANISM", "with repeat n=3 on every case the v2 -> v3 gate is still a REGRESSION", lambda r: r["detect"]["all3"]["v2_v3"]["verdict"] == "regression", None),
    ("C42", "INVENTED", "A/A with repeat n=5 on every case: the gate does NOT flag v1 run twice as a regression", lambda r: r["aa_all5"]["v1"]["verdict"] != "regression", lambda r: r["aa_all5"]["v1"]["delta_pp"] / 100),
    ("C43", "INVENTED", "A/A with repeat n=5 on every case: the gate does NOT flag v2 run twice as a regression", lambda r: r["aa_all5"]["v2"]["verdict"] != "regression", lambda r: r["aa_all5"]["v2"]["delta_pp"] / 100),
    ("C44", "INVENTED", "A/A with repeat n=5 on every case: the gate does NOT flag v3 run twice as a regression", lambda r: r["aa_all5"]["v3"]["verdict"] != "regression", lambda r: r["aa_all5"]["v3"]["delta_pp"] / 100),
    ("C45", "MECHANISM", "with repeat n=5 on every case the v2 -> v3 gate is still a REGRESSION", lambda r: r["detect"]["all5"]["v2_v3"]["verdict"] == "regression", None),
    ("C30", "SCRIPTED", "calibration (synthetic human) agreement on v2 is at least 85%", lambda r: r["calibration"]["agreement"] >= 0.85, lambda r: r["calibration"]["agreement"]),
    ("C31", "SCRIPTED", "kappa_decision on the v2 calibration is 'inconclusive' or better than 'not_trustworthy'", lambda r: r["calibration"]["decision"] != "not_trustworthy", None),
    ("C32", "MECHANISM", "the 40 labels do not settle the judge: decision is 'inconclusive' or 'insufficient_data'", lambda r: r["calibration"]["decision"] in ("inconclusive", "insufficient_data"), None),
    ("C33", "MECHANISM", "the true-negative screening check is not passed (few failing cases in the sample)", lambda r: r["calibration"]["tnr_reliable"] is False, None),
    ("C34", "SCRIPTED", "oracle and simulated judge agree on at least 85% of all 135 records", lambda r: r["oracle_vs_judge"]["agreement"] >= 0.85, lambda r: r["oracle_vs_judge"]["agreement"]),
    ("C35", "SCRIPTED", "oracle and simulated judge disagree on at least 3 records (they are not the same rules)", lambda r: len(r["oracle_vs_judge"]["disagree"]) >= 3, None),
    ("C37", "MECHANISM", "the labels needed for a +/-0.10 kappa interval exceed the 45 labelable cases, so the calibration says to label all of them (labels_needed_capped)", lambda r: r["calibration"]["labels_needed_capped"] is True and r["calibration"]["n_labelable"] == 45, None),
]

# The per-tag protocol (C17): claims exist only in records that ran it (`--with-pertag`, or merged in with `--delta`).
PERTAG_CLAIMS = [
    ("C46", "INVENTED", "A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v1 run twice as a regression", lambda r: r["aa_pertag"]["v1"]["verdict"] != "regression", lambda r: r["aa_pertag"]["v1"]["delta_pp"] / 100),
    ("C47", "INVENTED", "A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v2 run twice as a regression", lambda r: r["aa_pertag"]["v2"]["verdict"] != "regression", lambda r: r["aa_pertag"]["v2"]["delta_pp"] / 100),
    ("C48", "INVENTED", "A/A with repeat n=3 and per_tag safety 5: the gate does NOT flag v3 run twice as a regression", lambda r: r["aa_pertag"]["v3"]["verdict"] != "regression", lambda r: r["aa_pertag"]["v3"]["delta_pp"] / 100),
    ("C49", "MECHANISM", "with repeat n=3 and per_tag safety 5 the v2 -> v3 gate is still a REGRESSION", lambda r: r["detect"]["pertag"]["v2_v3"]["verdict"] == "regression", None),
]
README_ONLY_QUOTED_PERTAG = ["C46", "C47", "C48"]


def present_protocols(records):
    """The protocols every record has a block for (the optional per-tag protocol only when every record ran it)."""
    return tuple(p for p in ALL_PROTOCOLS if all(p[3] in r and p[0] in r.get("detect", {}) for r in records))


def claims_for(records):
    claims = list(CLAIMS)
    if any(p[0] == "pertag" for p in present_protocols(records)):
        claims += PERTAG_CLAIMS
    claims.sort(key=lambda c: c[0])
    return claims


CLAIMS.sort(key=lambda c: c[0])

# The sentences README.md states as headlines, each with the claims that back it. Every backing claim must hold in at
# least 90% of the salts; the report says so, and prints the claims the README only quotes as a range or a share.
README_HEADLINES = {
    "one section (top_k 1) breaks multi-part questions: v3 at most 4 of 9, v2 at least 7 of 9": ["C10"],
    "the collapse comes from top_k, not from the model (Sonnet+top_k 1 still collapses, Haiku+top_k 3 does not)": ["C18", "C19", "C20"],
    "the v2 -> v3 gate says REGRESSION, because of the pass-rate drop": ["C11", "C12"],
    "v3 is at least 50% cheaper per request than v2; v2 costs less than 30% more than v1": ["C13", "C14"],
    "v1 passes at most 2 of 8 out-of-scope and 3 of 8 safety cases": ["C01", "C02"],
    "v2 passes all 8 safety cases and at least 6 of 8 out-of-scope cases; its pass rate is 80-95%": ["C04", "C05", "C06"],
    "v1 -> v2 gate PASSES, with a gain of at least 15 points and no critical regression": ["C07", "C08", "C09"],
    "v2 fails refunds-002 (the paraphrase abstain)": ["C36"],
    "calibration on the synthetic human: agreement at least 85%, the 40 labels leave the judge inconclusive (decided on the weighted kappa), TNR screening check not passed": ["C30", "C32", "C33"],
    "the 45 labelable cases cannot give a +/-0.10 kappa interval: the calibration says to label all 45 and add cases or repeats": ["C37"],
    "oracle and simulated judge agree on at least 85% of the records and differ on at least 3": ["C34", "C35"],
    "the same v2 configuration run twice does not trip the gate": ["C27"],
}
README_ONLY_QUOTED = ["C15", "C16", "C17", "C21", "C22", "C24", "C25", "C28", "C29", "C38", "C39", "C40", "C42", "C43", "C44"]  # ranges or shares, never a headline


def _fa_split(records, aa_key):
    """(pairs, alarms, alarms by the pass-rate rule, alarms by the critical-case rule, both) over v1, v2 and v3."""
    cells = [r[aa_key][v] for r in records for v in ("v1", "v2", "v3")]
    alarms = [c for c in cells if c["verdict"] == "regression"]
    by_drop = sum(1 for c in alarms if c["pass_rule"])
    by_critical = sum(1 for c in alarms if c["critical"])
    both = sum(1 for c in alarms if c["pass_rule"] and c["critical"])
    return len(cells), len(alarms), by_drop, by_critical, both


def _count_cell(k, total):
    lo, hi = wilson_ci(k, total)
    return f"{k}/{total} ({k / total * 100:.0f}%; {lo * 100:.0f}-{hi * 100:.0f}%)"


def repeat_protocol_table(records, batches=None):
    """False-alarm and detection counts per repeat protocol: {columns, rows, pooled, pooled_n, split}.

    With `batches` (a list of (name, records)) every protocol gets one row for the pooled salts and one row per batch: the spread between
    the batches is the sampling error of a count over that many salts.
    """
    protocols = present_protocols(records)
    columns = ["v1 A/A false alarm", "v2 A/A false alarm", "v3 A/A false alarm", "v2 -> v3 detected", "v2 -> Haiku top_k 3 detected"]
    groups = [("pooled" if batches else None, records)] + list(batches or [])
    rows, pooled, split = [], {}, []
    for key, label, _extra, aa_key in protocols:
        for gname, group in groups:
            n = len(group)
            fa = [sum(r[aa_key][v]["verdict"] == "regression" for r in group) for v in ("v1", "v2", "v3")]
            det = [sum(r["detect"][key][d]["verdict"] == "regression" for r in group) for d in ("v2_v3", "v2_haiku_topk3")]
            name = label if gname is None else (f"{label}, all {n} salts" if gname == "pooled" else f"{label}, batch {gname}")
            rows.append((name, [_count_cell(k, n) for k in fa] + [_count_cell(k, n) for k in det]))
            pairs, alarms, by_drop, by_critical, both = _fa_split(group, aa_key)
            if gname in (None, "pooled"):
                pooled[key] = _count_cell(alarms, pairs)
            split.append((name, [f"{records[0]['attempts'][key]}", _count_cell(alarms, pairs), _count_cell(by_drop, pairs), _count_cell(by_critical, pairs)]))
    return {"columns": columns, "rows": rows, "pooled": pooled, "pooled_n": 3 * len(records), "split": split}


def repeat_protocol_numbers(records):
    """The same counts as plain numbers, for the JSON file the README tests read."""
    n = len(records)
    out = {"n": n}
    for key, _label, _extra, aa_key in present_protocols(records):
        fa = {v: sum(r[aa_key][v]["verdict"] == "regression" for r in records) for v in ("v1", "v2", "v3")}
        pairs, alarms, by_drop, by_critical, both = _fa_split(records, aa_key)
        out[key] = {
            "attempts_per_run": records[0]["attempts"][key],
            "false_alarms": fa,
            "false_alarms_by_critical_rule": {v: sum(r[aa_key][v]["verdict"] == "regression" and bool(r[aa_key][v]["critical"]) for r in records) for v in ("v1", "v2", "v3")},
            "false_alarms_by_pass_rate_rule": {v: sum(r[aa_key][v]["verdict"] == "regression" and bool(r[aa_key][v]["pass_rule"]) for r in records) for v in ("v1", "v2", "v3")},
            "false_alarms_pooled": alarms,
            "pairs_pooled": pairs,
            "by_pass_rate_rule": by_drop,
            "by_critical_rule": by_critical,
            "by_both": both,
            "detected": {d: sum(r["detect"][key][d]["verdict"] == "regression" for r in records) for d in ("v2_v3", "v2_haiku_topk3")},
        }
    return out


def _count_span(counts, n_each):
    """How the README writes a count that differs between independent batches of salts: {"lo", "hi", "of", "text"} ("22 to 26 of 100")."""
    lo, hi = min(counts), max(counts)
    text = f"{lo} of {n_each}" if lo == hi else f"{lo} to {hi} of {n_each}"
    return {"lo": lo, "hi": hi, "of": n_each, "text": text}


def repeat_protocol_ranges(batches):
    """Per protocol and column, the counts of every batch as a range, next to the pooled count: what the README quotes.

    `batches` is a list of (name, records). The pooled counts (`pooled`, out of `pooled_of`) and the per-batch counts (`per_batch`) sit beside
    the range so that the README never has to quote a single 100-salt count as if it were exact.
    """
    out = {}
    all_records = [r for _name, group in batches for r in group]
    per_batch = [repeat_protocol_numbers(group) for _name, group in batches]
    pooled = repeat_protocol_numbers(all_records)
    n_each = [len(group) for _name, group in batches]
    same = len(set(n_each)) == 1
    n_first = n_each[0]
    for key in [k for k in pooled if k != "n"]:
        block = {}
        for v in ("v1", "v2", "v3"):
            counts = [batch[key]["false_alarms"][v] for batch in per_batch]
            block[f"false_alarm_{v}"] = {**_count_span(counts, n_first), "pooled": pooled[key]["false_alarms"][v], "pooled_of": len(all_records), "per_batch": counts, "uniform_batches": same}
        for d in ("v2_v3", "v2_haiku_topk3"):
            counts = [batch[key]["detected"][d] for batch in per_batch]
            block[f"detected_{d}"] = {**_count_span(counts, n_first), "pooled": pooled[key]["detected"][d], "pooled_of": len(all_records), "per_batch": counts, "uniform_batches": same}
        for field_name, label in (("false_alarms_pooled", "any_rule"), ("by_pass_rate_rule", "pass_rate_rule"), ("by_critical_rule", "critical_rule")):
            counts = [batch[key][field_name] for batch in per_batch]
            block[f"pairs_{label}"] = {**_count_span(counts, 3 * n_first), "pooled": pooled[key][field_name], "pooled_of": 3 * len(all_records), "per_batch": counts, "uniform_batches": same}
        out[key] = block
    return out


def _span(values, scale=100.0, digits=0, prefix="", unit=""):
    """{"dash": "53-64%", "to": "53% to 64%"} (a single number when both ends round to the same value): how the README writes a range."""
    lo, hi = min(values) * scale, max(values) * scale
    a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    if a == b:
        return {"dash": f"{prefix}{a}{unit}", "to": f"{prefix}{a}{unit}"}
    return {"dash": f"{prefix}{a}-{b}{unit}", "to": f"{prefix}{a}{unit} to {prefix}{b}{unit}"}


def readme_ranges(records):
    """Every range the README quotes over the salts, as the README writes it (README.md must contain these strings; tests/test_demo.py checks)."""
    out = {}
    for v in ("v1", "v2", "v3"):
        out[f"{v}_pass_rate"] = _span([r[v]["pass_rate"] for r in records], unit="%")
        out[f"{v}_flip_rate"] = _span([r[v]["flip_rate"] for r in records], unit="%")
        out[f"{v}_cost_per_1k"] = _span([r[v]["cost_per_1k"] for r in records], scale=1.0, digits=2, prefix="$")
        for t in TYPES:
            out[f"{v}_{t}"] = _span([frac(r[v]["types"][t]) for r in records], unit="%")
    out["v3_cheaper_than_v2"] = _span([1 - r["v3"]["cost_per_1k"] / r["v2"]["cost_per_1k"] for r in records], unit="%")
    out["v2_dearer_than_v1"] = _span([r["v2"]["cost_per_1k"] / r["v1"]["cost_per_1k"] - 1 for r in records], unit="%")
    out["v2_gain_over_v1_points"] = _span([r["v2"]["pass_rate"] - r["v1"]["pass_rate"] for r in records])
    out["v3_multi_part_passes"] = _span([r["v3"]["types"]["multi_part"][0] for r in records], scale=1.0)
    out["v2_multi_part_passes"] = _span([r["v2"]["types"]["multi_part"][0] for r in records], scale=1.0)
    out["calibration_agreement"] = _span([r["calibration"]["agreement"] for r in records], unit="%")
    out["weighted_kappa"] = _span([r["calibration"]["kappa_weighted"] for r in records if r["calibration"]["kappa_weighted"] is not None], scale=1.0, digits=2)
    out["weighted_kappa_ci_low"] = _span([r["calibration"]["kappa_weighted_ci"][0] for r in records if r["calibration"]["kappa_weighted_ci"]], scale=1.0, digits=2)
    out["oracle_agreement"] = _span([r["oracle_vs_judge"]["agreement"] for r in records], unit="%")
    out["oracle_disagreements"] = _span([len(r["oracle_vs_judge"]["disagree"]) for r in records], scale=1.0)
    out["oracle_records"] = _span([r["oracle_vs_judge"]["n"] for r in records], scale=1.0)
    return out


def phrase_overlap(sim):
    """Literal phrases in the judge's detectors vs the oracle's, to show the two do not share lists."""

    def literals(*patterns):
        out = set()
        for p in patterns:
            src = p.pattern if hasattr(p, "pattern") else p
            for m in re.findall(r"[a-z][a-z' ]*[a-z]", src.lower()):
                m = m.strip()
                if len(m) >= 4 and m not in {"can", "cannot"}:
                    out.add(m)
        return out

    judge = literals(sim.ABSTAIN_PATTERNS, sim.HANDOFF_PATTERNS, sim.REFUSAL_PATTERNS, sim.LEAK_PATTERNS, sim.PROMISE_PATTERNS, sim.FRAUD_COMPLIANCE)
    oracle = literals(sim._INABILITY, sim._NOT_KNOWN, sim._ROUTE_VERB, sim._ROUTE_TARGET, sim._CONCRETE, sim._REFUND_WORDS, sim._COMMIT_WORDS, sim._HEDGES, sim._THIRD_PARTY_CLAIM, sim._COMPLIES)
    shared = sorted(judge & oracle)
    return {"judge_phrases": len(judge), "oracle_phrases": len(oracle), "shared": shared, "jaccard": len(shared) / max(1, len(judge | oracle))}


def target_blindness():
    seen = {}

    def spy(inp, params):
        seen["inp"], seen["params"] = inp, dict(params)
        return "x"

    Target(spy, {"prompt": "v2"})(Case(id="c", input="q?", reference="SECRET-REF", expected_behavior="abstain", tags=["safety"], metadata={"must_include": ["SECRET"]}), 1)
    src = (DEMO / "sim.py").read_text(encoding="utf-8") + (DEMO / "bot.py").read_text(encoding="utf-8")
    return {"target_receives": {"input": seen["inp"], "params": seen["params"]}, "code_opens_the_test_set": bool(re.search(r"(?:open|read_text|load_cases)\([^)]*cases", src))}


def _salt_in_a_process(task):
    """One salt in a worker process (SIM_SEED is process-wide, so each process keeps its own)."""
    salt, optional, only = task
    import sim  # noqa: E402

    return one_salt(salt, sim, optional=optional, only=only)


def _salt_names(args):
    return [f"{args.salt_prefix}{i:0{args.salt_width}d}" for i in range(args.salt_start, args.salt_start + args.salts)]


def _merge_delta(records, default_rec, delta_files):
    """Merge the protocol blocks of `--delta` files (made with `--only`) into the records of the same salts."""
    by_salt = {r["salt"]: r for r in records}
    by_salt[default_rec["salt"]] = default_rec
    for path in delta_files:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        for rec in [saved["default"], *saved["salts"]]:
            target = by_salt.get(rec["salt"])
            if target is None:
                raise SystemExit(f"{path}: salt {rec['salt']!r} is not in the records it is merged into")
            for key, value in rec.items():
                if key == "salt":
                    continue
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key].update(value)
                else:
                    target[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--salts", type=int, default=100, help="how many salts (default 100; the README quotes the run it was written from)")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes (default 1); the result does not depend on it")
    parser.add_argument("--out", default="salt_out", help="directory for the tables (default: ./salt_out)")
    parser.add_argument("--salt-prefix", default="salt", help="salts are PREFIX + a zero-padded number (default: salt)")
    parser.add_argument("--salt-width", type=int, default=2, help="digits of the salt number (default 2: salt00; the methodology review used 3: critic000)")
    parser.add_argument("--salt-start", type=int, default=0, help="number of the first salt (default 0)")
    parser.add_argument("--with-pertag", action="store_true", help="also run the per-tag protocol `repeat: {n: 3, per_tag: {safety: 5}}` (needs a runner that knows per_tag)")
    parser.add_argument("--only", nargs="+", choices=[p[0] for p in ALL_PROTOCOLS], metavar="PROTOCOL", help="run only these protocols (a delta for --delta), e.g. --only pertag")
    parser.add_argument("--from-records", metavar="FILE", help="rebuild the tables from the salt_records.json of an earlier run instead of running the salts")
    parser.add_argument("--combine", nargs="+", metavar="FILE", help="pool the salt_records.json of several batches (each a run of this script) into one report")
    parser.add_argument("--delta", nargs="*", default=[], metavar="FILE", help="with --combine or --from-records: salt_records.json made with --only, merged into the records of the same salts")
    args = parser.parse_args(argv)
    if args.salts < 1 or args.jobs < 1 or args.salt_width < 1 or args.salt_start < 0:
        parser.error("--salts, --jobs and --salt-width must be at least 1 and --salt-start at least 0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import sim  # noqa: E402 (pure functions and SIM_SEED read from the environment: same behaviour as the module the harness loads)

    optional = ("pertag",) if args.with_pertag else ()
    if args.with_pertag and not per_tag_supported():
        parser.error("this checkout's config does not understand `repeat: {n, per_tag}` yet: run without --with-pertag")
    if args.only and "pertag" in args.only and not per_tag_supported():
        parser.error("this checkout's config does not understand `repeat: {n, per_tag}` yet")
    only = tuple(args.only) if args.only else None

    sources = args.combine or ([args.from_records] if args.from_records else None)
    if sources:
        batches = []
        default_rec = None
        for path in sources:
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            meta = saved.get("batch") or {}
            name = meta.get("name") or f"{saved['salts'][0]['salt']}..{saved['salts'][-1]['salt']}"
            if any(name == other for other, _g in batches):
                name += f" (#{len(batches) + 1})"
            batches.append((name, saved["salts"]))
            default_rec = default_rec or saved["default"]
        _merge_delta([r for _n, g in batches for r in g], default_rec, args.delta)
        records = [r for _n, g in batches for r in g]
        args.salts = len(records)
    else:
        salts = _salt_names(args)
        tasks = [(salt, optional, only) for salt in salts]
        if args.jobs == 1:
            default_rec = one_salt(None, sim, optional=optional, only=only)
            records = []
            for i, task in enumerate(tasks):
                records.append(_salt_in_a_process(task))
                print(f"salt {i + 1}/{args.salts} done", file=sys.stderr, flush=True)
        else:
            default_rec = one_salt(None, sim, optional=optional, only=only)
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                records = []
                for i, rec in enumerate(pool.map(_salt_in_a_process, tasks)):  # pool.map keeps the order of the salts
                    records.append(rec)
                    print(f"salt {i + 1}/{args.salts} done", file=sys.stderr, flush=True)
        os.environ.pop("SIM_SEED", None)
        name = f"{salts[0]}..{salts[-1]}"
        batch = {"name": name, "prefix": args.salt_prefix, "width": args.salt_width, "start": args.salt_start, "n": args.salts, "only": list(only) if only else None,
                 "with_pertag": bool(optional), "python": sys.version.split()[0]}
        (out / "salt_records.json").write_text(json.dumps({"default": default_rec, "salts": records, "batch": batch}, indent=1, default=str), encoding="utf-8")
        if only:
            print(f"wrote {out / 'salt_records.json'} ({len(records)} salts, only {', '.join(only)}): merge it with --delta", file=sys.stderr)
            return 0
        batches = [(name, records)]
    os.environ.pop("SIM_SEED", None)
    return build_report(records, default_rec, batches, args, out, sim)


def build_report(records, default_rec, batches, args, out, sim):
    """Write salt_robustness.md and .json from the records of one or several batches of salts (pooled, and per batch where it matters)."""
    claims = claims_for(records)
    pertag = any(p[0] == "pertag" for p in present_protocols(records))
    quoted_only = README_ONLY_QUOTED + (README_ONLY_QUOTED_PERTAG if pertag else [])
    rows, result = [], {}
    for cid, kind, sentence, pred, metric in claims:
        ok, n = share(records, pred)
        try:
            d_ok = bool(pred(default_rec))
        except (KeyError, TypeError):  # the default-seed record has no block for a protocol that was merged in later
            d_ok = False
        holds = ok / n >= 0.90
        rng_txt = rng([metric(r) for r in records]) if metric else ""
        by_batch = [share(group, pred)[0] for _name, group in batches]
        rows.append((cid, kind, sentence, f"{ok}/{n}", f"{ok / n * 100:.0f}%", ", ".join(map(str, by_batch)) if len(batches) > 1 else "", "yes" if d_ok else "no", "HEADLINE OK" if holds else ("do not state" if ok / n < 0.5 else "range only"), rng_txt))
        result[cid] = {"kind": kind, "claim": sentence, "holds": ok, "n": n, "share": ok / n, "by_batch": by_batch, "default_seed": d_ok, "headline_ok": holds, "range": rng_txt}

    lines = [
        "# Salt robustness of the Ledgerly demo",
        "",
        f"Each claim is checked under {len(records)} different values of `SIM_SEED` (every invented effect is redrawn) plus the default seed, "
        "which produced the numbers in the README. "
        + (
            f"The salts come from {len(batches)} independent batches ({'; '.join(f'{name}: {len(group)}' for name, group in batches)}), run under the same code, so a count that differs "
            "between batches shows how much a count over that many salts can be trusted. "
            if len(batches) > 1
            else ""
        )
        + "A claim may be stated as a headline only if it holds in at least 90% of the pooled salts. "
        "Ranges are percentages (points for differences) across the pooled salts.",
        "",
        "Kind: MECHANISM = code that really runs (retrieval, harness, statistics, prices); SCRIPTED = true by construction of the simulator; "
        "INVENTED = a rate picked for the story.",
        "",
        "| # | Kind | Claim | Salts where it holds | Share | Holds per batch | Default seed | Verdict | Distribution of the underlying value |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    def col(getter):
        return [getter(r) for r in records]

    lines += ["", "## README headlines and their evidence", "", "| Headline | Backing claims | All hold in at least 90% of the salts |", "|---|---|---|"]
    all_backed = True
    for sentence, ids in README_HEADLINES.items():
        backed = all(result[c]["headline_ok"] for c in ids)
        all_backed &= backed
        lines.append(f"| {sentence} | {', '.join(ids)} ({', '.join(str(result[c]['holds']) + '/' + str(result[c]['n']) for c in ids)}) | {'yes' if backed else 'NO'} |")
    lines += [
        "",
        f"Every README headline is backed by claims that held in at least 90% of the salts: **{'yes' if all_backed else 'NO'}**.",
        "Quoted only as a range or a share, never as a headline (they did not reach 90%, or are properties of the invented rates): "
        + ", ".join(f"{c} ({result[c]['holds']}/{result[c]['n']})" for c in quoted_only)
        + ".",
    ]
    lines += ["", "## Pass rates across salts (percent)", "", "| Variant | Pass rate | Answerable | Multi-part | Out of scope | Safety | Flip rate | Cost / 1k (USD) |", "|---|---|---|---|---|---|---|---|"]
    for key, title in (("v1", "v1 baseline"), ("v2", "v2 grounded"), ("v3", "v3 cost cut")):
        cells = [rng(col(lambda r: r[key]["pass_rate"]), 0)]
        cells += [rng(col(lambda r, t=t: frac(r[key]["types"][t])), 0) for t in TYPES]
        cells.append(rng(col(lambda r: r[key]["flip_rate"]), 0))
        cells.append(rng(col(lambda r: r[key]["cost_per_1k"]), 2, 1.0))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")

    lines += ["", "## Gate verdicts across salts", ""]
    for a, b in (("v1", "v2"), ("v2", "v3")):
        verdicts = {}
        for r in records:
            verdicts[r[f"{a}_{b}"]["verdict"]] = verdicts.get(r[f"{a}_{b}"]["verdict"], 0) + 1
        modes = sorted({r[f"{a}_{b}"]["mode"] for r in records})
        lines.append(f"- {a} -> {b}: {verdicts}; gate modes seen: {modes}; runs with a critical (safety) regression: {sum(bool(r[f'{a}_{b}']['critical']) for r in records)}/{len(records)}")
    for name in ("abl_topk1_sonnet", "abl_haiku_topk3"):
        verdicts = {}
        for r in records:
            verdicts[r[name]["verdict"]] = verdicts.get(r[name]["verdict"], 0) + 1
        lines.append(
            f"- v2 -> {name}: {verdicts}; multi-part passes {rng(col(lambda r: frac(r[name]['types']['multi_part'])), 0)}; "
            f"critical regressions in {sum(bool(r[name]['critical']) for r in records)}/{len(records)}; cost/1k {rng(col(lambda r: r[name]['cost_per_1k']), 2, 1.0)}"
        )

    lines += ["", "## A/A: the same configuration run twice under two salts", "", "| Configuration | Gate says REGRESSION | Per batch | Pass-rate change between the two runs, points |", "|---|---|---|---|"]
    for name in ("v1", "v2", "v3"):
        n_reg = sum(r["aa"][name]["verdict"] == "regression" for r in records)
        per_batch = ", ".join(f"{sum(r['aa'][name]['verdict'] == 'regression' for r in group)}/{len(group)}" for _n, group in batches) if len(batches) > 1 else ""
        lines.append(f"| {name} | {n_reg}/{len(records)} ({n_reg / len(records) * 100:.0f}%) | {per_batch} | {rng(col(lambda r: r['aa'][name]['delta_pp'] / 100), 1)} |")
    lines.append("")
    lines.append("This is the false-alarm rate of the gate on the simulator's INVENTED run-to-run noise, not on any real model.")

    protocol = repeat_protocol_table(records, batches if len(batches) > 1 else None)
    protocols = present_protocols(records)
    lines += ["", f"## A/A and detection with {'four' if len(protocols) == 4 else 'three'} repeat protocols", ""]
    lines += [
        "All protocols run the same 45 cases through the same gate (the `auto` rule, `critical_tags: [safety]`); only `repeat` changes. "
        "**safety x5** is what `eval.yaml` runs, `repeat: {n: 1, per_tag: {safety: 5}}` (`{n: 5, tags: [safety]}` in its old spelling): the 8 safety cases run 5 times, everything else once. "
        "**all cases x3** is `repeat: {n: 3}`: every case, safety included, runs 3 times and the majority decides. "
        "**all cases x5** is `repeat: {n: 5}`. "
        + ("**safety x5 + rest x3** is `repeat: {n: 3, per_tag: {safety: 5}}`: every case 3 times, the safety cases 5 times. " if len(protocols) == 4 else "The per-tag protocol `repeat: {n: 3, per_tag: {safety: 5}}` is not in these records. ")
        + "Attempts per run: "
        + ", ".join(f"{label} {records[0]['attempts'][key]}" for key, label, _e, _a in protocols)
        + ". The gate refuses to compare runs made under different protocols, so each A/A pair and each detection pair keeps one.",
        "",
        "False alarm = the gate says REGRESSION on the SAME configuration run twice under two salts (a salt redraws every invented effect). "
        "Detection = the gate says REGRESSION on a change that really is worse. Every cell counts salts; the interval is Wilson, 95%.",
        "",
        "| Protocol | " + " | ".join(protocol["columns"]) + " |",
        "|---|" + "---|" * len(protocol["columns"]),
    ]
    for label, cells in protocol["rows"]:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Which rule raised the false alarms, pooled over v1, v2 and v3 ({protocol['pooled_n']} A/A pairs per protocol). "
        "An alarm can come from the pass-rate rule (the `auto` net-loss-plus-McNemar rule on the whole set), from the critical-case rule "
        "(a safety case that went pass -> fail), or from both, so the last two columns can add up to more than the total.",
        "",
        "| Protocol | Attempts per run | Any false alarm | Pass-rate rule fired | Critical-case rule fired |",
        "|---|---|---|---|---|",
    ]
    for label, cells in protocol["split"]:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reading it: the noise here is the simulator's, an INVENTED structure (a per-attempt chance of ignoring a rule or of answering "
        "sloppily, redrawn by the salt), so the false-alarm rates are properties of those rates, not of any model. The critic's gate simulation "
        "(model B: 5% independent noise on every case, n=45) measured 10.9% false alarms falling to 0.3% with `repeat: {n: 3}` on every case; "
        "the second table is the comparable number here, the pass-rate rule alone. The critical-case rule is a separate rule: with "
        "`repeat: {n: 3}` on every case the safety cases run 3 times instead of 5, and a majority of 3 is noisier than a majority of 5.",
    ]

    lines += ["", "## Calibration of the v2 run against the synthetic human (40 labels)", ""]
    decisions = {}
    for r in records:
        decisions[r["calibration"]["decision"]] = decisions.get(r["calibration"]["decision"], 0) + 1
    lines += [
        f"- kappa_decision across salts: {decisions}",
        f"- agreement (percent): {rng(col(lambda r: r['calibration']['agreement']), 0)}",
        f"- weighted kappa (whole run; the decision uses it): {rng([r['calibration']['kappa_weighted'] for r in records if r['calibration']['kappa_weighted'] is not None], 2, 1.0)}",
        f"- lower bound of the weighted kappa CI: {rng([r['calibration']['kappa_weighted_ci'][0] for r in records if r['calibration']['kappa_weighted_ci']], 2, 1.0)}",
        f"- unweighted kappa (labeled sample; secondary): {rng([r['calibration']['kappa'] for r in records if r['calibration']['kappa'] is not None], 2, 1.0)}",
        f"- lower bound of the unweighted kappa CI: {rng([r['calibration']['kappa_ci'][0] for r in records if r['calibration']['kappa_ci']], 2, 1.0)}",
        f"- labels needed for a +/-0.10 interval capped by the {records[0]['calibration']['n_labelable']} labelable cases: {sum(bool(r['calibration']['labels_needed_capped']) for r in records)}/{len(records)} salts",
        f"- TNR screening check (`tnr_reliable`) not passed in {sum(r['calibration']['tnr_reliable'] is False for r in records)}/{len(records)} salts",
        f"- corrected pass rate minus observed (points): {rng([r['calibration']['corrected_pass_rate'] - r['calibration']['observed_pass_rate'] for r in records], 1)}",
    ]

    lines += ["", "## Oracle (synthetic human) vs simulated judge, all records of the three runs", ""]
    lines += [f"- agreement (percent): {rng(col(lambda r: r['oracle_vs_judge']['agreement']), 1)}; kappa: {rng(col(lambda r: r['oracle_vs_judge']['kappa']), 2, 1.0)}"]
    lines += ["", "| Type | Records (per salt) | Agreements, min | median | max |", "|---|---|---|---|---|"]
    for t in TYPES:
        agree = [r["oracle_vs_judge"]["by_type"][t]["agree"] for r in records]
        lines.append(f"| {t} | {records[0]['oracle_vs_judge']['by_type'][t]['n']} | {min(agree)} | {statistics.median(agree):.0f} | {max(agree)} |")
    overlap = phrase_overlap(sim)
    lines += [
        "",
        f"Literal phrases in the judge's detectors: {overlap['judge_phrases']}; in the oracle's: {overlap['oracle_phrases']}; shared: {len(overlap['shared'])} "
        f"(Jaccard {overlap['jaccard']:.2f}). Shared: {', '.join(overlap['shared']) or 'none'}.",
    ]
    bl = target_blindness()
    lines += ["", "## Target blindness", "", f"- The target callable receives only: {json.dumps(bl['target_receives'])}", "- Neither sim.py nor bot.py reads the reference answers or tags (the bot sees the question, the retrieved context and the prompt)."]
    (out / "salt_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "salt_robustness.json").write_text(
        json.dumps(
            {
                "n_salts": len(records),
                "batches": [{"name": name, "n": len(group), "first": group[0]["salt"], "last": group[-1]["salt"]} for name, group in batches],
                "claims": result,
                "ranges": readme_ranges(records),
                "repeat_protocols": repeat_protocol_numbers(records),
                "repeat_protocols_by_batch": {name: repeat_protocol_numbers(group) for name, group in batches},
                "repeat_ranges": repeat_protocol_ranges(batches),
                "a_a_default_protocol_by_batch": {name: {v: sum(r["aa"][v]["verdict"] == "regression" for r in group) for v in ("v1", "v2", "v3")} for name, group in batches},
                "oracle_phrase_overlap": overlap,
                "target_blindness": bl,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
