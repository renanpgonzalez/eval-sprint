"""Human labeling tool: the ground truth that judge calibration needs.

`export_labeling_html` writes ONE self-contained page (inline CSS/JS, no network)
that a domain expert opens in any browser and clicks through, one case at a
time. `export_labeling_csv` is the spreadsheet alternative for people who prefer
Excel. Both are *blind*: they never contain the judge's verdict, score or
reasoning, because seeing them anchors the human and inflates agreement.

The labels come back as JSONL (or the same CSV, filled in) and are read by
`calibration.load_labels`. The `output_sha` written with every label lets the
calibration detect labels made on an output that has since changed.

Iterating on a rubric or on the AI must not cost a full labeling pass: both exports take
``previous_labels`` (a labels file, several, or loaded rows) and then leave out every case whose CURRENT output
already has a label there (`covered_case_ids`: same case id and same ``output_sha``). What is left is every case without a
label for its current output: the changed outputs and, unless the earlier labels already covered every unchanged case,
the unchanged cases that never had a label too. So only the changed outputs need a label when the earlier pass labeled
every case the run still has unchanged (for example a full pass); after a sampled pass the export also draws
never-labeled unchanged cases, because the new labels must stand for every case that has no carried label.
`calibration.calibrate` takes the old and the new label files together and counts the old labels as carried over (see there).
`plan_relabel` says how many cases that is before anything is written.

Every label row also records how the sample was drawn, so the calibration can weight
it correctly: `stratified_on` ("overall", a judge name, or "none" for a random sample)
and `stratum`. `stratum` is the judge verdict the case was stratified on, but the file
must stay blind, so it is written as an opaque per-row code (see
`calibration.stratum_token`) instead of `judge_pass` / `judge_fail`; the calibration
decodes it and checks it against the run. The code is an HMAC-SHA256 keyed with a per-run
secret that is created at the first export in `<run_dir>/.label_secret` (mode 0600) and is
never written into an export: a labeler who has only the file, and reads this code, cannot
compute the code of either verdict. A run without a folder on disk (or one where the secret
cannot be written) falls back to the legacy unkeyed code and says so in a warning. Set
`BLIND_STRATA = False` to write the plain values (for a labeling workflow where nobody is
blinded).
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .calibration import (
    LABEL_SECRET_FILE,
    _clean,
    _row_case_id,
    _sha_verified,
    _verdict,
    as_label_list,
    ensure_label_secret,
    neutralize_csv_id,
    normalize_label,
    stratum_of,
    stratum_token,
)

if TYPE_CHECKING:  # avoid importing the whole runner (and its providers) for a type hint
    from .runner import RunResult

BLIND_STRATA = True  # write `stratum` as an opaque code, not the readable judge_pass / judge_fail
CSV_COLUMNS = [
    "case_id", "tags", "input", "reference", "expected_behavior", "output", "output_sha", "run_id",
    "stratum", "stratified_on", "label", "note",
]
_CSV_TEXT_COLUMNS = ("tags", "input", "reference", "expected_behavior", "output")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


# ---------------------------------------------------------------------------- sampling
def _check_judge(run: RunResult, judge: str | None) -> None:
    if judge is None:
        return
    known = sorted({j["judge"] for r in run.results for j in r.get("judges", [])})
    if judge not in known:
        raise ValueError(f"judge {judge!r} did not run in this run; available: {known}")


class NothingToLabel(ValueError):
    """Every eligible case of the run already has a label for its current output (see ``previous_labels``)."""


def covered_case_ids(run: RunResult, previous_labels) -> frozenset[str]:
    """Ids of the run's cases whose CURRENT output already has a usable human label in ``previous_labels``.

    ``previous_labels`` is a labels file (``.jsonl`` / ``.csv``), a list of them, or already-loaded label rows. A case is
    covered when a row with a pass/fail label names it and its ``output_sha`` is present and equals the run's output sha
    for that case (the same prefix comparison the calibration uses): the human saw exactly this output. A row with no
    sha cannot be checked, so its case is NOT covered and is exported again; a case whose output changed since is not
    covered either. The run the row was made on does not matter.
    """
    by_case = run.by_case()
    covered: set[str] = set()
    for row in as_label_list(previous_labels):
        case_id = _row_case_id(row, by_case)
        record = by_case.get(case_id)
        if record is None or normalize_label(row.get("label")) is None:
            continue
        if _sha_verified(_clean(row.get("output_sha")), record["output"].get("sha")):
            covered.add(case_id)
    return frozenset(covered)


def select_sample(
    run: RunResult,
    sample: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    judge: str | None = None,
    previous_labels=None,
) -> list[dict]:
    """Pick the result records a human should label, in the order they will see them.

    Only cases with a verdict are eligible: records whose outcome is ``error`` (target or judge
    error; for records without an ``outcome`` field, a target error) are never sampled, and with
    ``judge`` set neither are cases that judge did not score. ``sample=None`` (or a sample larger
    than the eligible cases) labels every eligible case. With ``stratify`` the sample is balanced on
    the verdict: the overall verdict ``record["passed"]``, or the named ``judge``'s verdict when
    ``judge`` is given (label the judge you are about to calibrate). That gives about half fails
    and half passes, as far as the run has them. A random sample of a run that passes 90% would
    show the human only a handful of failures, too few to estimate how often the judge misses a
    bad output (TNR). The odd case goes to the failing side, and either side tops up the other if
    it runs short. `calibration.calibrate` re-weights (Horvitz-Thompson) for this, so stratifying
    does not bias TPR/TNR.

    ``previous_labels`` (see `covered_case_ids`) removes the cases that already have a label for their current output
    before anything is drawn: ``sample`` is then the number of NEW labels wanted, balanced on the verdict among the cases
    that have none. Those are the changed outputs and, unless the previous labels covered every unchanged case, the unchanged
    cases that never had a label: only changed outputs are drawn when the previous labels covered every unchanged case,
    otherwise the draw includes never-labeled unchanged ones. With no ``previous_labels`` the draw is exactly what it was
    before (same seed, same cases).

    The verdict is used for sampling only and is never shown. The selection is
    shuffled with ``seed`` so the order reveals neither dataset order nor verdict.
    """
    if sample is not None and sample < 1:
        raise ValueError("sample must be a positive number of cases")
    _check_judge(run, judge)
    rng = random.Random(seed)
    skip = covered_case_ids(run, previous_labels) if previous_labels is not None else frozenset()
    eligible = [r for r in run.results if _verdict(r, judge) is not None]
    records = [r for r in eligible if r["case_id"] not in skip]
    if eligible and not records:
        raise NothingToLabel(
            f"all {len(eligible)} labelable case(s) of this run already have a label for their current output: nothing to export "
            "(to label more of them, export without the previous labels)"
        )
    if sample is None or sample >= len(records):
        chosen = records
    elif stratify:
        fails = [r for r in records if not _verdict(r, judge)]
        passes = [r for r in records if _verdict(r, judge)]
        n_fail = min(len(fails), (sample + 1) // 2)
        n_pass = min(len(passes), sample - n_fail)
        n_fail = min(len(fails), sample - n_pass)
        chosen = rng.sample(fails, n_fail) + rng.sample(passes, n_pass)
    else:
        chosen = rng.sample(records, sample)
    rng.shuffle(chosen)
    return chosen


def plan_relabel(
    run: RunResult, previous_labels, sample: int | None = None, seed: int = 0, stratify: bool = True, judge: str | None = None
) -> dict:
    """What an export with ``previous_labels`` would contain, without writing anything.

    ``{"eligible", "covered", "uncovered", "to_label", "case_ids"}``: the run's labelable cases, how many of them already
    have a label for their current output, how many do not, how many the export would contain (``sample`` caps it) and their ids
    in export order. ``uncovered`` counts the changed outputs AND the never-labeled unchanged cases: it equals the number of
    changed outputs only when the previous labels covered every unchanged case (a full earlier pass). Unlike the exports
    it does not raise `NothingToLabel` when ``to_label`` is 0.
    """
    _check_judge(run, judge)
    eligible = [r for r in run.results if _verdict(r, judge) is not None]
    covered = covered_case_ids(run, previous_labels)
    n_covered = sum(1 for r in eligible if r["case_id"] in covered)
    try:
        chosen = select_sample(run, sample, seed, stratify, judge, previous_labels)
    except NothingToLabel:
        chosen = []
    return {
        "eligible": len(eligible),
        "covered": n_covered,
        "uncovered": len(eligible) - n_covered,
        "to_label": len(chosen),
        "case_ids": [r["case_id"] for r in chosen],
    }


def _stratified_on(stratify: bool, judge: str | None) -> str:
    """What the label rows record as the sampling design."""
    return (judge or "overall") if stratify else "none"


def _label_secret(run: RunResult, stratify: bool) -> bytes | None:
    """The run's stratum secret (created on first use, `<run_dir>/.label_secret`), or None when no keyed code is possible.

    None means the export falls back to the legacy unkeyed codes, with a warning: the run has no folder on disk,
    or the secret file cannot be written. Nothing is needed (and nothing is created) for a random sample or when
    `BLIND_STRATA` is off.
    """
    if not stratify or not BLIND_STRATA:
        return None
    run_dir = getattr(run, "run_dir", None)
    if run_dir is None:
        warnings.warn(
            "this run has no folder on disk, so there is no secret to key the stratum codes with: they use the unkeyed legacy "
            "code, which hides the judge's verdict from a reader but not from someone who reads the source. Export from a saved run.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    try:
        return ensure_label_secret(run_dir)
    except OSError as exc:
        warnings.warn(
            f"could not create {Path(run_dir) / LABEL_SECRET_FILE} ({exc}): the stratum codes use the unkeyed legacy code, which "
            "hides the judge's verdict from a reader but not from someone who reads the source.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _stratum_code(run: RunResult, record: dict, stratify: bool, judge: str | None, secret: bytes | None = None) -> str:
    """The `stratum` value of a label row: opaque by default, empty for a random sample."""
    if not stratify:
        return ""
    stratum = stratum_of(bool(_verdict(record, judge)))
    if not BLIND_STRATA:
        return stratum
    return stratum_token(run.run_id, _stratified_on(stratify, judge), record["case_id"], stratum, secret)


# ------------------------------------------------------------------------------ items
def _display_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _context_passages(meta: dict | None) -> list[str]:
    """Retrieved context as the LLM judge saw it (``output.meta["context"]``), one string per passage."""
    context = (meta or {}).get("context")
    if context is None or context == "" or context == []:
        return []
    passages = context if isinstance(context, list) else [context]
    return [str(p["text"]) if isinstance(p, dict) and "text" in p else _display_text(p) for p in passages]


def _labeling_item(record: dict, stratum: str = "") -> dict:
    """What the labeler may see. Deliberately excludes ``judges`` and ``passed``.

    ``stratum`` is an opaque code (see the module docstring), carried only so the labels file can
    record it; the page never displays it.
    """
    case, output = record["case"], record["output"]
    return {
        "case_id": record["case_id"],
        "tags": list(case.get("tags") or []),
        "input": _display_text(case.get("input")),
        "reference": case.get("reference") or "",
        "expected_behavior": case.get("expected_behavior") or "",
        "context": _context_passages(output.get("meta")),
        "output": output.get("text") or "",
        "output_sha": output.get("sha"),
        "error": output.get("error"),
        "stratum": stratum,
    }


# ------------------------------------------------------------------------------- HTML
def _embed_json(payload: dict) -> str:
    """JSON that is safe inside ``<script type="application/json">``.

    ``</`` becomes ``<\\/`` so a case that contains ``</script>`` cannot end the
    block, and ``<!--`` is escaped so it cannot switch the parser's script state.
    ASCII-only output also sidesteps any encoding problem with odd characters.
    """
    return json.dumps(payload).replace("</", "<\\/").replace("<!--", "<\\u0021--")


def export_labeling_html(
    run: RunResult,
    out_path: str | Path,
    sample: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    judge: str | None = None,
    previous_labels=None,
) -> Path:
    """Write the blind labeling page for (a sample of) the run and return its path.

    ``judge`` stratifies the sample on that judge's verdict instead of the overall one (see
    `select_sample`); every downloaded label row records ``stratified_on`` and ``stratum``.
    ``previous_labels`` exports only the cases that have no label for their current output yet (see `covered_case_ids`):
    the changed outputs and, unless the previous labels covered every unchanged case, the never-labeled unchanged ones too;
    `NothingToLabel` when there are none.
    """
    records = select_sample(run, sample, seed, stratify, judge, previous_labels)
    secret = _label_secret(run, stratify)
    payload = {
        "run_id": run.run_id,
        "suite": run.summary.get("suite"),
        "stratified_on": _stratified_on(stratify, judge),
        "items": [_labeling_item(r, _stratum_code(run, r, stratify, judge, secret)) for r in records],
    }
    values = {
        "TITLE": html.escape(f"Label AI outputs - {run.run_id}"),
        "RUN_ID": html.escape(run.run_id),
        "PAYLOAD": _embed_json(payload),
    }
    # One pass, so substituted text is never scanned for placeholders itself.
    page = re.sub(r"__(TITLE|RUN_ID|PAYLOAD)__", lambda m: values[m.group(1)], _HTML_TEMPLATE)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path


# -------------------------------------------------------------------------------- CSV
def _csv_safe(text: str) -> str:
    """Neutralise spreadsheet formulas: outputs are untrusted text and Excel runs cells starting with = + - @."""
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def export_labeling_csv(
    run: RunResult,
    out_path: str | Path,
    sample: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    judge: str | None = None,
    previous_labels=None,
) -> Path:
    """Write the blind labeling sheet as CSV (UTF-8 with BOM so Excel reads accents correctly).

    The labeler fills in ``label`` (pass/fail) and optionally ``note``; the file is
    read back by `calibration.load_labels`, which ignores every other column. The `stratum`
    and `stratified_on` columns record how the sample was drawn (leave them as they are).
    Text cells that start with a formula character get a leading apostrophe so
    Excel shows them instead of executing them. That includes ``case_id`` (an id is text the dataset
    author or the AI chose): it is written by `calibration.neutralize_csv_id` and read back exactly by
    `calibration.load_labels` (``restore_csv_id``), so a neutralised id still matches its case. Shas and
    run_id are untouched. ``previous_labels`` exports only the cases that have no label for their
    current output yet (see `covered_case_ids`): the changed outputs and, unless the previous labels covered every
    unchanged case, the never-labeled unchanged ones too; `NothingToLabel` when there are none.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = select_sample(run, sample, seed, stratify, judge, previous_labels)
    secret = _label_secret(run, stratify)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            item = _labeling_item(record, _stratum_code(run, record, stratify, judge, secret))
            row = {
                "case_id": item["case_id"],
                "tags": "; ".join(item["tags"]),
                "input": item["input"],
                "reference": item["reference"],
                "expected_behavior": item["expected_behavior"],
                "output": item["output"],
                "output_sha": item["output_sha"] or "",
                "run_id": run.run_id,
                "stratum": item["stratum"],
                "stratified_on": _stratified_on(stratify, judge),
                "label": "",
                "note": "",
            }
            row["case_id"] = neutralize_csv_id(row["case_id"])
            writer.writerow({k: _csv_safe(v) if k in _CSV_TEXT_COLUMNS else v for k, v in row.items()})
    return out_path


# ----------------------------------------------------------------------------- template
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f4f6f9;
  --surface: #ffffff;
  --text: #1b1f24;
  --muted: #566170;
  --border: #d3d9e1;
  --accent: #1f4fc4;
  --on-accent: #ffffff;
  --pass: #0f6b3a;
  --pass-bg: #e2f3ea;
  --fail: #a81e2e;
  --fail-bg: #fbe6e9;
  --on-solid: #ffffff;
  --focus: #1f4fc4;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151a;
    --surface: #1b2027;
    --text: #e8ebef;
    --muted: #9ba5b1;
    --border: #333b46;
    --accent: #7ea6ff;
    --on-accent: #0b1220;
    --pass: #5fd39a;
    --pass-bg: #12301f;
    --fail: #ff8a97;
    --fail-bg: #3a161c;
    --on-solid: #0b0f14;
    --focus: #9dbcff;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
h1, h2 { margin: 0; line-height: 1.25; }
h1 { font-size: 1.15rem; }
h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.35rem; }
code, kbd { font-family: var(--mono); font-size: 0.9em; }
kbd { border: 1px solid currentColor; border-radius: 4px; padding: 0 0.35em; opacity: 0.85; }

.top {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.6rem 16px;
  display: grid; gap: 0.5rem 1.5rem;
  grid-template-columns: 1fr;
}
.top .run { color: var(--muted); font-size: 0.85rem; }
.who { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.who label { font-weight: 600; font-size: 0.9rem; }
.who input { flex: 1 1 10rem; min-width: 0; }
.progress { display: flex; align-items: center; gap: 0.75rem; }
.progress progress { flex: 1; height: 0.8rem; accent-color: var(--accent); }
.progress span { font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 600; }
@media (min-width: 760px) {
  .top { position: sticky; top: 0; z-index: 2; grid-template-columns: auto 1fr 20rem; align-items: center; }
}

main { max-width: 52rem; margin: 0 auto; padding: 1rem 16px 2rem; display: grid; gap: 1rem; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.panel > * + * { margin-top: 1rem; }
.howto summary { cursor: pointer; font-weight: 600; }
.howto p { margin: 0.5rem 0 0; }

.meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.75rem; }
.meta .pos { font-weight: 600; }
.tags { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.tags li { border: 1px solid var(--border); border-radius: 999px; padding: 0 0.6rem; font-size: 0.8rem; color: var(--muted); }
.state { margin-left: auto; border: 2px solid var(--border); border-radius: 6px; padding: 0.05rem 0.6rem; font-weight: 600; font-size: 0.9rem; }
.state.pass { color: var(--pass); border-color: var(--pass); background: var(--pass-bg); }
.state.fail { color: var(--fail); border-color: var(--fail); background: var(--fail-bg); }
.link { background: none; border: 0; padding: 0.25rem; color: var(--accent); text-decoration: underline; cursor: pointer; font: inherit; font-size: 0.9rem; }

pre.text, .passage {
  margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word;
  font: inherit; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.85rem;
  max-height: 22rem; overflow: auto;
}
pre.text.empty { color: var(--muted); font-style: italic; }
.output pre.text { border-width: 2px; border-color: var(--accent); max-height: 32rem; }
.error { margin: 0 0 0.5rem; color: var(--fail); font-weight: 600; }
details.context summary { cursor: pointer; font-weight: 600; }
details.context ol { margin: 0.5rem 0 0; padding-left: 1.4rem; display: grid; gap: 0.5rem; }

label.field { display: block; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem; }
textarea, input[type="text"] {
  width: 100%; font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 0.65rem;
}
textarea { resize: vertical; min-height: 4.5rem; }
.panel > label.field + textarea { margin-top: 0; }

.btn {
  font: inherit; font-weight: 700; cursor: pointer; min-height: 48px; padding: 0.4rem 1rem; border-radius: 10px;
  border: 2px solid var(--border); background: var(--surface); color: var(--text);
  display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem;
}
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn.pass { color: var(--pass); border-color: var(--pass); }
.btn.fail { color: var(--fail); border-color: var(--fail); }
.btn.pass[aria-pressed="true"] { background: var(--pass); color: var(--on-solid); }
.btn.fail[aria-pressed="true"] { background: var(--fail); color: var(--on-solid); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }

.actions {
  position: sticky; bottom: 0; z-index: 2;
  background: var(--surface); border-top: 1px solid var(--border);
  padding: 0.6rem 16px calc(0.6rem + env(safe-area-inset-bottom));
}
.actions .row { max-width: 52rem; margin: 0 auto; display: grid; grid-template-columns: auto 1fr 1fr auto; gap: 0.5rem; }
.actions .btn { padding-left: 0.6rem; padding-right: 0.6rem; }
@media (max-width: 520px) { .actions kbd { display: none; } }
.opts { max-width: 52rem; margin: 0.45rem auto 0; display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; align-items: center; font-size: 0.9rem; }
.opts label { display: inline-flex; align-items: center; gap: 0.4rem; }
.opts input { width: 1.1rem; height: 1.1rem; }

.export .row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.export p { margin: 0.4rem 0 0; }
.muted { color: var(--muted); font-size: 0.9rem; }
#status { min-height: 1.5em; font-weight: 600; }
#fallback { margin-top: 0.5rem; font-family: var(--mono); font-size: 0.8rem; min-height: 8rem; }
.notice { border-left: 4px solid var(--fail); padding: 0.4rem 0.75rem; background: var(--fail-bg); border-radius: 4px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<header class="top">
  <div>
    <h1>Label AI outputs</h1>
    <div class="run">Run <code>__RUN_ID__</code></div>
  </div>
  <div class="who">
    <label for="labeler">Your name</label>
    <input type="text" id="labeler" autocomplete="name" placeholder="Added to every label you save">
  </div>
  <div class="progress">
    <progress id="bar" value="0" max="1" aria-labelledby="progressText"></progress>
    <span id="progressText">0 / 0 labeled</span>
  </div>
</header>

<main>
  <details class="panel howto">
    <summary>How to label</summary>
    <p>For each case, read the input and the AI output as a customer or a domain expert would. Choose <strong>Pass</strong> if you would be happy for a real user to receive this output. Choose <strong>Fail</strong> if it is wrong, not supported by the reference or context, unsafe or unhelpful.</p>
    <p>A short note on failures is the most useful thing you can add: the clearest ones become worked examples for the automated judge. If you are unsure, move on with the Next button and come back later.</p>
    <p>Keys: <kbd>P</kbd> pass, <kbd>F</kbd> fail, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous / next. Your work is saved in this browser as you go.</p>
  </details>

  <p id="empty" class="panel" hidden>There are no cases to label in this file.</p>

  <article id="card" class="panel">
    <div class="meta">
      <span id="position" class="pos"></span>
      <code id="caseId"></code>
      <ul id="tags" class="tags" aria-label="Tags"></ul>
      <span id="state" class="state" role="status"></span>
      <button type="button" id="clear" class="link" hidden>Clear label</button>
    </div>
    <section aria-labelledby="h-input">
      <h2 id="h-input">Input</h2>
      <pre class="text" id="input"></pre>
    </section>
    <section id="expectedWrap" aria-labelledby="h-expected">
      <h2 id="h-expected">Expected behavior</h2>
      <p id="expected" style="margin:0"></p>
    </section>
    <section id="referenceWrap" aria-labelledby="h-reference">
      <h2 id="h-reference">Reference answer</h2>
      <pre class="text" id="reference"></pre>
    </section>
    <details id="contextWrap" class="context">
      <summary>Retrieved context (<span id="contextCount"></span>)</summary>
      <ol id="context"></ol>
    </details>
    <section class="output" aria-labelledby="h-output">
      <h2 id="h-output">AI output (the thing to judge)</h2>
      <p id="targetError" class="error" hidden></p>
      <pre class="text" id="output"></pre>
    </section>
    <label class="field" for="note">Note (optional, for example why it fails)</label>
    <textarea id="note" rows="3"></textarea>
  </article>

  <section class="panel export" aria-labelledby="h-export">
    <h2 id="h-export">Save your labels</h2>
    <p id="doneNote" hidden><strong>Every case is labeled.</strong> Download your labels and send the file back.</p>
    <p class="muted">Labels are saved automatically in this browser. Download the file when you are done, or at any time to keep a copy.</p>
    <p id="storageWarning" class="notice" hidden>This browser is blocking local storage, so nothing is saved automatically. Download your labels before closing this tab.</p>
    <div class="row">
      <button type="button" class="btn primary" id="download">Download labels.jsonl</button>
      <button type="button" class="btn" id="copy">Copy JSONL</button>
    </div>
    <p id="status" role="status" aria-live="polite"></p>
    <textarea id="fallback" readonly hidden aria-label="Labels as JSONL: select all and copy"></textarea>
  </section>
</main>

<nav class="actions" id="actions" aria-label="Labeling actions">
  <div class="row">
    <button type="button" class="btn" id="prev" aria-keyshortcuts="ArrowLeft"><span aria-hidden="true">&larr;</span> Prev</button>
    <button type="button" class="btn pass" id="pass" aria-pressed="false" aria-keyshortcuts="P"><span aria-hidden="true">&#10003;</span> Pass <kbd aria-hidden="true">P</kbd></button>
    <button type="button" class="btn fail" id="fail" aria-pressed="false" aria-keyshortcuts="F"><span aria-hidden="true">&#10007;</span> Fail <kbd aria-hidden="true">F</kbd></button>
    <button type="button" class="btn" id="next" aria-keyshortcuts="ArrowRight">Next <span aria-hidden="true">&rarr;</span></button>
  </div>
  <div class="opts">
    <label><input type="checkbox" id="autoAdvance"> Go to the next case after labeling</label>
    <button type="button" class="link" id="nextUnlabeled">Next unlabeled</button>
  </div>
</nav>

<script type="application/json" id="items-data">__PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("items-data").textContent);
  const items = data.items;
  const runId = data.run_id;
  const total = items.length;
  const $ = (id) => document.getElementById(id);
  const STORE_KEY = "evalsprint:labels:" + runId;

  // labels: { [case_id]: { label: "pass" | "fail" | null, note: string } }
  const state = { labeler: "", autoAdvance: true, index: 0, labels: {} };
  let storageOk = true;   // false once localStorage proves unusable (blocked, private mode, quota)
  let unsaved = false;    // labels changed since the last download or copy

  // ---- persistence: every storage access is wrapped, the page works without it
  function loadState() {
    let raw = null;
    try { raw = window.localStorage.getItem(STORE_KEY); } catch (err) { storageOk = false; }
    if (!raw) return;
    let saved;
    try { saved = JSON.parse(raw); } catch (err) { return; }
    if (!saved || typeof saved !== "object") return;
    if (typeof saved.labeler === "string") state.labeler = saved.labeler;
    state.autoAdvance = saved.autoAdvance !== false;
    if (Number.isInteger(saved.index)) state.index = Math.max(0, Math.min(saved.index, total - 1));
    if (saved.labels && typeof saved.labels === "object") state.labels = saved.labels;
  }

  function saveState() {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch (err) {
      storageOk = false;
      $("storageWarning").hidden = false;
    }
  }

  // ---- model helpers
  function entryOf(id) {
    const e = state.labels[id];
    return e && typeof e === "object" ? e : null;
  }
  function labelOf(id) {
    const e = entryOf(id);
    return e && (e.label === "pass" || e.label === "fail") ? e.label : null;
  }
  function noteOf(id) {
    const e = entryOf(id);
    return e && typeof e.note === "string" ? e.note : "";
  }
  function countLabeled() {
    return items.filter((it) => labelOf(it.case_id)).length;
  }
  function ensureEntry(id) {
    let e = entryOf(id);
    if (!e) { e = { label: null, note: "" }; state.labels[id] = e; }
    return e;
  }

  // ---- rendering (data only ever goes through textContent)
  function fill(el, value, emptyText) {
    const has = typeof value === "string" && value.trim() !== "";
    el.textContent = has ? value : emptyText;
    el.classList.toggle("empty", !has);
  }

  function render() {
    const done = countLabeled();
    $("bar").max = Math.max(total, 1);
    $("bar").value = done;
    $("progressText").textContent = done + " / " + total + " labeled";
    $("doneNote").hidden = !(total > 0 && done === total);
    $("labeler").value = state.labeler;
    $("autoAdvance").checked = state.autoAdvance;
    $("storageWarning").hidden = storageOk;
    $("empty").hidden = total > 0;
    $("card").hidden = total === 0;
    $("actions").hidden = total === 0;
    if (total === 0) return;

    const item = items[state.index];
    const label = labelOf(item.case_id);
    $("position").textContent = "Case " + (state.index + 1) + " of " + total;
    $("caseId").textContent = item.case_id;
    const tags = $("tags");
    tags.replaceChildren();
    tags.hidden = !item.tags.length;
    item.tags.forEach((t) => { const li = document.createElement("li"); li.textContent = t; tags.appendChild(li); });

    const stateEl = $("state");
    stateEl.textContent = label ? "Labeled: " + (label === "pass" ? "Pass" : "Fail") : "Not labeled yet";
    stateEl.className = "state" + (label ? " " + label : "");
    $("clear").hidden = !label;

    fill($("input"), item.input, "(empty input)");
    $("expectedWrap").hidden = !item.expected_behavior;
    $("expected").textContent = item.expected_behavior;
    $("referenceWrap").hidden = !item.reference;
    fill($("reference"), item.reference, "");
    const ctx = $("context");
    ctx.replaceChildren();
    $("contextWrap").hidden = !item.context.length;
    $("contextCount").textContent = item.context.length + (item.context.length === 1 ? " passage" : " passages");
    item.context.forEach((p) => { const li = document.createElement("li"); li.className = "passage"; li.textContent = p; ctx.appendChild(li); });
    $("targetError").hidden = !item.error;
    $("targetError").textContent = item.error ? "The system returned an error instead of an answer: " + item.error : "";
    fill($("output"), item.output, "(empty output)");
    $("note").value = noteOf(item.case_id);

    $("pass").setAttribute("aria-pressed", label === "pass" ? "true" : "false");
    $("fail").setAttribute("aria-pressed", label === "fail" ? "true" : "false");
    $("prev").disabled = state.index === 0;
    $("next").disabled = state.index === total - 1;
  }

  // ---- actions
  function go(delta) {
    const next = Math.max(0, Math.min(state.index + delta, total - 1));
    if (total === 0 || next === state.index) return;
    state.index = next;
    saveState();
    render();
    window.scrollTo(0, 0);
  }

  function setLabel(value) {
    if (total === 0) return;
    ensureEntry(items[state.index].case_id).label = value;
    unsaved = true;
    const advance = state.autoAdvance && state.index < total - 1;
    if (advance) state.index += 1;
    saveState();
    render();
    if (advance) window.scrollTo(0, 0);
  }

  function clearLabel() {
    if (total === 0) return;
    ensureEntry(items[state.index].case_id).label = null;
    unsaved = true;
    saveState();
    render();
  }

  function goNextUnlabeled() {
    for (let step = 1; step <= total; step++) {
      const i = (state.index + step) % total;
      if (!labelOf(items[i].case_id)) {
        state.index = i;
        saveState();
        render();
        window.scrollTo(0, 0);
        return;
      }
    }
    setStatus("Every case already has a label.");
  }

  // ---- export
  function toJsonl() {
    const name = state.labeler.trim();
    const lines = items
      .filter((it) => labelOf(it.case_id))
      .map((it) => JSON.stringify({
        case_id: it.case_id,
        label: labelOf(it.case_id),
        output_sha: it.output_sha,
        note: noteOf(it.case_id),
        labeler: name || null,
        run_id: runId,
        stratum: it.stratum || null,
        stratified_on: data.stratified_on || null,
      }));
    return lines.length ? lines.join("\n") + "\n" : "";
  }

  function setStatus(message) { $("status").textContent = message; }

  function download() {
    const text = toJsonl();
    if (!text) { setStatus("Nothing to download yet: label at least one case."); return; }
    try {
      const blob = new Blob([text], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "labels-" + runId.replace(/[^A-Za-z0-9._-]+/g, "_") + ".jsonl";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      unsaved = false;
      setStatus("Downloaded " + countLabeled() + " labels.");
    } catch (err) {
      setStatus("The download was blocked. Use Copy JSONL instead.");
    }
  }

  function legacyCopy(text) {
    const box = $("fallback");
    box.hidden = false;
    box.value = text;
    box.focus();
    box.select();
    return document.execCommand && document.execCommand("copy");
  }

  function copy() {
    const text = toJsonl();
    if (!text) { setStatus("Nothing to copy yet: label at least one case."); return; }
    const done = () => { unsaved = false; setStatus("Copied " + countLabeled() + " labels to the clipboard."); };
    const manual = () => {
      let ok = false;
      try { ok = legacyCopy(text); } catch (err) { ok = false; }
      if (ok) { $("fallback").hidden = true; done(); }
      else setStatus("Copy blocked by the browser: select the text below and copy it manually.");
    };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, manual);
      } else {
        manual();
      }
    } catch (err) {
      manual();
    }
  }

  // ---- wiring
  $("pass").addEventListener("click", () => setLabel("pass"));
  $("fail").addEventListener("click", () => setLabel("fail"));
  $("clear").addEventListener("click", clearLabel);
  $("prev").addEventListener("click", () => go(-1));
  $("next").addEventListener("click", () => go(1));
  $("nextUnlabeled").addEventListener("click", goNextUnlabeled);
  $("download").addEventListener("click", download);
  $("copy").addEventListener("click", copy);
  $("note").addEventListener("input", (e) => {
    if (total === 0) return;
    ensureEntry(items[state.index].case_id).note = e.target.value;
    unsaved = true;
    saveState();
  });
  $("labeler").addEventListener("input", (e) => { state.labeler = e.target.value; unsaved = true; saveState(); });
  $("autoAdvance").addEventListener("change", (e) => { state.autoAdvance = e.target.checked; saveState(); });

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
    const t = e.target;
    const tag = t && t.tagName;
    const typing = tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable)
      || (tag === "INPUT" && t.type !== "checkbox" && t.type !== "radio");
    if (typing) return;  // letters belong to the text the person is entering
    const key = e.key;
    if (key === "p" || key === "P") { if (!e.repeat) setLabel("pass"); }
    else if (key === "f" || key === "F") { if (!e.repeat) setLabel("fail"); }
    else if (key === "ArrowLeft") go(-1);
    else if (key === "ArrowRight") go(1);
    else return;
    e.preventDefault();
  });

  window.addEventListener("beforeunload", (e) => {
    // Only when nothing is autosaved: otherwise the labels survive a reload anyway.
    if (!storageOk && unsaved && countLabeled() > 0) { e.preventDefault(); e.returnValue = ""; }
  });

  loadState();
  render();
})();
</script>
</body>
</html>
"""
