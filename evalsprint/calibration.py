"""Judge calibration: how far can the automated verdicts be trusted?

An LLM judge is only useful for a regression gate if its verdicts agree with
what a domain expert would say. This module compares the judge (or the overall
case verdict) with human labels collected via ``evalsprint label`` and reports

* agreement, Cohen's kappa (with a CI) and a confusion matrix,
* TPR / TNR (how often the judge passes good outputs / catches bad ones),
* a pass rate corrected for the judge's known error rates (Rogan-Gladen),
* ``kappa_decision``: trustworthy / usable / inconclusive / not_trustworthy /
  insufficient_data, decided on the kappa CI and never on the point estimate,
* the disagreements worth reading, warnings and a plain-English recommendation.

Labeling samples are stratified by verdict (see ``labeling.select_sample``), so
the labeled cases contain far more judge failures than the run as a whole.
Computing TPR/TNR naively on such a sample is biased: the judge-fail cases are
over-represented, which inflates TNR and deflates TPR. The estimates below are
therefore Horvitz-Thompson weighted: each labeled case counts N_h / n_h times,
where h is the sampling stratum recorded with the label (judge_pass /
judge_fail of whatever verdict the sample was stratified on), N_h the number of
scoreable run cases in that stratum and n_h the labeled ones. This stays exact
when a NAMED judge is calibrated from a sample that was stratified on the
overall verdict (the default workflow): the strata are the sampling strata, not
the verdict being calibrated. When every case is labeled, or the sample is
random, the weights are ~1 and the result equals the plain confusion-matrix rates.

Cases whose outcome is ``error`` (target error or any judge error) have no
verdict, for the overall verdict and for a named judge alike: they are never
sampled and never counted as judge fails.

The stratum a label records is an opaque token (HMAC-SHA256 with a per-run secret kept in
``<run_dir>/.label_secret``, never exported), so the labeler cannot decode the judge's verdict.
The weights never trust it: they use the stratum recomputed from the run; the token only
serves to warn about labels that belong to another run.

`export_fewshot` closes the rubric-improvement loop: it writes the clearest disagreements as
rubric ``examples:`` plus a random holdout pool of labeled cases to leave out of the next
calibration, so the judge is not graded on the cases it was shown.

Iterating on the rubric must not cost a full labeling pass. Human labels stay valid while the AI output they
judged is unchanged (same ``output_sha``), so `calibrate` accepts several label files and treats the labels
made on an EARLIER run whose output still matches as *carried over*: those cases count as a fully labeled
(census) part of this run, and the labels made on this run (`evalsprint label --previous-labels ...` exports only the
cases that have no label for their current output yet) are weighted to represent the remaining cases. Only the changed
outputs need a label when the earlier labels covered every unchanged case (a full earlier pass); after a sampled pass the
export also draws the never-labeled unchanged cases, since the new labels must stand for all cases without a carried label.
Pooling old and new labels under one set of weights instead would be biased, because the old sample was drawn on the old
run's verdicts (see `calibrate`, "Carried-over labels").

``tnr_reliable`` is a screening flag, not a guarantee: it says the true-negative rate is measured narrowly (enough
human fails, a tight interval) and that its point estimate is at least 75%. A judge whose true rate is a little
below 75% still passes the screen some of the time, because a threshold on the estimate selects lucky reads.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import hmac
import io
import json
import math
import os
import random
import re
import secrets
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from collections.abc import Iterable, Sequence

import yaml

from .stats import cohen_kappa, kappa_label, percentile, rogan_gladen

if TYPE_CHECKING:  # avoid importing the whole runner (and its providers) for a type hint
    from .runner import RunResult

# Thresholds behind the decision and the recommendation text (see `kappa_decision`, `_recommend`).
MIN_LABELS = 30  # below this, every metric is too noisy to act on (kappa_decision: insufficient_data)
MIN_HUMAN_FAILS = 10  # TNR needs at least this many human fails ...
TNR_MAX_HALF_WIDTH = 0.175  # ... its 95% interval (`_tnr_interval`) no wider than +/- this ...
TNR_MIN_RELIABLE = 0.75  # ... and a point estimate of at least this (below it the judge is weak, and the flag could not be monotone; see `_tnr_reliability`)
MIN_OVERLAP_FOR_IAA = 10  # cases two labelers must share before their kappa is reported
KAPPA_CI_SMOOTHING = 0.5  # pseudo-counts per possible confusion cell in the kappa bootstrap (Jeffreys); 0 = plain resampling
MIN_LABELS_RELEASE = 40  # fewer labeled cases than this can never release a judge (usable / trustworthy)
KAPPA_TRUSTWORTHY = 0.8
KAPPA_USABLE = 0.6
KAPPA_TARGET_HALF_WIDTH = 0.1  # for `labels_needed_estimate`
MAX_LABELS_NEEDED = 5000

BOOTSTRAP_ITERS = 2000
MAX_FAILED_DRAW_SHARE = 0.20  # a CI is dropped when more bootstrap draws than this are undefined
MAX_TEXT_CHARS = 500  # input/output excerpts in `disagreements`

STRATA = ("judge_pass", "judge_fail")
TNR_SCREEN_MEANING = (
    "tnr_reliable is a screening flag, not a guarantee: it says the true-negative rate is measured narrowly (enough human fails, an "
    "interval within the half-width) and that its point estimate is at least the minimum. It does not show that the true rate is at "
    "least the minimum: a threshold on an estimate also selects lucky reads, so a judge whose true rate is somewhat lower passes "
    "the screen some of the time (in the kit's simulations a judge whose true rate is 70% passed it in 2% to 28% of the runs, depending on "
    "the design)."
)
_POOLED_NOTE = (
    "That can bias TPR and TNR, because the earlier sample was drawn on the earlier run's verdicts, not on this run's: in a simulated "
    "rubric change (true-negative rate 55% to 85%) the pooled TNR read about 7 points too low, the TPR about 3 too high, and the 95% "
    "interval missed the truth in about 1 run in 5."
)
CORRECTED_CI_NOTE = (
    "The interval describes the population the test set was sampled from (it includes sampling noise "
    "in which cases were tested), not just this exact set."
)

_PASS_WORDS = {"pass", "1", "true", "yes"}
_FAIL_WORDS = {"fail", "0", "false", "no"}


# --------------------------------------------------------------------------- labels
class LabelList(list):
    """The label dicts returned by `load_labels`, plus bookkeeping about skipped rows.

    Behaves exactly like ``list[dict]``; `calibrate` reads the counters so that
    skipped rows show up in its output instead of vanishing silently.
    """

    def __init__(self, rows: Iterable[dict] = (), n_invalid: int = 0, n_unlabeled: int = 0):
        super().__init__(rows)
        self.n_invalid = n_invalid  # rows with a missing case_id or an unrecognised label
        self.n_unlabeled = n_unlabeled  # rows with an empty label (e.g. untouched CSV rows)

    def extend_from(self, other: Iterable[dict]) -> None:
        """Append another label list's rows and add its skipped-row counters (a plain iterable has none)."""
        self.extend(other)
        self.n_invalid += getattr(other, "n_invalid", 0)
        self.n_unlabeled += getattr(other, "n_unlabeled", 0)


def normalize_label(value) -> str | None:
    """Map pass/fail/1/0/true/false/yes/no (any case, bool or int) to "pass"/"fail", else None."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().lower()
    if text in _PASS_WORDS:
        return "pass"
    if text in _FAIL_WORDS:
        return "fail"
    return None


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


# A case id that starts with a formula character (after any apostrophes) is written to the labeling CSV with one
# leading apostrophe, so a spreadsheet shows it instead of running it; `restore_csv_id` undoes exactly that, for any id.
_FORMULA_ID_RE = re.compile(r"'*[=+\-@\t\r]")
_NEUTRALISED_ID_RE = re.compile(r"'+[=+\-@\t\r]")


def neutralize_csv_id(text: str) -> str:
    """The form of a case id that is safe in a spreadsheet cell: ``'`` + id when the id starts with ``= + - @``, a tab or a CR
    (optionally after apostrophes, which keeps `restore_csv_id` exact); every other id is unchanged."""
    return "'" + text if _FORMULA_ID_RE.match(text) else text


def restore_csv_id(text: str) -> str:
    """Inverse of `neutralize_csv_id`: drops one leading apostrophe when the rest starts with apostrophes and a formula character.

    ``restore_csv_id(neutralize_csv_id(x)) == x`` for every string ``x``. An id that an older evalsprint wrote into a CSV
    untouched is returned as it is, except the (practically nonexistent) id that itself starts with ``'`` and a formula character.
    """
    return text[1:] if _NEUTRALISED_ID_RE.match(text) else text


def _parse_label_row(row: dict, csv_ids: bool = False) -> dict | None:
    """Normalise one raw row. None = not labeled yet; ValueError = unusable row.

    ``csv_ids``: the row comes from a labeling CSV, whose ``case_id`` may be formula-neutralised (see `restore_csv_id`).
    A restored id is kept verbatim (an id that starts with a tab or CR would otherwise lose it to the whitespace strip).
    """
    row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    if "label" not in row:
        raise ValueError(f"row has no label field: {row!r}")
    if _clean(row["label"]) == "":
        return None
    raw_id = row.get("case_id")
    if csv_ids and isinstance(raw_id, str) and _NEUTRALISED_ID_RE.match(raw_id):
        case_id = restore_csv_id(raw_id)
    else:
        case_id = _clean(raw_id)
    label = normalize_label(row["label"])
    if not case_id or label is None:
        raise ValueError(f"unusable label row: {row!r}")
    return {
        "case_id": case_id,
        "label": label,
        "output_sha": _clean(row.get("output_sha")) or None,
        "note": _clean(row.get("note")),
        "labeler": _clean(row.get("labeler")) or None,
        "run_id": _clean(row.get("run_id")) or None,
        "stratum": _clean(row.get("stratum")) or None,
        "stratified_on": _clean(row.get("stratified_on")) or None,
    }


def _read_csv_rows(path: Path) -> list[dict]:
    """Read a labeling CSV as exported by `evalsprint label --csv` and re-saved from Excel.

    Excel may re-save it as cp1252 instead of UTF-8, and in many locales (most of
    the EU, Brazil) it separates columns with ";" - both are handled.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    header = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count)
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def _read_jsonl_rows(path: Path) -> Iterable[dict | None]:
    """Yield one dict per JSONL line; None for lines that are not a JSON object."""
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            yield row if isinstance(row, dict) else None


def _load_label_file(path: Path) -> LabelList:
    suffix = path.suffix.lower()
    csv_ids = suffix == ".csv"
    if suffix == ".csv":
        raw_rows: Iterable[dict | None] = _read_csv_rows(path)
    elif suffix == ".jsonl":
        raw_rows = _read_jsonl_rows(path)
    else:
        raise ValueError(f"unsupported labels file {path.name!r}: use .jsonl or .csv")

    labels = LabelList()
    for raw in raw_rows:
        if raw is None:  # a JSONL line that is not a JSON object
            labels.n_invalid += 1
            continue
        try:
            parsed = _parse_label_row(raw, csv_ids)
        except ValueError:
            labels.n_invalid += 1
            continue
        if parsed is None:
            labels.n_unlabeled += 1
        else:
            labels.append(parsed)
    if labels.n_invalid:
        print(f"warning: skipped {labels.n_invalid} unusable label row(s) in {path}", file=sys.stderr)
    return labels


def load_labels(path: str | os.PathLike | Sequence[str | os.PathLike]) -> LabelList:
    """Load human labels from a ``.jsonl`` or ``.csv`` file, or from several of them (the rows are concatenated in the order given).

    Each returned dict has ``case_id``, ``label`` ("pass"/"fail"), ``output_sha``,
    ``note``, ``labeler``, ``run_id``, ``stratum`` and ``stratified_on`` (all but the
    first two may be empty/None). Extra columns are ignored, so the labeling CSV
    round-trips (including a formula-neutralised ``case_id``, see `restore_csv_id`). Rows with an empty label are treated as "not
    labeled yet"; rows with a missing case_id or an unrecognised label are skipped and counted in
    ``.n_invalid`` (and reported on stderr). With several files the counters are summed; when the same labeler
    labeled the same case in two files, the later file wins (see `calibrate`).
    """
    if isinstance(path, (str, os.PathLike)):
        return _load_label_file(Path(path))
    merged = LabelList()
    for one in path:
        merged.extend_from(_load_label_file(Path(one)))
    return merged


def as_label_list(labels) -> LabelList:
    """Labels from whatever the caller has: a path, several paths, already-loaded label rows, or a mix (nested lists allowed).

    ``calibrate`` and ``labeling.export_labeling_*(previous_labels=...)`` take any of these. Order is kept; a `LabelList` is returned as it is.
    """
    if isinstance(labels, LabelList):
        return labels
    if isinstance(labels, (str, os.PathLike)):
        return load_labels(labels)
    if isinstance(labels, dict):
        raise TypeError("labels must be a path, a list of paths or a list of label dicts, not a single dict")
    merged = LabelList()
    for item in labels:
        if isinstance(item, dict):
            merged.append(item)
        elif isinstance(item, (str, os.PathLike)):
            merged.extend_from(load_labels(item))
        elif isinstance(item, (list, tuple, LabelList)):
            merged.extend_from(as_label_list(item))
        else:
            raise TypeError(f"labels must be paths or label dicts, got {type(item).__name__}")
    return merged


# --------------------------------------------------------------------- verdict access
def record_outcome(record: dict) -> str:
    """"pass" | "fail" | "error" of one result record.

    Uses the record's own ``outcome`` (contract C1). Records from older runs have none:
    they are an error when the target errored or any judge returned an error, otherwise
    the verdict is read from ``passed``.
    """
    outcome = record.get("outcome")
    if outcome in ("pass", "fail", "error"):
        return outcome
    if (record.get("output") or {}).get("error") or any(j.get("error") for j in record.get("judges") or []):
        return "error"
    return "pass" if record.get("passed") else "fail"


def _verdict(record: dict, judge: str | None) -> bool | None:
    """The automated verdict being calibrated; None when there is none.

    A case with outcome ``error`` has no verdict, whether the overall verdict or a named
    judge is calibrated: counting a default ``passed=False`` as a fail would corrupt TNR.
    A named judge that did not run on the case (or errored) has none either.
    """
    outcome = record_outcome(record)
    if outcome == "error":
        return None
    if judge is None:
        return outcome == "pass"
    for result in record.get("judges") or []:
        if result["judge"] == judge:
            return None if result.get("error") else bool(result["passed"])
    return None


def stratum_of(verdict: bool) -> str:
    return "judge_pass" if verdict else "judge_fail"


LABEL_SECRET_FILE = ".label_secret"  # per-run secret of the stratum tokens, in the run folder; never exported
HMAC_TOKEN_HEX = 16  # hex characters of the HMAC kept in a token ("s-" + 16); the legacy sha256 token kept 10
_LEGACY_TOKEN_RE = re.compile(r"s-[0-9a-f]{10}")
_HMAC_TOKEN_RE = re.compile(r"s-[0-9a-f]{%d}" % HMAC_TOKEN_HEX)
_SECRET_BYTES = 32


def _token_message(run_id: str | None, stratified_on: str, case_id: str, stratum: str) -> bytes:
    return "|".join(("evalsprint-stratum", str(run_id), str(stratified_on), str(case_id), stratum)).encode("utf-8")


def _secret_bytes(secret: bytes | str) -> bytes:
    return secret if isinstance(secret, bytes) else secret.encode("utf-8")


def stratum_token(run_id: str | None, stratified_on: str, case_id: str, stratum: str, secret: bytes | str | None = None) -> str:
    """Opaque per-label code for a stratum, written into the (blind) labeling files.

    The labeler must not see the judge verdict, and the CSV is the labeler's screen, so the
    files carry this code instead of ``judge_pass`` / ``judge_fail``. It differs from row to
    row (it covers the case id), so it reveals neither the verdict nor which cases share one.

    With a ``secret`` (the run's `<run_dir>/.label_secret`, see `ensure_label_secret`) the code is
    ``"s-"`` plus the first 16 hex characters of an HMAC-SHA256: someone who reads this code and
    has only the exported file cannot compute it, so cannot decode the judge's verdict. Without a
    ``secret`` it is the legacy 10-character sha256 code, which hides the verdict from a reader but
    not from someone who reads this code (kept so old labels still decode, and for a run that has no
    folder to keep a secret in).
    """
    message = _token_message(run_id, stratified_on, case_id, stratum)
    if secret is None:
        return "s-" + hashlib.sha256(message).hexdigest()[:10]
    return "s-" + hmac.new(_secret_bytes(secret), message, hashlib.sha256).hexdigest()[:HMAC_TOKEN_HEX]


def label_secret_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / LABEL_SECRET_FILE


def read_label_secret(run_dir: str | Path | None) -> bytes | None:
    """The run's stratum secret, or None when there is no run folder, no file or an unreadable one."""
    if run_dir is None:
        return None
    try:
        text = label_secret_path(run_dir).read_text(encoding="ascii").strip()
        secret = bytes.fromhex(text)
    except (OSError, ValueError):  # missing, unreadable, not hex (UnicodeDecodeError is a ValueError)
        return None
    return secret if len(secret) >= 16 else None


def ensure_label_secret(run_dir: str | Path) -> bytes:
    """The run's stratum secret; created (32 random bytes as hex, mode 0600) when the labels are first exported.

    An existing valid secret is reused, so labels exported earlier stay verifiable. The file is never part
    of an export; it lives next to the run's results and must not be shared or committed with them
    (the `.gitignore` template lists it). Raises OSError when it cannot be created.
    """
    existing = read_label_secret(run_dir)
    if existing is not None:
        return existing
    path = label_secret_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(_SECRET_BYTES)
    payload = (secret.hex() + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags | os.O_EXCL, 0o600)  # never follows or overwrites something that is already there
    except FileExistsError:
        raced = read_label_secret(run_dir)  # another process just created it, or the file is empty/garbled
        if raced is not None:
            return raced
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, flags | os.O_EXCL, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        os.replace(tmp, path)  # replaces an unusable file (not a symlink target); atomic
        return secret
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    return secret


def stratum_status(
    recorded: str | None, run_id: str | None, stratified_on: str, case_id: str, secret: bytes | str | None = None
) -> tuple[str | None, str]:
    """(stratum, how) for one label row's recorded stratum.

    ``how`` is ``"plain"`` (the readable value), ``"hmac"`` (a keyed token that verified), ``"legacy"`` (an old
    unkeyed token that verified), ``"unverifiable"`` (a keyed token, but the run's secret is not available:
    the stratum is None, recompute it from the run), ``"invalid"`` (a token that matches nothing: another run,
    another design or case, or a hand-edited value) or ``"none"`` (nothing recorded).
    """
    if not recorded:
        return None, "none"
    if recorded in STRATA:
        return recorded, "plain"
    if _LEGACY_TOKEN_RE.fullmatch(recorded):
        for stratum in STRATA:
            if hmac.compare_digest(recorded, stratum_token(run_id, stratified_on, case_id, stratum)):
                return stratum, "legacy"
    if _HMAC_TOKEN_RE.fullmatch(recorded):
        if secret is None:
            return None, "unverifiable"
        for stratum in STRATA:
            if hmac.compare_digest(recorded, stratum_token(run_id, stratified_on, case_id, stratum, secret)):
                return stratum, "hmac"
    return None, "invalid"


def decode_stratum(
    recorded: str | None, run_id: str | None, stratified_on: str, case_id: str, secret: bytes | str | None = None
) -> str | None:
    """The stratum a label row recorded (plain ``judge_pass`` / ``judge_fail``, a verified token), else None.

    Keyed tokens need the run's ``secret``; legacy unkeyed tokens and plain values decode without it.
    """
    return stratum_status(recorded, run_id, stratified_on, case_id, secret)[0]


def _judge_reasoning(record: dict, judge: str | None) -> str:
    """Why the automated verdict came out as it did (failing judges only when it failed)."""
    results = record.get("judges", [])
    if judge is not None:
        results = [j for j in results if j["judge"] == judge]
    elif not record["passed"]:
        results = [j for j in results if not j["passed"]] or results
    parts = []
    for result in results:
        text = result.get("reasoning") or (f"error: {result['error']}" if result.get("error") else "")
        if text:
            parts.append(text if judge is not None or len(results) == 1 else f"{result['judge']}: {text}")
    return " | ".join(parts)


def _as_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _row_case_id(row: dict, by_case: dict) -> str:
    """The case id of a label row: as written when that is a case of the run (so an id that starts or ends with whitespace, which
    the labeling CSV round-trips, still matches), else with the surrounding whitespace stripped (a padded id typed by hand)."""
    raw = row.get("case_id")
    return raw if isinstance(raw, str) and raw in by_case else _clean(raw)


def _sha_mismatch(label_sha: str | None, output_sha: str | None) -> bool:
    """True when the label was made on a different output than the one in the run.

    Compared as case-insensitive prefixes so that a full-length sha256 matches the
    12-character sha stored in the run. Missing shas cannot be verified.
    """
    if not label_sha or not output_sha:
        return False
    a, b = label_sha.lower(), output_sha.lower()
    return not (a.startswith(b) or b.startswith(a))


def _sha_verified(label_sha: str | None, output_sha: str | None) -> bool:
    """True when both shas exist and agree: the label is known to be about this very output (`_sha_mismatch` also passes a missing sha)."""
    return bool(label_sha and output_sha) and not _sha_mismatch(label_sha, output_sha)


# ---------------------------------------------------------------------------- stats
def _kappa(human: Sequence[bool], judge: Sequence[bool]) -> float | None:
    """Cohen's kappa; None when it is undefined (no items, or one identical verdict everywhere).

    `stats.cohen_kappa` returns 1.0 for the latter, which would read as "almost
    perfect" for a judge that was only ever tested on passing outputs.
    """
    if not human or len(set(human) | set(judge)) < 2:
        return None
    return cohen_kappa(list(human), list(judge))


_CELL = {(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}  # (human, judge)


def _ht_metrics(cells: dict[str, float]) -> dict[str, float | None]:
    """Rates from (weighted) confusion cells tp / fn / fp / tn."""
    tp, fn, fp, tn = (cells[k] for k in ("tp", "fn", "fp", "tn"))
    total = tp + fn + fp + tn
    if total <= 0:
        return {"tpr": None, "tnr": None, "agreement": None, "kappa": None, "human_pass": None}
    p_human, p_judge = (tp + fn) / total, (tp + fp) / total
    # Chance agreement of 1 (both raters constant on the same verdict) leaves kappa undefined. A constant rater against a
    # mixed one is defined: agreement equals chance agreement, kappa is 0 (a judge that passes everything has no skill).
    kappa = None
    pe = p_human * p_judge + (1 - p_human) * (1 - p_judge)
    if pe < 1 - 1e-12:
        kappa = ((tp + tn) / total - pe) / (1 - pe)
    return {
        "tpr": tp / (tp + fn) if tp + fn > 0 else None,
        "tnr": tn / (tn + fp) if tn + fp > 0 else None,
        "agreement": (tp + tn) / total,
        "kappa": kappa,
        "human_pass": p_human,
    }


class _Census(NamedTuple):
    """The cases that have a label carried over from an earlier run (same output): a group of the run with its own labels.

    ``marks`` is parallel to the frame: 0 = no carried label (the part the labels made on this run stand for, weighted N_u / n_u),
    1 = a carried label that is used, 2 = a carried label that is not used (held out for the few-shot examples, tied labelers).
    ``pairs`` holds the (human, judge) pairs of the marks-1 cases per stratum and ``sizes`` the number of marks-1 and marks-2 cases per
    stratum: the carried pairs count N_c / n_c times each, which is exactly once (a census) unless some carried labels were
    held out, in which case the remaining ones stand for the held-out ones (the hold-out is a random deletion of labels).
    """

    marks: Sequence[int]
    pairs: dict[str, list[tuple[bool, bool]]]
    sizes: dict[str, int]


def _weighted_cells(
    labeled: dict[str, list[tuple[bool, bool]]], frame_counts: dict[str, int], census: _Census | None = None
) -> dict[str, float] | None:
    """Horvitz-Thompson cells: every labeled pair of stratum h counts N_h / n_h times.

    None when a stratum that exists in the run has no labeled case: nothing then tells us how
    the judge behaves there, and dropping it would silently re-define the population.

    With a ``census`` (carried-over labels) its pairs count N_c / n_c times each (once, unless some were held out) and
    ``frame_counts`` / ``labeled`` describe only the cases without a carried label: N_u is the number of those in stratum h and n_u the
    labels made on this run for them.
    """
    cells = {"tp": 0.0, "fn": 0.0, "fp": 0.0, "tn": 0.0}
    if census is not None:
        for stratum, pairs in census.pairs.items():
            for pair in pairs:
                cells[_CELL[pair]] += census.sizes[stratum] / len(pairs)
    for stratum, size in frame_counts.items():
        if size <= 0:
            continue
        pairs = labeled.get(stratum) or []
        if not pairs:
            return None
        weight = size / len(pairs)
        for pair in pairs:
            cells[_CELL[pair]] += weight
    return cells


def _percentile_ci(draws: Sequence[float | None]) -> list[float] | None:
    """95% percentile interval of the defined draws; None if too many were undefined."""
    valid = [d for d in draws if d is not None]
    if not valid or len(valid) < (1 - MAX_FAILED_DRAW_SHARE) * len(draws):
        return None
    return [percentile(valid, 2.5), percentile(valid, 97.5)]


def _kappa_ci(
    pairs: Sequence[tuple[bool, bool]], iters: int, seed: int, smooth: float = 0.0, judge_verdicts: Sequence[bool] = (True, False)
) -> list[float] | None:
    """Percentile CI of the labeled sample's (unweighted) kappa; ``smooth`` > 0 adds pseudo-counts to the confusion
    cells the run allows (see ``_bootstrap``) so that a sample with no disagreement does not collapse to [1, 1]."""
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    if smooth > 0:
        counts = Counter(_CELL[pair] for pair in pairs)
        names = [_CELL[(human, v)] for v in sorted(set(judge_verdicts)) for human in (True, False)]
        names += [c for c in counts if c not in names]
        cum, total = [], 0.0
        for name in names:
            total += counts.get(name, 0) + smooth
            cum.append(total)
        for _ in range(iters):
            drawn = Counter(rng.choices(names, cum_weights=cum, k=len(pairs)))
            tp, fn, fp, tn = (drawn.get(c, 0) for c in ("tp", "fn", "fp", "tn"))
            draws.append(_ht_metrics({"tp": float(tp), "fn": float(fn), "fp": float(fp), "tn": float(tn)})["kappa"])
    else:
        for _ in range(iters):
            sample = rng.choices(pairs, k=len(pairs))
            draws.append(_kappa([h for h, _ in sample], [j for _, j in sample]))
    return _percentile_ci(draws)


def _bootstrap(
    frame: Sequence[tuple[str, bool]],
    labeled: dict[str, list[tuple[bool, bool]]],
    iters: int,
    seed: int,
    smooth: float = 0.0,
    census: _Census | None = None,
) -> dict[str, list[float] | None]:
    """Bootstrap CIs for TPR, TNR, agreement, kappa and the corrected pass rate, honouring the design.

    Each draw (1) resamples the run's scoreable cases (which sets the stratum sizes N_h and the
    observed pass rate: sampling noise of the test set itself) and (2) resamples the labeled
    pairs *within each stratum* (the sample was drawn that way, so n_h stays fixed), then
    re-computes the Horvitz-Thompson estimates. TPR, TNR and the corrected rate are computed
    from the same draw so the three stay consistent.

    With a ``census`` (carried-over labels) a draw also splits each stratum's resampled cases into those that have a carried label
    (their carried pairs are resampled and weighted by their number over the carried pairs: once each unless some were held out)
    and those without one (weighted by their number over the labels made on this run for them, as above).
    """
    rng = random.Random(seed)
    strata = sorted(labeled)
    n_frame = len(frame)
    stratum_ids = [s for s, _ in frame]
    verdicts = [v for _, v in frame]
    keys = ("tpr", "tnr", "agreement", "kappa", "corrected_pass_rate")
    draws: dict[str, list[float | None]] = {k: [] for k in keys}
    idx_range = range(n_frame)
    cell_names: dict[str, list[str]] = {}
    cum_weights: dict[str, list[float]] = {}
    census_names: dict[str, list[str]] = {}
    census_cum: dict[str, list[float]] = {}
    kinds = [(stratum_ids[i], census.marks[i]) for i in idx_range] if census is not None else []
    if smooth > 0:
        # Smoothed bootstrap: a confusion cell that can happen but was not seen still gets probability mass
        # (`smooth` pseudo-counts), so 0 or 1 observed disagreements do not collapse the interval. Only cells
        # the run allows are smoothed: with the sample stratified on the judge's own verdict, a judge-pass
        # stratum only holds (human pass, judge pass) and (human fail, judge pass) pairs.
        judge_verdicts: dict[str, set[bool]] = {}
        for stratum_id, verdict in frame:
            judge_verdicts.setdefault(stratum_id, set()).add(verdict)

        def table(pairs: Sequence[tuple[bool, bool]], stratum: str) -> tuple[list[str], list[float]]:
            allowed = [_CELL[(human, v)] for v in sorted(judge_verdicts.get(stratum, {True, False})) for human in (True, False)]
            counts = Counter(_CELL[pair] for pair in pairs)
            allowed += [c for c in counts if c not in allowed]  # defensive: never drop an observed cell
            cum: list[float] = []
            total = 0.0
            for name in allowed:
                total += counts.get(name, 0) + smooth
                cum.append(total)
            return allowed, cum

        for stratum in strata:
            cell_names[stratum], cum_weights[stratum] = table(labeled[stratum], stratum)
            # Carried cases of a stratum that also has cases without a carried label are smoothed through those (the sampled part);
            # only a stratum that is carried entirely needs pseudo-counts of its own (it is then a plain census).
            if census is not None and census.pairs.get(stratum) and not any(k == (stratum, 0) for k in kinds):
                census_names[stratum], census_cum[stratum] = table(census.pairs[stratum], stratum)
    for _ in range(iters):
        picked = rng.choices(idx_range, k=n_frame)
        p_star = sum(verdicts[i] for i in picked) / n_frame
        cells: dict[str, float] | None = {"tp": 0.0, "fn": 0.0, "fp": 0.0, "tn": 0.0}
        if census is None:
            counts = Counter(stratum_ids[i] for i in picked)
            sampled = {stratum: counts.get(stratum, 0) for stratum in strata}
        else:
            drawn = Counter(kinds[i] for i in picked)
            sampled = {stratum: drawn.get((stratum, 0), 0) for stratum in strata}
            for stratum in strata:
                size_carried = drawn.get((stratum, 1), 0) + drawn.get((stratum, 2), 0)
                if not size_carried:
                    continue
                carried = census.pairs[stratum]  # never empty: a stratum whose carried labels were all held out has no carried cases
                weight = size_carried / len(carried)
                if stratum in census_names:
                    for name in rng.choices(census_names[stratum], cum_weights=census_cum[stratum], k=len(carried)):
                        cells[name] += weight  # type: ignore[index]
                else:
                    for pair in rng.choices(carried, k=len(carried)):
                        cells[_CELL[pair]] += weight  # type: ignore[index]
        for stratum in strata:
            size = sampled[stratum]
            if size == 0:
                continue
            pairs = labeled[stratum]
            if not pairs:
                cells = None
                break
            weight = size / len(pairs)
            if smooth > 0:
                for name in rng.choices(cell_names[stratum], cum_weights=cum_weights[stratum], k=len(pairs)):
                    cells[name] += weight  # type: ignore[index]
            else:
                for pair in rng.choices(pairs, k=len(pairs)):
                    cells[_CELL[pair]] += weight  # type: ignore[index]
        if cells is None:
            for k in keys:
                draws[k].append(None)
            continue
        m = _ht_metrics(cells)
        draws["tpr"].append(m["tpr"])
        draws["tnr"].append(m["tnr"])
        draws["agreement"].append(m["agreement"])
        draws["kappa"].append(m["kappa"])
        draws["corrected_pass_rate"].append(
            None if m["tpr"] is None or m["tnr"] is None else rogan_gladen(p_star, m["tpr"], m["tnr"])
        )
    return {k: _percentile_ci(v) for k, v in draws.items()}


def _tnr_interval(
    frame: Sequence[tuple[str, bool]],
    labeled: dict[str, list[tuple[bool, bool]]],
    smooth: float = KAPPA_CI_SMOOTHING,
    z: float = 1.96,
    census: _Census | None = None,
) -> list[float] | None:
    """Closed-form 95% interval of the weighted TNR (Wilson at an effective n): [low, high], None when the TNR is undefined.

    TNR = A / (A + B), A = sum_h W_h tn_h (bad outputs the judge caught), B = sum_h W_h fp_h (bad outputs it passed),
    W_h = N_h / n_h the Horvitz-Thompson weight of sampling stratum h. Every count gets ``smooth`` pseudo-counts on the
    confusion cells the run allows (a judge-pass stratum cannot hold a caught bad output), so a judge with no observed
    miss still has an uncertain B: the rule of three, not a point estimate. With Poisson counts the delta method gives
    Var(TNR) = TNR^2 (1 - TNR)^2 (Var(A) / A^2 + Var(B) / B^2); the effective n is n_eff = TNR (1 - TNR) / Var(TNR) and the
    interval is the Wilson interval of TNR at that n_eff, clipped to [0, 1] and widened to contain the point estimate.

    Why a closed form. `tnr_reliable` is decided on this interval's half-width and has to be monotone in judge quality
    (a judge with fewer misses is never less reliable). A closed form has no resampling noise, so the flag does not
    depend on ``seed`` or ``iters`` and can be checked for monotonicity exactly (see the tests); the Wilson half-width
    falls as TNR rises above one half. Unlike a Kish effective n over the human-fail labels, this n_eff counts the
    uncertainty of the misses hiding among the judge-passes, so it does not call a judge with one lucky labeled sample
    "certain". Its coverage of the run's true TNR is measured in the cycle-3 simulation (see docs).

    With a ``census`` (carried-over labels) the carried pairs count N_c / n_c times each (once, like any census, unless some were
    held out) and N_u / n_u cover only the cases without a carried label; the pseudo-counts go on the part of a stratum that has
    cases without a carried label, or on the carried pairs of a stratum that has none (a plain census, as without carried labels).
    """
    frame_counts: Counter = Counter()
    verdicts: dict[str, set[bool]] = {}
    for i, (stratum, verdict) in enumerate(frame):
        verdicts.setdefault(stratum, set()).add(verdict)
        if census is None or census.marks[i] == 0:
            frame_counts[stratum] += 1
    caught = missed = var_caught = var_missed = raw_caught = raw_missed = 0.0
    if census is not None:
        for stratum, pairs in census.pairs.items():
            tn = sum(1 for human, judge in pairs if not human and not judge)
            fp = sum(1 for human, judge in pairs if not human and judge)
            if not pairs:
                continue
            weight = census.sizes[stratum] / len(pairs)
            raw_caught += weight * tn
            raw_missed += weight * fp
            sampled_part = frame_counts.get(stratum, 0) > 0
            tn_s = tn + (smooth if not sampled_part and False in verdicts[stratum] else 0.0)
            fp_s = fp + (smooth if not sampled_part and True in verdicts[stratum] else 0.0)
            caught += weight * tn_s
            missed += weight * fp_s
            var_caught += weight * weight * tn_s
            var_missed += weight * weight * fp_s
    for stratum, size in frame_counts.items():
        pairs = labeled.get(stratum) or []
        if not pairs:
            return None
        weight = size / len(pairs)
        tn = sum(1 for human, judge in pairs if not human and not judge)
        fp = sum(1 for human, judge in pairs if not human and judge)
        raw_caught += weight * tn
        raw_missed += weight * fp
        tn_s = tn + (smooth if False in verdicts[stratum] else 0.0)
        fp_s = fp + (smooth if True in verdicts[stratum] else 0.0)
        caught += weight * tn_s
        missed += weight * fp_s
        var_caught += weight * weight * tn_s
        var_missed += weight * weight * fp_s
    if raw_caught + raw_missed <= 0:
        return None
    point = raw_caught / (raw_caught + raw_missed)
    if caught <= 0 or missed <= 0:  # nothing can be missed (the judge failed everything) or caught: no uncertainty to add
        return [point, point]
    centre = caught / (caught + missed)
    n_eff = 1.0 / (centre * (1 - centre) * (var_caught / (caught * caught) + var_missed / (missed * missed)))
    denom = 1 + z * z / n_eff
    mid = (centre + z * z / (2 * n_eff)) / denom
    half = z * math.sqrt(centre * (1 - centre) / n_eff + z * z / (4 * n_eff * n_eff)) / denom
    return [min(max(0.0, mid - half), point), max(min(1.0, mid + half), point)]


# ------------------------------------------------------------------- label consensus
class _Vote(NamedTuple):
    is_pass: bool
    note: str
    labeler: str | None


class _Collected(NamedTuple):
    votes: dict[str, dict[str, _Vote]]
    names: dict[str, str]
    dropped: Counter
    strata: dict[str, list[tuple[str | None, str | None]]]  # case_id -> [(recorded stratum, stratified_on)] of the rows made on THIS run
    foreign_runs: Counter  # run_id -> kept label rows that were made for another run than the calibrated one (carried or not)
    carried_cases: frozenset[str]  # cases with a carried-over label: made on another run, the output sha verified unchanged (even if held out)
    carried_runs: Counter  # run_id -> kept carried-over label rows
    unverified_runs: Counter  # run_id -> kept rows of another run whose output sha could not be checked (they count as labels of this run)


def _collect_votes(
    labels: Sequence[dict], by_case: dict[str, dict], run_id: str | None = None, exclude: frozenset[str] = frozenset()
) -> _Collected:
    """Group usable labels per case and per labeler.

    ``dropped`` tallies the label rows that were dropped: "invalid" (label not pass/fail, for
    callers that skip `load_labels`), "unmatched" (unknown case id), "sha_mismatch" and "excluded"
    (the case is in ``exclude``, the few-shot holdout). A labeler
    who labeled a case twice counts once (their last label wins, e.g. after fixing a mistake);
    rows without a labeler name are separate voters because they cannot be told apart.

    ``foreign_runs`` counts the kept rows whose recorded ``run_id`` is not ``run_id``. Of those, a row whose output sha is
    present and equal to the run's is *carried over* (``carried_cases``, ``carried_runs``): the human judged exactly the output
    this run has, so the label is valid although it was made on an earlier run, and `calibrate` treats those cases as a fully
    labeled part of the run. A foreign row that cannot be verified (no sha on either side) is ``unverified_runs``: it is kept
    as before and counts as a label of this run. Carried rows do not feed ``strata``: the stratum token of another run cannot be
    decoded here and the sampling design of an earlier run is irrelevant to a census.
    """
    votes: dict[str, dict[str, _Vote]] = {}
    names: dict[str, str] = {}
    dropped: Counter = Counter()
    foreign: Counter = Counter()
    carried_runs: Counter = Counter()
    unverified: Counter = Counter()
    carried_cases: set[str] = set()
    strata: dict[str, list[tuple[str | None, str | None]]] = {}
    for i, row in enumerate(labels):
        verdict = normalize_label(row.get("label"))
        case_id = _row_case_id(row, by_case)
        if verdict is None or not case_id:
            dropped["invalid"] += 1
            continue
        record = by_case.get(case_id)
        if record is None:
            dropped["unmatched"] += 1
            continue
        label_sha, output_sha = _clean(row.get("output_sha")), record["output"].get("sha")
        row_run = _clean(row.get("run_id"))
        other_run = bool(row_run) and run_id is not None and row_run != str(run_id)
        carried = other_run and _sha_verified(label_sha, output_sha)
        if case_id in exclude:
            dropped["excluded"] += 1
            if carried:
                carried_cases.add(case_id)  # held out, but still a case that was labeled: never part of the "not labeled yet" cases
            continue
        if _sha_mismatch(label_sha, output_sha):
            dropped["sha_mismatch"] += 1
            continue
        if other_run:
            foreign[row_run] += 1
            (carried_runs if carried else unverified)[row_run] += 1
        if carried:
            carried_cases.add(case_id)
        labeler = _clean(row.get("labeler")) or None
        voter = labeler.casefold() if labeler else f"#anonymous-{i}"
        if labeler:
            names.setdefault(voter, labeler)
        votes.setdefault(case_id, {})[voter] = _Vote(verdict == "pass", _clean(row.get("note")), labeler)
        if not carried:
            strata.setdefault(case_id, []).append((_clean(row.get("stratum")) or None, _clean(row.get("stratified_on")) or None))
    return _Collected(votes, names, dropped, strata, foreign, frozenset(carried_cases), carried_runs, unverified)


def _consensus(case_votes: dict[str, _Vote]) -> bool | None:
    """Majority label of a case; None on a tie."""
    passes = sum(v.is_pass for v in case_votes.values())
    fails = len(case_votes) - passes
    return None if passes == fails else passes > fails


def _joined_notes(case_votes: dict[str, _Vote]) -> str:
    noted = [v for v in case_votes.values() if v.note]
    if len(case_votes) == 1:
        return noted[0].note if noted else ""
    return " | ".join(f"{v.labeler or 'anonymous'}: {v.note}" for v in noted)


def _inter_annotator(votes: dict[str, dict[str, _Vote]], names: dict[str, str]) -> dict | None:
    """Human-human kappa for the named labelers who overlap most (>= MIN_OVERLAP_FOR_IAA cases).

    It is the ceiling for any judge: a judge cannot agree with humans more than
    they agree with each other. With 3+ labelers the best-overlapping pair is reported.
    """
    best: tuple[int, str, str, list[str]] | None = None
    for a, b in combinations(sorted(names), 2):
        shared = [cid for cid, v in votes.items() if a in v and b in v]
        if len(shared) >= MIN_OVERLAP_FOR_IAA and (best is None or len(shared) > best[0]):
            best = (len(shared), a, b, shared)
    if best is None:
        return None
    n, a, b, shared = best
    kappa = _kappa([votes[c][a].is_pass for c in shared], [votes[c][b].is_pass for c in shared])
    return {"labelers": [names[a], names[b]], "n": n, "kappa": kappa}


# ------------------------------------------------------------------- decision helpers
def kappa_decision(
    n: int,
    kappa: float | None,
    kappa_ci: Sequence[float] | None,
    tnr_reliable: bool = True,
    min_labels_release: int | None = None,
) -> str:
    """Can this judge gate? Decided on the CI of the WEIGHTED kappa (the whole run's kappa), never on a point estimate.

    ``insufficient_data`` (fewer than MIN_LABELS labeled cases, or kappa undefined) wins over
    everything: a tiny sample can produce a lucky CI. A CI whose upper bound is < 0.6 is
    ``not_trustworthy`` at any n >= MIN_LABELS. A lower bound >= 0.6 releases the judge
    (``usable``) only with at least ``min_labels_release`` labeled cases (default MIN_LABELS_RELEASE,
    chosen by simulation: with fewer, judges of true kappa ~0.6 were released more than 10% of the
    time); below that it stays ``inconclusive``. A lower bound >= 0.8 is ``trustworthy`` only when the
    true-negative rate is also reliable (``tnr_reliable``: at least ``MIN_HUMAN_FAILS`` human fails, a TNR of at
    least ``TNR_MIN_RELIABLE`` and a narrow enough interval); otherwise it stays ``usable``. Anything that straddles 0.6 is ``inconclusive``.
    """
    if n < MIN_LABELS or kappa is None:
        return "insufficient_data"
    if not kappa_ci:
        return "inconclusive"
    lower, upper = kappa_ci
    if upper < KAPPA_USABLE:
        return "not_trustworthy"
    if lower < KAPPA_USABLE:
        return "inconclusive"
    if n < (MIN_LABELS_RELEASE if min_labels_release is None else min_labels_release):
        return "inconclusive"
    if lower >= KAPPA_TRUSTWORTHY and tnr_reliable:
        return "trustworthy"
    return "usable"


def _labels_needed(n: int, kappa_ci: Sequence[float] | None, cap: int | None = None) -> int | None:
    """Approximate total labels for a kappa CI half-width <= 0.1 (the CI shrinks like 1/sqrt(n)).

    Never below MIN_LABELS_RELEASE (fewer cannot release a judge) and never above ``cap``, the number
    of cases in the run that can be labeled at all.
    """
    if not kappa_ci or n < 2:
        return None
    half = (kappa_ci[1] - kappa_ci[0]) / 2
    needed = n if half <= KAPPA_TARGET_HALF_WIDTH else math.ceil(round(n * (half / KAPPA_TARGET_HALF_WIDTH) ** 2, 6))
    needed = int(min(max(needed, MIN_LABELS_RELEASE), MAX_LABELS_NEEDED))
    return min(needed, cap) if cap else needed


def _pick_design(rows: Sequence[tuple[str | None, str | None]]) -> tuple[str | None, bool]:
    """(stratified_on recorded by the labeling tool, whether the labels disagree on it)."""
    seen = Counter(so for _, so in rows if so)
    if not seen:
        return None, False
    return seen.most_common(1)[0][0], len(seen) > 1


# ---------------------------------------------------------------------- calibrate
def calibrate(
    run: RunResult,
    labels: Sequence[dict] | str | os.PathLike | Sequence,
    judge: str | None = None,
    seed: int = 0,
    iters: int = BOOTSTRAP_ITERS,
    exclude_case_ids: Iterable[str] | None = None,
) -> dict:
    """Compare a judge's verdict with human labels.

    ``judge=None`` calibrates the overall case verdict, otherwise the named judge's ``passed``;
    cases with outcome ``error`` and cases that judge did not run on have no verdict and are
    skipped. Counters ``n_unmatched``, ``n_excluded_sha_mismatch`` and ``n_invalid_labels`` count
    label rows; ``n_tied_cases`` and ``n_no_judge_verdict`` count cases. ``n`` is the number of
    cases used. Bootstrap intervals are seeded and reproducible. ``exclude_case_ids`` leaves those cases out
    (the ``holdout_case_ids`` of `export_fewshot`: the judge must not be graded on its own few-shot
    examples); the dropped label rows are counted in ``n_excluded_holdout``.

    Weights. TPR, TNR, ``agreement_weighted`` and the corrected pass rate use Horvitz-Thompson
    weights N_h / n_h on the sampling strata the labeling tool recorded (``stratified_on``: the
    overall verdict, or the judge the sample was stratified on). Each label's ``stratum`` is a
    code that is decoded and checked against the run; the weights use the run's own verdicts,
    so a stale or hand-edited label cannot skew them (the token is an HMAC keyed with the run's
    ``.label_secret``; without that file the stratum is recomputed and a warning says so). Labels without ``stratified_on`` (made
    before it existed) are assumed to come from a sample stratified on the overall verdict;
    that is exact for ``judge=None`` and is flagged in ``warnings`` for a named judge. A random
    sample (``stratified_on: "none"``) is post-stratified on the verdict being calibrated.

    ``labels`` is what `load_labels` returns, or a path, or several paths / label lists (they are concatenated in the order
    given; for the same labeler and case the last row wins), so the labels of an earlier run and the labels of this one can be
    calibrated together.

    Carried-over labels. A label row made on ANOTHER run (its ``run_id`` differs) whose ``output_sha`` equals the output this run
    has is carried over: the human judged exactly this output, so the label is valid for this run. Such cases are a fully labeled
    part of the run, counted once each (weight 1). The labels made on this run represent every case without a carried label and
    are weighted N_u,h / n_h on this run's strata. ``labeling.export_labeling_*(previous_labels=...)`` exports exactly those
    cases: the changed outputs and, unless the earlier labels covered every unchanged case (a full earlier pass), the never-labeled
    unchanged ones too. So labeling only the changed outputs is enough only when the carried labels cover every unchanged case;
    if some unchanged cases never had a label and the humans skip them, the new labels stand for cases they were not drawn from
    and the estimate can lean (simulated: TNR 6 to 7 points too high when the changed outputs passed more often than the rest).
    Given the labels of the export, the estimate is exact whatever the earlier sample looked like; pooling every label under one set
    of weights would not be, because the earlier sample was drawn on the earlier run's verdicts (``strata["estimator"]`` says
    which was used: ``"horvitz_thompson"`` with no carried labels, ``"carried_census"`` or ``"pooled"``). The price is precision: with
    few labels made on this run the carried census has wider intervals than pooling would show, and the TPR and TNR intervals can cover
    less than their nominal 95% (simulated coverage of the TPR interval with 20 new labels: from 0.85 to 1.00 over the scenarios, 0.93
    or more with 40), so label more of the cases without a carried label when a decision hangs on an interval (a warning says so
    when fewer than MIN_LABELS labels were made on this run). When a stratum has cases without a carried label but no label made on
    this run, the carried census cannot be completed: all labels are then pooled under one set of weights and a warning says that
    they may be biased and what to do. A carried case that is held out or tied is represented by the remaining carried labels (the
    hold-out is a random deletion of labels). ``strata["uncovered"]`` is, per stratum, the number of scoreable cases without a
    carried label (None without carried labels), ``strata["carried_weights"]`` the weight of a carried label (1 unless some were held
    out). ``n_carried_labels`` / ``n_carried_cases`` / ``carried_run_ids`` count them; the foreign-run counters still count every
    label row of another run.

    The result records ``judges_sha`` (the run's judge-configuration fingerprint) so that `stale_calibration` can tell later
    that the judges changed.
    """
    labels = as_label_list(labels)
    by_case = run.by_case()
    known = sorted({j["judge"] for r in run.results for j in r.get("judges", [])})
    if judge is not None and judge not in known:
        raise ValueError(f"judge {judge!r} did not run in this run; available: {known}")

    exclude = frozenset(str(c) for c in (exclude_case_ids or ()))
    collected = _collect_votes(labels, by_case, run.run_id, exclude)
    votes, names, dropped = collected.votes, collected.names, collected.dropped
    run_dir = getattr(run, "run_dir", None)
    secret = read_label_secret(run_dir)
    warnings: list[str] = []
    foreign_total = sum(collected.foreign_runs.values())
    unverified_total = sum(collected.unverified_runs.values())
    if unverified_total:
        ids = sorted(collected.unverified_runs)
        shown = ", ".join(repr(i) for i in ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
        warnings.append(
            f"{_plural(unverified_total, 'label row')} carry a run_id ({shown}) that is not this run ({run.run_id!r}) and have no output "
            "sha to check: they are matched by case id only. Make sure they were made on this run; otherwise export a labeling file from this "
            "run (`evalsprint label`) and label again."
        )

    # ---- sampling design -> the stratifying verdict
    design, mixed_design = _pick_design([row for rows in collected.strata.values() for row in rows])
    unknown_design = design not in (None, "overall", "none") and design not in known
    if design in (None, "overall") or unknown_design or design == "none" and judge is None:
        stratifier: str | None = None  # the overall verdict
    elif design == "none":
        stratifier = judge  # random sample: post-stratify on the verdict being calibrated
    else:
        stratifier = design
    if mixed_design:
        warnings.append(
            f"The labels were sampled with different designs; weights follow the most common one ({design!r}). "
            "Label from a single `evalsprint label` export for exact weights."
        )
    if unknown_design:
        warnings.append(f"The labels were stratified on judge {design!r}, which is not in this run: strata use the overall verdict.")
    carried_cases = collected.carried_cases
    if design is None and judge is not None and any(cid not in carried_cases for cid in votes):
        warnings.append(
            "The labels do not record how they were sampled (no `stratified_on`): assuming the sample was stratified on the "
            "overall verdict, as `evalsprint label` does without --judge. If it was stratified on this judge or not at all, "
            "label again from a fresh export."
        )
    strat_name = "overall" if stratifier is None else stratifier

    # ---- the run's scoreable cases (the frame the weights and the observed pass rate refer to)
    n_run_errors = sum(1 for r in by_case.values() if record_outcome(r) == "error")
    frame: list[tuple[str, bool]] = []
    frame_ids: list[str] = []
    for cid, r in by_case.items():
        verdict, strat_verdict = _verdict(r, judge), _verdict(r, stratifier)
        if verdict is not None and strat_verdict is not None:
            frame.append((stratum_of(strat_verdict), verdict))
            frame_ids.append(cid)
    run_verdicts = [v for _, v in frame]
    p_obs = sum(run_verdicts) / len(run_verdicts) if run_verdicts else None
    frame_counts = Counter(s for s, _ in frame)
    if n_run_errors:
        warnings.append(
            f"{_plural(n_run_errors, 'run case')} ended in an error (target or judge) and "
            "is excluded from the pass rates and from this calibration."
        )

    # ---- labeled cases with a verdict
    n_tied = n_no_verdict = 0
    used: list[tuple[str, bool, bool]] = []  # (case_id, human pass, judge pass)
    labeled: dict[str, list[tuple[bool, bool]]] = {s: [] for s in STRATA}  # made on this run (or unverifiable): the sampled part
    carried_pairs: dict[str, list[tuple[bool, bool]]] = {s: [] for s in STRATA}  # carried over from an earlier run: a census
    n_recorded = n_undecodable = n_disagree = n_unverifiable = 0
    for cid in by_case:  # run order keeps the output independent of the label file's order
        if cid not in votes:
            continue
        human = _consensus(votes[cid])
        if human is None:
            n_tied += 1
            continue
        record = by_case[cid]
        verdict, strat_verdict = _verdict(record, judge), _verdict(record, stratifier)
        if verdict is None or strat_verdict is None:
            n_no_verdict += 1
            continue
        used.append((cid, human, verdict))
        stratum = stratum_of(strat_verdict)
        (carried_pairs if cid in carried_cases else labeled)[stratum].append((human, verdict))
        recorded = [
            stratum_status(tok, run.run_id, so or design or "overall", cid, secret)
            for tok, so in ([] if cid in carried_cases else collected.strata.get(cid, ()))
            if tok
        ]
        if recorded and all(r[0] is None for r in recorded):
            if all(r[1] == "unverifiable" for r in recorded):
                n_unverifiable += 1
            else:
                n_undecodable += 1
        elif recorded:
            n_recorded += 1
            n_disagree += any(r[0] not in (None, stratum) for r in recorded)
    pairs = [(h, j) for _, h, j in used]
    if n_undecodable or n_disagree:
        warnings.append(
            f"{n_undecodable + n_disagree} label(s) record a stratum that does not match this run "
            "(labels from another run, or the run was re-judged): weights use this run's own verdicts."
        )
    if n_unverifiable:
        why = (
            f"{label_secret_path(run_dir)} is missing or unreadable"
            if run_dir is not None
            else "this run has no folder on disk, so there is no secret to read"
        )
        warnings.append(
            f"{_plural(n_unverifiable, 'label')} record a keyed stratum token that cannot be verified: the run's secret "
            f"({LABEL_SECRET_FILE}) is not available ({why}). The strata are recomputed from this run's own verdicts, which is "
            "exact for this run, but the check that catches labels from another run is off. Keep the file next to the run "
            "(do not delete it, do not share or commit it)."
        )

    n = len(used)
    tp = sum(1 for h, j in pairs if h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    fp = sum(1 for h, j in pairs if not h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    kappa = _kappa([h for h, _ in pairs], [j for _, j in pairs])

    # ---- which labels stand for which cases (carried-over labels make the labeled part of the run a census)
    n_carried_used = sum(len(v) for v in carried_pairs.values())
    census: _Census | None = None
    design_counts = frame_counts  # the cases N_h counts: the whole run, or only those without a carried label
    estimator = "horvitz_thompson"
    uncovered: dict[str, int] | None = None
    carried_weights: dict[str, float] = {}
    if n_carried_used:
        used_ids = {cid for cid, _, _ in used}
        marks = [(1 if cid in used_ids else 2) if cid in carried_cases else 0 for cid in frame_ids]
        # A stratum whose carried labels were all held out has none left to stand for its carried cases: they go with the cases
        # that have no carried label.
        has_label = {st for (st, _), m in zip(frame, marks) if m == 1}
        marks = [0 if m == 2 and st not in has_label else m for (st, _), m in zip(frame, marks)]
        uncovered_counts = Counter(st for (st, _), m in zip(frame, marks) if m == 0)
        uncovered = {s: uncovered_counts.get(s, 0) for s in STRATA}
        gaps = sorted(s for s in STRATA if uncovered[s] and not labeled[s])
        n_carried_rows = sum(collected.carried_runs.values())
        runs_shown = ", ".join(repr(i) for i in sorted(collected.carried_runs)[:3]) + (
            f" and {len(collected.carried_runs) - 3} more" if len(collected.carried_runs) > 3 else ""
        )
        if not gaps:
            estimator = "carried_census"
            sizes = Counter(st for (st, _), m in zip(frame, marks) if m in (1, 2))
            census = _Census(marks, carried_pairs, {s: sizes.get(s, 0) for s in STRATA})
            design_counts = uncovered_counts
            carried_weights = {s: sizes[s] / len(carried_pairs[s]) for s in STRATA if carried_pairs[s]}
            n_sampled = sum(len(v) for v in labeled.values())
            n_other = sum(uncovered.values())
            n_left_out = sum(1 for m in marks if m == 2)
            if n_other == 0:
                rest = "every scoreable case of the run has one"
            elif n_sampled >= n_other:
                rest = f"every other case ({n_other}) was labeled on this run"
            else:
                rest = f"the {_plural(n_other, 'other case')} (no carried label) are represented by the {_plural(n_sampled, 'label')} made on this run"
            left_out = (
                f" {_plural(n_left_out, 'carried case')} held out or tied {'is' if n_left_out == 1 else 'are'} represented by the other carried labels."
                if n_left_out
                else ""
            )
            few_new = (
                f" Only {_plural(n_sampled, 'label')} made on this run stand for them, so the intervals for TPR and TNR are wide and can cover "
                "less than their 95%: label more of the cases that have no carried label if a decision depends on them."
                if n_other > n_sampled and n_sampled < MIN_LABELS
                else ""
            )
            warnings.append(
                f"{_plural(n_carried_rows, 'label row')} ({_plural(n_carried_used, 'case')}) carried over from {runs_shown}: the output is "
                f"unchanged since they were made, so they are kept and count as a fully labeled part of this run; {rest}.{left_out}{few_new}"
            )
        else:
            estimator = "pooled"
            for st in STRATA:
                labeled[st] = carried_pairs[st] + labeled[st]
            warnings.append(
                f"{_plural(n_carried_used, 'case')} carry labels made on {runs_shown}, but no label made on this run stands for the "
                f"{' and '.join(gaps)} cases that have none: all labels are pooled under one set of weights. {_POOLED_NOTE} "
                "Export the cases that have no label yet (`evalsprint label` with the earlier labels as previous labels), label a sample "
                "of them and calibrate with both label files."
            )
    labeled_by_stratum = {s: len(labeled[s]) + (len(carried_pairs[s]) if estimator == "carried_census" else 0) for s in STRATA}

    # ---- Horvitz-Thompson estimates and their bootstrap intervals
    tpr = tnr = corrected = agreement_w = kappa_w = human_pass_w = None
    cells: dict[str, float] | None = None
    cis: dict[str, list[float] | None] = {
        k: None for k in ("tpr", "tnr", "agreement", "kappa", "corrected_pass_rate")
    }
    weights: dict[str, float] = {}
    if n and p_obs is not None:
        cells = _weighted_cells(labeled, design_counts, census)
        if cells is None:
            missing = sorted(s for s, size in design_counts.items() if size and not labeled.get(s))
            warnings.append(
                f"No labeled case in the {' and '.join(missing)} stratum, so TPR, TNR and the corrected pass rate "
                "cannot be estimated: label some cases from every stratum (the stratified labeling sample does this)."
            )
        else:
            weights = {s: design_counts[s] / len(labeled[s]) for s in STRATA if design_counts.get(s) and labeled[s]}
            m = _ht_metrics(cells)
            tpr, tnr, agreement_w, kappa_w, human_pass_w = m["tpr"], m["tnr"], m["agreement"], m["kappa"], m["human_pass"]
            if tpr is not None and tnr is not None:
                corrected = rogan_gladen(p_obs, tpr, tnr)
                if corrected is None:
                    warnings.append("The judge is no better than chance (TPR + TNR <= 1): the corrected pass rate is undefined.")
            cis = _bootstrap(frame, labeled, iters, seed + 1, census=census)
            for key, point in (("tpr", tpr), ("tnr", tnr), ("agreement", agreement_w), ("kappa", kappa_w), ("corrected_pass_rate", corrected)):
                if point is None:  # an interval around an undefined estimate would mislead
                    cis[key] = None

    human_fails = fp + tn
    n_labelable = len(frame)  # non-error cases the judge scored, one per case id: the most labels there can ever be
    n_disagree_total = fp + fn
    # Kappa intervals. The plain percentile bootstrap only ever sees a handful of disagreements when there are few (about
    # Poisson(d) per resample), so its lower bound is too optimistic and it collapses to [1, 1] at d = 0: a judge with 1
    # disagreement in 40 was released far more often than one with 0. The reported kappa intervals therefore use a
    # smoothed bootstrap (pseudo-counts on every confusion cell the run allows): monotone in the number of disagreements
    # and equal to the plain one as labels grow. The plain intervals stay in *_bootstrap for comparison.
    judge_verdicts = sorted({v for _, v in frame}) or [True, False]
    kappa_ci_bootstrap = _kappa_ci(pairs, iters, seed) if kappa is not None else None
    kappa_ci = (
        _kappa_ci(pairs, iters, seed, KAPPA_CI_SMOOTHING, judge_verdicts) if kappa is not None and KAPPA_CI_SMOOTHING > 0 else kappa_ci_bootstrap
    )
    kappa_w_ci_bootstrap = cis["kappa"]
    tnr_ci_bootstrap = cis["tnr"]
    kappa_w_ci = kappa_w_ci_bootstrap
    if kappa_w is not None and cells is not None and KAPPA_CI_SMOOTHING > 0:
        kappa_w_ci = _bootstrap(frame, labeled, iters, seed + 1, smooth=KAPPA_CI_SMOOTHING, census=census)["kappa"]
    cis["kappa"] = kappa_w_ci
    # The TNR interval is the closed-form Wilson interval at an effective n (see `_tnr_interval`), not a bootstrap: the plain
    # bootstrap collapses to a point where a good judge is (0 or 1 observed misses), and `tnr_reliable` has to be a
    # deterministic function of the counts to be monotone in judge quality. The plain bootstrap stays in `tnr_ci_bootstrap`.
    tnr_ci = _tnr_interval(frame, labeled, KAPPA_CI_SMOOTHING, census=census) if tnr is not None and cells is not None else None
    cis["tnr"] = tnr_ci
    tnr_reliable, tnr_reason = _tnr_reliability(tnr, tnr_ci, human_fails)
    strata_gap = sorted(s for s, size in design_counts.items() if size and not labeled.get(s)) if n and cells is None else []

    decision = kappa_decision(n, kappa_w, kappa_w_ci, tnr_reliable=tnr_reliable)
    qualified = kappa_decision(n, kappa_w, kappa_w_ci, tnr_reliable=True, min_labels_release=0)
    held_by = None  # why a judge whose interval qualifies was not released / not called trustworthy
    if qualified in ("usable", "trustworthy") and n < MIN_LABELS_RELEASE:
        held_by = "min_labels"
    elif qualified == "trustworthy" and decision == "usable":
        held_by = "tnr"
    labels_needed_uncapped = _labels_needed(n, kappa_w_ci)
    labels_needed = _labels_needed(n, kappa_w_ci, cap=n_labelable)
    labels_needed_capped = bool(labels_needed_uncapped and labels_needed is not None and labels_needed < labels_needed_uncapped)

    one_sided = None  # "pass" / "fail": every labeled case is one the judge passed / failed
    if pairs and len({j for _, j in pairs}) == 1:
        one_sided = "pass" if pairs[0][1] else "fail"
    run_constant = bool(frame) and len({v for _, v in frame}) == 1  # the judge gave one verdict on the whole run
    if one_sided:
        warnings.append(_one_sided_message(one_sided, n, run_constant, n_disagree_total))
    elif n and not tnr_reliable and not strata_gap:
        tnr_message = _tnr_message(tnr_reason, human_fails, tnr, tnr_ci)
        if tnr_message:
            warnings.append(tnr_message)
    if n and n < MIN_LABELS:
        warnings.append(f"Only {_plural(n, 'case')} labeled: every metric here is provisional (aim for 50-100 with failures included).")
    if n >= MIN_LABELS and n_disagree_total == 0 and kappa is not None:
        warnings.append(
            f"No disagreement in {_plural(n, 'label')}: the TPR and TNR intervals are degenerate and understate the uncertainty "
            f"(rule of three: the true disagreement rate could still be about {3 / n:.0%}); the kappa interval is smoothed "
            f"(pseudo-counts on every possible outcome), so it stays wide."
        )
    if held_by == "min_labels":
        warnings.append(
            f"The weighted kappa interval qualifies as {qualified}, but only {_plural(n, 'case')} are labeled: at least "
            f"{MIN_LABELS_RELEASE} are needed before a judge is released (with fewer, weak judges get released too often)."
        )
    elif held_by == "tnr":
        warnings.append(
            f"The weighted kappa interval is at or above {KAPPA_TRUSTWORTHY:.2f}, but the true-negative rate did not pass the "
            "screening check (see above), so the judge stays 'usable' and is not called 'trustworthy'."
        )
    if n and n_labelable < MIN_LABELS_RELEASE:
        warnings.append(
            f"Only {_plural(n_labelable, 'case')} of this run can be labeled (errors excluded), fewer than the "
            f"{MIN_LABELS_RELEASE} needed to release a judge: add cases or repeats."
        )
    for count, text in (
        (dropped["sha_mismatch"], "excluded because the AI output changed after labeling"),
        (dropped["unmatched"], "have an unknown case id"),
        (getattr(labels, "n_invalid", 0) + dropped["invalid"], "were unreadable"),
        (n_tied, "labeled case(s) skipped because the labelers tied"),
        (n_no_verdict, "labeled case(s) skipped because the judge gave no verdict (error or not applicable)"),
    ):
        if count:
            warnings.append(f"{count} label(s) {text}." if "case(s)" not in text else f"{count} {text}.")

    disagreements = [
        {
            "case_id": cid,
            "human": "pass" if human else "fail",
            "judge": "pass" if verdict else "fail",
            "judge_reasoning": _judge_reasoning(by_case[cid], judge),
            "note": _joined_notes(votes[cid]),
            "input": _truncate(_as_text(by_case[cid]["case"].get("input"))),
            "output": _truncate(_as_text(by_case[cid]["output"].get("text"))),
        }
        for cid, human, verdict in used
        if human != verdict
    ]
    disagreements.sort(key=lambda d: d["human"] != "fail")  # judge lets a bad output through: first

    inter_annotator = _inter_annotator(votes, names)
    return {
        "judge": judge or "overall",
        "run_id": run.run_id,
        "n": n,
        "n_labels": len(labels),
        "n_invalid_labels": getattr(labels, "n_invalid", 0) + dropped["invalid"],
        "n_excluded_sha_mismatch": dropped["sha_mismatch"],
        "n_unmatched": dropped["unmatched"],
        "n_excluded_holdout": dropped["excluded"],
        "n_foreign_run_labels": foreign_total,
        "foreign_run_ids": sorted(collected.foreign_runs),
        "n_carried_labels": sum(collected.carried_runs.values()),
        "n_carried_cases": n_carried_used,
        "carried_run_ids": sorted(collected.carried_runs),
        "judges_sha": (run.summary or {}).get("judges_sha"),
        "n_tied_cases": n_tied,
        "n_no_judge_verdict": n_no_verdict,
        "n_run_errors": n_run_errors,
        "agreement": (tp + tn) / n if n else None,
        "agreement_weighted": agreement_w,
        "agreement_weighted_ci": cis["agreement"],
        "kappa": kappa,
        "kappa_label": kappa_label(kappa),
        "kappa_ci": kappa_ci,
        "kappa_ci_bootstrap": kappa_ci_bootstrap,
        "kappa_weighted": kappa_w,
        "kappa_weighted_ci": cis["kappa"],
        "kappa_weighted_ci_bootstrap": kappa_w_ci_bootstrap,
        "kappa_ci_smoothing": KAPPA_CI_SMOOTHING,
        "kappa_decision": decision,
        "kappa_decision_basis": "weighted",
        "min_labels_release": MIN_LABELS_RELEASE,
        "n_labelable": n_labelable,
        "labels_needed_estimate": labels_needed,
        "labels_needed_capped": labels_needed_capped,
        "tpr": tpr,
        "tnr": tnr,
        "tpr_ci": cis["tpr"],
        "tnr_ci": cis["tnr"],
        "tnr_ci_bootstrap": tnr_ci_bootstrap,
        "tnr_reliable": tnr_reliable,
        "tnr_reliable_reason": tnr_reason,
        "tnr_reliable_rule": {
            "min_human_fails": MIN_HUMAN_FAILS,
            "max_half_width": TNR_MAX_HALF_WIDTH,
            "min_tnr": TNR_MIN_RELIABLE,
            "interval": "Wilson at the delta-method effective n",
            "kind": "screening flag",
            "meaning": TNR_SCREEN_MEANING,
        },
        "one_sided_labels": one_sided,
        "confusion": {
            "human_pass_judge_pass": tp,
            "human_pass_judge_fail": fn,
            "human_fail_judge_pass": fp,
            "human_fail_judge_fail": tn,
        },
        "human_pass_rate": (tp + fn) / n if n else None,
        "human_pass_rate_weighted": human_pass_w,
        "judge_pass_rate_labeled": (tp + fp) / n if n else None,
        "run_observed_pass_rate": p_obs,
        "corrected_pass_rate": corrected,
        "corrected_pass_rate_ci": cis["corrected_pass_rate"],
        "corrected_pass_rate_ci_note": CORRECTED_CI_NOTE,
        "strata": {
            "design": design,
            "stratified_on": strat_name,
            "source": "recorded"
            if n_recorded and not (n_undecodable or n_disagree or n_unverifiable) and n_recorded == n - n_carried_used
            else "recomputed",
            "estimator": estimator,
            "n_unverifiable_tokens": n_unverifiable,
            "run_cases": {s: frame_counts.get(s, 0) for s in STRATA},
            "labeled": labeled_by_stratum,
            "carried": {s: len(carried_pairs[s]) for s in STRATA},
            "uncovered": uncovered,
            "weights": weights,
            "carried_weights": carried_weights,
        },
        "inter_annotator": inter_annotator,
        "labeled_case_ids": [cid for cid, _, _ in used],
        "disagreements": disagreements,
        "warnings": warnings,
        "recommendation": _recommend(
            n=n,
            n_human_fail=human_fails,
            n_missed_bad=fp,
            tnr_reliable=tnr_reliable,
            tnr_reason=tnr_reason,
            tnr=tnr,
            tnr_ci=tnr_ci,
            one_sided=one_sided,
            run_constant=run_constant,
            n_disagree=n_disagree_total,
            kappa=kappa_w,
            kappa_ci=kappa_w_ci,
            decision=decision,
            labels_needed=labels_needed,
            labels_needed_capped=labels_needed_capped,
            n_labelable=n_labelable,
            held_by=held_by,
            strata_gap=strata_gap,
            kappa_defined_on_sample=kappa is not None,
            n_sha_mismatch=dropped["sha_mismatch"],
            inter_annotator=inter_annotator,
        ),
    }


STALE_CALIBRATION_TEXT = "calibration was made for a different judge configuration"


def _judges_sha_of(source) -> str | None:
    """A ``judges_sha`` from a string, a dict (calibration.json or summary.json) or a run (its ``summary``)."""
    if source is None:
        return None
    if isinstance(source, str):
        return source or None
    if isinstance(source, dict):
        value = source.get("judges_sha")
    else:
        value = (getattr(source, "summary", None) or {}).get("judges_sha")
    return value if isinstance(value, str) and value else None


def stale_calibration(cal, run) -> str | None:
    """The warning to show when ``cal`` was made for other judges than ``run`` has, else None.

    ``cal`` is what `calibrate` returned (or its calibration.json); ``run`` is a run, its summary or its ``judges_sha``. The two
    ``judges_sha`` fingerprints (judge config, rubrics and custom-judge code) must be equal; when they differ the agreement, kappa and
    TPR/TNR in ``cal`` describe judges that are no longer the ones that scored ``run``. None when they match and also when either
    side has no ``judges_sha`` (a calibration made by an older evalsprint, a run with no judges): nothing can be said then.
    It compares the whole configuration, so changing the rubric of another judge flags a calibration of this one too.
    """
    made_for, current = _judges_sha_of(cal), _judges_sha_of(run)
    if not made_for or not current or made_for == current:
        return None
    return (
        f"This {STALE_CALIBRATION_TEXT} (judges_sha {made_for} in the calibration, {current} in this run): the agreement, kappa and "
        "TPR/TNR it reports describe the earlier judges and do not say how well the current ones match the humans. Calibrate this run "
        "again; the human labels of outputs that have not changed still count."
    )


# ------------------------------------------------------------------------- labels budget
# How many labeled cases it takes before a judge can be released, by how often humans pass the output. Measured in the cycle-4
# simulations with the real `labeling.select_sample` and `calibrate`, with the number of human fails in a run drawn at random
# (Binomial), not fixed (see `labels_budget_table`).
_LABELS_BUDGET: dict = {'judge': 'perfect judge (agrees with the humans on every case): the best case, with the number of human fails drawn at random (Binomial over the run); plus, '
          'for human pass rates 70%, 80%, 90% and 95%, a strong judge (95% TPR and 95% TNR)',
 'python': '3.11.15',
 'reps': {'census': 'exact over the Binomial number of human fails (no draws): for each run size the range of fails at which the outcome holds, by bisection '
                    'with the real calibrate (2000 bootstrap iterations); run sizes 20 to 440 (every 5, every 1 near each figure); bootstrap seeds 0, 1, 2 '
                    '(seed 0 gives the figure)',
          'subset_perfect_judge': 'exact over the Binomial number of human fails: for each number of fails the fewest labels, with the real select_sample and '
                                  'calibrate; bootstrap seeds 0, 1, 2; re-measured with direct draws (human truth drawn per case): 100 draws per cell in 47 '
                                  'cells, 137 of 141 exact probabilities inside the 95% Wilson interval of the draws',
          'subset_strong_judge': '200 draws per cell'},
 'source': 'orchestration/2026-09-29_kit-v0.2/cycle4/loopback1/eval-scientist-judge/ (budget_enum.py, budget_exact.py, budget_direct.py, budget_final.py; '
           'budget_exact.json, budget_validate.json, budget_strong.json; real labeling.select_sample and calibrate, fixed seeds)',
 'how_to_read': 'Best case: a judge that agrees with the humans on every case; a real judge makes errors and needs more labels, so read these as a floor, not '
                'a promise. The number of human fails in a run is random (Binomial: a 200-case run at 90% human pass has 20 fails on average and anywhere from '
                'about 8 to 35), and fewer fails need more labels, so each figure is the smallest number for which at least 80% of the runs reach the outcome, '
                "not the number for an average run. 'Every case labeled' is the smallest run size, every case labeled, from which at least 80% of the runs "
                'reach the outcome at that and every larger size scanned (usable and trustworthy are the kappa decision, the third figure is the TNR screening '
                "check; the bootstrap seed can move a figure by up to about 10 cases, see 'spread'). 'Stratified sample' is a run of 200 or 400 cases of which "
                "only some are labeled with the labeling tool's default sample (stratified on the judge's verdict): the fewest labels for which at least 80% "
                "of the runs reach the outcome, rounded up to a multiple of 5 from the largest of three bootstrap seeds; 'more cases' means that not even "
                'labeling every case of the run reaches 80%. The higher the share of outputs the humans pass, the more labels it takes, because the decision '
                'needs enough human fails (at least 10 for the true-negative rate). In the strong-judge table (95% TPR and 95% TNR) the figure is the fewest '
                'labels on a grid of label counts (60, 100, 150 on 200 cases; 60 to 300 on 400) from which at least 80% of the draws reach the outcome at that '
                "and every larger count, and 'not reached' means that no count simulated gets there; those are 200 draws per cell (a 95% interval of about +-3 "
                'to +-7 points on each probability), not an exact enumeration. Fixed seeds; not a guarantee for a particular judge.',
 'rows': [{'human_pass_rate': 0.5,
           'census': {'usable': 40, 'trustworthy': 40, 'tnr_reliable': 24},
           'subset': {'200': {'usable': 40, 'trustworthy': 40, 'tnr_reliable': 20}, '400': {'usable': 40, 'trustworthy': 40, 'tnr_reliable': 20}},
           'strong_judge': {},
           'spread': {'census': {'usable': [40, 40], 'trustworthy': [40, 40], 'tnr_reliable': [24, 24]},
                      'subset': {'200': {'usable': [40, 40], 'trustworthy': [40, 40], 'tnr_reliable': [20, 20]},
                                 '400': {'usable': [40, 40], 'trustworthy': [40, 40], 'tnr_reliable': [20, 20]}}}},
          {'human_pass_rate': 0.7,
           'census': {'usable': 40, 'trustworthy': 48, 'tnr_reliable': 40},
           'subset': {'200': {'usable': 40, 'trustworthy': 50, 'tnr_reliable': 50}, '400': {'usable': 40, 'trustworthy': 50, 'tnr_reliable': 50}},
           'strong_judge': {'200': {'usable': 60, 'trustworthy': None, 'tnr_reliable': 60}, '400': {'usable': 60, 'trustworthy': None, 'tnr_reliable': 60}},
           'spread': {'census': {'usable': [40, 40], 'trustworthy': [45, 48], 'tnr_reliable': [40, 40]},
                      'subset': {'200': {'usable': [40, 40], 'trustworthy': [48, 48], 'tnr_reliable': [48, 48]},
                                 '400': {'usable': [40, 40], 'trustworthy': [46, 46], 'tnr_reliable': [46, 46]}}}},
          {'human_pass_rate': 0.8,
           'census': {'usable': 40, 'trustworthy': 67, 'tnr_reliable': 61},
           'subset': {'200': {'usable': 40, 'trustworthy': 80, 'tnr_reliable': 80}, '400': {'usable': 40, 'trustworthy': 80, 'tnr_reliable': 80}},
           'strong_judge': {'200': {'usable': 100, 'trustworthy': None, 'tnr_reliable': 100}, '400': {'usable': 100, 'trustworthy': None, 'tnr_reliable': 100}},
           'spread': {'census': {'usable': [40, 40], 'trustworthy': [67, 67], 'tnr_reliable': [61, 61]},
                      'subset': {'200': {'usable': [40, 40], 'trustworthy': [77, 77], 'tnr_reliable': [77, 77]},
                                 '400': {'usable': [40, 40], 'trustworthy': [80, 80], 'tnr_reliable': [80, 80]}}}},
          {'human_pass_rate': 0.9,
           'census': {'usable': 66, 'trustworthy': 124, 'tnr_reliable': 124},
           'subset': {'200': {'usable': 45, 'trustworthy': 120, 'tnr_reliable': 120}, '400': {'usable': 50, 'trustworthy': 130, 'tnr_reliable': 130}},
           'strong_judge': {'200': {'usable': None, 'trustworthy': None, 'tnr_reliable': 150},
                            '400': {'usable': None, 'trustworthy': None, 'tnr_reliable': 200}},
           'spread': {'census': {'usable': [66, 76], 'trustworthy': [124, 134], 'tnr_reliable': [124, 124]},
                      'subset': {'200': {'usable': [44, 45], 'trustworthy': [118, 118], 'tnr_reliable': [118, 118]},
                                 '400': {'usable': [46, 48], 'trustworthy': [128, 128], 'tnr_reliable': [128, 128]}}}},
          {'human_pass_rate': 0.95,
           'census': {'usable': 134, 'trustworthy': 249, 'tnr_reliable': 249},
           'subset': {'200': {'usable': 100, 'trustworthy': None, 'tnr_reliable': None}, '400': {'usable': 75, 'trustworthy': 230, 'tnr_reliable': 230}},
           'strong_judge': {'200': {'usable': None, 'trustworthy': None, 'tnr_reliable': None},
                            '400': {'usable': None, 'trustworthy': None, 'tnr_reliable': 300}},
           'spread': {'census': {'usable': [134, 135], 'trustworthy': [249, 249], 'tnr_reliable': [249, 249]},
                      'subset': {'200': {'usable': [85, 97], 'trustworthy': [None, None], 'tnr_reliable': [None, None]},
                                 '400': {'usable': [71, 71], 'trustworthy': [229, 229], 'tnr_reliable': [229, 229]}}}}]}


def labels_budget_table() -> dict:
    """How many human labels it takes to release a judge, by the share of outputs the humans pass (a copy, safe to change).

    ``rows`` has one entry per human pass rate (ascending: the fewer outputs humans fail, the more labels it takes to see enough of
    them). The number of human fails in a run is random (Binomial over the run's cases), and fewer fails need more labels, so every
    figure is the smallest number for which at least 80% of the runs reach the outcome (not the number for an average run).
    Each row has ``census`` (every case of a run is labeled: the first number of cases at which a judge that agrees with the humans
    on every case is ``usable`` / ``trustworthy`` and passes the ``tnr_reliable`` screen, and stays so for larger runs) and ``subset``
    (a run of 200 / 400 cases of which only some are labeled with the labeling tool's stratified sample: the fewest labels at which
    at least 80% of the runs reach it, rounded up to a multiple of 5; None when labeling every case is not enough). ``strong_judge``
    (only some rows) repeats the ``subset`` figures for a judge with 95% TPR and TNR, on a grid of label counts, where None means
    "not reached by 80% of the draws at any label count simulated". ``spread`` is how far the bootstrap seed moves each figure (the
    smallest and largest over three seeds; None when a seed never reaches it). The ``census`` and ``subset`` figures are BEST-CASE
    figures for a judge that never errs: a judge with errors needs more, and the figures are not a promise for a particular judge.
    ``how_to_read``, ``reps`` and ``source`` say what was measured and where the raw numbers are. `labels_budget_markdown` renders the
    table for docs.
    """
    return copy.deepcopy(_LABELS_BUDGET)


def labels_budget_hint(human_pass_rate: float | None) -> dict | None:
    """The `labels_budget_table` row for a human pass rate: the first row at or above it (None when the rate is unknown).

    Rounding up errs on the side of more labels, because the rarer human fails are, the more labels it takes.
    """
    rows = _LABELS_BUDGET.get("rows") or []
    if human_pass_rate is None or not rows:
        return None
    for row in rows:
        if human_pass_rate <= row["human_pass_rate"] + 1e-9:
            return copy.deepcopy(row)
    return copy.deepcopy(rows[-1])


def labels_budget_markdown() -> str:
    """The labels budget as Markdown tables (what the docs render): the best case (a judge that never errs) and, where measured, a strong judge."""
    table = labels_budget_table()

    def triple(sub, none_text: str) -> str:
        if sub is None:
            return "n/a"
        return " / ".join(none_text if sub[key] is None else str(sub[key]) for key in ("usable", "trustworthy", "tnr_reliable"))

    sizes = sorted({size for row in table["rows"] for size in row["subset"]}, key=int)
    head = ["Human pass rate", "Every case labeled (usable / trustworthy / TNR screen)"] + [f"{size}-case run, stratified sample" for size in sizes]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in table["rows"]:
        c = row["census"]
        cells = [f"{row['human_pass_rate']:.0%}", f"{c['usable']} / {c['trustworthy']} / {c['tnr_reliable']}"]
        cells += [triple(row["subset"].get(size), "more cases") for size in sizes]
        lines.append("| " + " | ".join(cells) + " |")
    strong = [row for row in table["rows"] if row.get("strong_judge")]
    if strong:
        ssizes = sorted({size for row in strong for size in row["strong_judge"]}, key=int)
        lines += ["", "A strong but imperfect judge (95% TPR and TNR), stratified sample:", ""]
        head = ["Human pass rate"] + [f"{size}-case run" for size in ssizes]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for row in strong:
            lines.append("| " + " | ".join([f"{row['human_pass_rate']:.0%}"] + [triple(row["strong_judge"].get(size), "not reached") for size in ssizes]) + " |")
    return "\n".join(lines) + "\n\n_" + table["how_to_read"] + "_\n"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _fmt_kappa(kappa: float, ci: Sequence[float] | None) -> str:
    text = f"{kappa:.2f} ({kappa_label(kappa)}"
    return text + (f", 95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else ")")


def _tnr_reliability(tnr: float | None, ci: Sequence[float] | None, human_fails: int) -> tuple[bool, str]:
    """(reliable, reason) for the true-negative rate; reason is "ok" or what is missing.

    A SCREENING flag, not a guarantee (`TNR_SCREEN_MEANING`): "reliable" means narrowly measured and a point estimate of at least
    TNR_MIN_RELIABLE, not proof that the true rate is. It is the selection on the estimate that lets a judge whose true rate is a
    little lower read "reliable" now and then; the simulations in the cycle-3 and cycle-4 evidence put a number on it.

    Reliable = at least MIN_HUMAN_FAILS human fails, an interval (`_tnr_interval`) no wider than +/- TNR_MAX_HALF_WIDTH
    (it folds in how many labeled judge-passes the estimate rests on and the sampling weights) and a TNR of at least
    TNR_MIN_RELIABLE. The last clause is what makes the flag monotone in judge quality, "a judge with fewer misses is never
    less reliable": the half-width of an interval for a proportion is largest near one half, so at low TNR a judge
    that catches more bad outputs can have a WIDER interval; from 0.75 up it only narrows (checked exhaustively on
    thousands of designs, see the tests and the cycle-3 scan; the same scan finds violations for floors of 0.5 to 0.6).
    The flag says how well the TNR is measured, and a judge below the floor is reported as ``low_tnr``,
    not as merely imprecise: it is a weak judge, whatever the interval says.
    """
    if human_fails < MIN_HUMAN_FAILS:
        return False, "few_human_fails"
    if tnr is None or not ci:
        return False, "undefined"
    if tnr < TNR_MIN_RELIABLE:
        return False, "low_tnr"
    if (ci[1] - ci[0]) / 2 > TNR_MAX_HALF_WIDTH:
        return False, "wide_ci"
    return True, "ok"


def _tnr_message(reason: str, n_human_fail: int, tnr: float | None = None, ci: Sequence[float] | None = None) -> str:
    """Why the true-negative rate did not pass the screening check and what to do about it ("" when there is nothing to say)."""
    head = "The true-negative rate (how often the judge catches bad outputs) did not pass the screening check: "
    where = f" ({_pct(tnr)}, 95% CI {_pct(ci[0])} to {_pct(ci[1])})" if tnr is not None and ci else ""
    if reason == "few_human_fails":
        return (
            head
            + f"only {_plural(n_human_fail, 'case')} labeled fail by humans (at least {MIN_HUMAN_FAILS} are needed). "
            "Label more cases: the ones the judge failed (most bad outputs are there) and the ones it passed (the bad outputs "
            "it missed hide there). The human fails are the only evidence the TNR and its interval rest on, so how many labels "
            "this takes depends on how often humans fail an output (see the labels budget, `calibration.labels_budget_table()`)."
        )
    if reason == "wide_ci":
        return (
            head
            + f"the estimate{where} is too imprecise, its interval must be within +/-{TNR_MAX_HALF_WIDTH * 100:.0f} points. "
            "Label more cases, especially ones the judge passed: each one narrows the interval, and the bad outputs it "
            "missed (or the lack of them) are the evidence it rests on."
        )
    if reason == "low_tnr":
        text = (
            "The true-negative rate (how often the judge catches bad outputs) is low: "
            + f"{_pct(tnr)}{f' (95% CI {_pct(ci[0])} to {_pct(ci[1])})' if ci else ''}, below {TNR_MIN_RELIABLE:.0%}, so the judge "
            f"appears to miss more than {1 - TNR_MIN_RELIABLE:.0%} of the bad outputs. Review the disagreements where a human failed what the judge passed "
            "(listed first) and tighten the rubric or add them as few-shot examples, then calibrate again."
        )
        if ci and (ci[1] - ci[0]) / 2 > TNR_MAX_HALF_WIDTH:
            text += " The estimate is also imprecise: labeling more cases the judge passed would sharpen it."
        return text
    if reason == "undefined":
        return head + "its interval could not be computed from these labels (label cases from every stratum, including ones the judge passed)."
    return ""


def _one_sided_message(side: str, n: int, run_constant: bool, n_disagree: int = 0) -> str:
    """Every labeled case is one the judge passed (``side="pass"``) or failed: kappa and one of TPR / TNR are degenerate.

    ``n_disagree`` is how many of them the humans judged the other way (bad outputs the judge passed, or good ones it
    failed): that is evidence about the judge (it lets bad outputs through), only not enough to measure it.
    The advice is always to label the other kind of case, never to revise the rubric.
    """
    one = n == 1
    count = "The only labeled case is" if one else f"All {n} labeled cases are"
    if side == "pass":
        head = f"{count} {'one' if one else 'ones'} the judge passed"
        if n_disagree:
            head += (
                f" and humans failed {n_disagree} of them: the judge lets bad outputs through, but how often it catches them "
                "cannot be measured, so kappa and the true-negative rate are degenerate. "
            )
        else:
            head += ", so nothing in these labels shows whether it can catch a bad output: kappa and the true-negative rate are degenerate. "
        if run_constant:
            return (
                head + "The judge also passed every case of this run, so there is nothing it failed to label yet: add cases that "
                "should fail (known-bad outputs) or check that the judge can fail at all, then label cases the judge failed."
            )
        return head + "To measure it, label cases the judge failed (`evalsprint label` samples them on purpose)."
    head = f"{count} {'one' if one else 'ones'} the judge failed"
    if n_disagree:
        head += (
            f" and humans passed {n_disagree} of them: the judge rejects good outputs, but how often it accepts them "
            "cannot be measured, so kappa and the true-positive rate are degenerate. "
        )
    else:
        head += ", so nothing in these labels shows whether it lets a good output through: kappa and the true-positive rate are degenerate. "
    if run_constant:
        return (
            head + "The judge also failed every case of this run: add cases that should pass, or check that the judge can pass "
            "at all, then label cases the judge passed."
        )
    return head + "To measure it, label cases the judge passed (`evalsprint label` samples them on purpose)."


def _recommend(
    *,
    n: int,
    n_human_fail: int,
    n_missed_bad: int,
    tnr_reliable: bool,
    kappa: float | None,
    kappa_ci: Sequence[float] | None,
    decision: str,
    labels_needed: int | None,
    n_sha_mismatch: int,
    inter_annotator: dict | None,
    labels_needed_capped: bool = False,
    n_labelable: int | None = None,
    held_by: str | None = None,
    strata_gap: Sequence[str] = (),
    kappa_defined_on_sample: bool = True,
    tnr_reason: str | None = None,
    tnr: float | None = None,
    tnr_ci: Sequence[float] | None = None,
    one_sided: str | None = None,
    run_constant: bool = False,
    n_disagree: int = 0,
) -> str:
    """Plain-English next steps that follow ``kappa_decision``; every applicable message is included.

    ``kappa`` and ``kappa_ci`` are the weighted (whole-run) ones, the estimand the decision uses.
    ``one_sided`` ("pass" / "fail") says every labeled case is one the judge passed / failed: the advice is then to
    label the other kind, never to revise the rubric (the labels say nothing about the rubric).
    """
    if n == 0:
        return (
            "No usable labels matched this run. Label 50-100 cases (include failures) with "
            "`evalsprint label`, then run the calibration again."
        )
    messages = []
    fmt = f"Weighted kappa {_fmt_kappa(kappa, kappa_ci)}" if kappa is not None else "Weighted kappa"
    if one_sided:
        messages.append(_one_sided_message(one_sided, n, run_constant, n_disagree))
    if decision == "insufficient_data":
        if one_sided and n >= MIN_LABELS:
            pass  # said above: the labels are one-sided, that is why nothing can be decided
        elif n < MIN_LABELS:
            messages.append(
                f"Only {_plural(n, 'case')} labeled: label more (aim for 50-100, and include failures) "
                "before relying on these numbers."
            )
        elif strata_gap:
            messages.append(
                f"No labeled case in the {' and '.join(strata_gap)} stratum, so the kappa of the whole run cannot be "
                "estimated: label some cases from every stratum (the stratified labeling sample does this)."
            )
        elif not kappa_defined_on_sample:
            messages.append(
                "Kappa is undefined because humans and judge gave the same verdict on every labeled case: "
                "label a mix of passing and failing outputs."
            )
        else:  # pragma: no cover - insufficient_data implies one of the branches above
            messages.append("Not enough data to decide.")
    elif decision == "trustworthy":
        messages.append(
            f"{fmt}: the judge is trustworthy for gating (the whole 95% CI is at or above {KAPPA_TRUSTWORTHY:.2f} and the "
            f"true-negative rate passed the screening check: at least {MIN_HUMAN_FAILS} human fails, an interval within "
            f"+/-{TNR_MAX_HALF_WIDTH * 100:.0f} points and a point estimate of at least {TNR_MIN_RELIABLE:.0%}). The check is a screen, "
            "not a guarantee that the true-negative rate is that high: keep labeling some judge passes now and then, and calibrate "
            "again when the rubric or the model changes."
        )
    elif decision == "usable":
        messages.append(
            f"{fmt}: usable (the 95% CI lower bound is at or above {KAPPA_USABLE:.2f}), but "
            "review the disagreements and add the clearest ones as few-shot `examples:` in the rubric "
            "(`evalsprint calibrate --export-fewshot` writes them, and the cases to leave out of the next calibration), "
            "then calibrate again."
        )
        if held_by == "tnr":
            messages.append(
                f"It is not called trustworthy although the interval is at or above {KAPPA_TRUSTWORTHY:.2f}: the true-negative "
                "rate did not pass the screening check (see the warning above for why and what to do)."
            )
    elif decision == "not_trustworthy":
        if one_sided:  # a judge that never fails anything: the labels cannot say the rubric is wrong
            messages.append(f"{fmt}: do not gate on this judge yet (the whole 95% CI is below {KAPPA_USABLE:.2f}).")
        else:
            messages.append(
                f"{fmt}: do not gate on this judge yet (the whole 95% CI is below "
                f"{KAPPA_USABLE:.2f}). Revise the rubric criteria (make each one specific and binary), review the "
                "disagreements (or add the clearest as few-shot examples), then calibrate again."
            )
    else:  # inconclusive
        if held_by == "min_labels":
            messages.append(
                f"{fmt}: the interval looks good, but only {_plural(n, 'case')} are labeled and at least "
                f"{MIN_LABELS_RELEASE} are needed before a judge is released (with fewer, weak judges get released too often). "
                "Do not gate on it yet: label more cases."
            )
        else:
            messages.append(
                f"{fmt}: inconclusive, the 95% CI straddles {KAPPA_USABLE:.2f}, so these labels "
                "cannot say whether the judge is good enough. Do not gate on it yet: label more cases."
            )
        if labels_needed_capped and n_labelable:
            if n >= n_labelable:
                messages.append(
                    f"All {n_labelable} labelable cases are already labeled: add cases or repeats for more precision."
                )
            else:
                messages.append(
                    f"This run has only {n_labelable} labelable cases, fewer than the labels needed for a +/-0.10 interval: "
                    f"label all {n_labelable} cases; add cases or repeats for more precision."
                )
        elif labels_needed:
            messages.append(f"About {labels_needed} labels in total would narrow it to +/-0.10.")
    if not tnr_reliable and not one_sided and not strata_gap:
        tnr_text = _tnr_message(tnr_reason or ("few_human_fails" if n_human_fail < MIN_HUMAN_FAILS else "undefined"), n_human_fail, tnr, tnr_ci)
        if tnr_text:
            messages.append(tnr_text)
    if inter_annotator and inter_annotator["kappa"] is not None and round(inter_annotator["kappa"], 2) < KAPPA_USABLE:
        messages.append(
            f"The labelers agree with each other only at kappa {inter_annotator['kappa']:.2f}, so "
            "the pass/fail definition is ambiguous: settle it with them before tuning the judge."
        )
    if n_sha_mismatch:
        messages.append(
            f"{_plural(n_sha_mismatch, 'label')} excluded because the AI output changed after labeling: "
            "export the labeling file again from this run."
        )
    return " ".join(messages)


# ---------------------------------------------------------------------- markdown
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _with_ci(text: str, ci: Sequence[float] | None, fmt=_num) -> str:
    return text if not ci else f"{text} (95% CI {fmt(ci[0])} to {fmt(ci[1])})"


def _one_line(text: str, limit: int = 300) -> str:
    return _truncate(" ".join((text or "").split()), limit)


_TNR_UNRELIABLE_TEXT = {
    "few_human_fails": f"screening check not passed: fewer than {MIN_HUMAN_FAILS} bad outputs labeled",
    "wide_ci": f"screening check not passed: interval wider than +/-{TNR_MAX_HALF_WIDTH * 100:.0f} points",
    "low_tnr": f"low: the judge misses more than {1 - TNR_MIN_RELIABLE:.0%} of the bad outputs (below {TNR_MIN_RELIABLE:.0%})",
    "undefined": "screening check not passed: interval undefined",
}

_DECISION_TEXT = {
    "trustworthy": "trustworthy (weighted kappa CI lower bound >= 0.80 and the TNR screening check passed)",
    "usable": "usable (weighted kappa CI lower bound >= 0.60)",
    "inconclusive": "inconclusive (weighted kappa CI straddles 0.60, or too few labels to release)",
    "not_trustworthy": "not trustworthy (weighted kappa CI upper bound < 0.60)",
    "insufficient_data": "insufficient data",
}


def to_markdown(cal: dict, max_disagreements: int = 10) -> str:
    """Compact report for a PR comment, `$GITHUB_STEP_SUMMARY` or the client deliverable."""
    conf = cal["confusion"]
    lines = [f"## Judge calibration: {cal['judge']} (run `{cal['run_id']}`)", ""]

    skipped = [
        (cal.get("n_excluded_sha_mismatch"), "labels on an output that has since changed"),
        (cal.get("n_unmatched"), "labels with an unknown case id"),
        (cal.get("n_invalid_labels"), "unreadable label rows"),
        (cal.get("n_tied_cases"), "cases with tied labels"),
        (cal.get("n_no_judge_verdict"), "cases the judge did not score"),
        (cal.get("n_excluded_holdout"), "labels held out (few-shot examples)"),
    ]
    skipped_text = ", ".join(f"{count} {what}" for count, what in skipped if count)
    notes = []
    if cal.get("n_carried_cases"):
        notes.append(f"{cal['n_carried_cases']} carried over from an earlier run")
    if skipped_text:
        notes.append(f"skipped: {skipped_text}")
    lines.append(f"**{cal['n']}** labeled cases used" + (f" ({'; '.join(notes)})." if notes else "."))
    lines.append("")

    decision = cal.get("kappa_decision")
    if decision:
        lines += [f"**Decision: {_DECISION_TEXT.get(decision, decision)}.**", ""]

    human_fail = conf["human_fail_judge_pass"] + conf["human_fail_judge_fail"]
    kappa_w = cal.get("kappa_weighted")
    if kappa_w is not None:
        kappa_text = f"{_with_ci(_num(kappa_w), cal.get('kappa_weighted_ci'))}, {kappa_label(kappa_w)}"
    else:  # calibration dicts written before the weighted kappa existed
        kappa_text = "n/a" if cal["kappa"] is None else f"{_with_ci(_num(cal['kappa']), cal.get('kappa_ci'))}, {cal['kappa_label']}"
    sample_kappa_text = "n/a" if cal["kappa"] is None else f"{_with_ci(_num(cal['kappa']), cal.get('kappa_ci'))}, {cal['kappa_label']}"
    agreement = cal.get("agreement_weighted")
    agreement_text = _pct(cal["agreement"]) if agreement is None else f"{_pct(agreement)} (labeled sample: {_pct(cal['agreement'])})"
    tnr_text = _with_ci(_pct(cal["tnr"]), cal.get("tnr_ci"), _pct)
    if cal.get("tnr_reliable") is False and cal["tnr"] is not None:
        tnr_text += " - " + _TNR_UNRELIABLE_TEXT.get(cal.get("tnr_reliable_reason"), "screening check not passed")
    elif cal.get("tnr_reliable") is True:
        tnr_text += " - screening check passed (a screen, not a guarantee)"
    kappa_rows = (
        [
            f"| Cohen's kappa (whole run, weighted; the decision uses it) | {kappa_text} |",
            f"| Cohen's kappa (labeled sample, unweighted; secondary) | {sample_kappa_text} |",
        ]
        if kappa_w is not None
        else [f"| Cohen's kappa | {kappa_text} |"]
    )
    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Agreement | {agreement_text} |",
        *kappa_rows,
        f"| TPR: judge passes what humans pass | {_with_ci(_pct(cal['tpr']), cal.get('tpr_ci'), _pct)} |",
        f"| TNR: judge fails what humans fail | {tnr_text} |",
        f"| Human labels | {cal['n'] - human_fail} pass, {human_fail} fail |",
        f"| Pass rate as judged, whole run | {_pct(cal['run_observed_pass_rate'])} |",
        f"| Pass rate corrected for judge errors | {_with_ci(_pct(cal['corrected_pass_rate']), cal.get('corrected_pass_rate_ci'), _pct)} |",
    ]
    ia = cal.get("inter_annotator")
    if ia:
        lines.append(
            f"| Human-human kappa (ceiling for any judge) | {_num(ia['kappa'])} ({' & '.join(ia['labelers'])}, n={ia['n']}) |"
        )
    lines.append("")
    if cal.get("corrected_pass_rate") is not None:
        lines += [f"_{cal.get('corrected_pass_rate_ci_note') or CORRECTED_CI_NOTE}_", ""]

    labeled_rate, run_rate = cal.get("judge_pass_rate_labeled"), cal.get("run_observed_pass_rate")
    if labeled_rate is not None and run_rate is not None and abs(labeled_rate - run_rate) >= 0.02:
        lines += [
            f"_Agreement, TPR, TNR and the corrected pass rate are re-weighted: the labeled sample has a {_pct(labeled_rate)} "
            f"judge pass rate versus {_pct(run_rate)} in the whole run._",
            "",
        ]

    lines += [
        "| | Judge pass | Judge fail |",
        "|---|---:|---:|",
        f"| **Human pass** | {conf['human_pass_judge_pass']} | {conf['human_pass_judge_fail']} |",
        f"| **Human fail** | {conf['human_fail_judge_pass']} | {conf['human_fail_judge_fail']} |",
        "",
        f"**Recommendation:** {cal['recommendation']}",
    ]

    warnings = cal.get("warnings") or []
    if warnings:
        lines += ["", "### Warnings", ""] + [f"- {_one_line(w, 600)}" for w in warnings]

    disagreements = cal.get("disagreements") or []
    if disagreements:
        shown = disagreements[:max_disagreements]
        lines += ["", f"### Disagreements (showing {len(shown)} of {len(disagreements)}; judge too lenient first)", ""]
        for i, d in enumerate(shown, 1):
            lines.append(f"{i}. `{d['case_id']}`: human **{d['human']}**, judge **{d['judge']}**")
            lines.append(f"   - Input: {_one_line(d['input'])}")
            lines.append(f"   - AI output: {_one_line(d['output'])}")
            if d["judge_reasoning"]:
                lines.append(f"   - Judge reasoning: {_one_line(d['judge_reasoning'])}")
            if d["note"]:
                lines.append(f"   - Human note: {_one_line(d['note'])}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ few-shot loop
FEWSHOT_MAX_FIELD_CHARS = 2000  # an example's input / output is cut at this many characters (the rubric pastes them into every judge call)
_HOLDOUT_KEY = "holdout_case_ids"


class _FewshotDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line text as a literal block, so an example stays readable and editable."""


def _represent_str(dumper: yaml.SafeDumper, value: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_FewshotDumper.add_representer(str, _represent_str)


def _cut(text: str, limit: int = FEWSHOT_MAX_FIELD_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}\n[... truncated {len(text) - limit} chars ...]"


def _judge_score(record: dict, judge: str | None) -> float | None:
    """The judge's 0..1 score for a case (the mean of the judges' scores for the overall verdict), when it has one."""
    scores = [
        j.get("score")
        for j in record.get("judges") or []
        if (judge is None or j.get("judge") == judge) and isinstance(j.get("score"), (int, float)) and not isinstance(j.get("score"), bool)
    ]
    return sum(scores) / len(scores) if scores else None


def _holdout_pool(cal: dict, run: RunResult, frac: float, seed: int) -> list[str]:
    """A seeded random share of the labeled cases, drawn within each sampling stratum (never a whole stratum)."""
    stratified_on = (cal.get("strata") or {}).get("stratified_on")
    stratifier = None if stratified_on in (None, "overall", "none") else stratified_on
    by_case = run.by_case()
    groups: dict[str, list[str]] = {}
    for cid in cal["labeled_case_ids"]:
        record = by_case.get(cid)
        verdict = _verdict(record, stratifier) if record is not None else None
        groups.setdefault("?" if verdict is None else stratum_of(verdict), []).append(cid)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for name in sorted(groups):
        members = groups[name]
        take = max(0, min(len(members) - 1, int(frac * len(members) + 0.5)))
        chosen.update(rng.sample(members, take))
    return [cid for cid in cal["labeled_case_ids"] if cid in chosen]


def export_fewshot(
    cal: dict, run: RunResult, out_path: str | Path, max_examples: int = 6, holdout_frac: float = 0.3, seed: int = 0
) -> Path:
    """Write the clearest human-vs-judge disagreements as rubric ``examples:`` and the cases to keep out of the next calibration.

    The YAML has two keys. ``examples:`` is ready to paste under ``examples:`` in the rubric file (``verdict`` is the
    HUMAN's verdict and comes first, then ``input``, ``output``, ``reasoning``; ``case_id`` is extra and ignored by
    the judge). ``holdout_case_ids:`` lists the labeled cases the examples were drawn from: leave them out of the next
    calibration (``calibrate(..., exclude_case_ids=load_holdout_ids(path))``), or the judge is graded on cases it was shown
    with the answer and the kappa comes out too high.

    Why a random pool. Excluding only the example cases would remove the hardest disagreements from the next
    calibration and flatter the judge. So ``holdout_frac`` of the labeled cases (a seeded random draw within each
    sampling stratum, never a whole stratum) is set aside, the examples are chosen among the disagreements inside
    that pool, and the rest of the labels stay an unbiased random subsample for the next calibration. The price is
    that the next calibration has fewer labels (label more cases if the interval gets too wide).

    Which examples. Within the pool, disagreements with a human note first (the note becomes ``reasoning``), then the
    ones where the judge was most confident in its mistake (by its score), then run order; up to ``max_examples``,
    balanced between "human fail, judge pass" (listed first: the judge was too lenient) and "human pass, judge fail".
    A disagreement without a human note has an empty ``reasoning``: write one sentence before using it.
    ``input`` / ``output`` are the full case texts (cut at FEWSHOT_MAX_FIELD_CHARS), not the excerpts of ``cal``.

    ``cal`` is the dict `calibrate` returned for ``run`` (it needs ``labeled_case_ids``: an older calibration.json
    without them must be recomputed). Deterministic for a given ``seed``.
    """
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be strictly between 0 and 1: the examples come from cases that are then left out of the next calibration")
    if "labeled_case_ids" not in cal:
        raise ValueError("this calibration has no labeled_case_ids (it was made by an older evalsprint): run calibrate again")
    if cal.get("run_id") is not None and cal["run_id"] != run.run_id:
        raise ValueError(f"the calibration is of run {cal['run_id']!r}, not {run.run_id!r}")
    judge = None if cal.get("judge") in (None, "overall") else cal["judge"]
    by_case = run.by_case()
    pool = _holdout_pool(cal, run, holdout_frac, seed)
    in_pool = set(pool)

    candidates = []
    for order, d in enumerate(cal.get("disagreements") or []):
        record = by_case.get(d["case_id"])
        if d["case_id"] not in in_pool or record is None:
            continue
        score = _judge_score(record, judge)
        wrong_confidence = 0.5 if score is None else (score if d["judge"] == "pass" else 1 - score)
        candidates.append((d["human"], not (d.get("note") or "").strip(), -wrong_confidence, order, d, record))
    candidates.sort(key=lambda c: c[:4])
    by_human = {h: [c for c in candidates if c[0] == h] for h in ("fail", "pass")}
    picked: list[tuple] = []
    turn = ["fail", "pass"]
    while len(picked) < max_examples and (by_human["fail"] or by_human["pass"]):
        side = turn[len(picked) % 2] if by_human[turn[len(picked) % 2]] else turn[(len(picked) + 1) % 2]
        picked.append(by_human[side].pop(0))
    picked.sort(key=lambda c: (c[0] != "fail", c[1:4]))  # human fail (judge too lenient) first, then the other order kept
    examples = [
        {
            "verdict": human,
            "input": _cut(_as_text(record["case"].get("input"))),
            "output": _cut(_as_text(record["output"].get("text"))),
            "reasoning": (d.get("note") or "").strip(),
            "case_id": d["case_id"],
        }
        for human, _, _, _, d, record in picked
    ]

    n_labeled = len(cal["labeled_case_ids"])
    n_no_note = sum(1 for e in examples if not e["reasoning"])
    header = [
        f"# Few-shot examples from the calibration of run {run.run_id!r} (judge: {(cal.get('judge') or 'overall')!r}), seed {seed}.",
        "# examples: paste under `examples:` in the rubric file. `verdict` is the human's verdict; `case_id` is ignored by the judge.",
        f"# holdout_case_ids: {len(pool)} of the {n_labeled} labeled cases (a random {holdout_frac:.0%}). The examples come from them.",
        "#   Leave them OUT of the next calibration (calibrate(..., exclude_case_ids=load_holdout_ids(<this file>))),",
        "#   or the judge is graded on cases it was shown with the answer. Keep the others as they are.",
    ]
    if not examples:
        header.append("# No disagreement fell inside the pool, so there are no examples: raise holdout_frac or label more cases.")
    elif len(examples) < max_examples:
        header.append(f"# Only {len(examples)} disagreement(s) fell inside the pool (max_examples was {max_examples}).")
    if n_no_note:
        header.append(f"# {n_no_note} example(s) have an empty `reasoning`: write one sentence on why the human verdict is right.")
    body = yaml.dump(
        {"examples": examples, _HOLDOUT_KEY: pool}, Dumper=_FewshotDumper, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
    return out_path


def load_holdout_ids(path: str | Path) -> list[str]:
    """The case ids to leave out of a calibration, from `export_fewshot`'s YAML, a JSON/YAML list or a text file (one id per line)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        if _HOLDOUT_KEY not in data:
            raise ValueError(f"{path} has no {_HOLDOUT_KEY!r} key")
        data = data[_HOLDOUT_KEY]
    elif not isinstance(data, list):
        data = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if data is not None and not isinstance(data, list):
        raise ValueError(f"{path}: {_HOLDOUT_KEY!r} must be a list of case ids")
    return [str(item) for item in data or [] if item is not None and str(item).strip()]
