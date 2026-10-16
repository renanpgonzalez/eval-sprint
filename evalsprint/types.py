"""Core data structures shared across the kit."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------- text hygiene
# A lone UTF-16 surrogate (what JSON.stringify emits for a truncated emoji) is a valid
# Python str but cannot be written as UTF-8. One such character used to lose a whole
# paid run at save time, so text is cleaned when it enters the kit.
_SURROGATES = re.compile("[\ud800-\udfff]")


def sanitize_text(text: str) -> str:
    """Replace lone surrogates with U+FFFD so the text can be encoded as UTF-8."""
    return _SURROGATES.sub("�", text) if isinstance(text, str) else text


def sanitize_obj(value: Any) -> Any:
    """`sanitize_text` applied to every string (and dict key) inside nested dicts, lists and tuples."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {sanitize_text(k) if isinstance(k, str) else k: sanitize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_obj(v) for v in value]
    return value


def suggest(word: str, options, n: int = 1) -> str:
    """` - did you mean 'x'?` for the closest option, or an empty string."""
    close = difflib.get_close_matches(str(word), [str(o) for o in options], n=n, cutoff=0.6)
    if not close:
        return ""
    return f" - did you mean {' or '.join(repr(c) for c in close)}?"


def case_sha(case: "Case | dict") -> str:
    """Fingerprint of a case: first 12 hex chars of the sha256 of the canonical JSON (sorted keys, default
    separators, ASCII-escaped, `default=str`) of the WHOLE case dict except `id`.

    That is input, reference, expected_behavior, tags, checks and every metadata / extra field (a judge-facing
    field such as `must_include` changes what is being tested just as much as the input does). Two runs can only
    be paired case by case when the case itself did not change; the id is left out so a rename is not an edit.
    Accepts a `Case` or the `case` dict stored in a result record (`Case.to_dict()`).
    """
    data = case.to_dict() if isinstance(case, Case) else case
    payload = {key: value for key, value in data.items() if key != "id"}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @classmethod
    def from_dict(cls, d: dict | None) -> "Usage":
        d = d or {}
        return cls(
            int(d.get("input_tokens", 0) or 0),
            int(d.get("output_tokens", 0) or 0),
            int(d.get("cache_read_tokens", 0) or 0),
            int(d.get("cache_write_tokens", 0) or 0),
        )


_CASE_FIELDS = {"id", "input", "reference", "expected_behavior", "tags", "checks", "metadata"}


@dataclass
class Case:
    """One test case. `input` is whatever the target expects (str or dict)."""

    id: str
    input: Any
    reference: str | None = None
    expected_behavior: str | None = None  # e.g. answer | abstain | refuse
    tags: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)  # per-case heuristic checks
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        """Build a case from one dataset row, validating the schema so a typo fails at load time
        (a `tags: "safety"` string used to become the tags s, a, f, e, t, y)."""
        if not isinstance(d, dict) or "id" not in d or "input" not in d:
            raise ValueError(f"case needs 'id' and 'input': {d!r}")
        d = sanitize_obj(d)
        raw_id = d["id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError(f"case 'id' must be a non-empty string, got {raw_id!r}")
        case_id = str(raw_id)
        if not case_id.strip():
            raise ValueError("case 'id' must be a non-empty string")
        if d["input"] is None:
            raise ValueError(f"case {case_id!r}: 'input' must not be null")

        where = f"case {case_id!r}"
        tags = d.get("tags")
        if tags is None:
            tags = []
        if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
            hint = f' - write it as a list: ["{tags}"]' if isinstance(tags, str) else ""
            raise ValueError(f"{where}: 'tags' must be a list of non-empty strings, got {tags!r}{hint}")

        checks = d.get("checks")
        if checks is None:
            checks = []
        if not isinstance(checks, list):
            raise ValueError(f"{where}: 'checks' must be a list of {{type: ..., ...}} objects, got {type(checks).__name__}")
        from .judges.heuristics import validate_check  # deferred: judges import this module

        for i, check in enumerate(checks):
            validate_check(check, f"{where}, check #{i}")

        behavior = d.get("expected_behavior")
        if behavior is not None and not isinstance(behavior, str):
            raise ValueError(f"{where}: 'expected_behavior' must be a string, got {behavior!r}")
        metadata = d.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{where}: 'metadata' must be an object, got {type(metadata).__name__}")

        extra = {k: v for k, v in d.items() if k not in _CASE_FIELDS}
        meta = dict(metadata)
        meta.update(extra)  # keep unknown fields (e.g. must_include) available to judges
        return cls(
            id=case_id,
            input=d["input"],
            reference=d.get("reference"),
            expected_behavior=behavior,
            tags=list(tags),
            checks=[dict(c) for c in checks],
            metadata=meta,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha(self) -> str:
        return case_sha(self)


@dataclass
class TargetOutput:
    text: str
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_s: float | None = None
    meta: dict = field(default_factory=dict)  # e.g. retrieved context
    error: str | None = None

    @property
    def sha(self) -> str:
        return hashlib.sha256(sanitize_text(self.text or "").encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class JudgeResult:
    judge: str
    passed: bool
    score: float | None = None  # normalised 0..1 when available
    reasoning: str = ""
    details: dict = field(default_factory=dict)  # per-criterion results etc.
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


OUTCOMES = ("pass", "fail", "error")


def _count(value: Any) -> int:
    """A non-negative whole number from a record field that an older file may not have (or a hand edit may have broken)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return 0
    return int(value)


def derive_outcome(target_error: str | None, judges: list, passed: bool) -> tuple[str, str | None]:
    """(outcome, error_source) for one case attempt.

    A case is an `error` when the target failed or any applicable judge errored: neither says
    anything about quality, so errors are kept out of the pass rate. Target errors win because
    the judges are skipped (and marked as errored) when the target failed.
    """
    if target_error:
        return "error", "target"
    if any(getattr(j, "error", None) if not isinstance(j, dict) else j.get("error") for j in judges):
        return "error", "judge"
    return ("pass" if passed else "fail"), None


@dataclass
class CaseResult:
    case: Case
    output: TargetOutput
    judges: list[JudgeResult]
    passed: bool
    outcome: str = ""  # "pass" | "fail" | "error"; derived from the other fields when left empty
    error_source: str | None = None  # None | "target" | "judge"
    attempts: list[dict] | None = None  # only for repeated cases
    stability: float | None = None
    target_retries_used: int = 0  # times the target was called AGAIN after it errored (all attempts of a repeated case together)

    def resolved(self) -> tuple[str, str | None]:
        if self.outcome:
            return self.outcome, self.error_source
        return derive_outcome(self.output.error, self.judges, self.passed)

    def to_dict(self) -> dict:
        outcome, source = self.resolved()
        record = {
            "case_id": self.case.id,
            "case": self.case.to_dict(),
            "case_sha": self.case.sha,
            "output": {
                "text": self.output.text,
                "sha": self.output.sha,
                "model": self.output.model,
                "usage": asdict(self.output.usage),
                "cost_usd": self.output.cost_usd,
                "latency_s": self.output.latency_s,
                "meta": self.output.meta,
                "error": self.output.error,
            },
            "judges": [
                {
                    "judge": j.judge,
                    "passed": j.passed,
                    "score": j.score,
                    "reasoning": j.reasoning,
                    "details": j.details,
                    "cost_usd": j.cost_usd,
                    "usage": asdict(j.usage),
                    "error": j.error,
                }
                for j in self.judges
            ],
            "outcome": outcome,
            "error_source": source,
            "passed": outcome == "pass",
            # errored ATTEMPTS: 0 or 1 for a case that ran once, 0..n for a repeated case. A crash in one attempt is
            # invisible in `outcome` when the majority still passes; this count keeps it visible.
            "n_attempt_errors": (sum(1 for a in self.attempts if a.get("outcome") == "error") if self.attempts is not None else int(outcome == "error")),
            # C18: the paid calls repeated after a crash, inside the run (target_retries): 0 when the first call of every attempt worked
            "target_retries_used": int(self.target_retries_used or 0),
        }
        if self.attempts is not None:
            record["attempts"] = self.attempts
            record["stability"] = self.stability
        return record

    @classmethod
    def from_record(cls, record: dict, case: Case) -> "CaseResult":
        """Rebuild a result from one line of results.jsonl / results.partial.jsonl (`run --resume`).
        `case` is the CURRENT dataset case: the caller has already checked that its case_sha matches the record's."""
        out = record.get("output") or {}
        output = TargetOutput(
            text=out.get("text") or "",
            model=out.get("model"),
            usage=Usage.from_dict(out.get("usage")),
            cost_usd=out.get("cost_usd"),
            latency_s=out.get("latency_s"),
            meta=dict(out.get("meta") or {}),
            error=out.get("error"),
        )
        judges = [
            JudgeResult(
                judge=str(j.get("judge")),
                passed=bool(j.get("passed")),
                score=j.get("score"),
                reasoning=j.get("reasoning") or "",
                details=dict(j.get("details") or {}),
                cost_usd=float(j.get("cost_usd") or 0.0),
                usage=Usage.from_dict(j.get("usage")),
                error=j.get("error"),
            )
            for j in record.get("judges") or []
            if isinstance(j, dict)
        ]
        outcome = record.get("outcome")
        source = record.get("error_source")
        if outcome not in OUTCOMES:
            outcome, source = derive_outcome(output.error, judges, bool(record.get("passed")))
        attempts = record.get("attempts")
        return cls(
            case=case,
            output=output,
            judges=judges,
            passed=outcome == "pass",
            outcome=outcome,
            error_source=source if outcome == "error" else None,
            attempts=[dict(a) for a in attempts if isinstance(a, dict)] if isinstance(attempts, list) else None,
            stability=record.get("stability") if isinstance(attempts, list) else None,
            target_retries_used=_count(record.get("target_retries_used")),
        )
