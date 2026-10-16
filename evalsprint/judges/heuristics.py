"""Deterministic judges: cheap, fast, no false positives from model noise.

Use them for hard requirements (format, forbidden strings, length) and keep the
LLM judge for semantic quality.
"""

from __future__ import annotations

import inspect
import json
import re
import unicodedata

from ..loader import load_callable
from ..types import Case, JudgeResult, TargetOutput, suggest
from .base import Judge, JudgeConfigError


def _text_list(values, what: str = "values") -> list[str]:
    """`values` as a non-empty list of strings (a bare string means one value)."""
    items = [values] if isinstance(values, str) else values
    if not isinstance(items, (list, tuple)) or not items or not all(isinstance(v, str) and v != "" for v in items):
        raise ValueError(f"'{what}' must be a non-empty string or a list of non-empty strings, got {values!r}")
    return list(items)


def _flag(value, what: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{what}' must be true or false, got {value!r}")
    return value


def _norm(text: str, case_insensitive: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if case_insensitive else text


def _count(value) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)  # `--set judges.0.value=5` and ${MAX_CHARS} may arrive as text
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 0:
        raise ValueError(f"'value' must be a whole number >= 0, got {value!r}")
    return int(value)


class Contains(Judge):
    kind = "contains"

    def __init__(self, name, values, mode: str = "any", case_insensitive: bool = True, **kw):
        super().__init__(name, **kw)
        self.values = _text_list(values)
        if mode not in ("any", "all"):
            raise ValueError(f"'mode' must be 'any' or 'all', got {mode!r}")  # anything else used to mean "all"
        self.mode = mode
        self.ci = _flag(case_insensitive, "case_insensitive")

    def evaluate(self, case: Case, output: TargetOutput) -> JudgeResult:
        text = _norm(output.text, self.ci)
        hits = [v for v in self.values if _norm(v, self.ci) in text]
        passed = bool(hits) if self.mode == "any" else len(hits) == len(self.values)
        missing = [v for v in self.values if v not in hits]
        reason = f"found {hits}" if passed else f"missing {missing if self.mode == 'all' else self.values}"
        return JudgeResult(self.name, passed, score=len(hits) / len(self.values), reasoning=reason)


class NotContains(Judge):
    kind = "not_contains"

    def __init__(self, name, values, case_insensitive: bool = True, **kw):
        super().__init__(name, **kw)
        self.values = _text_list(values)
        self.ci = _flag(case_insensitive, "case_insensitive")

    def evaluate(self, case, output):
        text = _norm(output.text, self.ci)
        hits = [v for v in self.values if _norm(v, self.ci) in text]
        return JudgeResult(self.name, not hits, reasoning=f"forbidden text present: {hits}" if hits else "ok")


class Regex(Judge):
    kind = "regex"

    def __init__(self, name, pattern: str, must_match: bool = True, flags: str = "i", **kw):
        super().__init__(name, **kw)
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"'pattern' must be a non-empty string, got {pattern!r}")
        if not isinstance(flags, str) or set(flags) - set("ism"):
            raise ValueError(f"'flags' may only contain i, s and m, got {flags!r}")
        f = 0
        f |= re.I if "i" in flags else 0
        f |= re.S if "s" in flags else 0
        f |= re.M if "m" in flags else 0
        self.pattern = re.compile(pattern, f)  # re.error on a bad pattern; build_judges / validate_check report it as a config error
        self.must_match = _flag(must_match, "must_match")

    def evaluate(self, case, output):
        matched = bool(self.pattern.search(output.text or ""))
        passed = matched if self.must_match else not matched
        return JudgeResult(self.name, passed, reasoning=f"pattern {'matched' if matched else 'not matched'}")


class MaxChars(Judge):
    kind = "max_chars"

    def __init__(self, name, value: int, **kw):
        super().__init__(name, **kw)
        self.value = _count(value)

    def evaluate(self, case, output):
        n = len(output.text or "")
        return JudgeResult(self.name, n <= self.value, reasoning=f"{n} chars (max {self.value})")


class MinChars(Judge):
    kind = "min_chars"

    def __init__(self, name, value: int, **kw):
        super().__init__(name, **kw)
        self.value = _count(value)

    def evaluate(self, case, output):
        n = len((output.text or "").strip())
        return JudgeResult(self.name, n >= self.value, reasoning=f"{n} chars (min {self.value})")


class JsonValid(Judge):
    kind = "json_valid"

    def __init__(self, name, required_keys: list[str] | None = None, **kw):
        super().__init__(name, **kw)
        if isinstance(required_keys, str):
            required_keys = [required_keys]
        if required_keys is not None and (not isinstance(required_keys, (list, tuple)) or not all(isinstance(k, str) for k in required_keys)):
            raise ValueError(f"'required_keys' must be a list of strings, got {required_keys!r}")
        self.required = list(required_keys or [])

    def evaluate(self, case, output):
        text = (output.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            return JudgeResult(self.name, False, reasoning=f"invalid JSON: {exc}")
        missing = [k for k in self.required if not (isinstance(obj, dict) and k in obj)]
        return JudgeResult(self.name, not missing, reasoning=f"missing keys {missing}" if missing else "valid JSON")


class ExactMatch(Judge):
    kind = "exact_match"

    def __init__(self, name, case_insensitive: bool = True, **kw):
        super().__init__(name, **kw)
        self.ci = case_insensitive

    def evaluate(self, case, output):
        if case.reference is None:
            return JudgeResult(self.name, False, error="case has no reference")
        ok = _norm(output.text, self.ci) == _norm(case.reference, self.ci)
        return JudgeResult(self.name, ok, reasoning="exact match" if ok else "differs from reference")


HEURISTICS: dict[str, type[Judge]] = {
    "contains": Contains,
    "not_contains": NotContains,
    "regex": Regex,
    "max_chars": MaxChars,
    "min_chars": MinChars,
    "json_valid": JsonValid,
    "exact_match": ExactMatch,
}


def config_keys(cls: type) -> tuple[set[str], set[str]]:
    """(allowed, required) constructor settings of a judge class, read from its __init__ signatures.

    `name` is always allowed (build_judges fills it in from the type) and never required.
    """
    allowed: set[str] = set()
    required: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None or klass is object:
            continue
        for pname, param in inspect.signature(init).parameters.items():
            if pname == "self" or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            allowed.add(pname)
            if param.default is param.empty:
                required.add(pname)
    return allowed, required - {"name"}


def check_settings(settings: dict, cls: type, where: str, *, ignore: tuple[str, ...] = ()) -> None:
    """Reject unknown and missing settings of a judge or check, suggesting the closest valid key."""
    allowed, required = config_keys(cls)
    allowed = allowed - set(ignore)
    for key in settings:
        if key not in allowed:
            raise JudgeConfigError(f"{where}: unknown setting {key!r}{suggest(key, allowed)} (valid: {', '.join(sorted(allowed - {'name'}))})")
    missing = sorted(required - set(settings))
    if missing:
        raise JudgeConfigError(f"{where}: missing required setting(s) {', '.join(repr(m) for m in missing)}")


def validate_check(check, where: str) -> None:
    """Validate one per-case check (`{"type": "contains", "values": [...]}`) at dataset load time,
    so a typo fails `describe` instead of surfacing mid-run as a judge error."""
    if not isinstance(check, dict):
        raise ValueError(f"{where}: a check must be an object like {{\"type\": \"contains\", \"values\": [...]}}, got {type(check).__name__}")
    ctype = check.get("type")
    if ctype is None:
        raise ValueError(f"{where}: a check needs a 'type' (valid types: {', '.join(sorted(HEURISTICS))})")
    if not isinstance(ctype, str) or ctype not in HEURISTICS:
        raise ValueError(f"{where}: check type {ctype!r} is not valid{suggest(ctype, HEURISTICS) if isinstance(ctype, str) else ''} (valid types: {', '.join(sorted(HEURISTICS))})")
    settings = {k: v for k, v in check.items() if k != "type"}
    cls = HEURISTICS[ctype]
    label = f"{where} ({ctype})"
    check_settings(settings, cls, label, ignore=("name", "when_tags", "when_behavior"))
    try:
        cls(name=f"{ctype}#check", **settings)
    except (TypeError, ValueError, re.error) as exc:
        raise ValueError(f"{label}: {exc}") from exc


class CaseChecksJudge(Judge):
    """Runs the per-case `checks` declared in the dataset, e.g.
    {"checks": [{"type": "contains", "values": ["Settings > Billing"]}]}"""

    kind = "case_checks"

    def applies(self, case: Case) -> bool:
        return bool(case.checks)

    def evaluate(self, case, output):
        results = []
        for i, check in enumerate(case.checks):
            check = dict(check)
            ctype = check.pop("type")
            judge = HEURISTICS[ctype](name=f"{ctype}#{i}", **check)
            results.append(judge.evaluate(case, output))
        failed = [r for r in results if not r.passed]
        reason = "; ".join(f"{r.judge}: {r.reasoning}" for r in failed) or "all case checks passed"
        return JudgeResult(
            self.name,
            not failed,
            score=sum(r.passed for r in results) / len(results),
            reasoning=reason,
            details={r.judge: r.passed for r in results},
        )


class CustomJudge(Judge):
    """Python judge: fn(case, output) -> bool | dict(passed, score, reasoning)."""

    kind = "custom"

    def __init__(self, name, fn, data_files=None, **kw):
        super().__init__(name, **kw)
        self.fn = fn
        self.data_files = [data_files] if isinstance(data_files, str) else list(data_files or [])  # the files fn reads: only recorded here, they count in judges_sha (runner.judges_fingerprint)

    @classmethod
    def from_config(cls, cfg: dict, base_dir=None, **_):
        cfg = dict(cfg)
        fn = load_callable(cfg.pop("callable"), base_dir)
        return cls(fn=fn, **cfg)

    def evaluate(self, case, output):
        res = self.fn(case, output)
        if isinstance(res, bool):
            return JudgeResult(self.name, res)
        return JudgeResult(self.name, bool(res.get("passed")), score=res.get("score"), reasoning=res.get("reasoning", ""))
