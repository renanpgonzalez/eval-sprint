"""Test-set loading and validation (JSONL, one case per line)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .types import Case, suggest


def load_cases(path: str | Path, behaviors: list[str] | None = None) -> list[Case]:
    """Load and validate a JSONL test set. Every row is checked at load time (id, tags, checks, ...);
    with `behaviors` (config `behaviors: [answer, abstain, refuse]`) `expected_behavior` must be one of them.
    Lone UTF-16 surrogates in the text become U+FFFD so the run can always be saved."""
    cases: list[Case] = []
    seen: set[str] = set()
    allowed = list(behaviors or [])
    try:
        with open(path, encoding="utf-8-sig") as fh:  # -sig: tolerate the BOM that Excel/Windows tools add
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    case = Case.from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
                if allowed and case.expected_behavior is not None and case.expected_behavior not in allowed:
                    raise ValueError(
                        f"{path}:{lineno}: case {case.id!r}: expected_behavior {case.expected_behavior!r} "
                        f"is not one of {allowed}{suggest(case.expected_behavior, allowed)}"
                    )
                if case.id in seen:
                    raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
                seen.add(case.id)
                cases.append(case)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8 (byte {exc.start}: {exc.reason}); save the file as UTF-8") from exc
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def filter_cases(cases: list[Case], tags: list[str] | None = None, ids: list[str] | None = None, limit: int | None = None) -> list[Case]:
    out = cases
    if tags:
        wanted = set(tags)
        out = [c for c in out if wanted & set(c.tags)]
    if ids:
        wanted_ids = set(ids)
        out = [c for c in out if c.id in wanted_ids]
    if limit:
        out = out[:limit]
    return out


def describe(cases: list[Case]) -> dict:
    tags = Counter(t for c in cases for t in c.tags)
    behaviors = Counter(c.expected_behavior or "unspecified" for c in cases)
    return {
        "n_cases": len(cases),
        "tags": dict(tags.most_common()),
        "expected_behavior": dict(behaviors),
        "with_reference": sum(1 for c in cases if c.reference),
    }
