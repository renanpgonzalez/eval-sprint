"""Judges score one (case, output) pair. Build them from eval.yaml `judges:`."""

from __future__ import annotations

import re
from pathlib import Path

from ..pricing import PriceTable
from ..types import Case, JudgeResult, TargetOutput, Usage, suggest
from .base import Judge
from .heuristics import HEURISTICS, CaseChecksJudge, CustomJudge, check_settings
from .llm_judge import LLMJudge

__all__ = ["Judge", "LLMJudge", "build_judges", "run_judges", "validate_pass_rule"]

JUDGE_TYPES = ("llm", "custom", *HEURISTICS)
_CUSTOM_KEYS = ("callable", "name", "when_tags", "when_behavior", "data_files")


def data_file_paths(value, base_dir: str | Path, where: str = "data_files") -> list[Path]:
    """The files of a custom judge's `data_files: [path, ...]` (relative to the eval.yaml directory), checked: a list of non-empty
    strings, each an existing FILE. They are hashed into `judges_sha`, so a judge that reads a word list changes when the list does;
    a path that does not exist is a configuration error now, not an exception in the middle of a paid run."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{where}: 'data_files' must be a list of file paths (relative to eval.yaml), got {value!r}")
    paths = []
    for item in value:
        path = Path(base_dir) / item
        if not path.is_file():
            raise ValueError(f"{where}: data file {item!r} not found (paths are relative to the eval.yaml directory, {Path(base_dir)})")
        paths.append(path)
    return paths



def _split_config(cfg: dict) -> tuple[str, str, dict]:
    """(type, name, settings) of one `judges:` entry, with the basic shape validated."""
    if not isinstance(cfg, dict):
        raise ValueError(f"each judge must be a mapping with at least a 'type', got {cfg!r}")
    cfg = dict(cfg)
    jtype = cfg.pop("type", None)
    name = cfg.get("name") or jtype
    if not jtype:
        raise ValueError(f"judge {name!r} needs a 'type'")
    if not isinstance(jtype, str) or jtype not in JUDGE_TYPES:
        raise ValueError(f"unknown judge type {jtype!r}{suggest(jtype, JUDGE_TYPES)}; use llm, custom or one of {sorted(HEURISTICS)}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"judge of type {jtype!r} needs a non-empty string 'name'")
    cfg["name"] = name
    return jtype, name, cfg


def _check_keys(jtype: str, name: str, cfg: dict, base_dir: str | Path = ".") -> None:
    """Unknown or missing settings of heuristic and custom judges fail here with a suggestion.
    LLM judge settings are validated by `LLMJudge.from_config` (its signature is the source of truth)."""
    where = f"judge {name!r} ({jtype})"
    if jtype == "custom":
        for key in cfg:
            if key not in _CUSTOM_KEYS:
                raise ValueError(f"{where}: unknown setting {key!r}{suggest(key, _CUSTOM_KEYS)} (valid: callable, data_files, when_behavior, when_tags)")
        if not cfg.get("callable"):
            raise ValueError(f"{where}: missing required setting 'callable' (e.g. callable: my_judges.py:check)")
        data_file_paths(cfg.get("data_files"), base_dir, where)
    elif jtype == "llm":
        for key in ("model", "rubric"):
            if not cfg.get(key):
                raise ValueError(f"{where}: missing required setting {key!r}" + (" (e.g. model: claude-sonnet-5-5)" if key == "model" else " (a rubric file or an inline rubric)"))
    else:
        check_settings(cfg, HEURISTICS[jtype], where)


def build_judges(judge_cfgs: list[dict], base_dir: str | Path, prices: PriceTable, cases: list[Case] | None = None) -> list[Judge]:
    if judge_cfgs is None:
        judge_cfgs = []
    if not isinstance(judge_cfgs, list):
        raise ValueError(f"'judges' must be a list, got {type(judge_cfgs).__name__}")
    judges: list[Judge] = []
    names: set[str] = set()
    for raw in judge_cfgs:
        jtype, name, cfg = _split_config(raw)
        if name in names:
            raise ValueError(f"duplicate judge name {name!r}")
        names.add(name)
        _check_keys(jtype, name, cfg, base_dir)
        if jtype == "llm":
            judges.append(LLMJudge.from_config(cfg, base_dir=base_dir, prices=prices))  # its ValueErrors are config errors: not swallowed
            continue
        try:
            if jtype == "custom":
                judges.append(CustomJudge.from_config(cfg, base_dir=base_dir))
            else:
                judges.append(HEURISTICS[jtype].from_config(cfg))
        except (ValueError, re.error) as exc:
            raise ValueError(f"judge {name!r} ({jtype}): {exc}") from exc
    if cases and any(c.checks for c in cases) and "case_checks" not in names:
        judges.append(CaseChecksJudge(name="case_checks"))
    return judges


def validate_pass_rule(rule, judge_names) -> None:
    """A `pass_rule` naming a judge that does not exist would silently make every case fail."""
    if rule in ("all", "any"):
        return
    names = [rule] if isinstance(rule, str) else rule
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"pass_rule must be 'all', 'any', a judge name or a list of judge names, got {rule!r}")
    known = list(judge_names)
    for n in names:
        if n not in known:
            raise ValueError(f"pass_rule names judge {n!r}, which is not configured{suggest(n, known)} (judges: {', '.join(known) or 'none'})")


def run_judges(judges: list[Judge], case: Case, output: TargetOutput) -> list[JudgeResult]:
    results = []
    for judge in judges:
        if not judge.applies(case):
            continue
        if output.error:
            results.append(JudgeResult(judge=judge.name, passed=False, reasoning="target error", error=output.error))
            continue
        try:
            results.append(judge.evaluate(case, output))
        except Exception as exc:  # noqa: BLE001
            # An exception that carries `.usage` / `.cost_usd` (a judge that spent tokens before giving up)
            # keeps that spend on the errored result: failed calls are still billed.
            usage = getattr(exc, "usage", None)
            cost = getattr(exc, "cost_usd", None)
            results.append(
                JudgeResult(
                    judge=judge.name,
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                    usage=usage if isinstance(usage, Usage) else Usage(),
                    cost_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0,
                )
            )
    return results
