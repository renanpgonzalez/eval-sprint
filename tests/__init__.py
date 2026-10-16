"""Shared helpers for the test suite.

Tests are stdlib `unittest` (they also run under pytest). Everything is offline:
LLM calls go to the `mock` provider or to a throwaway local HTTP server.
"""

from __future__ import annotations

from pathlib import Path

from evalsprint.runner import RunResult, summarize
from evalsprint.types import Case, TargetOutput, Usage

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "evalsprint" / "templates"  # shipped inside the package (evalsprint init reads them from there)
FIXTURES = Path(__file__).parent / "fixtures"
SUITE_DIR = FIXTURES / "suite"


def make_case(id: str = "c1", input="hello", **kw) -> Case:
    return Case(id=id, input=input, **kw)


def make_output(text: str = "hello", **kw) -> TargetOutput:
    return TargetOutput(text=text, **kw)


def make_record(
    case_id: str,
    passed: bool,
    tags: list[str] | None = None,
    cost: float | None = None,
    latency: float | None = None,
    error: str | None = None,
    judges: list[dict] | None = None,
    text: str | None = None,
    model: str | None = None,
    usage: dict | None = None,
    **case_kw,
) -> dict:
    """A result record shaped exactly like one line of results.jsonl."""
    text = f"answer to {case_id}" if text is None else text
    return {
        "case_id": case_id,
        "case": {
            "id": case_id,
            "input": f"question {case_id}",
            "reference": case_kw.get("reference"),
            "expected_behavior": case_kw.get("expected_behavior"),
            "tags": tags or [],
            "checks": [],
            "metadata": {},
        },
        "output": {
            "text": text,
            "sha": TargetOutput(text=text).sha,
            "model": model,
            "usage": usage or {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0},
            "cost_usd": cost,
            "latency_s": latency,
            "meta": {},
            "error": error,
        },
        "judges": judges
        if judges is not None
        else [{"judge": "j", "passed": passed, "score": 1.0 if passed else 0.0, "reasoning": "", "details": {}, "cost_usd": 0.0, "usage": {}, "error": None}],
        "passed": passed,
    }


def make_run(records: list[dict], run_id: str = "run-x", **summary_extra) -> RunResult:
    summary = {"run_id": run_id, "suite": "unit", "label": None, **summarize(records), **summary_extra}
    return RunResult(run_dir=None, summary=summary, results=records)


__all__ = ["ROOT", "TEMPLATES", "FIXTURES", "SUITE_DIR", "Usage", "make_case", "make_output", "make_record", "make_run"]
