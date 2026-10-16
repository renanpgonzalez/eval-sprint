"""System-under-test adapters.

A target is any Python callable with the signature

    def my_target(input, params: dict) -> str | dict

Return a plain string, or a dict with any of:
    text (required), model, usage {input_tokens, output_tokens, cache_read_tokens,
    cache_write_tokens}, cost_usd, latency_s, meta (e.g. retrieved context).

The target never sees the reference answer or tags - only the case input.
Wrap an HTTP endpoint, a LangChain chain, or a function from the client's repo.
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .loader import load_callable
from .pricing import PriceTable
from .providers import Completion
from .types import Case, TargetOutput, Usage, sanitize_obj, sanitize_text

_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent).replace("\\", "/")
_FILE_LINE = re.compile(r'File "([^"]+)"')


def _forward(path: str) -> str:
    return path.replace("\\", "/")


def relativize_path(path: str, base_dir: str | Path | None = None) -> str:
    """A path safe to commit: relative to the eval.yaml directory when inside it, `evalsprint/...` for the kit,
    `<site-packages>/...` / `<stdlib>/...` for libraries, otherwise just `.../<file name>`.
    Handles Windows paths too (`C:\\Users\\me\\proj\\bot.py` -> `bot.py`)."""
    p = _forward(path)
    if not (p.startswith("/") or re.match(r"[A-Za-z]:/", p)):  # already relative (or <frozen importlib._bootstrap>, <string>): idempotent
        return path
    base = _forward(str(base_dir)).rstrip("/") if base_dir else ""
    if base and (p == base or p.lower().startswith(base.lower() + "/")):
        return p[len(base) + 1 :] or "."
    if _PACKAGE_PARENT and p.startswith(_PACKAGE_PARENT + "/"):
        return p[len(_PACKAGE_PARENT) + 1 :]
    for marker in ("site-packages/", "dist-packages/"):
        if marker in p:
            return "<site-packages>/" + p.split(marker, 1)[1]
    match = re.search(r"/lib/python3\.\d+/(.+)$", p)
    if match:
        return "<stdlib>/" + match.group(1)
    return ".../" + p.rsplit("/", 1)[-1]


def scrub_text(text: str, base_dir: str | Path | None = None) -> str:
    """Remove the local user's directories from free text (error messages, tracebacks)."""
    if not text:
        return text
    out = _FILE_LINE.sub(lambda m: f'File "{relativize_path(m.group(1), base_dir)}"', text)
    for root, repl in ((str(base_dir) if base_dir else "", "."), (str(Path.home()), "~")):
        if len(root) > 3:  # never replace "/" or "C:\\"
            out = out.replace(root, repl)
            out = out.replace(_forward(root), repl)
    return out


def trimmed_traceback(exc: BaseException, base_dir: str | Path | None = None, limit: int = 5) -> str:
    """The last `limit` frames of `exc` (without the kit's own calling frame), with repo-relative paths."""
    tb = exc.__traceback__
    tb = tb.tb_next if tb is not None and tb.tb_next is not None else tb  # skip Target._call itself
    return scrub_text("".join(traceback.format_exception(type(exc), exc, tb, limit=-limit)), base_dir)


def from_completion(completion: Completion, meta: dict | None = None) -> dict:
    """Helper for target authors: turn a provider Completion into a target result."""
    return {
        "text": completion.text,
        "model": completion.model,
        "usage": asdict(completion.usage),
        "latency_s": completion.latency_s or None,
        "meta": meta or {},
    }


class Target:
    def __init__(self, fn: Callable, params: dict | None = None, prices: PriceTable | None = None, base_dir: str | Path | None = None):
        self.fn = fn
        self.params = dict(params or {})
        self.prices = prices or PriceTable()
        # resolved like load_callable resolves it: the traceback names the callable's file by its resolved path, so a base
        # directory given as "." or with a Windows 8.3 short name (C:\Users\RENANG~1\...) or through a symlink would
        # match nothing and the absolute path would reach the committed baseline
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    @classmethod
    def from_config(cls, target_cfg: dict, base_dir, prices: PriceTable) -> "Target":
        return cls(load_callable(target_cfg["callable"], base_dir), target_cfg.get("params"), prices, base_dir=base_dir)

    def __call__(self, case: Case, attempt: int | None = None) -> TargetOutput:
        """Call the system under test. For a repeated case the runner passes the 0-based `attempt`,
        which reaches the callable as `params["_attempt"]` (targets may ignore it)."""
        return self._clean(self._call(case, attempt))

    def _clean(self, out: TargetOutput) -> TargetOutput:
        """Whatever the client's code returned must be storable: lone surrogates become U+FFFD and
        local paths are scrubbed from errors and tracebacks (they end up in the committed baseline)."""
        out.text = sanitize_text(out.text)
        out.error = scrub_text(sanitize_text(out.error), self.base_dir) if out.error else out.error
        out.meta = sanitize_obj(out.meta)
        if isinstance(out.meta.get("traceback"), str):
            out.meta["traceback"] = scrub_text(out.meta["traceback"], self.base_dir)
        if isinstance(out.model, str):
            out.model = sanitize_text(out.model)
        return out

    def _call(self, case: Case, attempt: int | None) -> TargetOutput:
        t0 = time.perf_counter()
        params = dict(self.params)
        if attempt is not None:
            params["_attempt"] = attempt
        try:
            raw: Any = self.fn(case.input, params)
        except Exception as exc:  # noqa: BLE001 - we record any failure of the system under test
            return TargetOutput(
                text="",
                latency_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
                meta={"traceback": trimmed_traceback(exc, self.base_dir)},
            )
        wall = time.perf_counter() - t0
        if isinstance(raw, str):
            raw = {"text": raw}
        if not isinstance(raw, dict) or "text" not in raw:
            return TargetOutput(text="", latency_s=wall, error=f"target returned {type(raw).__name__} without 'text'")
        try:  # a malformed result from the adapter must fail this case, not crash the whole run
            usage = Usage.from_dict(raw.get("usage"))
            model = raw.get("model") or self.params.get("model")
            if model is not None and not isinstance(model, str):
                raise TypeError(f"'model' must be a string, got {type(model).__name__}")
            cost = raw.get("cost_usd")
            if cost is not None:
                cost = float(cost)
            elif usage.input_tokens or usage.output_tokens:
                cost = self.prices.cost(model, usage)
            latency = raw.get("latency_s")
            return TargetOutput(
                text=str(raw.get("text") or ""),
                model=model,
                usage=usage,
                cost_usd=cost,
                latency_s=float(latency) if latency is not None else wall,
                meta=dict(raw.get("meta") or {}),
                error=str(raw["error"]) if raw.get("error") else None,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            return TargetOutput(text="", latency_s=wall, error=f"target returned a malformed result: {type(exc).__name__}: {exc}")
