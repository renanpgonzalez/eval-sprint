"""Token pricing and cost computation."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import yaml

from .types import Usage

_DEFAULT_FILE = Path(__file__).with_name("pricing.yaml")
_warned: set[str] = set()


@lru_cache(maxsize=1)
def _defaults() -> dict[str, dict]:
    with open(_DEFAULT_FILE, encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh)["models"])


class PriceTable:
    def __init__(self, overrides: dict[str, dict] | None = None):
        self.models = dict(_defaults())
        for model, price in (overrides or {}).items():
            self.models[model] = {**self.models.get(model, {}), **price}

    def lookup(self, model: str | None) -> dict | None:
        if not model:
            return None
        if model in self.models:
            return self.models[model]
        candidates = [m for m in self.models if model.startswith(m)]
        if candidates:
            return self.models[max(candidates, key=len)]
        return None

    def cost(self, model: str | None, usage: Usage) -> float | None:
        """Cost in USD, or None when the model has no known price.

        `usage.input_tokens` is the *uncached* input. Providers normalise to this
        convention (Anthropic reports it that way; OpenAI includes cached tokens
        in prompt_tokens, so the OpenAI provider subtracts them).
        """
        price = self.lookup(model)
        if price is None:
            if model and model not in _warned:
                _warned.add(model)
                print(f"[evalsprint] warning: no price for model '{model}', cost left empty", file=sys.stderr)
            return None
        inp = price.get("input", 0.0)
        cost = (
            usage.input_tokens * inp
            + usage.output_tokens * price.get("output", 0.0)
            + usage.cache_read_tokens * price.get("cache_read", inp)
            + usage.cache_write_tokens * price.get("cache_write", inp)
        ) / 1_000_000
        if price.get("batch"):
            cost *= 0.5
        return cost
