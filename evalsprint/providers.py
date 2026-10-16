"""Minimal LLM provider layer (stdlib only).

Every provider returns a `Completion` with normalised token usage so cost can
be computed the same way for the system under test and for the judges.

Providers:
  anthropic  - Claude Messages API (ANTHROPIC_API_KEY)
  openai     - OpenAI Chat Completions, or any OpenAI-compatible server via
               `base_url` (OpenRouter, vLLM, Ollama, Azure-compatible gateways)
  mock       - deterministic, offline; uses a `responder` callable
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .loader import load_callable
from .types import Usage

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for offline/mock runs."""
    return max(1, len(text) // 4)


class BaseProvider:
    name = "base"

    def complete(
        self,
        system: str | None,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> Completion:
        raise NotImplementedError


class _HTTPProvider(BaseProvider):
    def __init__(self, timeout: float = 120.0, max_retries: int = 4, **_ignored):
        # _ignored: options meant for other providers (e.g. a mock `responder`)
        # so one eval.yaml can switch providers via ${LLM_PROVIDER}.
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_err = ProviderError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise last_err from exc
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2**attempt
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_err = ProviderError(f"network error calling {url}: {exc}")
                if attempt == self.max_retries:
                    raise last_err from exc
                wait = 2**attempt
            time.sleep(min(wait, 30) + random.uniform(0, 0.5))
        raise last_err or ProviderError("unreachable")


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.environ.get("EVALSPRINT_ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")

    def complete(self, system, messages, model, max_tokens=1024, temperature=None) -> Completion:
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        raw = self._post(f"{self.base_url}/v1/messages", headers, body)
        latency = time.perf_counter() - t0
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        u = raw.get("usage", {}) or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        )
        return Completion(text=text, model=raw.get("model", model), usage=usage, latency_s=latency, raw=raw)


class OpenAIProvider(_HTTPProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get(api_key_env)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not self.api_key and "api.openai.com" in self.base_url:
            raise ProviderError(f"{api_key_env} is not set")

    def complete(self, system, messages, model, max_tokens=1024, temperature=None) -> Completion:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        body: dict[str, Any] = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        t0 = time.perf_counter()
        raw = self._post(f"{self.base_url}/chat/completions", headers, body)
        latency = time.perf_counter() - t0
        choice = (raw.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = raw.get("usage", {}) or {}
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens", 0) or 0) - cached,
            output_tokens=int(u.get("completion_tokens", 0) or 0),
            cache_read_tokens=cached,
        )
        return Completion(text=text, model=raw.get("model", model), usage=usage, latency_s=latency, raw=raw)


Responder = Callable[..., Any]


class MockProvider(BaseProvider):
    """Offline provider. `responder(system=..., messages=..., model=...)` returns
    either a string or a dict with keys text / usage / latency_s."""

    name = "mock"

    def __init__(self, responder: Responder | str | None = None, base_dir: str | None = None, **_):
        if isinstance(responder, str):
            responder = load_callable(responder, base_dir)
        self.responder = responder

    def complete(self, system, messages, model, max_tokens=1024, temperature=None) -> Completion:
        if self.responder is None:
            last = messages[-1]["content"] if messages else ""
            out: Any = f"MOCK RESPONSE: {str(last)[:200]}"
        else:
            out = self.responder(system=system, messages=messages, model=model)
        if isinstance(out, str):
            out = {"text": out}
        text = out.get("text", "")
        prompt_chars = len(system or "") + sum(len(str(m.get("content", ""))) for m in messages)
        usage = Usage.from_dict(out.get("usage")) if out.get("usage") else Usage(
            input_tokens=max(1, prompt_chars // 4), output_tokens=estimate_tokens(text)
        )
        return Completion(text=text, model=model, usage=usage, latency_s=float(out.get("latency_s", 0.0)), raw=out)


_PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}
_cache: dict[tuple, BaseProvider] = {}
_lock = threading.Lock()


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    _PROVIDERS[name] = cls


def get_provider(name: str, **options) -> BaseProvider:
    """Return a (cached) provider instance. Options are provider-specific."""
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; known: {sorted(_PROVIDERS)}")
    key = (name, json.dumps(options, sort_keys=True, default=str))
    with _lock:
        if key not in _cache:
            _cache[key] = _PROVIDERS[name](**options)
        return _cache[key]
