"""Target adapters: the glue between Eval Sprint and the client's LLM feature.

A target is any function with this signature:

    def my_target(input, params: dict) -> str | dict

  input   the case's `input` field exactly as written in cases.jsonl (a string, or a dict)
  params  the `target.params` block of eval.yaml (a fresh copy on every call)

Return a plain string, or a dict with any of these keys (only `text` is required):

    text        the answer the client's users would see
    model       model id that produced it (used to look the price up)
    usage       {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"}
                input_tokens means UNCACHED input tokens; cached tokens go in cache_read_tokens
    cost_usd    total cost of the call in USD. Only needed when your feature makes several
                model calls (rewrite + retrieve + answer + rerank...): sum them yourself.
                When absent, cost is computed from `model` + `usage` with the price table
    latency_s   seconds. When absent, the wall-clock time of your function is used
    meta        anything worth keeping. meta["context"] (retrieved documents, tool results)
                is shown to the LLM judge so it can check groundedness

Rules of thumb
  * Call the feature the way production does: same prompt, same retrieval, same parameters.
    A test that passes through a different code path measures the wrong thing.
  * Never raise on purpose. If the feature fails, let the exception escape: Eval Sprint records it
    as an error for that case (counted in error_rate, case fails) and continues.
  * Do not read the reference answer here. The target only receives the case input.
  * Keep it thread-safe: `concurrency` cases run at the same time. Build clients at module level,
    not per call, and do not mutate shared state.
  * Secrets come from environment variables, never from this file.

This file ships with four adapters. Point `target.callable` in eval.yaml at the one you need,
for example `target_adapter.py:call_http`:

    offline_stub    canned answers, no network. Lets you run the whole pipeline before the
                    client's code is wired up (this is the default in the template)
    call_function   call a Python function inside the client's codebase
    call_http       call the client's HTTP endpoint with urllib
    call_llm        the feature is "a prompt + a model": call the provider directly

It also ships `offline_judge`, which is NOT a target: it is the placeholder the LLM judge uses when
you run with `JUDGE_PROVIDER=mock` (see eval.yaml). Together with `offline_stub` it lets
`evalsprint init`, `describe` and `run` work on a machine with no API key and no client code. That
proves the pipeline runs here. It measures nothing: never commit a baseline made that way.

Tip: make sure the client's package is importable (`pip install -e ../client-repo`, or add it
to PYTHONPATH) before running `evalsprint`.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _question(input) -> str:
    """Most features take a question. Accept both `"text"` and `{"question": "text", ...}`."""
    return input["question"] if isinstance(input, dict) else str(input)


# -----------------------------------------------------------------------------
# 1. Offline stub: keeps the template runnable end to end without keys or client code.
# -----------------------------------------------------------------------------
def offline_stub(input, params):
    text = "Open Settings > Billing and click Download next to the invoice."
    if "?" not in _question(input) and len(_question(input)) < 12:
        text = "Could you tell me a bit more about what you need?"
    return {
        "text": text,
        "model": params.get("model"),
        "usage": {"input_tokens": 220, "output_tokens": 24},
        "meta": {"context": ["Help center: Settings > Billing lists every invoice with a Download button."]},
    }


# -----------------------------------------------------------------------------
# The placeholder LLM judge for offline runs (JUDGE_PROVIDER=mock).
#
# eval.yaml hands this function to the `mock` provider as its `responder`; the real providers
# (anthropic, openai) ignore it, so a run that uses a real judge never reaches this code and a CI
# gate can never silently be graded by it. It is called as offline_judge(system=..., messages=...,
# model=...) and must return the judge's reply: JSON with one verdict per rubric criterion.
#
# It reads the criterion ids from the judge's system prompt (one "- <id> [critical]: <question>"
# line per criterion) and passes every one of them, unless the answer under test is empty. It does
# NOT look at quality. The reasoning text says so, so nobody mistakes the output for a measurement.
# -----------------------------------------------------------------------------
_CRITERION_LINE = re.compile(r"^- ([A-Za-z0-9_.\-]+)(?: \[critical\])?: ", re.MULTILINE)
_AI_OUTPUT = re.compile(r"<ai_output_[0-9a-f]+>\n?(.*?)\n?</ai_output_[0-9a-f]+>", re.DOTALL)


def offline_judge(system="", messages=(), model=None):
    user = str(messages[-1].get("content", "")) if messages else ""
    match = _AI_OUTPUT.search(user)
    answered = bool(match and match.group(1).strip())
    reasoning = "offline placeholder judge: not a measurement"
    criteria = {cid: {"reasoning": reasoning, "pass": answered} for cid in _CRITERION_LINE.findall(system or "")}
    return json.dumps({"criteria": criteria})


# -----------------------------------------------------------------------------
# 2. Call a function in the client's code base.
#
# Use this when the feature is a Python function (or a chain, an agent, a service class...).
# Adapt the import and the return handling. The three things to get right:
#   - call the PRODUCTION entry point (not a re-implementation),
#   - expose retrieved context through meta["context"],
#   - return real token usage so cost is measured, not guessed.
# -----------------------------------------------------------------------------
def call_function(input, params):
    from my_app.support import answer_question  # <- the client's code; adapt this import

    result = answer_question(
        _question(input),
        prompt_version=params.get("prompt", "current"),
        top_k=params.get("top_k", 4),
    )
    # Adapt the attribute names below to what the client's function returns.
    return {
        "text": result.answer,
        "model": result.model,
        "usage": usage_from_anthropic(result.usage),  # or usage_from_openai(result.usage), see helpers below
        "meta": {"context": [doc.text for doc in result.documents]},
    }


# -----------------------------------------------------------------------------
# 3. Call the client's HTTP endpoint with nothing but the standard library.
#
# params used (all optional except url):
#   url          endpoint, e.g. "${TARGET_URL}" in eval.yaml so staging/prod can be switched
#   api_key_env  name of the env var that holds the bearer token (default TARGET_API_KEY)
#   question_key JSON field for the question in the request body     (default "question")
#   answer_key   JSON field of the answer in the response            (default "answer")
#   timeout_s    per-request timeout                                 (default 60)
#
# Point it at staging, never at production, and check with the client how much
# traffic the endpoint tolerates before raising `concurrency`.
# -----------------------------------------------------------------------------
def call_http(input, params):
    url = params["url"]
    body = {params.get("question_key", "question"): _question(input)}
    if isinstance(input, dict):  # forward extra structured fields (user, locale, history...)
        body.update({k: v for k, v in input.items() if k != "question"})
    headers = {"content-type": "application/json"}
    token = os.environ.get(params.get("api_key_env", "TARGET_API_KEY"))
    if token:
        headers["authorization"] = f"Bearer {token}"

    data = json.dumps(body).encode("utf-8")
    attempts = 3
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=params.get("timeout_s", 60)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2**attempt)  # be gentle with the client's staging environment
                continue
            raise  # recorded as an error for this case

    answer = payload[params.get("answer_key", "answer")]
    return {
        "text": answer,
        "model": payload.get("model"),
        # If the endpoint reports usage, pass it through so cost is real. Adapt the key names.
        "usage": payload.get("usage") or {},
        # If it returns its sources, pass them to the judge: {"sources": [{"text": "..."}]}
        "meta": {"context": [s.get("text", "") for s in payload.get("sources", [])]},
    }


# -----------------------------------------------------------------------------
# 4. The feature is "a prompt + a model": call the provider directly.
#
# Handy when the client's feature is a single LLM call and you want to evaluate a prompt change
# without touching their repo. params used:
#   provider     anthropic | openai | mock             (default anthropic)
#   model        model id                              (required)
#   prompt_file  system prompt, relative to this file  (default prompts/system.txt)
#   max_tokens   output cap                            (default 600)
#   temperature  optional
# -----------------------------------------------------------------------------
def call_llm(input, params):
    from evalsprint.providers import get_provider
    from evalsprint.targets import from_completion

    system = (HERE / params.get("prompt_file", "prompts/system.txt")).read_text(encoding="utf-8")
    provider = get_provider(params.get("provider", "anthropic"))
    completion = provider.complete(
        system,
        [{"role": "user", "content": _question(input)}],
        params["model"],
        max_tokens=params.get("max_tokens", 600),
        temperature=params.get("temperature"),
    )
    return from_completion(completion)  # text + model + normalised usage + latency


# -----------------------------------------------------------------------------
# Helpers for reporting usage from vendor SDK objects. The price table needs
# UNCACHED input tokens in `input_tokens` and cached tokens separately.
# -----------------------------------------------------------------------------
def usage_from_anthropic(usage) -> dict:
    """Anthropic SDK `response.usage` (or the raw dict). input_tokens already excludes cached tokens."""
    get = usage.get if isinstance(usage, dict) else lambda k, default=0: getattr(usage, k, default)
    return {
        "input_tokens": get("input_tokens", 0) or 0,
        "output_tokens": get("output_tokens", 0) or 0,
        "cache_read_tokens": get("cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": get("cache_creation_input_tokens", 0) or 0,
    }


def usage_from_openai(usage) -> dict:
    """OpenAI SDK `response.usage` (or the raw dict). prompt_tokens INCLUDES cached tokens: subtract them."""
    if not isinstance(usage, dict):
        details = getattr(usage, "prompt_tokens_details", None)
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": getattr(details, "cached_tokens", 0) if details else 0},
        }
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return {
        "input_tokens": (usage.get("prompt_tokens", 0) or 0) - cached,
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cache_read_tokens": cached,
    }
