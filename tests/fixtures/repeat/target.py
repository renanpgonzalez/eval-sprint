"""Deterministic target for the repeat tests: what it does depends on params["_attempt"].

SCRIPTS maps an input to the behaviour of attempt 0, 1, 2, ... (the list wraps around):
  good  -> answers "good answer"          (passes the `contains good` judge)
  bad   -> answers "bad answer"           (fails it)
  boom  -> raises, an error outcome
A case that is not repeated gets no `_attempt`; it behaves as attempt 0.
"""

SCRIPTS = {
    "always-good": ["good"],
    "always-bad": ["bad"],
    "mostly-good": ["good", "bad", "good"],  # 3 attempts: pass (2 of 3)
    "late-good": ["bad", "good", "good"],  # majority pass, first passing attempt is #1
    "mostly-bad": ["bad", "good", "bad"],  # 3 attempts: fail
    "tie": ["good", "bad"],  # 2 attempts (or 4): tie -> fail
    "one-error": ["good", "good", "boom"],  # 3 attempts: 1 error < half -> pass
    "mostly-error": ["good", "boom", "boom"],  # 3 attempts: 2 errors >= half -> error
    "half-error": ["good", "boom"],  # 2 attempts: 1 error >= half -> error
}


def answer(input, params):
    script = SCRIPTS.get(input, ["good"])
    attempt = params.get("_attempt")
    step = script[(attempt or 0) % len(script)]
    if step == "boom":
        raise RuntimeError("upstream timeout")
    return {
        "text": "good answer" if step == "good" else "bad answer",
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "latency_s": 0.1,
        "meta": {"attempt": attempt},
    }
