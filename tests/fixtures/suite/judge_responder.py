"""Deterministic fake judge model for the mock provider.

Rules (so tests can predict the verdict from the target text):
  grounded  -> pass unless the output says "cannot help" or "don't know"
  polite    -> pass unless the output contains "!!!"   (non-critical in rubric.yaml)
"""

import json
import re


def _ai_output(messages):
    """The graded text. The judge wraps it in delimiters that carry a per-call nonce (<ai_output_7f3a9c>)."""
    match = re.search(r"<ai_output(?:_[0-9a-f]+)?>\n?(.*?)\n?</ai_output(?:_[0-9a-f]+)?>", messages[-1]["content"], re.S)
    return match.group(1).strip()


def judge_responder(system, messages, model):
    text = _ai_output(messages).lower()
    grounded = "cannot help" not in text and "don't know" not in text
    polite = "!!!" not in text
    payload = {
        "criteria": {
            "grounded": {"reasoning": "checked against the reference", "pass": grounded},
            "polite": {"reasoning": "tone check", "pass": polite},
        }
    }
    return {"text": json.dumps(payload), "usage": {"input_tokens": 400, "output_tokens": 60}}
