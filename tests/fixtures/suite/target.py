"""Deterministic fake target for tests: a canned FAQ bot with fake usage/latency."""

import time
import zlib

ANSWERS = {
    "How do I download my invoice?": "Open Settings > Billing and click Download invoice.",
    "Can I change my billing email?": "Sorry, I cannot help with that.",  # deliberately wrong
    "Where is my order?": "Use the tracking link in your confirmation email.",
    "I want a refund for last month.": "Refunds are available within 30 days of purchase.",
    "What is the weather in Paris?": "I can only help with account and order questions.",
    "How do I reset my password?": "We sent a reset link to your email.",
    "hello": "Hi! How can I help you today?",
}


def answer(input, params):
    if input == "BOOM":
        raise RuntimeError("upstream exploded")
    # Vary completion order across threads without making results non-deterministic.
    if params.get("jitter"):
        time.sleep((zlib.crc32(str(input).encode()) % 7) / 1000)
    text = ANSWERS.get(input, "I don't know.")
    prefix = params.get("prefix", "")
    return {
        "text": prefix + text,
        "model": params.get("model", "claude-haiku-4-5"),
        "usage": {"input_tokens": 100 + len(str(input)), "output_tokens": 20 + len(text) // 4},
        "latency_s": 0.1 + (zlib.crc32(str(input).encode()) % 10) / 100,
        "meta": {"context": ["Settings > Billing lets you download invoices."]},
    }
