"""Ledgerly support bot: the system under test in the Eval Sprint portfolio demo.

A deliberately ordinary RAG bot, small enough to read in five minutes:

    question -> BM25 retrieval over help-center sections -> prompt -> LLM -> answer

It is a stand-in for "the LLM feature a client shipped". The eval harness calls
`answer(input, params)` and never looks inside. Everything the eval varies lives
in `params`:

    prompt      "v1" | "v2"  -> prompts/<prompt>.txt (the system prompt)
    top_k       how many KB sections to put in the context (default 3)
    model       model id sent to the provider
    provider    "mock" (offline simulator, default) | "anthropic" | "openai"
    max_tokens  output cap (default 400)
    temperature optional; left to the provider default when unset
    sim_noise   mock only: chance that a simulated sample is sloppy (default 0.03, 0 = every attempt identical)

The harness adds `_attempt` (0-based) to `params` when a case is repeated. With a real model it is ignored (the
model is what varies between attempts); the simulator uses it so repeated attempts can differ.

With provider "mock" the completion comes from `sim.py:bot_responder`, a
simulator with deterministic, hash-drawn behaviour. It is NOT a language model. See sim.py.
"""

from __future__ import annotations

import functools
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evalsprint.loader import load_callable
from evalsprint.providers import get_provider
from evalsprint.targets import from_completion

HERE = Path(__file__).resolve().parent
KB_DIR = HERE / "kb"
PROMPT_DIR = HERE / "prompts"

DEFAULT_SIM_NOISE = 0.03  # keep equal to sim.DEFAULT_NOISE (tests/test_demo.py checks it)

BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does did for from had has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to us was we were what when
    where which who why will with would you your yours am been being get got just also please hi hello thanks
    thank should may might about any some""".split()
)


# ----------------------------------------------------------------------------- text
def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokens, stopwords removed, light plural stripping."""
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(_singular(raw))
    return tokens


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# ----------------------------------------------------------------------------- knowledge base
@dataclass(frozen=True)
class Chunk:
    id: str  # "<article-file-stem>#<section-slug>"
    article: str  # article title, e.g. "Refunds and cancellations"
    section: str  # section heading, e.g. "Annual plans"
    body: str

    @property
    def text(self) -> str:
        """Chunk text as shown to the LLM. It carries the article title so the
        model can cite it."""
        return f"{self.article} > {self.section}\n{self.body}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    """One chunk per `##` section of every article in kb/*.md."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        title = path.stem
        section: str | None = None
        body: list[str] = []

        def flush() -> None:
            if section is not None and any(line.strip() for line in body):
                chunks.append(Chunk(f"{path.stem}#{_slug(section)}", title, section, "\n".join(body).strip()))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, body = line[3:].strip(), []
            elif section is not None:
                body.append(line)
        flush()
    return chunks


class BM25Index:
    """Okapi BM25 over section chunks (title + heading + body are indexed)."""

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.article} {c.section} {c.body}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        doc, length = self.docs[i], self.lengths[i]
        total = 0.0
        for t in set(query_tokens):
            tf = doc.get(t, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[t] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        ranked = sorted(((self.score(q, i), i) for i in range(len(self.chunks))), key=lambda p: (-p[0], p[1]))
        return [(self.chunks[i], s) for s, i in ranked[: max(0, top_k)] if s > 0]


_index: BM25Index | None = None
_index_lock = threading.Lock()


def get_index() -> BM25Index:
    global _index
    with _index_lock:
        if _index is None:
            _index = BM25Index(load_chunks())
        return _index


# ----------------------------------------------------------------------------- the target
def build_user_message(question: str, chunks: list[Chunk]) -> str:
    blocks = "\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks) or "(no matching articles)"
    return f"<context>\n{blocks}\n</context>\n\n<question>\n{question}\n</question>"


def answer(input, params: dict) -> dict:
    """Target entry point: `fn(input, params) -> dict` (see evalsprint/targets.py)."""
    question = input.get("question", "") if isinstance(input, dict) else str(input)
    prompt_version = str(params.get("prompt", "v1"))
    top_k = int(params.get("top_k", 3))
    model = params.get("model") or "claude-sonnet-5-5"
    provider_name = params.get("provider") or "mock"

    system = (PROMPT_DIR / f"{prompt_version}.txt").read_text(encoding="utf-8").strip()
    hits = get_index().search(question, top_k=top_k)
    chunks = [c for c, _ in hits]

    options: dict = {}
    if provider_name == "mock":
        # The simulator gets the attempt number and the noise level; real providers get neither.
        simulator = load_callable("sim.py:bot_responder", str(HERE))
        options["responder"] = functools.partial(
            simulator, attempt=int(params.get("_attempt") or 0), noise=float(params.get("sim_noise", DEFAULT_SIM_NOISE))
        )
    provider = get_provider(provider_name, **options)
    completion = provider.complete(
        system,
        [{"role": "user", "content": build_user_message(question, chunks)}],
        model,
        max_tokens=int(params.get("max_tokens", 400)),
        temperature=params.get("temperature"),
    )
    return from_completion(
        completion,
        meta={
            "context": [c.text for c in chunks],
            "retrieved": [{"id": c.id, "score": round(s, 3)} for c, s in hits],
            "prompt_version": prompt_version,
            "simulated": provider_name == "mock",  # True: the answer came from sim.py, not a language model
        },
    )
