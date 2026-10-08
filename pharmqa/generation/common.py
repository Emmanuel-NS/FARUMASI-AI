"""Prompt format and context helpers shared by generator training, evaluation and the app.

The generator is a *reader*: it receives the question plus three candidate passages and must
answer from the relevant one. During training the gold passage is mixed with two BM25 hard
negatives in random order, so the model learns passage selection as well as concise
phrasing. (Giving only the gold passage would make the task trivial copying, because the
reference answers are drawn from the gold passage.)
"""
from __future__ import annotations

import random
import re

PROMPT = (
    "You are a community pharmacist. Read the passages and answer the patient's question briefly "
    "and clearly, using only the passage that answers it. If none answers it, say you are not sure.\n"
    "Question: {question}\n{passages}\nAnswer:"
)
WORDS_PER_PASSAGE = 110
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def format_passages(passages: list[str]) -> str:
    return "\n".join(f"Passage {i}: {' '.join(p.split()[:WORDS_PER_PASSAGE])}" for i, p in enumerate(passages, 1))


def build_prompt(question: str, passages: list[str]) -> str:
    return PROMPT.format(question=question.strip(), passages=format_passages(passages))


def mix_with_negatives(gold: str, negatives: list[str], rng: random.Random) -> list[str]:
    items = [gold] + negatives
    rng.shuffle(items)
    return items


def extractive_answer(passage: str, n: int = 3) -> str:
    """No-learning baseline: return the first n sentences of a passage."""
    body = passage.split(". ", 1)[1] if ". " in passage[:120] else passage  # drop the "Title." prefix
    return " ".join(_SENT.split(body)[:n])
