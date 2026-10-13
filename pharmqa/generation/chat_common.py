"""Prompt format and training targets for the chat reader (Qwen2.5-Instruct + LoRA).

The Flan-T5 reader was trained to copy the first sentences of the right passage, so its answers
start mid-leaflet ("Use For women to ..."). The chat reader is a decoder-only instruction model
fine-tuned to answer like a pharmacist: one direct opening sentence, then the details from the
passages, and an explicit "my sources don't cover this" when no passage answers the question.

Training targets are built from the same grounded QA pairs (no external LLM): the opening sentence
comes from the question type (e.g. side effects -> "Metformin can cause side effects."), the body is
the cleaned gold passage, cut at a sentence boundary. Three kinds of examples are mixed:

  answer      question + gold passage + 2 hard negatives -> opener + gold passage body
  overview    "tell me everything about X" + 3-5 sections of X -> headed multi-section answer
  no answer   question + 3 unrelated passages -> "my sources don't cover this" (teaches abstaining)
"""
from __future__ import annotations

import random
import re

from pharmqa.query import clean_passage

SYSTEM_PROMPT = (
    "You are FARUMASI's pharmacy assistant for patients in Rwanda. Answer only from the numbered passages. "
    "Start with one clear sentence that directly answers the question, then give the important details "
    "(doses, warnings, who should not take it) in plain language. If the passages do not answer the "
    "question, say that your sources do not cover it and suggest asking a pharmacist or doctor. "
    "Never invent doses, brands or facts that are not in the passages."
)
NO_ANSWER = (
    "My sources don't cover this question, so I can't answer it reliably. Please ask a pharmacist or "
    "doctor, who can advise you based on your situation."
)
WORDS_PER_PASSAGE = 120
BODY_WORDS = 130

_SENT_END = re.compile(r"(?<=[.!?])\s+")

# Opening sentence per question type; {d} = medicine or condition name.
OPENERS = {
    "indications_and_usage": "{D} is used for the following.", "uses": "{D} is used for the following.",
    "indication": "{D} is used for the following.", "purpose": "{D} is used for the following.",
    "dosage_and_administration": "Here is how {d} is usually taken, according to its official leaflet.",
    "dosage": "Here is how {d} is usually taken, according to its official leaflet.",
    "dose": "Here is the dosing information for {d}.", "usage": "Here is how {d} should be used.",
    "directions": "Here is how {d} should be used.",
    "adverse_reactions": "{D} can cause side effects.", "side_effects": "{D} can cause side effects.",
    "side effects": "{D} can cause side effects.",
    "contraindications": "Some people should not take {d}.", "do_not_use": "Some people should not use {d}.",
    "before": "Before taking {d}, check the following.",
    "warnings": "There are important warnings for {d}.", "warnings_and_cautions": "There are important warnings for {d}.",
    "boxed_warning": "{D} carries a serious (boxed) warning.",
    "drug_interactions": "{D} can interact with other medicines.", "interaction": "{D} can interact with other medicines.",
    "overdosage": "Taking too much {d} can be dangerous.",
    "pregnancy": "Here is what is known about {d} in pregnancy.",
    "pregnancy_or_breast_feeding": "Here is the advice on {d} during pregnancy and breastfeeding.",
    "use_in_specific_populations": "{D} needs extra care in some groups of people.",
    "information_for_patients": "Here is what patients taking {d} should know.",
    "storage": "Here is how to store {d}.", "storage_and_handling": "Here is how to store {d}.",
    "treatment": "Here is how {d} is treated.", "symptoms": "These are the usual symptoms of {d}.",
    "causes": "These are the main causes of {d}.", "prevention": "Here is how {d} can be prevented.",
    "exams and tests": "Here is how {d} is diagnosed.", "outlook": "Here is the usual outlook for {d}.",
}

OVERVIEW_SECTIONS = [
    ("What it is for", ("indications_and_usage", "purpose", "uses")),
    ("How to take it", ("dosage_and_administration", "directions", "dosage")),
    ("Who should not take it", ("contraindications", "do_not_use", "before")),
    ("Possible side effects", ("adverse_reactions", "side_effects")),
    ("Warnings", ("warnings", "warnings_and_cautions", "boxed_warning")),
    ("Other medicines", ("drug_interactions",)),
    ("Storage", ("storage_and_handling", "storage")),
]


def format_passages(passages: list[str]) -> str:
    return "\n".join(f"[{i}] {' '.join(clean_passage(p).split()[:WORDS_PER_PASSAGE])}" for i, p in enumerate(passages, 1))


def build_messages(question: str, passages: list[str], detailed: bool = False) -> list[dict]:
    ask = "Give a complete overview with a short heading for each topic." if detailed else ""
    user = f"Passages:\n{format_passages(passages)}\n\nQuestion: {question.strip()}\n{ask}".strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def passage_body(full: str, max_words: int = BODY_WORDS) -> str:
    """Cleaned passage text without its "Title." prefix, cut at a sentence boundary."""
    body = clean_passage(full).partition(". ")[2] or clean_passage(full)
    words = body.split()
    if len(words) <= max_words:
        return body
    cut = " ".join(words[:max_words])
    sentences = _SENT_END.split(cut)
    return " ".join(sentences[:-1]) if len(sentences) > 1 else cut + " ..."


def opener(qtype: str, focus: str) -> str:
    template = OPENERS.get(str(qtype).lower())
    if not template or not focus:
        return ""
    name = focus.strip()
    return template.format(d=name, D=name[:1].upper() + name[1:])


def answer_target(qtype: str, focus: str, gold_full: str) -> str:
    first = opener(qtype, focus)
    body = passage_body(gold_full)
    return f"{first} {body}".strip() if first else body


def overview_target(name: str, sections: list[tuple[str, str]]) -> str:
    """sections = [(heading, passage_full)] in OVERVIEW_SECTIONS order."""
    parts = [f"Here is an overview of {name}, based on its official leaflet."]
    parts += [f"**{heading}:** {passage_body(full, 70)}" for heading, full in sections]
    return "\n\n".join(parts)


def overview_question(name: str, rng: random.Random) -> str:
    return rng.choice([
        "Tell me everything about {d}.", "I need full details about {d}.", "Can you give me more information on {d}?",
        "What should I know about {d}?", "I want a detailed answer about {d}, not just one line.",
    ]).format(d=name)
