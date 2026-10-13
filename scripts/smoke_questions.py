"""Run a handful of realistic patient questions through the full assistant and print what it does."""
from __future__ import annotations

import sys

from pharmqa.pipeline import PharmacistAssistant

QUESTIONS = [
    "hey",
    "thank you",
    "okay, i want to know more about after sex pills for pregnancy prevention",
    "How soon should I take the morning-after pill?",
    "What is metformin used for?",
    "What are the side effects of metformin?",
    "What is the exchange rate of the Rwandan franc?",
    "What is paracetamol used for?",
    "How many ibuprofen can I take in a day?",
    "Should Coartem be taken with food?",
    "I have a headache since morning, what can I take?",
    "Can I take panadol while pregnant?",
    "My son swallowed a bottle of paracetamol",
]


def main() -> None:
    questions = sys.argv[1:] or QUESTIONS
    assistant = PharmacistAssistant()
    for q in questions:
        r = assistant.answer(q)
        print(f"Q: {q}\n   intent={r.intent} ({r.intent_confidence:.2f}) risk={r.risk_level}")
        for s in r.sources:
            print(f"   [{s['score']:.3f}] {s['title']} ({s['source']})")
        print(f"A: {r.answer}\n")


if __name__ == "__main__":
    main()
