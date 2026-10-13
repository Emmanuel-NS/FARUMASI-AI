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


# One conversation from testing the live app; later turns depend on earlier ones.
CONVERSATION = [
    "hey",
    "i want more about after sex pills",
    "i need more and examples of those pills",
    "okay, which pills/ drugs of before sex to get best expireience",
    "what do they do?",
    "What are the side effects of metformin?",
    "and how should I take it?",
]


def main() -> None:
    chat = sys.argv[1:] == ["--chat"]
    questions = CONVERSATION if chat else (sys.argv[1:] or QUESTIONS)
    assistant = PharmacistAssistant()
    earlier: list[str] = []
    for q in questions:
        r = assistant.answer(q, earlier if chat else ())
        earlier.append(q)
        print(f"Q: {q}\n   intent={r.intent} ({r.intent_confidence:.2f}) risk={r.risk_level}")
        if r.interpreted and r.interpreted != q:
            print(f"   understood as: {r.interpreted}")
        for s in r.sources:
            print(f"   [{s['score']:.3f}] {s['title']} ({s['source']})")
        print(f"A: {r.answer}\n")


if __name__ == "__main__":
    main()
