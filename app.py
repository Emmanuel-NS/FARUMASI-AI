"""Gradio web app for the FARUMASI Pharmacist Assistant (deployed on Hugging Face Spaces)."""
from __future__ import annotations

import gradio as gr

from pharmqa.pipeline import PharmacistAssistant

assistant = PharmacistAssistant()

EXAMPLES = [
    "What is paracetamol used for?",
    "How many ibuprofen can I take in a day?",
    "What are the side effects of metformin?",
    "Should Coartem be taken with food?",
    "I have a headache since morning, what can I take?",
    "How do I clean a small wound from a fall?",
    "Is it safe to take ibuprofen while breastfeeding?",
    "My son swallowed a bottle of paracetamol",
    "What is the exchange rate of the Rwandan franc?",
]

INTENT_NAMES = {"medicine_info": "Medicine information", "condition_advice": "Symptom / condition advice",
                "out_of_domain": "Outside pharmacy scope"}
RISK_NAMES = {"routine": "Routine", "prescription": "Prescription-only medicine",
              "special_group": "Special population", "emergency": "Possible emergency"}


def _source_label(s: dict) -> str:
    return f"[{s['source']}]({s['url']})" if s.get("url") else s["source"]


def respond(message: str, history: list[dict]):
    r = assistant.answer(message)
    parts = []
    if r.risk_level == "emergency":
        parts.append(f"**{r.warnings[0]}**")
    parts.append(r.answer)
    rest = r.warnings[1:] if r.risk_level == "emergency" else r.warnings
    if rest:
        parts.append("\n".join(f"> {w}" for w in rest))
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "\n\n".join(parts)}]

    details = (
        f"**Predicted intent:** {INTENT_NAMES[r.intent]} ({r.intent_confidence:.0%} confidence)  \n"
        f"**Risk level:** {RISK_NAMES[r.risk_level]}  \n"
        f"**Models used:** intent = `{r.components['intent_model']}`, retriever = `{r.components['retriever']}`, "
        f"generator = `{r.components['generator']}`"
    )
    sources = "\n\n".join(
        f"**{i}. {s['title']}** ({_source_label(s)}, similarity {s['score']})  \n{s['text'][:400]}..."
        for i, s in enumerate(r.sources, 1)
    ) or "_No passages retrieved (question was outside the pharmacy scope)._"
    return history, "", details, sources


with gr.Blocks(title="FARUMASI Pharmacist Assistant", theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown(
        "# FARUMASI Pharmacist Assistant\n"
        "Ask about medicines (uses, dosage, side effects, interactions) or minor health problems such as "
        "headaches, wounds or diarrhoea. Answers come from trusted sources (Rwanda FDA patient information leaflets, "
        "US FDA drug labels, NIH MedlinePlus, MedicationQA). **This is a student research prototype and not a substitute for a pharmacist or doctor.** "
        "In an emergency in Rwanda call **912**."
    )
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(type="messages", height=460, label="Conversation")
            box = gr.Textbox(placeholder="e.g. What are the side effects of amoxicillin?", label="Your question")
            with gr.Row():
                send = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            with gr.Accordion("How this answer was produced", open=True):
                details = gr.Markdown()
            with gr.Accordion("Retrieved sources", open=True):
                sources = gr.Markdown()

    send.click(respond, [box, chat], [chat, box, details, sources])
    box.submit(respond, [box, chat], [chat, box, details, sources])
    clear.click(lambda: ([], "", "", ""), None, [chat, box, details, sources])

if __name__ == "__main__":
    demo.launch()
