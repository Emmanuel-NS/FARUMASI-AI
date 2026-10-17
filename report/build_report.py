"""Build the summative research report PDF (reportlab).

Run from the repo root:
    python report/build_report.py
Output:
    report/Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf"

NAVY = colors.HexColor("#0F2A44")
TEAL = colors.HexColor("#1B6B5A")
LIGHT = colors.HexColor("#F4F7F6")
LINE = colors.HexColor("#C9D4CF")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "T", parent=base["Title"], fontName="Times-Bold", fontSize=16, leading=20,
            textColor=NAVY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ST", parent=base["Normal"], fontName="Times-Roman", fontSize=11, leading=14,
            alignment=TA_CENTER, textColor=colors.HexColor("#334455"), spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "M", parent=base["Normal"], fontName="Times-Italic", fontSize=10, leading=13,
            alignment=TA_CENTER, textColor=colors.HexColor("#556677"), spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=13, leading=16,
            textColor=NAVY, spaceBefore=14, spaceAfter=6, borderPadding=2,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.5, leading=14,
            textColor=TEAL, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "B", parent=base["Normal"], fontName="Times-Roman", fontSize=10, leading=13.5,
            alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "C", parent=base["Normal"], fontName="Times-Italic", fontSize=8.5, leading=11,
            alignment=TA_CENTER, textColor=colors.HexColor("#445566"), spaceBefore=2, spaceAfter=10,
        ),
        "ref": ParagraphStyle(
            "R", parent=base["Normal"], fontName="Times-Roman", fontSize=9, leading=11.5,
            leftIndent=12, firstLineIndent=-12, spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "A", parent=base["Normal"], fontName="Times-Roman", fontSize=9.5, leading=12.5,
            alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BU", parent=base["Normal"], fontName="Times-Roman", fontSize=10, leading=13,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "F", parent=base["Normal"], fontName="Times-Roman", fontSize=8,
            alignment=TA_CENTER, textColor=colors.HexColor("#667788"),
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontName="Times-Roman", fontSize=8, leading=10,
        ),
        "cellb": ParagraphStyle(
            "CellB", parent=base["Normal"], fontName="Times-Bold", fontSize=8, leading=10,
        ),
    }
    return s


def P(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", " "), style)


def bullets(items, s):
    return ListFlowable(
        [ListItem(Paragraph(i, s["bullet"]), leftIndent=8, bulletColor=TEAL) for i in items],
        bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9,
    )


def table(headers, rows, col_widths=None):
    s = styles()
    data = [[Paragraph(h, s["cellb"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["cell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#667788"))
    canvas.drawString(2 * cm, 0.9 * cm, "FARUMASI Pharmacist QA — Summative Project Sep26")
    canvas.drawRightString(A4[0] - 2 * cm, 0.9 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    s = styles()
    story = []

    # ----- Title block -----
    story.append(Spacer(1, 8 * mm))
    story.append(P(
        "FARUMASI Pharmacist Assistant: A Grounded Domain-Specific "
        "Question-Answering System for Community Pharmacy in Rwanda",
        s["title"],
    ))
    story.append(P("Emmanuel Nsabagasani", s["subtitle"]))
    story.append(P(
        "Summative Project — Natural Language Processing (Sep26)<br/>"
        "African Leadership University<br/>"
        "GitHub: https://github.com/Emmanuel-NS/FARUMASI-AI<br/>"
        "Live app: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa",
        s["meta"],
    ))

    story.append(P("<b>Abstract</b>", s["h2"]))
    story.append(P(
        "Patients in Rwanda often use community pharmacies as a first point of care, yet reliable, "
        "plain-language answers about medicines are hard to obtain online, and general-purpose chatbots "
        "can invent doses. This project builds a <b>domain-specific English question-answering (QA) "
        "system</b> for routine medicine and minor-ailment questions, grounded only in trusted passages "
        "(US openFDA labels for WHO essential medicines, MedQuAD, MedicationQA, and Rwanda FDA patient "
        "leaflets). The pipeline combines a DistilBERT intent classifier, a fine-tuned MiniLM dense "
        "retriever, and a LoRA-adapted Qwen2.5-1.5B-Instruct chat reader (with a Flan-T5-base + LoRA "
        "fallback), plus rule-based safety and query understanding. Systematic experiments against "
        "TF-IDF, BM25, zero-shot Transformers, and extractive baselines show that LoRA fine-tuning "
        "roughly doubles ROUGE-L over zero-shot Qwen (0.327 → 0.643 with gold context) while raising "
        "passage-groundedness (support) from 0.62 to 0.96 and abstention on unrelated passages from "
        "25% to 73%. The system is deployed as a Gradio web app on Hugging Face Spaces (ZeroGPU). "
        "This report documents the problem, data, methodology, experiments, evaluation, error analysis, "
        "deployment, and limitations.",
        s["abstract"],
    ))
    story.append(P(
        "<b>Keywords:</b> domain-specific QA; retrieval-augmented generation; LoRA; biomedical NLP; "
        "pharmacy; Rwanda; DistilBERT; MiniLM; Qwen2.5; Flan-T5",
        s["abstract"],
    ))

    # ----- 1. Introduction -----
    story.append(P("1. Introduction and Problem Definition", s["h1"]))
    story.append(P(
        "Community pharmacies in Rwanda are a primary access point for medicine advice, but pharmacists "
        "are time-constrained and online information is fragmented between US brand names, clinician-facing "
        "labels, and informal social media advice. A general chatbot is unsuitable for this setting: large "
        "language models (LLMs) can hallucinate doses, interactions, and contraindications [1], [2]. "
        "The research problem addressed here is therefore:",
        s["body"],
    ))
    story.append(P(
        "<i>How can we build an English-language, domain-specific QA assistant that answers routine "
        "medicine and minor-ailment questions for Rwandan patients using only trusted leaflet text, "
        "refuses out-of-scope queries, and flags risky situations—without relying on an external "
        "commercial chatbot API?</i>",
        s["body"],
    ))
    story.append(P("1.1 Task, language, and target users", s["h2"]))
    story.append(P(
        "<b>Task.</b> Extractive / generative open-domain QA over a fixed pharmacy corpus (retrieval + "
        "grounded answer generation), with an upstream 3-way intent classifier "
        "(medicine_info / condition_advice / out_of_domain) and a rule-based risk detector. "
        "This is <b>not</b> machine translation; language choice is justified by data availability "
        "and by the fact that most official medicine labels and NIH health topics used in training "
        "are English, while Kinyarwanda/French leaflets are sparse and often scanned images.",
        s["body"],
    ))
    story.append(P(
        "<b>Language.</b> English only for this version. Rwandan community pharmacy practice commonly "
        "uses English brand and generic names alongside Kinyarwanda speech; the assistant expands "
        "local lay terms and brand names (e.g. <i>panadol</i> → paracetamol, <i>coartem</i> → "
        "artemether/lumefantrine, <i>after sex pills</i> → levonorgestrel) so English retrieval "
        "still serves local phrasing. Multilingual QA is left as future work because half of the "
        "Rwanda FDA PDFs lack a text layer.",
        s["body"],
    ))
    story.append(P(
        "<b>Target users.</b> Adult patients and caregivers seeking routine information about "
        "WHO essential medicines and minor ailments, plus pharmacy students and partners of the "
        "FARUMASI digital pharmacy platform. The assistant is explicitly <b>not</b> a substitute "
        "for a pharmacist or doctor; every answer carries a disclaimer, and emergencies are directed "
        "to SAMU (912).",
        s["body"],
    ))
    story.append(P("1.2 Research questions", s["h2"]))
    story.append(bullets([
        "<b>RQ1.</b> Does DistilBERT fine-tuning outperform TF-IDF baselines on informal consumer questions?",
        "<b>RQ2.</b> Does dense MiniLM fine-tuning with in-batch negatives beat BM25 and zero-shot MiniLM on Recall@k / MRR?",
        "<b>RQ3.</b> Does LoRA fine-tuning of a reader (Flan-T5, then Qwen2.5-Instruct) improve grounded answer quality over extractive and zero-shot baselines?",
        "<b>RQ4.</b> Does including Rwanda FDA leaflet questions improve retrieval on local brands, or does US/NIH training already transfer?",
    ], s))

    # ----- 2. Related work -----
    story.append(P("2. Related Work", s["h1"]))
    story.append(P("2.1 Biomedical and consumer health QA", s["h2"]))
    story.append(P(
        "Biomedical QA has progressed from IR over curated corpora to neural readers. "
        "<b>MedQuAD</b> [3] aggregates large NIH/MedlinePlus-style QA collections and frames "
        "answering partly as question–entailment; <b>MedicationQA</b> [4] focuses specifically "
        "on real consumer medicine questions with answers drawn from trusted resources—closer "
        "to our target user phrasing than clinician notes. Earlier clinical QA often assumed "
        "expert users; consumer health QA must handle lay synonyms, incomplete questions, and "
        "higher stakes when the system is wrong [2]. Our contribution is not a new dataset "
        "alone, but a <b>closed-corpus RAG pipeline</b> adapted to Rwandan pharmacy practice "
        "(local brands, emergency contraception lay terms, SAMU referral) on top of these "
        "public resources plus Rwanda FDA leaflets [20].",
        s["body"],
    ))
    story.append(P("2.2 Retrieval and grounded generation", s["h2"]))
    story.append(P(
        "Dense bi-encoders such as Sentence-BERT and MiniLM [5], [6] map questions and passages "
        "into a shared vector space for cosine ranking; classical <b>BM25</b> [7] remains strong "
        "when rare brand tokens dominate matching—which we observe for Rwanda leaflet names. "
        "<b>Retrieval-augmented generation (RAG)</b> [11] conditions a generator on retrieved "
        "documents to reduce hallucination. Instruction-tuned generators including <b>Flan-T5</b> "
        "[8] and <b>Qwen2.5</b> [9] can be adapted with <b>LoRA</b> [10], updating a small set of "
        "low-rank matrices instead of all weights—essential on free Colab T4 hardware. Our "
        "experiments connect to this literature by showing that (i) domain LoRA roughly doubles "
        "zero-shot Qwen ROUGE-L while raising a simple groundedness proxy (support) from 0.62 "
        "to 0.96, and (ii) zero-shot instruction models paraphrase freely and under-abstain, "
        "consistent with known hallucination risks [1], [2].",
        s["body"],
    ))
    story.append(P("2.3 Scope control and safety", s["h2"]))
    story.append(P(
        "Out-of-scope detection for dialogue systems is studied in CLINC150 [13]. We treat "
        "scope as a first-class classifier before retrieval so non-health questions never "
        "reach the reader. Commercial clinical chatbots and closed APIs are deliberately "
        "avoided: the assessment rubric forbids simply wiring an existing chatbot API, and "
        "auditable grounding requires that every answer cite passages from a known corpus. "
        "Rule-based emergency overlays complement learned modules because safety actions "
        "(e.g. “call 912”) must be predictable rather than sampled [12].",
        s["body"],
    ))

    # ----- 3. Dataset -----
    story.append(P("3. Dataset", s["h1"]))
    story.append(P("3.1 Sources, size, and characteristics", s["h2"]))
    story.append(P(
        "All training and evaluation text is human-written. No LLM-generated passages were used "
        "as gold answers. The passage corpus contains <b>8,840</b> passages; the QA set contains "
        "<b>9,518</b> question–answer pairs grouped by medicine/condition so the same entity does "
        "not appear in more than one split (train 7,200 / val 1,136 / test 1,182).",
        s["body"],
    ))
    story.append(table(
        ["Dataset", "Role", "Size used", "Licence / notes"],
        [
            ["openFDA drug labels (US FDA)", "Medicine passages for ~140 WHO essential medicines",
             "2,049 passages; 2,523 templated QA", "Public domain"],
            ["MedQuAD [3]", "Condition / health-topic QA", "5,546 passages; 5,580 QA", "CC BY 4.0"],
            ["MedicationQA [4]", "Real consumer medicine questions", "655 QA", "CC BY 4.0"],
            ["Rwanda FDA Patient Information Leaflets", "Local brands (e.g. Coartem, Agycin)",
             "187 usable leaflets → 590 passages; 760 QA", "Public regulatory PDFs"],
            ["CLINC150 [13]", "Out-of-domain intent negatives", "23,700 queries (sampled)", "CC BY 3.0"],
            ["ChatDoctor-HealthCareMagic [14]", "Weakly labelled consumer questions (intent train only)",
             "3,416 questions", "Apache 2.0"],
            ["Challenge set (this project)", "56 informal consumer questions; eval only",
             "56", "Author-labelled"],
        ],
        col_widths=[3.2 * cm, 4.2 * cm, 4.0 * cm, 3.4 * cm],
    ))
    story.append(P("Table 1. Datasets used in the project.", s["caption"]))

    story.append(P("3.2 Preprocessing and quality", s["h2"]))
    story.append(bullets([
        "Label sections cleaned and split into ≤160-word passages; mean passage length ≈ 118 words.",
        "openFDA and Rwanda leaflet questions are templated over real section text (documented limitation); MedicationQA and MedQuAD questions are natural.",
        "Splits are <b>entity-grouped</b> to avoid leakage (a medicine in train cannot appear in test).",
        "Rwanda PDFs: of 416 cached files, 213 scanned images without a text layer were excluded; two-column layouts sometimes interleave columns (noise).",
        "Supplementary corrected labels for metformin (single ingredient), levonorgestrel (OTC emergency pill), and ulipristal (120-hour emergency pill) replace wrong combination-product downloads.",
        "Passage cleaning removes cross-references such as “[see Warnings…]” and US MedWatch boilerplate before the reader sees text.",
    ], s))
    story.append(P("3.3 Dataset limitations", s["h2"]))
    story.append(P(
        "Medicine answers rely heavily on FDA wording written for clinicians; MedQuAD drug answers "
        "were removed by their authors for copyright. Templated questions inflate lexical overlap "
        "metrics. US brand names and doses may differ from Rwandan practice. English-only coverage "
        "excludes many real patient utterances. These limitations motivate grounded generation, "
        "honest abstention for unknown local brands, and a strong disclaimer.",
        s["body"],
    ))

    # ----- 4. Methodology -----
    story.append(P("4. Methodology", s["h1"]))
    story.append(P(
        "Figure conceptually: question → (optional) small-talk / feedback rules → risk detector → "
        "intent classifier → query expansion (lay/brand terms) → dense retriever with medicine/section "
        "re-ranking → chat reader (Qwen + LoRA) grounded on on-topic passages → safety localisation "
        "(e.g. US poison number → 912) + disclaimer + sources.",
        s["body"],
    ))
    story.append(P("4.1 Intent / scope classifier", s["h2"]))
    story.append(P(
        "<b>Architecture.</b> DistilBERT [15] (6-layer Transformer encoder, distilled from BERT) with "
        "a linear classification head over three labels. Input: tokenised question (max length 64). "
        "Output: label + softmax confidence. "
        "<b>Baselines.</b> TF-IDF + logistic regression with word and word+char n-grams. "
        "<b>Training.</b> Fine-tuned on MedQuAD/CLINC, then on a consumer-augmented set mixing real "
        "patient questions (HealthCareMagic) with heuristic labels. "
        "<b>Why DistilBERT.</b> Course-appropriate Transformer, fits free Colab hardware, strong "
        "text classification baseline, and small enough for ZeroGPU alongside retrieval and generation.",
        s["body"],
    ))
    story.append(P("4.2 Retriever", s["h2"]))
    story.append(P(
        "<b>Architecture.</b> all-MiniLM-L6-v2 bi-encoder [6]: question and passage mapped to 384-d "
        "normalised embeddings; ranking by cosine similarity. "
        "<b>Fine-tuning.</b> Contrastive training with in-batch negatives (2 epochs, batch 64, lr 2e-5). "
        "<b>Inference.</b> Top-50 dense hits are re-ranked with boosts when the passage medicine "
        "matches a drug named in the query (+0.15) and when the section matches the asked aspect "
        "(+0.08). Stub/empty passages are masked. "
        "<b>Baselines.</b> BM25 [7]; zero-shot MiniLM; hybrid BM25+dense via Reciprocal Rank Fusion (RRF).",
        s["body"],
    ))
    story.append(P("4.3 Answer readers (Flan-T5 LoRA and Qwen LoRA)", s["h2"]))
    story.append(P(
        "<b>Flan-T5-base + LoRA.</b> Encoder–decoder instruction model [8]. LoRA adapters (rank 16) "
        "on query/value projections (~0.7% of parameters). Trained to produce short answers from "
        "a prompt containing three passages (gold + hard negatives). Hyperparameters for the "
        "reported run: 2 epochs, lr 3e-4, effective batch 16, ~25 minutes on a Colab T4.",
        s["body"],
    ))
    story.append(P(
        "<b>Qwen2.5-1.5B-Instruct + LoRA (deployed reader).</b> Decoder-only chat model [9]. "
        "LoRA (r=16, α=32, dropout 0.05) on attention and MLP projections (q/k/v/o, gate/up/down). "
        "Chat template with a system prompt forbidding invention; loss masked on prompt tokens "
        "(answer tokens only). Three example types: (i) answer with opener + passage body; "
        "(ii) multi-section overview; (iii) ~8% “my sources don’t cover this” with unrelated "
        "passages. Hyperparameters: 1 epoch, max_train 3,000, batch 2, grad_accum 8, max_len 1536, "
        "lr 2e-4, fp16, gradient checkpointing, Colab T4 (~1–2 h). "
        "At inference the model sees on-topic retrieved passages only; overviews are written "
        "per section in one batch so headings stay aligned.",
        s["body"],
    ))
    story.append(P("4.4 Query understanding and safety (non-learned)", s["h2"]))
    story.append(P(
        "Lay/brand expansion, follow-up resolution (“what do they do?” inherits the previous "
        "medicine), progressive overviews, unknown-product detection (e.g. Kopavitamin), fuzzy "
        "suggestion of close names (quarteme → Coartem), and rule-based risk levels "
        "(routine / prescription / special group / emergency) are implemented in "
        "<font face='Courier'>pharmqa/query.py</font> and <font face='Courier'>pharmqa/safety.py</font>. "
        "These components are transparent by design and were refined from live-app failures that "
        "benchmarks alone did not expose.",
        s["body"],
    ))
    story.append(P("4.5 Training and inference hyperparameters (summary)", s["h2"]))
    story.append(table(
        ["Component", "Base model", "Adaptation", "Key hyperparameters", "Hardware"],
        [
            ["Intent", "DistilBERT-base-uncased", "Full fine-tune + consumer data",
             "max_len 64; 3 labels", "Colab T4"],
            ["Retriever", "all-MiniLM-L6-v2", "Contrastive fine-tune",
             "2 epochs; batch 64; lr 2e-5; in-batch negatives", "Colab T4"],
            ["Reader (legacy)", "Flan-T5-base", "LoRA r=16 (q/v)",
             "2 epochs; lr 3e-4; eff. batch 16; max_src 512", "Colab T4 ~25 min"],
            ["Reader (deployed)", "Qwen2.5-1.5B-Instruct", "LoRA r=16 (attn+MLP)",
             "1 epoch; lr 2e-4; batch 2×8; max_len 1536; answer-token loss",
             "Colab T4 ~1–2 h"],
        ],
        col_widths=[2.8 * cm, 3.6 * cm, 3.2 * cm, 4.8 * cm, 2.4 * cm],
    ))
    story.append(P("Table 1b. Modelling choices used for the reported results and the live app.", s["caption"]))

    # ----- 5. Experiments -----
    story.append(P("5. Baseline and Experimentation", s["h1"]))
    story.append(P(
        "Each component has a meaningful baseline and controlled ablations. Experiments were run "
        "on Colab T4 GPUs; metrics are stored under <font face='Courier'>results/</font> for "
        "reproducibility.",
        s["body"],
    ))
    story.append(P("5.1 Intent experiments", s["h2"]))
    story.append(P(
        "<b>Baseline rationale.</b> TF-IDF + logistic regression is a strong classical text "
        "classifier and isolates whether Transformers help beyond bag-of-words. "
        "<b>What changed.</b> (i) word vs word+char n-grams; (ii) DistilBERT without consumer "
        "augmentation; (iii) DistilBERT with HealthCareMagic questions. "
        "<b>Why.</b> Models trained only on formal MedQuAD/CLINC text scored &gt;99% on the "
        "held-out test split but collapsed on the informal challenge set—evidence they learned "
        "<i>style</i> rather than topic. Consumer augmentation tests whether realistic phrasing "
        "closes that gap (RQ1).",
        s["body"],
    ))
    story.append(P("5.2 Retrieval experiments", s["h2"]))
    story.append(P(
        "<b>Baseline rationale.</b> BM25 is the standard sparse IR baseline; zero-shot MiniLM "
        "shows pretrained semantic matching without domain fine-tuning. "
        "<b>What changed.</b> Fine-tune MiniLM on the pharmacy QA pairs; train with vs without "
        "RwandaFDA-PIL questions; fuse BM25 with dense ranks via RRF. "
        "<b>Why.</b> RQ2–RQ4: measure gains from domain adaptation and whether local leaflets "
        "are necessary for local brands.",
        s["body"],
    ))
    story.append(P("5.3 Generation experiments", s["h2"]))
    story.append(P(
        "<b>Baselines.</b> Extractive (first sentences of top passage); Flan-T5 zero-shot; "
        "Qwen2.5-1.5B zero-shot. "
        "<b>Proposed.</b> Flan-T5 + LoRA; Qwen2.5 + LoRA. "
        "<b>Context conditions.</b> <i>Gold</i> = gold passage + 2 BM25 hard negatives (isolates "
        "the reader); <i>Retrieved</i> = top-3 BM25 (end-to-end with a weaker retriever so "
        "Flan/Qwen tables remain comparable). "
        "<b>Why.</b> RQ3: test whether LoRA teaches the model to select the right passage and "
        "phrase a grounded answer, and whether a chat-style reader improves openings and abstention.",
        s["body"],
    ))

    # ----- 6. Results -----
    story.append(P("6. Results", s["h1"]))
    story.append(P("6.1 Intent classification", s["h2"]))
    story.append(table(
        ["Model", "Test acc", "MedicationQA acc", "Challenge acc", "Challenge macro-F1"],
        [
            ["TF-IDF + LR (word)", "0.993", "0.528", "0.393", "0.372"],
            ["TF-IDF + LR (word+char)", "0.995", "0.694", "0.464", "0.420"],
            ["TF-IDF + LR + patient Qs", "0.979", "0.556", "0.768", "0.764"],
            ["DistilBERT (original)", "0.998", "0.917", "0.554", "0.486"],
            ["<b>DistilBERT + patient Qs (deployed)</b>", "<b>0.994</b>", "<b>0.889</b>",
             "<b>0.804</b>", "<b>0.782</b>"],
        ],
        col_widths=[5.5 * cm, 2.5 * cm, 3.0 * cm, 2.8 * cm, 3.0 * cm],
    ))
    story.append(P(
        "Table 2. Intent results. High formal-test accuracy is misleading; the challenge set "
        "is the realistic criterion. Consumer augmentation is the decisive factor.",
        s["caption"],
    ))

    story.append(P("6.2 Retrieval", s["h2"]))
    story.append(table(
        ["Model", "R@1", "R@5", "MRR@10", "Topic@1", "Rwanda R@1"],
        [
            ["BM25", "0.322", "0.706", "0.478", "0.852", "0.444"],
            ["MiniLM zero-shot", "0.436", "0.804", "0.589", "0.882", "0.400"],
            ["MiniLM FT (no Rwanda)", "0.694", "0.954", "0.809", "0.977", "0.667"],
            ["<b>MiniLM FT (all, deployed)</b>", "<b>0.695</b>", "<b>0.958</b>",
             "<b>0.810</b>", "<b>0.977</b>", "<b>0.678</b>"],
            ["Hybrid RRF", "0.551", "0.867", "0.684", "0.951", "0.689"],
        ],
        col_widths=[4.8 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm],
    ))
    story.append(P(
        "Table 3. Retrieval on 1,182 test questions / 8,840 passages. Fine-tuning more than "
        "doubles BM25 Recall@1. Adding Rwanda questions helps only ~1 point on Rwanda R@1 "
        "(within noise for n=180): the model mainly learns question→section mapping that transfers.",
        s["caption"],
    ))

    story.append(P("6.3 Answer generation", s["h2"]))
    story.append(table(
        ["Reader", "Context", "ROUGE-L", "BERTScore", "Support", "Words", "Abstain unrelated"],
        [
            ["Extractive*", "gold", "0.931*", "0.985*", "—", "62", "no"],
            ["Flan-T5 zero-shot", "gold", "0.191", "0.761", "—", "9", "no"],
            ["Flan-T5 + LoRA", "gold", "0.608", "0.900", "—", "37", "no"],
            ["Qwen zero-shot", "gold", "0.327", "0.829", "0.624", "58", "25%"],
            ["<b>Qwen + LoRA (deployed)</b>", "gold", "<b>0.643</b>", "<b>0.911</b>",
             "<b>0.964</b>", "104", "<b>73%</b>"],
            ["Extractive", "retrieved", "0.395", "0.833", "—", "60", "no"],
            ["Flan-T5 + LoRA", "retrieved", "0.414", "0.841", "—", "35", "no"],
            ["<b>Qwen + LoRA</b>", "retrieved", "<b>0.432</b>", "<b>0.847</b>",
             "0.930", "99", "73%"],
        ],
        col_widths=[3.8 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm, 1.5 * cm, 2.9 * cm],
    ))
    story.append(P(
        "Table 4. Generation results. *Extractive on gold is an upper bound (references are "
        "the first sentences of the gold passage). Support = share of answer content words "
        "found in the given passages (groundedness). Qwen LoRA never falsely abstained when "
        "the gold passage was present (false abstention = 0).",
        s["caption"],
    ))
    story.append(P("6.4 Safety rules", s["h2"]))
    story.append(P(
        "On the labelled challenge risk set, the rule-based detector achieved accuracy 1.0 "
        "across routine, prescription, special-group, and emergency classes. This score is "
        "optimistic because the same set helped refine the rules; it is reported for "
        "transparency, not as an independent clinical validation.",
        s["body"],
    ))
    story.append(P("6.5 Discussion: connecting results to the literature", s["h2"]))
    story.append(P(
        "Three findings align with prior work. First, high in-domain classification accuracy "
        "can mask poor generalisation to consumer style—consistent with intent OOS research "
        "on CLINC150 [13] and with the need for realistic evaluation beyond clean test splits. "
        "Second, BM25’s strength on rare brand tokens vs dense models’ strength after "
        "fine-tuning mirrors classic sparse/dense trade-offs [7], [5]. Hybrid RRF helping "
        "only when the dense model is weak matches the intuition that fusion cannot rescue "
        "a weaker ranker once the strong ranker already dominates. Third, zero-shot "
        "instruction models produce fluent but under-grounded answers (support 0.62), while "
        "LoRA on grounded targets restores fidelity—empirically supporting RAG + parameter-"
        "efficient adaptation [10], [11] over “prompt the base model and hope”. The chat "
        "reader’s higher abstention rate (73% vs 25%) is particularly important for pharmacy "
        "use: refusing is safer than inventing a local brand’s ingredients.",
        s["body"],
    ))

    # ----- 7. Error analysis -----
    story.append(P("7. Evaluation Metrics and Error Analysis", s["h1"]))
    story.append(P("7.1 Why these metrics", s["h2"]))
    story.append(bullets([
        "<b>Accuracy / macro-F1</b> for intent: classes are imbalanced; macro-F1 penalises ignoring a class.",
        "<b>Recall@k and MRR@10</b> for retrieval: standard IR metrics; Topic@1 checks whether the right medicine is found even if the wrong section is ranked first.",
        "<b>ROUGE-L</b> [16] for lexical overlap with reference answers; we also report <b>ROUGE-L recall</b> because chat answers are longer by design.",
        "<b>BERTScore</b> [17] (DistilBERT) for semantic similarity when wording differs.",
        "<b>Support</b> and <b>abstention rate</b> for groundedness and safety of “I don’t know” behaviour—lexical metrics alone cannot capture hallucination.",
    ], s))
    story.append(P(
        "None of these metrics measure clinical correctness; they measure agreement with trusted "
        "source text and discrimination of scope. That limitation is discussed in Section 9.",
        s["body"],
    ))
    story.append(P("7.2 Successful examples (live system)", s["h2"]))
    story.append(bullets([
        "<b>“What are the side effects of metformin?”</b> → opens with “Metformin can cause side effects.” then lists diarrhoea, nausea, etc. from the single-ingredient label (earlier bug: metoclopramide text).",
        "<b>“after sex pills” then “any of 120hrs, I passed 72hrs”</b> → overview of levonorgestrel, then ulipristal dosing within 120 hours.",
        "<b>“what do you know about quarteme”</b> → “closest name is Coartem” + malaria overview (artemether/lumefantrine).",
        "<b>“exchange rate of the Rwandan franc”</b> → low-confidence / out-of-scope refusal (no medicine answer invented).",
    ], s))
    story.append(P("7.3 Failure modes and explanations", s["h2"]))
    story.append(bullets([
        "<b>Style overfitting (intent).</b> Without consumer questions, DistilBERT was near-perfect on formal test text but ~55% on the challenge set—misreading casual health wording as out-of-domain.",
        "<b>Aspect errors (retrieval).</b> BM25 Topic@1 is high (0.85) but R@1 is low (0.32): it finds the drug but the wrong leaflet section. Fine-tuning mainly fixes aspect ranking.",
        "<b>Wrong openFDA product.</b> The “richest label” download sometimes selected combination products; silent wrong answers until supplementary labels were added.",
        "<b>Unknown local brands.</b> “Kopavitamin” triggered generic vitamin text until NameChecker forced an honest “product not in sources” reply.",
        "<b>Reader opener mismatch.</b> Qwen sometimes started with a side-effects sentence for a “what is that?” follow-up; a post-hoc opener correction aligns the announced section with the question.",
        "<b>Clinician wording.</b> Even correct answers retain FDA phrasing; symptom questions may explain the condition before naming OTC options (acetaminophen/ibuprofen).",
        "<b>Loose abstention.</b> On deliberately unrelated passages, Qwen LoRA still answers ~27% of the time from loosely related text—hence the retrieval similarity threshold remains a second guard.",
    ], s))

    # ----- 8. Deployment -----
    story.append(P("8. Web Deployment", s["h1"]))
    story.append(P(
        "The final system is a <b>Gradio 5</b> web application hosted on Hugging Face Spaces "
        "with ZeroGPU. At startup, DistilBERT, MiniLM embeddings, and the Qwen base model + "
        "merged LoRA adapter are loaded on CPU; a <font face='Courier'>@spaces.GPU</font> "
        "wrapper moves compute to the GPU only for each answer (~5–10 s). The UI shows the "
        "chat, an “Understood as” follow-up interpretation, model component tags, and "
        "source passages with similarity scores. Deployment artefacts and "
        "<font face='Courier'>scripts/deploy_space.py</font> reproduce the Space contents. "
        "Live URL: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa",
        s["body"],
    ))

    # ----- 9. Limitations -----
    story.append(P("9. Limitations and Ethical Considerations", s["h1"]))
    story.append(bullets([
        "English only; Kinyarwanda and French are out of scope for this version.",
        "Templated openFDA/Rwanda questions and clinician-facing label wording limit ecological validity.",
        "ROUGE/BERTScore ≠ clinical correctness; no clinician adjudication was performed.",
        "Challenge and safety sets are small; safety accuracy is optimistic.",
        "Local Rwandan brands absent from leaflets cannot be described beyond honest abstention.",
        "ZeroGPU free quota can interrupt anonymous demos; assessors should use a signed-in session.",
        "The assistant must not be used as medical advice; emergencies require SAMU 912 / a pharmacist.",
    ], s))
    story.append(P(
        "<b>AI assistance.</b> AI coding tools helped scaffold code and draft challenge items; "
        "all training, evaluation, and live behaviour were run and verified by the author, who "
        "takes full responsibility for the submission. Commits are authored solely as Emmanuel-NS.",
        s["body"],
    ))

    # ----- 10. Conclusion -----
    story.append(P("10. Conclusion", s["h1"]))
    story.append(P(
        "This project delivers a complete, reproducible domain-specific pharmacy QA system: "
        "trusted data, DistilBERT intent control, a fine-tuned MiniLM retriever, and a "
        "LoRA-adapted Qwen2.5 chat reader deployed on the web. Experiments show that "
        "<b>domain fine-tuning and consumer-style data</b> matter more than raw formal-test "
        "scores, that <b>LoRA roughly doubles</b> zero-shot Qwen answer quality while "
        "sharply improving groundedness and abstention, and that <b>live-user testing</b> "
        "surfaces failure modes (wrong labels, unknown brands, follow-ups) that offline "
        "benchmarks miss. Future work includes multilingual support, clinician evaluation, "
        "and expanding Rwanda leaflet coverage with OCR for scanned PDFs.",
        s["body"],
    ))

    # ----- References -----
    story.append(P("References", s["h1"]))
    refs = [
        "[1] Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys.",
        "[2] Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature.",
        "[3] Ben Abacha, A., &amp; Demner-Fushman, D. (2019). A Question-Entailment Approach to Question Answering. BMC Bioinformatics. (MedQuAD)",
        "[4] Ben Abacha, A., et al. (2019). Bridging the Gap Between Consumers’ Medication Questions and Trusted Answers. MEDINFO. (MedicationQA)",
        "[5] Reimers, N., &amp; Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.",
        "[6] Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS.",
        "[7] Robertson, S., &amp; Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR.",
        "[8] Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. (Flan-T5)",
        "[9] Qwen Team. (2024). Qwen2.5 Technical Report. arXiv.",
        "[10] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.",
        "[11] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP. NeurIPS.",
        "[12] Amodei, D., et al. (2016). Concrete Problems in AI Safety. arXiv. (motivation for predictable safety behaviour)",
        "[13] Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP. (CLINC150)",
        "[14] Li, Y., et al. (2023). ChatDoctor: A Medical Chat Model Fine-tuned on a Large Language Model Meta-AI (LLaMA). (HealthCareMagic corpus)",
        "[15] Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS EMC² Workshop.",
        "[16] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop.",
        "[17] Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.",
        "[18] World Health Organization. Model List of Essential Medicines. (domain scoping)",
        "[19] U.S. Food and Drug Administration. openFDA Drug Label API. https://open.fda.gov/",
        "[20] Rwanda Food and Drugs Authority. Patient Information Leaflets (public PDF library).",
    ]
    for r in refs:
        story.append(P(r, s["ref"]))

    # ----- Appendix -----
    story.append(P("Appendix A. Reproducibility checklist", s["h1"]))
    story.append(bullets([
        "Clone https://github.com/Emmanuel-NS/FARUMASI-AI and install <font face='Courier'>requirements.txt</font>.",
        "Data: <font face='Courier'>python -m pharmqa.data.download && python -m pharmqa.data.prepare</font>.",
        "Full Colab pipeline: <font face='Courier'>notebooks/train_on_colab.ipynb</font>; chat reader: <font face='Courier'>notebooks/train_chat_reader_colab.ipynb</font>.",
        "Reported metrics are committed under <font face='Courier'>results/*.json</font>.",
        "Local UI: <font face='Courier'>python app.py</font>; Space deploy: <font face='Courier'>scripts/deploy_space.py</font>.",
        "Demo speaking outline for the required video: <font face='Courier'>report/DEMO_VIDEO_SCRIPT.md</font>.",
    ], s))
    story.append(P("Appendix B. How this report maps to the marking criteria", s["h1"]))
    story.append(table(
        ["Criterion (pts)", "Where addressed in this submission"],
        [
            ["Problem, language &amp; dataset (8)", "§1, §3; README data tables; dataset_stats.json"],
            ["Model development (10)", "§4; train_* scripts; explained again in the demo video"],
            ["Baseline &amp; experiments (8)", "§5–§6; results/*.json; video must narrate why/what-learned"],
            ["Evaluation &amp; error analysis (7)", "§6–§7; live success/fail cases; support &amp; abstention"],
            ["Research report (10)", "This PDF: related work, method, results, discussion, refs"],
            ["Web deployment (7)", "§8; live Space URL; video workflow"],
            ["Code &amp; GitHub (5)", "Repo layout, README reproduce, notebooks"],
            ["Technical defense (5)", "Demo video Q&amp;A using DEMO_VIDEO_SCRIPT.md"],
        ],
        col_widths=[5.5 * cm, 11.3 * cm],
    ))
    story.append(P("Table 5. Submission map for assessors.", s["caption"]))
    story.append(P(
        "<i>Supplementary materials: GitHub repository, Colab notebooks, results/*.json, "
        "live Hugging Face Space, and this PDF under report/.</i>",
        s["caption"],
    ))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="FARUMASI Pharmacist QA — Research Report",
        author="Emmanuel Nsabagasani",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
