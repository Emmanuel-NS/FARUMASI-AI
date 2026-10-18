"""Build the summative research report PDF (reportlab).

Run from the repo root:
    python report/build_report.py
Output:
    report/Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "T",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ST",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=3,
        ),
        "meta": ParagraphStyle(
            "M",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.black,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            firstLineIndent=0,
        ),
        "quote": ParagraphStyle(
            "Q",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            leftIndent=18,
            rightIndent=18,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "C",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "ref": ParagraphStyle(
            "R",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=5,
        ),
        "abstract": ParagraphStyle(
            "A",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "BU",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14.5,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "F",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
        ),
        "cellb": ParagraphStyle(
            "CellB",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=9,
            leading=11,
        ),
        "page_num": ParagraphStyle(
            "PN",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            alignment=TA_RIGHT,
        ),
    }


def P(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", " "), style)


def bullets(items, s):
    return ListFlowable(
        [
            ListItem(Paragraph(i, s["bullet"]), leftIndent=6, bulletColor=colors.black)
            for i in items
        ],
        bulletType="bullet",
        start="-",
        leftIndent=16,
        bulletFontSize=10,
        bulletFontName="Times-Roman",
    )


def table(headers, rows, col_widths=None):
    s = styles()
    data = [[Paragraph(h, s["cellb"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["cell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.4)
    canvas.line(2.2 * cm, 1.5 * cm, A4[0] - 2.2 * cm, 1.5 * cm)
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(A4[0] / 2.0, 0.95 * cm, str(doc.page))
    canvas.restoreState()


def build():
    s = styles()
    story = []

    story.append(Spacer(1, 6 * mm))
    story.append(
        P(
            "FARUMASI Pharmacist Assistant: A Domain-Specific Question-Answering "
            "System for Community Pharmacy in Rwanda",
            s["title"],
        )
    )
    story.append(P("Emmanuel Nsabagasani", s["subtitle"]))
    story.append(
        P(
            "Summative Project, Natural Language Processing (Sep26)<br/>"
            "African Leadership University<br/>"
            "GitHub: https://github.com/Emmanuel-NS/FARUMASI-AI<br/>"
            "Live application: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa",
            s["meta"],
        )
    )

    story.append(P("Abstract", s["h2"]))
    story.append(
        P(
            "This report describes a domain-specific English question-answering system built to "
            "support routine medicine and minor-ailment questions in a Rwandan community pharmacy "
            "setting. The work was inspired by FARUMASI, a digital pharmacy platform the author "
            "has been building, and by repeated observation that pharmacists spend much of their "
            "day answering the same basic questions while attending to dispensing and counselling. "
            "The system retrieves answers only from trusted leaflet text (US openFDA labels for "
            "WHO essential medicines, MedQuAD, MedicationQA, and Rwanda FDA patient leaflets). "
            "It combines a DistilBERT intent classifier, a fine-tuned MiniLM retriever, and a "
            "LoRA-adapted Qwen2.5-1.5B-Instruct chat reader, with a Flan-T5 LoRA model kept as a "
            "fallback. Across controlled experiments, LoRA fine-tuning raised gold-context "
            "ROUGE-L from 0.327 (zero-shot Qwen) to 0.643, raised passage support from 0.62 to "
            "0.96, and raised abstention on unrelated passages from 25 percent to 73 percent. "
            "The finished model is deployed as a Gradio web application on Hugging Face Spaces.",
            s["abstract"],
        )
    )
    story.append(
        P(
            "Keywords: domain-specific QA, retrieval-augmented generation, LoRA, biomedical NLP, "
            "pharmacy, Rwanda, FARUMASI",
            s["abstract"],
        )
    )

    # ----- 1 -----
    story.append(P("1. Introduction and Problem Definition", s["h1"]))
    story.append(P("1.1 Motivation from FARUMASI", s["h2"]))
    story.append(
        P(
            "The idea for this project did not start as an abstract NLP exercise. It grew out of "
            "FARUMASI, a digital pharmacy marketplace and patient-facing platform the author has "
            "been developing for Rwanda. While working with partner pharmacies and watching how "
            "patients actually ask for help, one pattern kept returning: many people approach the "
            "counter with the same kinds of questions. What is this medicine for? Can I take it "
            "with food? What are the side effects? Is this the same as Panadol? Pharmacists are "
            "trained to answer these questions, but they are also busy. In a typical shift they "
            "must verify prescriptions, manage stock, counsel on dosing, and keep the queue "
            "moving. When the same informational questions arrive again and again, they take time "
            "away from the work that only a pharmacist can do.",
            s["body"],
        )
    )
    story.append(
        P(
            "FARUMASI already helps patients find medicines and connect with pharmacies. What it "
            "still lacked was a careful, grounded way to answer basic medicine questions before "
            "those questions reach a busy counter. The inspiration for this summative project was "
            "therefore practical: build an assistant that can take routine informational questions, "
            "answer them from trusted leaflet text when the answer is available, and refuse honestly "
            "when it is not. If such a tool works well, it can reduce the volume of repeated "
            "questions directed at pharmacists, free them for higher-value counselling, and give "
            "patients a safer first stop than social media or a general chatbot that may invent "
            "doses.",
            s["body"],
        )
    )
    story.append(
        P(
            "General-purpose language models are a poor fit for that role. They can invent "
            "interactions, contraindications, and dosing details [1], [2]. The research problem "
            "addressed in this report is therefore:",
            s["body"],
        )
    )
    story.append(
        P(
            "How can we build an English-language, domain-specific QA assistant that answers "
            "routine medicine and minor-ailment questions for Rwandan patients using only trusted "
            "leaflet text, refuses out-of-scope queries, and flags risky situations, without relying "
            "on an external commercial chatbot API?",
            s["quote"],
        )
    )

    story.append(P("1.2 Task, language, and target users", s["h2"]))
    story.append(
        P(
            "The main task is open-domain question answering over a fixed pharmacy corpus: retrieve "
            "relevant passages and generate a short answer that stays faithful to those passages. "
            "An upstream three-way intent classifier labels each question as medicine information, "
            "condition advice, or out of domain. A separate rule-based detector flags higher-risk "
            "situations such as emergencies. This is not a machine translation project. English was "
            "chosen because most of the official medicine labels and NIH health topics available for "
            "training are written in English, while usable Kinyarwanda and French leaflet text is "
            "still sparse and often locked inside scanned PDFs.",
            s["body"],
        )
    )
    story.append(
        P(
            "Even so, Rwandan patients often mix English brand names with everyday local phrasing. "
            "The assistant therefore expands common lay terms and local brands before retrieval. "
            "For example, panadol is treated as paracetamol, coartem as artemether with lumefantrine, "
            "and after sex pills as levonorgestrel. Multilingual answering is left for later work, "
            "because roughly half of the Rwanda FDA PDFs examined for this project had no text layer.",
            s["body"],
        )
    )
    story.append(
        P(
            "The target users are adult patients and caregivers who need routine information about "
            "WHO essential medicines and minor ailments, together with pharmacy students and "
            "FARUMASI partners who want a first-line informational tool. The assistant is not a "
            "substitute for a pharmacist or a doctor. Every answer carries a disclaimer, and "
            "emergencies are directed to SAMU on 912.",
            s["body"],
        )
    )

    story.append(P("1.3 Research questions", s["h2"]))
    story.append(
        bullets(
            [
                "RQ1. Does DistilBERT fine-tuning outperform TF-IDF baselines on informal consumer questions?",
                "RQ2. Does dense MiniLM fine-tuning with in-batch negatives beat BM25 and zero-shot MiniLM on Recall at k and MRR?",
                "RQ3. Does LoRA fine-tuning of a reader (Flan-T5, then Qwen2.5-Instruct) improve grounded answer quality over extractive and zero-shot baselines?",
                "RQ4. Does including Rwanda FDA leaflet questions improve retrieval on local brands, or does US and NIH training already transfer?",
            ],
            s,
        )
    )

    # ----- 2 -----
    story.append(P("2. Related Work", s["h1"]))
    story.append(P("2.1 Biomedical and consumer health QA", s["h2"]))
    story.append(
        P(
            "Biomedical QA has moved from classical information retrieval over curated corpora "
            "toward neural readers. MedQuAD [3] brings together large NIH and MedlinePlus-style "
            "collections and frames answering partly through question entailment. MedicationQA [4] "
            "is closer to the present use case because it focuses on real consumer medicine "
            "questions answered from trusted resources. Earlier clinical QA systems often assumed "
            "expert users. Consumer health QA must cope with lay synonyms, incomplete questions, "
            "and higher stakes when the system is wrong [2]. The contribution of this project is "
            "not a new public dataset on its own. It is a closed-corpus retrieval-augmented "
            "pipeline adapted to Rwandan pharmacy practice, including local brands, emergency "
            "contraception lay terms, and SAMU referral, built on public resources plus Rwanda "
            "FDA leaflets [20].",
            s["body"],
        )
    )
    story.append(P("2.2 Retrieval and grounded generation", s["h2"]))
    story.append(
        P(
            "Dense bi-encoders such as Sentence-BERT and MiniLM [5], [6] map questions and passages "
            "into a shared vector space for cosine ranking. Classical BM25 [7] remains strong when "
            "rare brand tokens dominate matching, which is common for Rwanda leaflet names. "
            "Retrieval-augmented generation [11] conditions a generator on retrieved documents to "
            "reduce hallucination. Instruction-tuned generators including Flan-T5 [8] and Qwen2.5 "
            "[9] can be adapted with LoRA [10], which updates a small set of low-rank matrices "
            "instead of all weights. That efficiency mattered here because training ran on free "
            "Colab T4 hardware. The experiments connect to this literature in two ways. Domain "
            "LoRA roughly doubles zero-shot Qwen ROUGE-L while raising a simple groundedness "
            "proxy from 0.62 to 0.96. Zero-shot instruction models paraphrase freely and under-"
            "abstain, which is consistent with known hallucination risks [1], [2].",
            s["body"],
        )
    )
    story.append(P("2.3 Scope control and safety", s["h2"]))
    story.append(
        P(
            "Out-of-scope detection for dialogue systems is studied in CLINC150 [13]. This project "
            "treats scope as a first-class classifier before retrieval so non-health questions never "
            "reach the reader. Commercial clinical chatbots and closed APIs are deliberately "
            "avoided. The assessment brief forbids simply wiring an existing chatbot API, and "
            "auditable grounding requires that every answer cite passages from a known corpus. "
            "Rule-based emergency overlays sit beside the learned modules because safety actions "
            "such as directing a caller to 912 must be predictable rather than sampled [12].",
            s["body"],
        )
    )

    # ----- 3 -----
    story.append(P("3. Dataset", s["h1"]))
    story.append(P("3.1 Sources, size, and characteristics", s["h2"]))
    story.append(
        P(
            "All training and evaluation text is human-written. No model-generated passages were "
            "used as gold answers. The passage corpus contains 8,840 passages. The QA set contains "
            "9,518 question-answer pairs. Pairs are grouped by medicine or condition so the same "
            "entity does not appear in more than one split. The split sizes are 7,200 for training, "
            "1,136 for validation, and 1,182 for testing.",
            s["body"],
        )
    )
    story.append(
        table(
            ["Dataset", "Role", "Size used", "Licence or notes"],
            [
                [
                    "openFDA drug labels (US FDA)",
                    "Medicine passages for about 140 WHO essential medicines",
                    "2,049 passages; 2,523 templated QA pairs",
                    "Public domain",
                ],
                [
                    "MedQuAD [3]",
                    "Condition and health-topic QA",
                    "5,546 passages; 5,580 QA pairs",
                    "CC BY 4.0",
                ],
                [
                    "MedicationQA [4]",
                    "Real consumer medicine questions",
                    "655 QA pairs",
                    "CC BY 4.0",
                ],
                [
                    "Rwanda FDA Patient Information Leaflets",
                    "Local brands such as Coartem and Agycin",
                    "187 usable leaflets, 590 passages, 760 QA pairs",
                    "Public regulatory PDFs",
                ],
                [
                    "CLINC150 [13]",
                    "Out-of-domain intent negatives",
                    "23,700 queries (sampled)",
                    "CC BY 3.0",
                ],
                [
                    "ChatDoctor-HealthCareMagic [14]",
                    "Weakly labelled consumer questions for intent training only",
                    "3,416 questions",
                    "Apache 2.0",
                ],
                [
                    "Challenge set (this project)",
                    "Informal consumer questions used for evaluation only",
                    "56 questions",
                    "Author-labelled",
                ],
            ],
            col_widths=[3.4 * cm, 4.4 * cm, 4.0 * cm, 3.0 * cm],
        )
    )
    story.append(P("Table 1. Datasets used in the project.", s["caption"]))

    story.append(P("3.2 Preprocessing and quality", s["h2"]))
    story.append(
        bullets(
            [
                "Label sections were cleaned and split into passages of at most 160 words. Mean passage length is about 118 words.",
                "openFDA and Rwanda leaflet questions are templated over real section text. This is a documented limitation. MedicationQA and MedQuAD questions are natural.",
                "Splits are entity-grouped to avoid leakage. A medicine that appears in training cannot appear in the test set.",
                "Of 416 cached Rwanda FDA PDFs, 213 scanned images without a text layer were excluded. Two-column layouts sometimes interleave columns and introduce noise.",
                "Supplementary corrected labels for metformin (single ingredient), levonorgestrel (OTC emergency pill), and ulipristal (120-hour emergency pill) replace wrong combination-product downloads.",
                "Passage cleaning removes cross-references such as see Warnings and US MedWatch boilerplate before the reader sees the text.",
            ],
            s,
        )
    )
    story.append(P("3.3 Dataset limitations", s["h2"]))
    story.append(
        P(
            "Medicine answers rely heavily on FDA wording written for clinicians. MedQuAD drug "
            "answers were removed by their authors for copyright reasons. Templated questions "
            "inflate lexical overlap metrics. US brand names and doses may differ from Rwandan "
            "practice. English-only coverage excludes many real patient utterances. These limits "
            "motivate grounded generation, honest abstention for unknown local brands, and a clear "
            "disclaimer on every answer.",
            s["body"],
        )
    )

    # ----- 4 -----
    story.append(P("4. Methodology", s["h1"]))
    story.append(
        P(
            "The pipeline is organised as follows. A question first passes optional small-talk and "
            "feedback rules, then a risk detector, then the intent classifier. Lay and brand terms "
            "are expanded. A dense retriever ranks passages and applies medicine and section "
            "re-ranking. The chat reader (Qwen with LoRA) writes an answer from on-topic passages "
            "only. Finally, safety localisation replaces US poison numbers with 912 where needed, "
            "and the interface adds a disclaimer and source list.",
            s["body"],
        )
    )
    story.append(P("4.1 Intent and scope classifier", s["h2"]))
    story.append(
        P(
            "The classifier is DistilBERT [15], a six-layer Transformer encoder distilled from BERT, "
            "with a linear classification head over three labels. The input is a tokenised question "
            "with a maximum length of 64 tokens. The output is a label and a softmax confidence. "
            "Baselines are TF-IDF with logistic regression using word n-grams and word-plus-character "
            "n-grams. Training first used MedQuAD and CLINC material, then a consumer-augmented set "
            "that mixes real patient questions from HealthCareMagic with heuristic labels. DistilBERT "
            "was chosen because it is a course-appropriate Transformer, fits free Colab hardware, "
            "and is small enough to sit beside retrieval and generation on ZeroGPU.",
            s["body"],
        )
    )
    story.append(P("4.2 Retriever", s["h2"]))
    story.append(
        P(
            "The retriever is all-MiniLM-L6-v2 [6]. Questions and passages are mapped to 384-"
            "dimensional normalised embeddings and ranked by cosine similarity. Fine-tuning used "
            "contrastive training with in-batch negatives for two epochs, batch size 64, and "
            "learning rate 2e-5. At inference, the top 50 dense hits are re-ranked. A passage "
            "receives a boost of 0.15 when its medicine matches a drug named in the query, and "
            "0.08 when its section matches the asked aspect. Empty stub passages are masked. "
            "Baselines are BM25 [7], zero-shot MiniLM, and a hybrid of BM25 with dense ranks "
            "through Reciprocal Rank Fusion.",
            s["body"],
        )
    )
    story.append(P("4.3 Answer readers", s["h2"]))
    story.append(
        P(
            "Two readers were trained. Flan-T5-base with LoRA [8] is an encoder-decoder "
            "instruction model. LoRA adapters of rank 16 were placed on query and value "
            "projections, covering about 0.7 percent of parameters. The model was trained to "
            "produce short answers from a prompt that contains three passages (one gold passage "
            "and two hard negatives). The reported run used two epochs, learning rate 3e-4, "
            "effective batch size 16, and about 25 minutes on a Colab T4.",
            s["body"],
        )
    )
    story.append(
        P(
            "The deployed reader is Qwen2.5-1.5B-Instruct with LoRA [9]. Adapters use rank 16, "
            "alpha 32, and dropout 0.05 on attention and MLP projections. Training used a chat "
            "template with a system prompt that forbids invention. Loss was computed on answer "
            "tokens only. Three example types were used: ordinary answers with an opener and "
            "passage body, multi-section overviews, and about 8 percent no-answer examples where "
            "the passages are unrelated and the target is a refusal. Hyperparameters were one "
            "epoch, at most 3,000 training examples, batch size 2 with gradient accumulation 8, "
            "maximum length 1,536, learning rate 2e-4, fp16, and gradient checkpointing on a "
            "Colab T4 for roughly one to two hours. At inference the model sees only on-topic "
            "retrieved passages. Overviews are written per section in one batch so headings stay "
            "aligned with the content.",
            s["body"],
        )
    )
    story.append(P("4.4 Query understanding and safety", s["h2"]))
    story.append(
        P(
            "Several non-learned components sit around the neural modules. They include lay and "
            "brand expansion, follow-up resolution (for example, what do they do inherits the "
            "previous medicine), progressive overviews, unknown-product detection, fuzzy suggestion "
            "of close names (for example, quarteme suggested as Coartem), and rule-based risk "
            "levels for routine, prescription, special-group, and emergency cases. These pieces "
            "live in pharmqa/query.py and pharmqa/safety.py. They are deliberately transparent "
            "and were refined from live-app failures that benchmarks alone did not expose.",
            s["body"],
        )
    )
    story.append(P("4.5 Training and inference summary", s["h2"]))
    story.append(
        table(
            ["Component", "Base model", "Adaptation", "Key hyperparameters", "Hardware"],
            [
                [
                    "Intent",
                    "DistilBERT-base-uncased",
                    "Full fine-tune plus consumer data",
                    "max length 64; 3 labels",
                    "Colab T4",
                ],
                [
                    "Retriever",
                    "all-MiniLM-L6-v2",
                    "Contrastive fine-tune",
                    "2 epochs; batch 64; lr 2e-5; in-batch negatives",
                    "Colab T4",
                ],
                [
                    "Reader (legacy)",
                    "Flan-T5-base",
                    "LoRA rank 16 on q and v",
                    "2 epochs; lr 3e-4; effective batch 16; max source 512",
                    "Colab T4, about 25 min",
                ],
                [
                    "Reader (deployed)",
                    "Qwen2.5-1.5B-Instruct",
                    "LoRA rank 16 on attention and MLP",
                    "1 epoch; lr 2e-4; batch 2 times 8; max length 1536; answer-token loss",
                    "Colab T4, about 1 to 2 h",
                ],
            ],
            col_widths=[2.6 * cm, 3.6 * cm, 3.4 * cm, 4.6 * cm, 2.6 * cm],
        )
    )
    story.append(
        P(
            "Table 2. Modelling choices used for the reported results and the live application.",
            s["caption"],
        )
    )

    # ----- 5 -----
    story.append(P("5. Baseline and Experimentation", s["h1"]))
    story.append(
        P(
            "Each component has a meaningful baseline and controlled ablations. Experiments were "
            "run on Colab T4 GPUs. Metrics are stored under the results folder so the numbers in "
            "this report can be checked against the committed JSON files.",
            s["body"],
        )
    )
    story.append(P("5.1 Intent experiments", s["h2"]))
    story.append(
        P(
            "TF-IDF with logistic regression was chosen as the baseline because it is a strong "
            "classical text classifier and isolates whether Transformers help beyond bag-of-words "
            "features. The experiments compared word n-grams with word-plus-character n-grams, "
            "DistilBERT without consumer augmentation, and DistilBERT with HealthCareMagic "
            "questions. Models trained only on formal MedQuAD and CLINC text scored above "
            "99 percent on the held-out test split but collapsed on the informal challenge set. "
            "That pattern suggested they had learned style rather than topic. Consumer "
            "augmentation tests whether realistic phrasing closes the gap (RQ1).",
            s["body"],
        )
    )
    story.append(P("5.2 Retrieval experiments", s["h2"]))
    story.append(
        P(
            "BM25 is the standard sparse information-retrieval baseline. Zero-shot MiniLM shows "
            "what pretrained semantic matching can do without domain fine-tuning. The experiments "
            "then fine-tuned MiniLM on the pharmacy QA pairs, compared training with and without "
            "Rwanda FDA leaflet questions, and fused BM25 with dense ranks through Reciprocal Rank "
            "Fusion. These runs address RQ2 and RQ4: how much domain adaptation helps, and whether "
            "local leaflets are necessary for local brands.",
            s["body"],
        )
    )
    story.append(P("5.3 Generation experiments", s["h2"]))
    story.append(
        P(
            "Generation baselines were an extractive first-sentence reader, Flan-T5 zero-shot, and "
            "Qwen2.5-1.5B zero-shot. The proposed systems were Flan-T5 with LoRA and Qwen2.5 with "
            "LoRA. Two context conditions were used. Gold context gives the gold passage plus two "
            "BM25 hard negatives and isolates the reader. Retrieved context gives the top three "
            "BM25 passages and tests end-to-end behaviour with a weaker retriever so the Flan and "
            "Qwen tables stay comparable. These runs address RQ3: whether LoRA teaches the model "
            "to select the right passage and phrase a grounded answer, and whether a chat-style "
            "reader improves openings and abstention.",
            s["body"],
        )
    )

    # ----- 6 -----
    story.append(P("6. Results", s["h1"]))
    story.append(P("6.1 Intent classification", s["h2"]))
    story.append(
        table(
            ["Model", "Test acc.", "MedicationQA acc.", "Challenge acc.", "Challenge macro-F1"],
            [
                ["TF-IDF + LR (word)", "0.993", "0.528", "0.393", "0.372"],
                ["TF-IDF + LR (word + char)", "0.995", "0.694", "0.464", "0.420"],
                ["TF-IDF + LR + patient questions", "0.979", "0.556", "0.768", "0.764"],
                ["DistilBERT (original)", "0.998", "0.917", "0.554", "0.486"],
                [
                    "DistilBERT + patient questions (deployed)",
                    "0.994",
                    "0.889",
                    "0.804",
                    "0.782",
                ],
            ],
            col_widths=[5.2 * cm, 2.4 * cm, 3.2 * cm, 2.8 * cm, 3.2 * cm],
        )
    )
    story.append(
        P(
            "Table 3. Intent results. High formal-test accuracy is misleading. The challenge set "
            "is the more realistic criterion. Consumer augmentation is the decisive factor.",
            s["caption"],
        )
    )

    story.append(P("6.2 Retrieval", s["h2"]))
    story.append(
        table(
            ["Model", "R@1", "R@5", "MRR@10", "Topic@1", "Rwanda R@1"],
            [
                ["BM25", "0.322", "0.706", "0.478", "0.852", "0.444"],
                ["MiniLM zero-shot", "0.436", "0.804", "0.589", "0.882", "0.400"],
                ["MiniLM fine-tuned (no Rwanda)", "0.694", "0.954", "0.809", "0.977", "0.667"],
                [
                    "MiniLM fine-tuned (all, deployed)",
                    "0.695",
                    "0.958",
                    "0.810",
                    "0.977",
                    "0.678",
                ],
                ["Hybrid RRF", "0.551", "0.867", "0.684", "0.951", "0.689"],
            ],
            col_widths=[5.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm],
        )
    )
    story.append(
        P(
            "Table 4. Retrieval on 1,182 test questions and 8,840 passages. Fine-tuning more than "
            "doubles BM25 Recall at 1. Adding Rwanda questions helps only about one point on "
            "Rwanda Recall at 1, which is within noise for n equals 180. The model mainly learns "
            "a question-to-section mapping that transfers across sources.",
            s["caption"],
        )
    )

    story.append(P("6.3 Answer generation", s["h2"]))
    story.append(
        table(
            [
                "Reader",
                "Context",
                "ROUGE-L",
                "BERTScore",
                "Support",
                "Words",
                "Abstain unrelated",
            ],
            [
                ["Extractive*", "gold", "0.931*", "0.985*", "-", "62", "no"],
                ["Flan-T5 zero-shot", "gold", "0.191", "0.761", "-", "9", "no"],
                ["Flan-T5 + LoRA", "gold", "0.608", "0.900", "-", "37", "no"],
                ["Qwen zero-shot", "gold", "0.327", "0.829", "0.624", "58", "25%"],
                ["Qwen + LoRA (deployed)", "gold", "0.643", "0.911", "0.964", "104", "73%"],
                ["Extractive", "retrieved", "0.395", "0.833", "-", "60", "no"],
                ["Flan-T5 + LoRA", "retrieved", "0.414", "0.841", "-", "35", "no"],
                ["Qwen + LoRA", "retrieved", "0.432", "0.847", "0.930", "99", "73%"],
            ],
            col_widths=[3.6 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm, 1.5 * cm, 3.1 * cm],
        )
    )
    story.append(
        P(
            "Table 5. Generation results. The starred extractive gold score is an upper bound "
            "because the references are the first sentences of the gold passage. Support is the "
            "share of answer content words found in the given passages. Qwen with LoRA never "
            "falsely abstained when the gold passage was present.",
            s["caption"],
        )
    )
    story.append(P("6.4 Safety rules", s["h2"]))
    story.append(
        P(
            "On the labelled challenge risk set, the rule-based detector reached accuracy 1.0 "
            "across routine, prescription, special-group, and emergency classes. This score is "
            "optimistic because the same set helped refine the rules. It is reported for "
            "transparency, not as an independent clinical validation.",
            s["body"],
        )
    )
    story.append(P("6.5 Discussion", s["h2"]))
    story.append(
        P(
            "Three findings sit comfortably with prior work. First, high in-domain classification "
            "accuracy can hide poor generalisation to consumer style. That is consistent with "
            "out-of-scope research on CLINC150 [13] and with the need for realistic evaluation "
            "beyond clean test splits. Second, BM25 remains strong on rare brand tokens, while "
            "dense models pull ahead after fine-tuning. That trade-off is familiar from sparse "
            "and dense retrieval literature [7], [5]. Hybrid fusion helped mainly when the dense "
            "model was still weak. Third, zero-shot instruction models produce fluent but under-"
            "grounded answers (support 0.62), while LoRA on grounded targets restores fidelity. "
            "That result supports retrieval-augmented generation with parameter-efficient "
            "adaptation [10], [11] over prompting a base model and hoping for the best.",
            s["body"],
        )
    )
    story.append(
        P(
            "The chat reader's higher abstention rate, 73 percent against 25 percent for zero-"
            "shot Qwen, matters especially for pharmacy use and for FARUMASI. Refusing an unknown "
            "local brand is safer than inventing its ingredients, and it is also more useful to a "
            "busy pharmacist: patients who need product-specific advice are steered back to a "
            "human rather than given a confident but unsupported answer.",
            s["body"],
        )
    )

    # ----- 7 -----
    story.append(P("7. Evaluation Metrics and Error Analysis", s["h1"]))
    story.append(P("7.1 Choice of metrics", s["h2"]))
    story.append(
        bullets(
            [
                "Accuracy and macro-F1 for intent. Classes are imbalanced, so macro-F1 penalises ignoring a class.",
                "Recall at k and MRR at 10 for retrieval. These are standard information-retrieval metrics. Topic at 1 checks whether the right medicine is found even if the wrong section is ranked first.",
                "ROUGE-L [16] for lexical overlap with reference answers. ROUGE-L recall is also reported because chat answers are longer by design.",
                "BERTScore [17] with DistilBERT for semantic similarity when wording differs.",
                "Support and abstention rate for groundedness and for the safety of I do not know behaviour. Lexical metrics alone cannot capture hallucination.",
            ],
            s,
        )
    )
    story.append(
        P(
            "None of these metrics measure clinical correctness. They measure agreement with "
            "trusted source text and discrimination of scope. That limitation is discussed in "
            "Section 9.",
            s["body"],
        )
    )
    story.append(P("7.2 Successful examples from the live system", s["h2"]))
    story.append(
        bullets(
            [
                "What are the side effects of metformin? The system opens with a side-effects framing and lists diarrhoea, nausea, and related effects from the single-ingredient label. An earlier bug had returned metoclopramide text.",
                "after sex pills followed by any of 120hrs, I passed 72hrs. The system first gives a levonorgestrel overview, then ulipristal dosing within 120 hours.",
                "what do you know about quarteme. The system suggests Coartem as the closest name and returns a malaria overview for artemether with lumefantrine.",
                "exchange rate of the Rwandan franc. The system refuses as out of scope and does not invent a medicine answer.",
            ],
            s,
        )
    )
    story.append(P("7.3 Failure modes", s["h2"]))
    story.append(
        bullets(
            [
                "Style overfitting in intent. Without consumer questions, DistilBERT was near perfect on formal test text but only about 55 percent accurate on the challenge set. Casual health wording was often read as out of domain.",
                "Aspect errors in retrieval. BM25 Topic at 1 is high (0.85) but Recall at 1 is low (0.32). The model finds the drug but the wrong leaflet section. Fine-tuning mainly fixes aspect ranking.",
                "Wrong openFDA product. The richest-label download sometimes selected combination products. Wrong answers stayed silent until supplementary labels were added.",
                "Unknown local brands. Kopavitamin triggered generic vitamin text until a name checker forced an honest product not in sources reply.",
                "Reader opener mismatch. Qwen sometimes started with a side-effects sentence for a what is that follow-up. A post-hoc opener correction now aligns the announced section with the question.",
                "Clinician wording. Even correct answers retain FDA phrasing. Symptom questions may explain the condition before naming over-the-counter options such as acetaminophen or ibuprofen.",
                "Loose abstention. On deliberately unrelated passages, Qwen with LoRA still answers about 27 percent of the time from loosely related text. A retrieval similarity threshold remains a second guard.",
            ],
            s,
        )
    )

    # ----- 8 -----
    story.append(P("8. Web Deployment", s["h1"]))
    story.append(
        P(
            "The final system is a Gradio 5 web application hosted on Hugging Face Spaces with "
            "ZeroGPU. At startup, DistilBERT, MiniLM embeddings, and the Qwen base model with its "
            "merged LoRA adapter are loaded on CPU. A GPU wrapper moves compute to the GPU only "
            "for each answer, typically for about five to ten seconds. The interface shows the "
            "chat, an Understood as line for follow-up interpretation, model component tags, and "
            "source passages with similarity scores. The script scripts/deploy_space.py reproduces "
            "the Space contents. The live URL is "
            "https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa.",
            s["body"],
        )
    )
    story.append(
        P(
            "From a FARUMASI perspective, the deployment matters because the assistant is meant "
            "to sit in front of real users, not only in a notebook. A patient can ask a routine "
            "question before contacting a pharmacy. If the answer is covered by the leaflets, the "
            "patient gets a grounded reply. If it is not, the system abstains and the patient is "
            "still directed to a pharmacist. That design is intentional: the goal is to reduce "
            "repeated informational load on busy pharmacies, not to replace professional care.",
            s["body"],
        )
    )

    # ----- 9 -----
    story.append(P("9. Limitations and Ethical Considerations", s["h1"]))
    story.append(
        bullets(
            [
                "English only. Kinyarwanda and French are out of scope for this version.",
                "Templated openFDA and Rwanda questions, together with clinician-facing label wording, limit ecological validity.",
                "ROUGE and BERTScore are not clinical correctness. No clinician adjudication was performed.",
                "The challenge and safety sets are small. Safety accuracy is optimistic.",
                "Local Rwandan brands absent from leaflets cannot be described beyond honest abstention.",
                "ZeroGPU free quota can interrupt anonymous demos. Assessors should use a signed-in session.",
                "The assistant must not be used as medical advice. Emergencies require SAMU 912 or a pharmacist.",
            ],
            s,
        )
    )
    story.append(
        P(
            "AI coding tools helped scaffold code and draft some challenge items. All training, "
            "evaluation, and live behaviour were run and verified by the author, who takes full "
            "responsibility for the submission. Commits are authored solely as Emmanuel-NS.",
            s["body"],
        )
    )

    # ----- 10 -----
    story.append(P("10. Conclusion", s["h1"]))
    story.append(
        P(
            "This project delivers a complete and reproducible domain-specific pharmacy QA system "
            "grounded in the practical needs of FARUMASI and of busy community pharmacies in Rwanda. "
            "It combines trusted data, DistilBERT intent control, a fine-tuned MiniLM retriever, "
            "and a LoRA-adapted Qwen2.5 chat reader deployed on the web. The experiments show that "
            "domain fine-tuning and consumer-style data matter more than raw formal-test scores, "
            "that LoRA roughly doubles zero-shot Qwen answer quality while improving groundedness "
            "and abstention, and that live-user testing surfaces failure modes such as wrong "
            "labels, unknown brands, and follow-ups that offline benchmarks miss.",
            s["body"],
        )
    )
    story.append(
        P(
            "If the assistant is used as intended, patients get safer first answers for routine "
            "questions, and pharmacists keep more of their time for the counselling that only they "
            "can provide. Future work includes multilingual support, clinician evaluation, closer "
            "integration with FARUMASI patient flows, and broader Rwanda leaflet coverage with OCR "
            "for scanned PDFs.",
            s["body"],
        )
    )

    # ----- References -----
    story.append(P("References", s["h1"]))
    refs = [
        "[1] Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys.",
        "[2] Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature.",
        "[3] Ben Abacha, A., and Demner-Fushman, D. (2019). A Question-Entailment Approach to Question Answering. BMC Bioinformatics. (MedQuAD)",
        "[4] Ben Abacha, A., et al. (2019). Bridging the Gap Between Consumers' Medication Questions and Trusted Answers. MEDINFO. (MedicationQA)",
        "[5] Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.",
        "[6] Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS.",
        "[7] Robertson, S., and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.",
        "[8] Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. (Flan-T5)",
        "[9] Qwen Team. (2024). Qwen2.5 Technical Report. arXiv.",
        "[10] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.",
        "[11] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP. NeurIPS.",
        "[12] Amodei, D., et al. (2016). Concrete Problems in AI Safety. arXiv.",
        "[13] Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP. (CLINC150)",
        "[14] Li, Y., et al. (2023). ChatDoctor: A Medical Chat Model Fine-tuned on a Large Language Model Meta-AI (LLaMA). (HealthCareMagic corpus)",
        "[15] Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS EMC2 Workshop.",
        "[16] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop.",
        "[17] Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.",
        "[18] World Health Organization. Model List of Essential Medicines.",
        "[19] U.S. Food and Drug Administration. openFDA Drug Label API. https://open.fda.gov/",
        "[20] Rwanda Food and Drugs Authority. Patient Information Leaflets (public PDF library).",
    ]
    for r in refs:
        story.append(P(r, s["ref"]))

    # ----- Appendix -----
    story.append(P("Appendix A. Reproducibility checklist", s["h1"]))
    story.append(
        bullets(
            [
                "Clone https://github.com/Emmanuel-NS/FARUMASI-AI and install the packages listed in requirements.txt.",
                "Prepare data with: python -m pharmqa.data.download and then python -m pharmqa.data.prepare.",
                "Full Colab pipeline: notebooks/train_on_colab.ipynb. Chat reader: notebooks/train_chat_reader_colab.ipynb.",
                "Reported metrics are committed under results as JSON files.",
                "Local interface: python app.py. Space deploy: scripts/deploy_space.py.",
                "Demo speaking outline for the required video: report/DEMO_VIDEO_SCRIPT.md.",
            ],
            s,
        )
    )
    story.append(P("Appendix B. Mapping to the marking criteria", s["h1"]))
    story.append(
        table(
            ["Criterion", "Where addressed"],
            [
                [
                    "Problem, language and dataset (8)",
                    "Sections 1 and 3; README data tables; dataset_stats.json",
                ],
                [
                    "Model development (10)",
                    "Section 4; training scripts; also explained in the demo video",
                ],
                [
                    "Baseline and experiments (8)",
                    "Sections 5 and 6; results JSON files; video narrates why and what was learned",
                ],
                [
                    "Evaluation and error analysis (7)",
                    "Sections 6 and 7; live success and failure cases; support and abstention",
                ],
                [
                    "Research report (10)",
                    "This PDF: related work, method, results, discussion, references",
                ],
                [
                    "Web deployment (7)",
                    "Section 8; live Space URL; video workflow",
                ],
                [
                    "Code and GitHub (5)",
                    "Repository layout, README reproduce steps, notebooks",
                ],
                [
                    "Technical defense (5)",
                    "Demo video and questions using DEMO_VIDEO_SCRIPT.md",
                ],
            ],
            col_widths=[5.5 * cm, 11.3 * cm],
        )
    )
    story.append(P("Table 6. Submission map for assessors.", s["caption"]))
    story.append(
        P(
            "Supplementary materials: GitHub repository, Colab notebooks, results JSON files, "
            "live Hugging Face Space, and this PDF under the report folder.",
            s["caption"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        title="FARUMASI Pharmacist QA Research Report",
        author="Emmanuel Nsabagasani",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
