"""Stage 2: build the passage corpus and the question-answer pairs.

Outputs (data/processed/):
  corpus.jsonl    passages the assistant can retrieve from (id, title, text, source)
  qa_{train,val,test}.jsonl   question -> gold passage id + short reference answer

Splits are grouped by *focus* (the medicine or condition a question is about), so no
medicine/condition appears in more than one split. This prevents the retriever and
generator from being tested on entities they were trained on.

Usage:  python -m pharmqa.data.prepare
"""
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pharmqa.config import PROCESSED, RAW, SEED

MAX_WORDS = 160          # passage length cap (fits MiniLM / T5 context comfortably)
CHUNKS_PER_SECTION = 2   # long label sections are split; keep the first two chunks
REF_SENTENCES = 3        # reference answers = first sentences of the gold passage

# MedQuAD subsets that still contain answers and cover common conditions.
# (10-12 had answers removed for copyright; 2_GARD and 3_GHR are rare genetic diseases.)
MEDQUAD_DIRS = [
    "1_CancerGov_QA", "4_MPlus_Health_Topics_QA", "5_NIDDK_QA", "6_NINDS_QA",
    "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA",
]

SECTION_TITLES = {
    "purpose": "Purpose", "indications_and_usage": "Uses", "directions": "Directions",
    "dosage_and_administration": "Dosage", "contraindications": "Contraindications",
    "do_not_use": "Do not use", "warnings_and_cautions": "Warnings", "warnings": "Warnings",
    "boxed_warning": "Boxed warning", "adverse_reactions": "Side effects",
    "drug_interactions": "Interactions", "overdosage": "Overdose", "pregnancy": "Pregnancy",
    "pregnancy_or_breast_feeding": "Pregnancy and breastfeeding",
    "use_in_specific_populations": "Special populations",
    "information_for_patients": "Patient information", "ask_doctor": "Ask a doctor",
    "ask_doctor_or_pharmacist": "Ask a doctor or pharmacist", "stop_use": "Stop use",
    "when_using": "When using", "storage_and_handling": "Storage",
}

# Question templates for openFDA label sections. The *answers* are real FDA label text;
# only the question wording is templated (documented as a limitation in the report).
TEMPLATES = {
    "indications_and_usage": ["What is {d} used for?", "What does {d} treat?", "Why would someone take {d}?", "What are the uses of {d}?"],
    "purpose": ["What is {d} used for?", "What does {d} do?"],
    "dosage_and_administration": ["How should I take {d}?", "What is the dose of {d}?", "How much {d} should I take?", "How often should {d} be taken?"],
    "directions": ["How do I use {d}?", "What is the right dose of {d}?", "How often can I take {d}?"],
    "contraindications": ["Who should not take {d}?", "When should {d} not be used?", "What are the contraindications of {d}?"],
    "do_not_use": ["Who should not use {d}?", "When should I avoid {d}?"],
    "warnings_and_cautions": ["What warnings should I know about {d}?", "What precautions should I take with {d}?", "Is {d} safe to use?"],
    "warnings": ["What are the warnings for {d}?", "Is {d} dangerous?", "What should I be careful about with {d}?"],
    "boxed_warning": ["Does {d} have a serious warning?", "What is the black box warning for {d}?"],
    "adverse_reactions": ["What are the side effects of {d}?", "Does {d} have side effects?", "What adverse reactions can {d} cause?"],
    "drug_interactions": ["Can I take {d} with other medicines?", "What drugs interact with {d}?", "Does {d} interact with anything?"],
    "overdosage": ["What happens if I take too much {d}?", "What are the signs of a {d} overdose?"],
    "pregnancy": ["Can I take {d} while pregnant?", "Is {d} safe in pregnancy?"],
    "pregnancy_or_breast_feeding": ["Can I use {d} when pregnant or breastfeeding?", "Is {d} safe while breastfeeding?"],
    "use_in_specific_populations": ["Can children or older adults take {d}?", "Is {d} safe for pregnant women and children?"],
    "information_for_patients": ["What should patients know about {d}?", "What advice is given to people taking {d}?"],
    "ask_doctor": ["When should I ask a doctor before using {d}?"],
    "ask_doctor_or_pharmacist": ["When should I talk to a pharmacist before taking {d}?"],
    "stop_use": ["When should I stop taking {d} and see a doctor?", "When should I stop using {d}?"],
    "when_using": ["What should I avoid while using {d}?"],
    "storage_and_handling": ["How should I store {d}?"],
}

# Leading label headings such as "10 OVERDOSAGE" or "WARNINGS AND PRECAUTIONS (5.1):" -
# whole upper-case words only, so the first real word ("Adverse ...") is never clipped.
_HEADING = re.compile(
    r"^\s*(\d+(\.\d+)*\s+)?(?:[A-Z][A-Z&/,\-()]+\s+)*[A-Z][A-Z&/,\-()]{3,}(\s*\(\s*\d+(\.\d+)*\s*\))?\s*:?(?=\s)"
)
_SPACES = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def clean(text: str) -> str:
    text = text.replace("\u2022", " ").replace("\u00a0", " ")
    text = _SPACES.sub(" ", text).strip()
    return _HEADING.sub("", text, count=1).strip()


def chunk_words(text: str, max_words: int = MAX_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def first_sentences(text: str, n: int = REF_SENTENCES, max_words: int = 80) -> str:
    sents = _SENT.split(text)
    out = " ".join(sents[:n])
    words = out.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def build_openfda(corpus: list[dict], qa: list[dict], rng: random.Random) -> None:
    for line in (RAW / "openfda_labels.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        drug, us = rec["drug"], rec["us_name"]
        names = [drug] if drug == us else [drug, drug, us]  # mostly INN, sometimes the US name
        for section, raw in rec["sections"].items():
            text = clean(raw)
            if len(text.split()) < 8 or section not in SECTION_TITLES:
                continue
            title = f"{drug.title()}: {SECTION_TITLES[section]}"
            chunks = chunk_words(text)[:CHUNKS_PER_SECTION]
            for k, ch in enumerate(chunks):
                corpus.append({"id": f"fda:{drug}:{section}:{k}", "title": title, "text": ch, "source": "openFDA"})
            for tmpl in rng.sample(TEMPLATES.get(section, []), k=min(2, len(TEMPLATES.get(section, [])))):
                qa.append({
                    "question": tmpl.format(d=rng.choice(names)),
                    "gold_id": f"fda:{drug}:{section}:0",
                    "answer": first_sentences(chunks[0]),
                    "focus": drug, "qtype": section, "source": "openFDA-templated",
                })


def build_medquad(corpus: list[dict], qa: list[dict]) -> None:
    base = next((RAW / "medquad").glob("MedQuAD-*"))
    for d in MEDQUAD_DIRS:
        for path in (base / d).glob("*.xml"):
            root = ET.parse(path).getroot()
            focus = (root.findtext("Focus") or "").strip()
            for pair in root.iter("QAPair"):
                q_el, a_el = pair.find("Question"), pair.find("Answer")
                if q_el is None or a_el is None or not (a_el.text or "").strip():
                    continue
                answer = clean(re.sub(r"^\s*(Summary|What is \(are\)[^?]*\?)\s*:?", "", a_el.text))
                if len(answer.split()) < 8:
                    continue
                pid = f"mq:{d.split('_')[0]}:{q_el.get('qid')}"  # qids restart in each subset
                corpus.append({"id": pid, "title": focus, "text": chunk_words(answer)[0], "source": f"MedQuAD/{d}"})
                qa.append({
                    "question": _SPACES.sub(" ", q_el.text or "").strip(),
                    "gold_id": pid, "answer": first_sentences(answer),
                    "focus": focus.lower(), "qtype": q_el.get("qtype"), "source": "MedQuAD",
                })


def build_medicationqa(corpus: list[dict], qa: list[dict]) -> None:
    for i, line in enumerate((RAW / "medicationqa.jsonl").open(encoding="utf-8")):
        row = json.loads(line)
        q, a = (row.get("Question") or "").strip(), clean(row.get("Answer") or "")
        if not q or len(a.split()) < 5:
            continue
        focus = (row.get("Focus (Drug)") or "").strip().lower()
        pid = f"mqa:{i}"
        corpus.append({"id": pid, "title": focus.title(), "text": chunk_words(a)[0], "source": "MedicationQA"})
        qa.append({
            "question": q, "gold_id": pid, "answer": first_sentences(a),
            "focus": focus, "qtype": (row.get("Question Type") or "").lower(), "source": "MedicationQA",
        })


def main() -> None:
    rng = random.Random(SEED)
    corpus: list[dict] = []
    qa: list[dict] = []
    build_openfda(corpus, qa, rng)
    build_medquad(corpus, qa)
    build_medicationqa(corpus, qa)

    corpus_df = pd.DataFrame(corpus).drop_duplicates("id")
    qa_df = pd.DataFrame(qa).drop_duplicates(["question", "gold_id"])
    qa_df = qa_df[qa_df["gold_id"].isin(corpus_df["id"])].reset_index(drop=True)

    # Group-aware split: 80 / 10 / 10 by focus entity.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, rest_idx = next(outer.split(qa_df, groups=qa_df["focus"]))
    rest = qa_df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(inner.split(rest, groups=rest["focus"]))
    splits = {"train": qa_df.iloc[train_idx], "val": rest.iloc[val_idx], "test": rest.iloc[test_idx]}

    corpus_df.to_json(PROCESSED / "corpus.jsonl", orient="records", lines=True, force_ascii=False)
    for name, df in splits.items():
        df.to_json(PROCESSED / f"qa_{name}.jsonl", orient="records", lines=True, force_ascii=False)

    stats = {
        "corpus_passages": len(corpus_df),
        "corpus_by_source": dict(Counter(s.split("/")[0] for s in corpus_df["source"])),
        "qa_total": len(qa_df),
        "qa_by_source": dict(qa_df["source"].value_counts()),
        "splits": {k: {"n": len(v), "focus_entities": v["focus"].nunique(), "by_source": dict(v["source"].value_counts())} for k, v in splits.items()},
        "passage_words_mean": round(corpus_df["text"].str.split().str.len().mean(), 1),
        "question_words_mean": round(qa_df["question"].str.split().str.len().mean(), 1),
    }
    (PROCESSED / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
