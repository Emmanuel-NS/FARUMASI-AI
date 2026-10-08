"""Stage 3: build the labelled data for the domain/scope classifier.

Labels
  medicine_info     questions about a medicine (use, dose, side effects, interactions...)
  condition_advice  questions about a symptom, condition or injury (headache, wounds...)
  out_of_domain     anything else (banking, travel, small talk...)

Sources (all real questions, labels come from each dataset's own metadata):
  medicine_info     MedicationQA + MedQuAD 11_MPlusDrugs + 12_MPlusHerbsSupplements
  condition_advice  MedQuAD condition subsets incl. 10_MPlus_ADAM (A.D.A.M. encyclopedia)
  out_of_domain     CLINC150 (all 150 intents + its out-of-scope class)

An augmented training file (intent_train_consumer.jsonl) adds short real patient questions
from HealthCareMagic with weak heuristic labels; evaluation sets are never augmented.

MedQuAD subsets 10-12 had their *answers* removed for copyright, but the questions remain
and are valid classifier inputs. Splits for MedQuAD/MedicationQA are grouped by focus;
CLINC150 keeps its official train/validation/test split.

Usage:  python -m pharmqa.data.intent_data
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pharmqa.config import PROCESSED, RAW, SEED
from pharmqa.data.essential_medicines import ESSENTIAL_MEDICINES

MED_DIRS = ["11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"]
COND_DIRS = [
    "10_MPlus_ADAM_QA", "1_CancerGov_QA", "4_MPlus_Health_Topics_QA", "5_NIDDK_QA",
    "6_NINDS_QA", "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA",
]
MAX_PER_CLASS_TRAIN = 4000
_WS = re.compile(r"\s+")


def _medquad_questions(dirs: list[str], label: str) -> list[dict]:
    base = next((RAW / "medquad").glob("MedQuAD-*"))
    rows = []
    for d in dirs:
        for path in (base / d).glob("*.xml"):
            root = ET.parse(path).getroot()
            focus = (root.findtext("Focus") or "").strip().lower()
            for q in root.iter("Question"):
                text = _WS.sub(" ", q.text or "").strip()
                if text:
                    rows.append({"text": text, "label": label, "group": f"{label}:{focus}", "source": f"MedQuAD/{d}"})
    return rows


def _cap_per_label(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    parts = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby("label")]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    rows = _medquad_questions(MED_DIRS, "medicine_info") + _medquad_questions(COND_DIRS, "condition_advice")
    for line in (RAW / "medicationqa.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r.get("Question") or "").strip():
            focus = (r.get("Focus (Drug)") or "").strip().lower()
            rows.append({"text": r["Question"].strip(), "label": "medicine_info", "group": f"medicine_info:{focus}", "source": "MedicationQA"})

    health = pd.DataFrame(rows).drop_duplicates("text")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, rest = next(outer.split(health, groups=health["group"]))
    rest_df = health.iloc[rest]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    va, te = next(inner.split(rest_df, groups=rest_df["group"]))
    health_splits = {"train": health.iloc[tr], "val": rest_df.iloc[va], "test": rest_df.iloc[te]}

    clinc = pd.read_json(RAW / "clinc150.jsonl", lines=True)
    clinc_split = {"train": "train", "val": "validation", "test": "test"}

    stats = {}
    for name, hdf in health_splits.items():
        ood = clinc[clinc["split"] == clinc_split[name]].assign(label="out_of_domain", source="CLINC150")
        ood = ood.assign(group="ood:" + ood["intent"])[["text", "label", "group", "source"]]
        df = pd.concat([hdf, ood], ignore_index=True)
        # Cap each class so no label dominates; evaluation sets stay roughly balanced so
        # macro-F1 is meaningful.
        cap = MAX_PER_CLASS_TRAIN if name == "train" else int(df["label"].value_counts().min() * 1.5)
        df = _cap_per_label(df, cap).sample(frac=1, random_state=SEED).reset_index(drop=True)
        df.to_json(PROCESSED / f"intent_{name}.jsonl", orient="records", lines=True, force_ascii=False)
        stats[name] = {k: int(v) for k, v in df["label"].value_counts().items()}

    consumer = consumer_questions()
    aug = pd.concat([pd.read_json(PROCESSED / "intent_train.jsonl", lines=True), consumer], ignore_index=True)
    aug = aug.sample(frac=1, random_state=SEED).reset_index(drop=True)
    aug.to_json(PROCESSED / "intent_train_consumer.jsonl", orient="records", lines=True, force_ascii=False)
    stats["train_consumer"] = {k: int(v) for k, v in aug["label"].value_counts().items()}
    stats["consumer_added"] = {k: int(v) for k, v in consumer["label"].value_counts().items()}

    (PROCESSED / "intent_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


_GREETING = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening))\b[^,.!?]*[,.!]?\s*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MED_WORDS = re.compile(
    r"\b(tablets?|pills?|capsules?|dose|dosage|mg|medicines?|medications?|syrup|drugs?|injections?|"
    r"antibiotics?|side effects?|prescribed|panadol|coartem|augmentin|flagyl|" + "|".join(
        re.escape(n) for n in sorted({w for k, v in ESSENTIAL_MEDICINES.items() for w in (k, v) if " " not in w}, key=len, reverse=True)
    ) + r")\b",
    re.I,
)


def consumer_questions(per_class: int = 2500) -> pd.DataFrame:
    """Turn long HealthCareMagic patient messages into short chat-style questions.

    Picks the patient's first short question sentence (or first short sentence) and labels
    it with a transparent heuristic: mentions a medicine / dose word -> medicine_info,
    otherwise condition_advice. These weak labels are used for *training only*.
    """
    rows = []
    for line in (RAW / "healthcaremagic_20k.jsonl").open(encoding="utf-8"):
        text = _GREETING.sub("", json.loads(line)["text"]).strip()
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if 4 <= len(s.split()) <= 30]
        if not sents:
            continue
        q = next((s for s in sents if s.endswith("?")), sents[0])
        label = "medicine_info" if _MED_WORDS.search(q) else "condition_advice"
        rows.append({"text": q, "label": label, "group": "hcm", "source": "HealthCareMagic"})
    df = pd.DataFrame(rows).drop_duplicates("text")
    return _cap_per_label(df, per_class)


if __name__ == "__main__":
    main()
