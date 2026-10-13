"""Fetch a few extra openFDA labels the original download got wrong or missed.

The main download keeps, per medicine, the openFDA label with the richest patient sections. For a
few medicines that picks the wrong product (metformin -> a sitagliptin/metformin combination;
levonorgestrel -> a daily combined pill instead of the morning-after pill). These passages are
appended to the corpus at app start-up and embedded with the trained retriever; no model is
retrained and the evaluation corpus is unchanged.

Usage:  python scripts/fetch_supplementary.py
"""
from __future__ import annotations

import json

import requests

from pharmqa.config import DATA
from pharmqa.data.download import OPENFDA_URL, PATIENT_SECTIONS
from pharmqa.data.prepare import CHUNKS_PER_SECTION, SECTION_TITLES, chunk_words, clean

OUT = DATA / "external" / "supplementary_labels.jsonl"

# (corpus drug name, display note, openFDA search)
LABELS = [
    ("levonorgestrel", "emergency contraceptive (morning-after pill)",
     'openfda.generic_name:"levonorgestrel" AND openfda.product_type:"HUMAN OTC DRUG"'),
    ("metformin", "single-ingredient tablets",
     'openfda.generic_name.exact:"METFORMIN HYDROCHLORIDE" AND _exists_:adverse_reactions'),
]


def main() -> None:
    rows = []
    for drug, note, search in LABELS:
        r = requests.get(OPENFDA_URL, params={"search": search, "limit": 5}, timeout=30)
        r.raise_for_status()
        label = max(r.json()["results"], key=lambda x: sum(1 for s in PATIENT_SECTIONS if x.get(s)))
        url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={label['set_id']}"
        for section in PATIENT_SECTIONS:
            if not label.get(section) or section not in SECTION_TITLES:
                continue
            text = clean(" ".join(label[section]))
            if len(text.split()) < 8:
                continue
            for k, chunk in enumerate(chunk_words(text)[:CHUNKS_PER_SECTION]):
                rows.append({"id": f"fdax:{drug}:{section}:{k}", "title": f"{drug.title()}: {SECTION_TITLES[section]}",
                             "text": chunk, "source": f"openFDA ({note})", "url": url})
        print(drug, label.get("openfda", {}).get("brand_name"), label.get("openfda", {}).get("generic_name"))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} passages -> {OUT}")


if __name__ == "__main__":
    main()
