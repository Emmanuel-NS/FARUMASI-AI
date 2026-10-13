"""Fetch a few extra openFDA labels the original download got wrong or missed.

The main download keeps, per medicine, the openFDA label with the richest patient sections. For a
few medicines that picks the wrong product (metformin -> a sitagliptin/metformin combination;
levonorgestrel -> a daily combined pill instead of the morning-after pill), and ulipristal (the
emergency pill that works up to 120 hours after sex) was missing. These passages are
appended to the corpus at app start-up and embedded with the trained retriever; no model is
retrained and the evaluation corpus is unchanged. For the same medicines an "Examples" passage
lists brand names found on US FDA labels, so "examples of those pills" can be answered.

Usage:  python scripts/fetch_supplementary.py            (fetch + embed)
        python scripts/fetch_supplementary.py --embed    (re-embed after retraining the retriever)
"""
from __future__ import annotations

import json
import re

import requests

from pharmqa.config import RETRIEVER_MODEL_DIR, SUPPLEMENTARY_EMB_PATH, SUPPLEMENTARY_PATH as OUT
from pharmqa.data.download import OPENFDA_URL, PATIENT_SECTIONS
from pharmqa.data.prepare import CHUNKS_PER_SECTION, SECTION_TITLES, chunk_words, clean

# (corpus drug name, display note, openFDA search)
LABELS = [
    ("levonorgestrel", "emergency contraceptive (morning-after pill)",
     'openfda.generic_name:"levonorgestrel" AND openfda.product_type:"HUMAN OTC DRUG"'),
    ("metformin", "single-ingredient tablets",
     'openfda.generic_name.exact:"METFORMIN HYDROCHLORIDE" AND _exists_:adverse_reactions'),
    ("ulipristal", "emergency contraceptive for up to 120 hours (5 days) after sex",
     'openfda.generic_name:"ulipristal"'),
]
MAX_BRANDS = 12


def brand_names(search: str) -> list[str]:
    r = requests.get(OPENFDA_URL, params={"search": search, "count": "openfda.brand_name.exact", "limit": 50}, timeout=30)
    r.raise_for_status()
    names = []
    for row in r.json()["results"]:
        name = re.sub(r"\s+(Tm|Er|Xr)$", "", row["term"].strip().title())
        if name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return names


def is_generic(name: str, drug: str) -> bool:
    low = name.lower()
    return drug in low or low in {"morning after", "morning after pill", "afterpill"} or low.endswith(" morning after pill")


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
        brands = [b for b in brand_names(search) if not is_generic(b, drug)][:MAX_BRANDS]
        if brands:
            rows.append({
                "id": f"fdax:{drug}:examples:0", "title": f"{drug.title()}: Examples",
                "text": (f"Examples of {drug} products ({note}) on US FDA labels: {', '.join(brands)}. "
                         "Brand names differ between countries, and these sources do not include a list of "
                         f"products registered in Rwanda; ask a pharmacist which {drug} products are available "
                         "near you."),
                "source": f"openFDA ({note})", "url": "https://open.fda.gov/apis/drug/label/",
            })
        print(drug, label.get("openfda", {}).get("brand_name"), "| brands:", brands)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} passages -> {OUT}")
    embed()


def embed() -> None:
    """Embed the passages with the fine-tuned retriever, so the app does no model compute at start-up."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in OUT.open(encoding="utf-8")]
    model = SentenceTransformer(str(RETRIEVER_MODEL_DIR), device="cpu")
    emb = model.encode([f"{r['title']}. {r['text']}" for r in rows], batch_size=64, normalize_embeddings=True)
    np.save(SUPPLEMENTARY_EMB_PATH, emb)
    print(f"embeddings {emb.shape} -> {SUPPLEMENTARY_EMB_PATH}")


if __name__ == "__main__":
    import sys

    embed() if "--embed" in sys.argv else main()
