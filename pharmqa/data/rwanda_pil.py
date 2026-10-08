"""Rwanda FDA Patient Information Leaflets (PILs) - Rwanda-registered medicines.

Source: https://rwandafda.gov.rw/patient-information-leaflet-pil-2/ (official PDFs, one per
registered product). These are patient-facing documents with the brand names, strengths and
wording Rwandan patients actually see, so they complement the US-centric openFDA/MedQuAD data.

The FDA site rate-limits and blocks scripted downloads, so the text is extracted once from a
local PDF cache and committed as data/external/rwanda_fda_pil.jsonl.gz (text only, with the
official URL of every leaflet for attribution). prepare.py then calls build().

About half of the cached PDFs are scanned images with no text layer. They are excluded rather
than OCR'd: OCR took ~40 s/page on CPU, merged words ("Eachfilmcoated"), and the scans are
mostly older prescriber-style inserts without the patient-leaflet section headings.

Usage (one-off, needs the PDF cache):
    python -m pharmqa.data.rwanda_pil extract --source <dir with checkpoint.json + PDFs>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from pharmqa.config import DATA
from pharmqa.data.essential_medicines import ESSENTIAL_MEDICINES

PIL_PATH = DATA / "external" / "rwanda_fda_pil.jsonl.gz"
SOURCE = "RwandaFDA-PIL"
MAX_WORDS = 160
CHUNKS_PER_SECTION = 2
MIN_WORDS = 400  # drops scanned PDFs (no text layer) and title-only stubs

# Headings of the patient-leaflet template, plus the prescriber-style (SmPC) headings that
# many older Rwanda-registered inserts use -> section key.
SECTION_HEADINGS: list[tuple[str, str]] = [
    (r"what\s+[^.?\n]{0,120}?\s+(?:is|are)\s+and\s+(?:what|when)\s+(?:it|they)\s+(?:is|are)\s+used\s+for", "uses"),
    (r"what\s+[^.?\n]{0,120}?\s+(?:is|are)\s+and\s+when\s+to\s+use\s+it", "uses"),
    (r"what\s+[^.?\n]{0,120}?\s+(?:is|are)\s+used\s+for", "uses"),
    (r"therapeutic\s+indications?|indications?\s+and\s+usage|\bindications?\s*:", "uses"),
    (r"what\s+you\s+need\s+to\s+know\s+before\s+(?:you\s+)?(?:take|use|are\s+given)", "before"),
    (r"before\s+you\s+(?:take|use)", "before"),
    (r"\bcontra-?indications?\b", "before"),
    (r"how\s+to\s+(?:take|use)\b", "dosage"),
    (r"posology\s+and\s+method\s+of\s+administration|dosage\s+and\s+administration", "dosage"),
    (r"possible\s+side\s+effects?|undesirable\s+effects|adverse\s+(?:reactions|effects)\s*:?", "side_effects"),
    (r"how\s+to\s+store|special\s+precautions\s+for\s+storage|storage\s+conditions?", "storage"),
]
_USE_HINT = re.compile(r"\b(used|indicated|treat(?:s|ment)?|prevent(?:s|ion)?)\b", re.I)
_HEADINGS_RE = re.compile("|".join(f"(?P<h{i}>{p})" for i, (p, _) in enumerate(SECTION_HEADINGS)), re.I)

SECTION_TITLES = {"uses": "Uses", "before": "Before you take", "dosage": "How to take",
                  "side_effects": "Side effects", "storage": "Storage"}

# Question wording is templated (as for openFDA); answers are the real leaflet text.
TEMPLATES = {
    "uses": ["What is {d} used for?", "What does {d} treat?", "Why would I be given {d}?"],
    "before": ["Who should not take {d}?", "What should I know before taking {d}?", "Is {d} safe for me to take?"],
    "dosage": ["How should I take {d}?", "What is the dose of {d}?", "How often do I take {d}?"],
    "side_effects": ["What are the side effects of {d}?", "Can {d} cause side effects?", "What problems can {d} cause?"],
    "storage": ["How should I store {d}?", "How do I keep {d} at home?"],
}

_STOP_WORDS = set(
    "film coated tablet tablets tab tabs caps capsule capsules cream ointment gel syrup suspension solution "
    "injection injectable infusion intravenous iv oral powder drop drops eye ear nasal spray for and sachet "
    "sachets dispersible chewable lotion inhaler vial vials ampoule ampoules pil bp usp eur wv w v mg mcg g ml "
    "iu per eq in of with pessary pessaries emulsion concentrate sterile single dose".split())
# INN stems (WHO stem book): a word with one of these endings is almost certainly a generic name.
_STRONG_STEM = re.compile(r"(cillin|mycin|micin|floxacin|azole|vir|mab|pril|sartan|olol|dipine|statin|tidine|"
                          r"prazole|xaban|gliptin|formin|profen|fenac|caine|cycline|cef\w+|azepam|tinib|platin|"
                          r"setron|parin|lukast|triptan|semide|thiazide|prost)$", re.I)
_WEAK_STEM = re.compile(r"(ine|ide|ate|one|ol|in|ium)$", re.I)


def _words(title: str) -> list[str]:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])(\d)|(\d)([A-Za-z])", lambda m: " ".join(g for g in m.groups() if g), t)
    return re.sub(r"[^A-Za-z0-9+ ]", " ", t).split()


def _clean(text: str) -> str:
    text = text.replace("\u2022", " ").replace("\u00a0", " ").replace("\uf0b7", " ")
    text = re.sub(r"Page \d+ of \d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_from_url(url: str) -> str:
    name = unquote(re.split(r"[/#]", url)[-1]).replace("_20", " ")
    name = re.sub(r"\.pdf$", "", name, flags=re.I).replace("_", " ")
    name = re.sub(r"\s*-?\s*PIL\s*$", "", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip(" -")


def _is_generic(word: str, first_is_generic: bool) -> bool:
    w = word.lower()
    return w in _INN_WORDS or bool(_STRONG_STEM.search(w)) or (not first_is_generic and bool(_WEAK_STEM.search(w)) and len(w) > 5)


_INN_WORDS = {w for inn in ESSENTIAL_MEDICINES for w in inn.split() if len(w) > 4}


def names_for(title: str, text: str) -> tuple[str, str]:
    """(brand shown in questions, active-ingredient key used to group splits)."""
    words = _words(title)
    lead: list[str] = []
    for w in words:
        if w.lower() in _STOP_WORDS or w[0].isdigit() or len(lead) == 3:
            break
        lead.append(w)
    lead = lead or words[:1]
    first_generic = _is_generic(lead[0], False)
    for i in range(1, len(lead)):
        if _is_generic(lead[i], first_generic):
            lead = lead[:i]
            break
    brand = " ".join(w.title() if w.isupper() and len(w) > 3 else w for w in lead)

    rest = [w.lower() for w in words[len(lead):] if w.lower() not in _STOP_WORDS and not w[0].isdigit()]
    head = (title + " " + " ".join(text.split()[:200])).lower()
    hits = [inn for inn in ESSENTIAL_MEDICINES if re.search(rf"\b{re.escape(inn)}\b", head)]
    generic_rest = [w for w in rest if _is_generic(w, False)]
    if hits:
        ingredient = max(hits, key=len)
    elif generic_rest:
        ingredient = generic_rest[0]
    else:
        ingredient = brand.lower()
    return brand, ingredient


def split_sections(text: str) -> dict[str, str]:
    """Longest segment after each heading wins (the leaflet's table of contents repeats
    every heading with almost no text in between)."""
    marks = []
    for m in _HEADINGS_RE.finditer(text):
        idx = int(next(k for k, v in m.groupdict().items() if v)[1:])
        marks.append((m.start(), m.end(), SECTION_HEADINGS[idx][1]))
    out: dict[str, str] = {}
    for i, (_, end, key) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = re.sub(r"^[\s:.\-)]*", "", text[end:stop])
        seg = re.sub(r"\s*\d\s*[.)]?\s*$", "", seg).strip()
        if len(seg.split()) > len(out.get(key, "").split()):
            out[key] = seg
    # Leaflets usually open by saying what the medicine is and what it treats.
    if "uses" not in out:
        opening = " ".join(text.split()[:MAX_WORDS])
        if _USE_HINT.search(opening):
            out["uses"] = opening
    return {k: v for k, v in out.items() if len(v.split()) >= 15}


def build(corpus: list[dict], qa: list[dict], rng: random.Random) -> None:
    if not PIL_PATH.exists():
        print(f"[rwanda_pil] {PIL_PATH} missing - skipping Rwanda FDA leaflets")
        return
    seen_text: set[str] = set()
    with gzip.open(PIL_PATH, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            rec = json.loads(line)
            brand, ingredient = rec["brand"], rec["ingredient"]
            for section, body in split_sections(rec["text"]).items():
                words = body.split()
                chunks = [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)][:CHUNKS_PER_SECTION]
                if chunks[0] in seen_text:  # identical leaflets reused across strengths
                    continue
                seen_text.add(chunks[0])
                for k, ch in enumerate(chunks):
                    corpus.append({"id": f"rw:{n}:{section}:{k}", "title": f"{brand}: {SECTION_TITLES[section]}",
                                   "text": ch, "source": SOURCE, "url": rec["url"]})
                for tmpl in rng.sample(TEMPLATES[section], k=2):
                    qa.append({
                        "question": tmpl.format(d=brand), "gold_id": f"rw:{n}:{section}:0",
                        "answer": " ".join(chunks[0].split()[:80]),
                        "focus": ingredient, "qtype": section, "source": SOURCE,
                    })


# ---------------------------------------------------------------- one-off extraction

def _batch_texts(source: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for f in source.glob("agent_batches/*/*.json"):
        for item in json.loads(f.read_text(encoding="utf-8")).get("items", []):
            if item.get("pil_text"):
                texts[item["source_url"]] = item["pil_text"]
    return texts


def _pdf_by_digest(source: Path) -> dict[str, Path]:
    return {p.name[:24]: p for p in source.rglob("*") if p.is_file() and re.match(r"[0-9a-f]{24}_", p.name)}


def _pdf_text(pdf: Path) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    return "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))


def extract(source: Path) -> None:
    urls = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))["processed_pdf_urls"]
    batch, pdfs = _batch_texts(source), _pdf_by_digest(source)
    # Batch texts first; then every cached PDF (matched to its URL by the cache's sha256 prefix,
    # otherwise attributed to the PIL index page).
    jobs: dict[str, tuple[str, Path | None]] = {}
    for url in dict.fromkeys(urls):
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        if url in batch or digest in pdfs:
            jobs[url] = (url, pdfs.pop(digest, None))
    for digest, pdf in pdfs.items():
        jobs[f"file:{pdf.name}"] = ("https://rwandafda.gov.rw/patient-information-leaflet-pil-2/#" + pdf.name[25:], pdf)

    rows, stats = [], {"from_batch": 0, "from_pdf": 0, "too_short": 0, "failed": 0}
    for key, (url, pdf) in jobs.items():
        text = batch.get(key)
        if text:
            stats["from_batch"] += 1
        else:
            try:
                text = _pdf_text(pdf)
                stats["from_pdf"] += 1
            except Exception:  # noqa: BLE001 - some FDA-hosted PDFs are corrupt
                stats["failed"] += 1
                continue
        text = _clean(text)
        if len(text.split()) < MIN_WORDS:
            stats["too_short"] += 1
            continue
        title = _title_from_url(url)
        brand, ingredient = names_for(title, text)
        rows.append({"url": url, "title": title, "brand": brand, "ingredient": ingredient, "text": text})

    PIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(PIL_PATH, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "kept": len(rows), "ingredients": len({r["ingredient"] for r in rows})}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--source", type=Path, required=True)
    args = ap.parse_args()
    extract(args.source)


if __name__ == "__main__":
    main()
