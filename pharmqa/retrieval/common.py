"""Corpus loading, BM25 index and retrieval metrics shared by all retrievers."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from pharmqa.config import PROCESSED, RESULTS

KS = (1, 3, 5, 10)
# openFDA titles look like "Paracetamol: Dosage"; the topic is the part before the colon.
_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on for and or is are be can i my me do does how what when which who why with "
           "should it this that if you your there about from at by as was were".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def load_corpus() -> pd.DataFrame:
    df = pd.read_json(PROCESSED / "corpus.jsonl", lines=True)
    df["full"] = df["title"].fillna("") + ". " + df["text"]
    return df.reset_index(drop=True)


def load_qa(split: str) -> pd.DataFrame:
    return pd.read_json(PROCESSED / f"qa_{split}.jsonl", lines=True)


class BM25Retriever:
    def __init__(self, corpus: pd.DataFrame):
        self.ids = corpus["id"].tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus["full"]])

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in top]


def rrf(rankings: list[list[str]], k: int = 60, top: int = 10) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009) of several ranked id lists."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r + 1)
    return [pid for pid, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]


def retrieval_metrics(qa: pd.DataFrame, ranked: list[list[str]]) -> dict:
    gold = qa["gold_id"].tolist()
    hits = np.array([[g in r[:k] for k in KS] for g, r in zip(gold, ranked)])
    rr = [1.0 / (r[:10].index(g) + 1) if g in r[:10] else 0.0 for g, r in zip(gold, ranked)]
    out = {f"recall@{k}": round(float(hits[:, i].mean()), 4) for i, k in enumerate(KS)}
    out["mrr@10"] = round(float(np.mean(rr)), 4)
    out["by_source"] = {}
    src = qa["source"].tolist()
    for s in sorted(set(src)):
        idx = [i for i, x in enumerate(src) if x == s]
        out["by_source"][s] = {
            "n": len(idx),
            "recall@1": round(float(hits[idx, 0].mean()), 4),
            "recall@5": round(float(hits[idx, 2].mean()), 4),
            "mrr@10": round(float(np.mean([rr[i] for i in idx])), 4),
        }
    return out


def save_retrieval(name: str, qa: pd.DataFrame, ranked: list[list[str]], corpus: pd.DataFrame) -> dict:
    m = retrieval_metrics(qa, ranked)
    # Topic accuracy: is the top passage about the right medicine/condition, even if it is a
    # different section (e.g. "side effects" instead of "dosage")? Separates entity errors
    # from section/aspect errors in the error analysis.
    title = dict(zip(corpus["id"], corpus["title"].fillna("").str.lower().str.split(":").str[0].str.strip()))
    m["topic@1"] = round(float(np.mean([title.get(r[0]) == title.get(g) for g, r in zip(qa["gold_id"], ranked)])), 4)
    m["model"] = name
    (RESULTS / f"retrieval_{name}.json").write_text(json.dumps(m, indent=2))
    text = dict(zip(corpus["id"], corpus["full"]))
    rows = []
    for (_, q), r in zip(qa.iterrows(), ranked):
        rows.append({
            "source": q["source"], "question": q["question"], "gold_id": q["gold_id"],
            "top1_id": r[0], "hit@1": q["gold_id"] == r[0], "hit@5": q["gold_id"] in r[:5],
            "top1_text": text.get(r[0], "")[:200],
        })
    pd.DataFrame(rows).to_csv(RESULTS / f"retrieval_{name}_predictions.csv", index=False)
    print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in m.items() if k not in ("by_source", "model")))
    return m
