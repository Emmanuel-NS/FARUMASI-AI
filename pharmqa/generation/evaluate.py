"""Evaluate answer generation.

Systems
  extractive  first 3 sentences of the top-ranked passage (no learning)
  zeroshot    Flan-T5-base reader with the same prompt, no fine-tuning
  lora        Flan-T5-base reader + our LoRA adapter

Context (always three passages)
  gold        gold passage + 2 BM25 hard negatives, shuffled ("oracle retrieval"): isolates
              the reader's ability to pick the right passage and phrase the answer. For the
              extractive system this condition uses the gold passage itself and is reported
              as an upper bound, since copying is what the references reward.
  retrieved   top-3 passages from the fine-tuned retriever (end-to-end system quality; falls
              back to BM25 if the retriever has not been trained)

Metrics: ROUGE-L F1 (Lin, 2004) for lexical overlap and BERTScore F1 (Zhang et al., 2020)
for semantic similarity with the reference answer. Both are reported because medical
answers can be correct with different wording, and lexical overlap alone under-rewards that.

Usage:  python -m pharmqa.generation.evaluate --system lora --context retrieved [--n 400]
"""
from __future__ import annotations

import argparse
import json
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pharmqa.config import (CORPUS_EMB_PATH, GENERATOR_ADAPTER_DIR, GENERATOR_BASE_MODEL, RESULTS,
                            RETRIEVER_MODEL_DIR, SEED)
from pharmqa.generation.common import build_prompt, extractive_answer, mix_with_negatives
from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa


def retrieved_top3(qa: pd.DataFrame, corpus: pd.DataFrame) -> list[list[str]]:
    ids = corpus["id"].tolist()
    if RETRIEVER_MODEL_DIR.exists() and CORPUS_EMB_PATH.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(RETRIEVER_MODEL_DIR))
        emb = np.load(CORPUS_EMB_PATH)
        q = model.encode(qa["question"].tolist(), normalize_embeddings=True, batch_size=128)
        return [[ids[j] for j in row] for row in np.argsort(-(q @ emb.T), axis=1)[:, :3]]
    bm25 = BM25Retriever(corpus)
    return [[pid for pid, _ in bm25.search(q, 3)] for q in tqdm(qa["question"], desc="BM25")]


def gold_mixed(qa: pd.DataFrame, corpus: pd.DataFrame) -> list[list[str]]:
    bm25, rng = BM25Retriever(corpus), random.Random(SEED)
    out = []
    for q, g in tqdm(zip(qa["question"], qa["gold_id"]), total=len(qa), desc="hard negatives"):
        negs = [pid for pid, _ in bm25.search(q, 4) if pid != g][:2]
        out.append(mix_with_negatives(g, negs, rng))
    return out


def load_generator(system: str, adapter: str = str(GENERATOR_ADAPTER_DIR)):
    tok = AutoTokenizer.from_pretrained(GENERATOR_BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_BASE_MODEL)
    if system == "lora":
        from pharmqa.compat import hide_incompatible_torchao
        hide_incompatible_torchao()
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device).eval(), device


@torch.no_grad()
def generate(tok, model, device, prompts: list[str], batch: int = 16) -> list[str]:
    out = []
    for i in tqdm(range(0, len(prompts), batch), desc="generate"):
        enc = tok(prompts[i:i + batch], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=128, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
        out.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["extractive", "zeroshot", "lora"])
    ap.add_argument("--context", default="retrieved", choices=["gold", "retrieved"])
    ap.add_argument("--n", type=int, default=400, help="test questions to evaluate (stratified by source)")
    ap.add_argument("--adapter", default=str(GENERATOR_ADAPTER_DIR))
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. _no_rwandafda")
    args = ap.parse_args()

    corpus = load_corpus()
    text = dict(zip(corpus["id"], corpus["full"]))
    qa = load_qa("test")
    qa = pd.concat(
        [g.sample(min(len(g), max(1, round(args.n * len(g) / len(qa)))), random_state=SEED) for _, g in qa.groupby("source")],
        ignore_index=True,
    )
    ctx_ids = gold_mixed(qa, corpus) if args.context == "gold" else retrieved_top3(qa, corpus)

    if args.system == "extractive":
        first = qa["gold_id"].tolist() if args.context == "gold" else [c[0] for c in ctx_ids]
        preds = [extractive_answer(text[pid]) for pid in first]
    else:
        tok, model, device = load_generator(args.system, args.adapter)
        prompts = [build_prompt(q, [text[pid] for pid in c]) for q, c in zip(qa["question"], ctx_ids)]
        preds = generate(tok, model, device, prompts)

    refs = qa["answer"].tolist()
    rouge = evaluate.load("rouge").compute(predictions=preds, references=refs, use_aggregator=False)["rougeL"]
    bs = evaluate.load("bertscore").compute(predictions=preds, references=refs, model_type="distilbert-base-uncased")["f1"]
    qa = qa.assign(prediction=preds, rougeL=rouge, bertscore_f1=bs,
                   gold_in_context=[g in c for g, c in zip(qa["gold_id"], ctx_ids)],
                   context_ids=[" | ".join(c) for c in ctx_ids])

    name = f"gen_{args.system}_{args.context}{args.tag}"
    summary = {
        "system": args.system + args.tag, "context": args.context, "n": len(qa),
        "rougeL": round(float(np.mean(rouge)), 4), "bertscore_f1": round(float(np.mean(bs)), 4),
        "avg_words": round(float(np.mean([len(p.split()) for p in preds])), 1),
        "gold_in_context_rate": round(float(qa["gold_in_context"].mean()), 4),
        "rougeL_when_gold_in_context": round(float(qa.loc[qa["gold_in_context"], "rougeL"].mean()), 4),
        "by_source": {s: {"rougeL": round(float(g["rougeL"].mean()), 4), "bertscore_f1": round(float(g["bertscore_f1"].mean()), 4)}
                      for s, g in qa.groupby("source")},
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    qa[["source", "question", "answer", "prediction", "rougeL", "bertscore_f1", "gold_in_context", "context_ids"]].to_csv(
        RESULTS / f"{name}_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
