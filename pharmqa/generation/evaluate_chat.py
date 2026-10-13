"""Evaluate the Qwen2.5-1.5B-Instruct chat reader, zero-shot vs. LoRA fine-tuned.

Uses the same stratified test questions and context conditions as generation/evaluate.py
(gold + 2 hard negatives, or the retriever's top 3), so results compare directly with the
Flan-T5 reader. Metrics:

  rougeL / rougeL_recall   lexical overlap with the reference (recall matters here because the
                           chat answers are longer than the references by design)
  bertscore_f1             semantic similarity (distilbert, as for Flan-T5)
  support                  share of answer content words that appear in the given passages, a
                           simple groundedness proxy (1.0 = every content word is in the context)
  abstain_on_unrelated     share of questions answered with "my sources don't cover this" when the
                           three passages are about other topics (higher = safer)
  false_abstain            share of gold-context questions wrongly answered with "don't cover"

Usage:  python -m pharmqa.generation.evaluate_chat --system lora --context gold [--n 200]
"""
from __future__ import annotations

import argparse
import json
import re

import evaluate
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

from pharmqa.config import CHAT_ADAPTER_DIR, RESULTS, SEED
from pharmqa.generation.chat_common import build_messages
from pharmqa.generation.chat_model import chat_generate, load_chat_model
from pharmqa.generation.evaluate import gold_mixed, retrieved_top3
from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of to and or in on for with is are be by as at it this that from you your may can
not if do does should will was were has have had which who what when how than then their them they its into
about any all more most other such these those also only i me my we our us""".split())
_ABSTAIN = re.compile(r"(?:don't|do not|doesn't|does not) (?:cover|answer|mention|contain)|can't answer|cannot answer", re.I)


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def support(answer: str, context: str) -> float:
    words = content_words(answer)
    if not words:
        return 1.0
    ctx = set(content_words(context))
    return sum(w in ctx for w in words) / len(words)


def unrelated_contexts(qa: pd.DataFrame, corpus: pd.DataFrame, text: dict) -> list[list[str]]:
    """Three BM25 hits that do not mention the question's focus (the answer is not in them)."""
    bm25 = BM25Retriever(corpus)
    out = []
    for q, g, focus in tqdm(zip(qa["question"], qa["gold_id"], qa["focus"]), total=len(qa), desc="unrelated"):
        name = str(focus).lower()
        hits = [pid for pid, _ in bm25.search(q, 30)
                if pid.split(":")[:2] != g.split(":")[:2] and name not in text[pid][:400].lower()]
        out.append(hits[:3])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["zeroshot", "lora"])
    ap.add_argument("--context", default="gold", choices=["gold", "retrieved"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_unrelated", type=int, default=60)
    ap.add_argument("--adapter", default=str(CHAT_ADAPTER_DIR))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    corpus = load_corpus()
    text = dict(zip(corpus["id"], corpus["full"]))
    qa = load_qa("test")
    qa = pd.concat(
        [g.sample(min(len(g), max(1, round(args.n * len(g) / len(qa)))), random_state=SEED) for _, g in qa.groupby("source")],
        ignore_index=True,
    )
    ctx_ids = gold_mixed(qa, corpus) if args.context == "gold" else retrieved_top3(qa, corpus)
    tok, model = load_chat_model(use_adapter=args.system == "lora", adapter=args.adapter)
    contexts = [[text[pid] for pid in c] for c in ctx_ids]
    preds = chat_generate(tok, model, [build_messages(q, c) for q, c in tqdm(zip(qa["question"], contexts), total=len(qa))])

    unrel_qa = qa.sample(min(args.n_unrelated, len(qa)), random_state=SEED)
    unrel_ctx = unrelated_contexts(unrel_qa, corpus, text)
    unrel_preds = chat_generate(tok, model, [build_messages(q, [text[p] for p in c]) for q, c in zip(unrel_qa["question"], unrel_ctx)])

    refs = qa["answer"].tolist()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [scorer.score(r, p)["rougeL"] for r, p in zip(refs, preds)]
    bs = evaluate.load("bertscore").compute(predictions=preds, references=refs, model_type="distilbert-base-uncased")["f1"]
    sup = [support(p, " ".join(c)) for p, c in zip(preds, contexts)]
    abstained = [bool(_ABSTAIN.search(p)) for p in preds]
    qa = qa.assign(prediction=preds, rougeL=[s.fmeasure for s in rl], rougeL_recall=[s.recall for s in rl],
                   bertscore_f1=bs, support=sup, abstained=abstained,
                   gold_in_context=[g in c for g, c in zip(qa["gold_id"], ctx_ids)])

    name = f"chat_{args.system}_{args.context}{args.tag}"
    summary = {
        "system": f"qwen2.5-1.5b_{args.system}{args.tag}", "context": args.context, "n": len(qa),
        "rougeL": round(float(qa["rougeL"].mean()), 4), "rougeL_recall": round(float(qa["rougeL_recall"].mean()), 4),
        "bertscore_f1": round(float(np.mean(bs)), 4), "support": round(float(np.mean(sup)), 4),
        "avg_words": round(float(np.mean([len(p.split()) for p in preds])), 1),
        "false_abstain": round(float(qa.loc[qa["gold_in_context"], "abstained"].mean()), 4),
        "abstain_on_unrelated": round(float(np.mean([bool(_ABSTAIN.search(p)) for p in unrel_preds])), 4),
        "by_source": {s: {"rougeL": round(float(g["rougeL"].mean()), 4), "bertscore_f1": round(float(g["bertscore_f1"].mean()), 4)}
                      for s, g in qa.groupby("source")},
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    qa[["source", "question", "answer", "prediction", "rougeL", "rougeL_recall", "bertscore_f1", "support", "abstained",
        "gold_in_context"]].to_csv(RESULTS / f"{name}_predictions.csv", index=False)
    pd.DataFrame({"question": unrel_qa["question"].tolist(), "prediction": unrel_preds}).to_csv(
        RESULTS / f"{name}_unrelated.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
