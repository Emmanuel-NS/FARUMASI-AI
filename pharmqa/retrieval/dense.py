"""Dense retrieval with a sentence-transformer bi-encoder (Reimers & Gurevych, 2019).

Model: all-MiniLM-L6-v2 (6-layer MiniLM, 384-d mean-pooled embeddings). Questions and
passages are encoded independently; relevance = cosine similarity.

Experiments
  zeroshot   pretrained MiniLM, no domain training
  finetune   MiniLM fine-tuned on (question, gold passage) pairs from qa_train with
             MultipleNegativesRankingLoss (in-batch negatives; Henderson et al., 2017)
  hybrid     fine-tuned MiniLM + BM25 combined with reciprocal rank fusion

Usage:  python -m pharmqa.retrieval.dense --mode all [--epochs 2 --batch 64 --lr 2e-5]
"""
from __future__ import annotations

import argparse

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from tqdm import tqdm

from pharmqa.config import CORPUS_EMB_PATH, RETRIEVER_BASE_MODEL, RETRIEVER_MODEL_DIR, SEED
from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa, rrf, save_retrieval


def dense_rank(model: SentenceTransformer, corpus, questions: list[str], k: int = 10, emb=None):
    if emb is None:
        emb = model.encode(corpus["full"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    q = model.encode(questions, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    sims = q @ emb.T
    top = np.argsort(-sims, axis=1)[:, :k]
    ids = corpus["id"].tolist()
    return [[ids[j] for j in row] for row in top], emb


def finetune(epochs: float, batch: int, lr: float) -> SentenceTransformer:
    corpus = load_corpus()
    text = dict(zip(corpus["id"], corpus["full"]))
    tr, va = load_qa("train"), load_qa("val")
    to_ds = lambda df: Dataset.from_dict({"anchor": df["question"].tolist(), "positive": [text[g] for g in df["gold_id"]]})

    model = SentenceTransformer(RETRIEVER_BASE_MODEL)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(RETRIEVER_MODEL_DIR / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch, learning_rate=lr,
        warmup_steps=int(0.1 * epochs * len(tr) / batch), batch_sampler=BatchSamplers.NO_DUPLICATES,  # avoid false in-batch negatives
        eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to="none", seed=SEED,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=to_ds(tr), eval_dataset=to_ds(va),
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save(str(RETRIEVER_MODEL_DIR))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["zeroshot", "finetune", "all"])
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    corpus, qa = load_corpus(), load_qa("test")
    questions = qa["question"].tolist()

    if args.mode in ("zeroshot", "all"):
        ranked, _ = dense_rank(SentenceTransformer(RETRIEVER_BASE_MODEL), corpus, questions)
        save_retrieval("minilm_zeroshot", qa, ranked, corpus)

    if args.mode in ("finetune", "all"):
        model = finetune(args.epochs, args.batch, args.lr)
        ranked, emb = dense_rank(model, corpus, questions, k=50)
        np.save(CORPUS_EMB_PATH, emb.astype(np.float32))
        save_retrieval("minilm_finetuned", qa, [r[:10] for r in ranked], corpus)

        bm25 = BM25Retriever(corpus)
        bm = [[pid for pid, _ in bm25.search(q, 50)] for q in tqdm(questions, desc="BM25 for hybrid")]
        save_retrieval("hybrid_rrf", qa, [rrf([d, b]) for d, b in zip(ranked, bm)], corpus)


if __name__ == "__main__":
    main()
