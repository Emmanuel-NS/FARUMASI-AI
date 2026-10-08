"""Intent model: fine-tune DistilBERT (Sanh et al., 2019) for 3-way sequence classification.

Architecture: DistilBERT encoder (6 Transformer layers, 768-d, 66M params) -> [CLS] vector
-> dropout -> linear layer (768 -> 3) -> softmax. All weights are fine-tuned with
cross-entropy loss.

Usage:  python -m pharmqa.intent.transformer [--epochs 3 --lr 2e-5 --batch 32]
Run on a GPU (Colab/Kaggle); CPU works but is slow.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from pharmqa.config import INTENT_BASE_MODEL, INTENT_LABELS, INTENT_MODEL_DIR, SEED
from pharmqa.metrics import evaluate_intent_model, load_intent

L2I = {l: i for i, l in enumerate(INTENT_LABELS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--name", default="distilbert")
    ap.add_argument("--train_file", default="train", choices=["train", "train_consumer"])
    args = ap.parse_args()
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(INTENT_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_BASE_MODEL, num_labels=len(INTENT_LABELS),
        id2label=dict(enumerate(INTENT_LABELS)), label2id=L2I,
    )

    def to_ds(split: str) -> Dataset:
        df = load_intent(split)
        ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": [L2I[l] for l in df["label"]]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    train_ds, val_ds = to_ds(args.train_file), to_ds("val")

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    out_dir = INTENT_MODEL_DIR if args.name == "distilbert" else INTENT_MODEL_DIR.with_name(f"intent_{args.name}")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=128, weight_decay=0.01,
            warmup_steps=int(0.1 * args.epochs * len(train_ds) / args.batch), eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="accuracy", save_total_limit=1, logging_steps=50, report_to="none",
            fp16=torch.cuda.is_available(), seed=SEED,
        ),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    model.eval()
    device = model.device

    @torch.no_grad()
    def predict(texts: list[str]) -> list[str]:
        preds = []
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i + 128], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(device)
            preds.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
        return [INTENT_LABELS[p] for p in preds]

    evaluate_intent_model(args.name, predict)


if __name__ == "__main__":
    main()
