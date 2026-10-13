"""Fine-tune Flan-T5-base (Chung et al., 2022) with LoRA (Hu et al., 2022) as a *reader*
that writes short, patient-friendly answers from retrieved passages.

Input  : prompt = instruction + question + 3 passages (gold + 2 BM25 hard negatives,
         shuffled; see generation/common.py)
Output : short reference answer (first sentences of the trusted answer)

LoRA freezes the 248M base weights and learns low-rank updates (rank r) for the attention
query/value projections, so only ~0.7% of parameters are trained. This keeps training fast
on a free Colab T4 and avoids catastrophic forgetting of Flan-T5's instruction following.

Usage:  python -m pharmqa.generation.train_lora [--epochs 3 --lr 3e-4 --rank 16]
"""
from __future__ import annotations

import argparse
import random

import os

from pharmqa.compat import alias_renamed_seq2seq_methods, hide_incompatible_torchao

hide_incompatible_torchao()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

from pharmqa.config import GENERATOR_ADAPTER_DIR, GENERATOR_BASE_MODEL, SEED
from pharmqa.generation.common import build_prompt, mix_with_negatives
from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa

MAX_SRC, MAX_TGT = 512, 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    # 16 x 512-token sequences in fp32 overflow a 15 GB T4 (no bf16 there), so the default is
    # 4 per step with 4 accumulation steps: same effective batch of 16.
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="subsample training pairs (0 = all)")
    ap.add_argument("--max_val", type=int, default=400)
    ap.add_argument("--out", default=str(GENERATOR_ADAPTER_DIR))
    ap.add_argument("--exclude_source", default="", help="train without this source (before/after experiment; set --out too)")
    args = ap.parse_args()
    set_seed(SEED)

    corpus = load_corpus()
    text = dict(zip(corpus["id"], corpus["full"]))
    tok = AutoTokenizer.from_pretrained(GENERATOR_BASE_MODEL)
    bm25 = BM25Retriever(corpus)
    rng = random.Random(SEED)

    def to_ds(split: str, cap: int = 0) -> Dataset:
        df = load_qa(split)
        if args.exclude_source:
            df = df[df["source"] != args.exclude_source]
        if cap:
            df = df.sample(min(cap, len(df)), random_state=SEED)
        src = []
        for q, g in tqdm(zip(df["question"], df["gold_id"]), total=len(df), desc=f"hard negatives ({split})"):
            negs = [text[pid] for pid, _ in bm25.search(q, 4) if pid != g][:2]
            src.append(build_prompt(q, mix_with_negatives(text[g], negs, rng)))
        ds = Dataset.from_dict({"src": src, "tgt": df["answer"].tolist()})

        def enc(b):
            x = tok(b["src"], max_length=MAX_SRC, truncation=True)
            x["labels"] = tok(text_target=b["tgt"], max_length=MAX_TGT, truncation=True)["input_ids"]
            return x

        return ds.map(enc, batched=True, remove_columns=["src", "tgt"])

    train_ds, val_ds = to_ds("train", args.max_train), to_ds("val", args.max_val)

    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_BASE_MODEL)
    alias_renamed_seq2seq_methods(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q", "v"],
    ))
    model.print_trainable_parameters()

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=f"{args.out}/checkpoints", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=2 * args.batch,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=int(0.05 * args.epochs * len(train_ds) / (args.batch * args.grad_accum)), weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss", save_total_limit=1,
            logging_steps=50, report_to="none", seed=SEED,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        ),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
