"""Fine-tune Qwen2.5-1.5B-Instruct (Qwen Team, 2024) with LoRA (Hu et al., 2022) as a chat reader.

Same grounded data as the Flan-T5 reader, but the targets are pharmacist-style answers (direct
opening sentence + passage details), plus multi-section overviews and "my sources don't cover
this" examples; see generation/chat_common.py. The loss is computed on the answer tokens only
(the system prompt, passages and question are masked with -100).

LoRA adapts the attention and MLP projections (r=16), ~1% of the 1.5B parameters, in fp16 with
gradient checkpointing so it fits a free Colab T4 (15 GB).

Usage:  python -m pharmqa.generation.train_chat_lora [--epochs 1 --max_train 3000]
"""
from __future__ import annotations

import argparse
import os
import random

from pharmqa.compat import hide_incompatible_torchao

hide_incompatible_torchao()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed  # noqa: E402

from pharmqa.config import CHAT_ADAPTER_DIR, CHAT_BASE_MODEL, SEED  # noqa: E402
from pharmqa.generation.chat_common import (NO_ANSWER, OVERVIEW_SECTIONS, answer_target, build_messages,  # noqa: E402
                                            overview_question, overview_target)
from pharmqa.generation.common import mix_with_negatives  # noqa: E402
from pharmqa.query import is_stub  # noqa: E402
from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa  # noqa: E402

NO_ANSWER_SHARE = 0.08
OVERVIEW_SOURCES = ("openFDA-templated", "RwandaFDA-PIL")


def prefix(pid: str) -> str:
    """'fda:metformin:adverse_reactions:0' -> 'fda:metformin' (one medicine / leaflet)."""
    return ":".join(pid.split(":")[:2])


def build_examples(split: str, cap: int, corpus, bm25, rng: random.Random, max_overviews: int) -> list[tuple[list[dict], str]]:
    text = dict(zip(corpus["id"], corpus["full"]))
    stub = {i for i, t in zip(corpus["id"], corpus["text"].fillna("")) if is_stub(i, t)}
    df = load_qa(split)
    df = df[~df["gold_id"].isin(stub)]
    if cap:
        df = df.sample(min(cap, len(df)), random_state=SEED)
    out = []
    for q, g, qtype, focus in tqdm(zip(df["question"], df["gold_id"], df["qtype"], df["focus"]), total=len(df), desc=f"answers ({split})"):
        hits = [pid for pid, _ in bm25.search(q, 15)]
        negs = [pid for pid in hits if pid != g][:2]
        out.append((build_messages(q, mix_with_negatives(text[g], [text[n] for n in negs], rng)),
                    answer_target(qtype, str(focus), text[g])))
        if rng.random() < NO_ANSWER_SHARE:
            name = str(focus).lower()
            other = [text[pid] for pid in hits
                     if prefix(pid) != prefix(g) and name not in text[pid][:400].lower()][:3]
            if len(other) == 3:
                out.append((build_messages(q, other), NO_ANSWER))

    # Overviews: one per medicine / leaflet whose sections appear in this split.
    by_prefix: dict[str, dict[str, str]] = {}
    for pid in corpus["id"]:
        parts = pid.split(":")
        if len(parts) == 4 and parts[3] == "0":
            by_prefix.setdefault(prefix(pid), {})[parts[2]] = pid
    prefixes = sorted({prefix(g) for g, s in zip(df["gold_id"], df["source"]) if s in OVERVIEW_SOURCES})
    rng.shuffle(prefixes)
    for p in prefixes[:max_overviews]:
        sections = by_prefix.get(p, {})
        chosen = []
        for heading, keys in OVERVIEW_SECTIONS:
            pid = next((sections[k] for k in keys if k in sections and sections[k] not in stub), None)
            if pid:
                chosen.append((heading, pid))
        if len(chosen) < 3:
            continue
        chosen = chosen[:5]
        name = text[chosen[0][1]].split(":")[0]
        passages = [text[pid] for _, pid in chosen]
        rng.shuffle(passages)
        out.append((build_messages(overview_question(name, rng), passages, detailed=True),
                    overview_target(name, [(h, text[pid]) for h, pid in chosen])))
    rng.shuffle(out)
    return out


def tokenize(tok, examples, max_len: int) -> Dataset:
    rows = {"input_ids": [], "labels": []}
    skipped = 0
    for messages, target in examples:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            skipped += 1
            continue
        rows["input_ids"].append(p_ids + t_ids)
        rows["labels"].append([-100] * len(p_ids) + t_ids)
    print(f"{len(rows['input_ids'])} examples ({skipped} longer than {max_len} tokens skipped)")
    return Dataset.from_dict(rows)


def collate(tok):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--max_train", type=int, default=3000, help="answer examples sampled from qa_train (0 = all)")
    ap.add_argument("--max_overviews", type=int, default=400)
    ap.add_argument("--max_val", type=int, default=200)
    ap.add_argument("--out", default=str(CHAT_ADAPTER_DIR))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)

    corpus = load_corpus()
    bm25 = BM25Retriever(corpus)
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    train_ds = tokenize(tok, build_examples("train", args.max_train, corpus, bm25, rng, args.max_overviews), args.max_len)
    val_ds = tokenize(tok, build_examples("val", args.max_val, corpus, bm25, rng, 40), args.max_len)

    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp16 mixed precision cannot unscale fp16 gradients
    model.config.use_cache = False
    model.print_trainable_parameters()

    steps = args.epochs * len(train_ds) / (args.batch * args.grad_accum)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{args.out}/checkpoints", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, warmup_steps=int(0.05 * steps), weight_decay=0.0,
            lr_scheduler_type="cosine", eval_strategy="steps", eval_steps=max(20, int(steps / 4)),
            save_strategy="steps", save_steps=max(20, int(steps / 4)), save_total_limit=1,
            logging_steps=10, report_to="none", seed=SEED, fp16=fp16, bf16=dtype == torch.bfloat16,
            remove_unused_columns=False,
        ),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate(tok),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
