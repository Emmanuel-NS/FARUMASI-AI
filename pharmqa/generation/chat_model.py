"""Load the Qwen chat reader (base model + optional LoRA adapter) and generate grounded answers."""
from __future__ import annotations

from pathlib import Path

import torch

from pharmqa.config import CHAT_ADAPTER_DIR, CHAT_BASE_MODEL


def has_adapter(adapter: str | Path = CHAT_ADAPTER_DIR) -> bool:
    return (Path(adapter) / "adapter_config.json").exists()


def load_chat_model(use_adapter: bool = True, adapter: str | Path = CHAT_ADAPTER_DIR, device: str | None = None):
    """Returns (tokenizer, model). The adapter is merged on CPU so ZeroGPU only sees a .to(device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pharmqa.compat import hide_incompatible_torchao
    hide_incompatible_torchao()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(CHAT_BASE_MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(CHAT_BASE_MODEL, dtype=dtype)
    if use_adapter:
        if not has_adapter(adapter):
            raise SystemExit(f"chat LoRA adapter not found at {adapter}: run pharmqa.generation.train_chat_lora first")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), torch_device="cpu").merge_and_unload()
    return tok, model.to(device).eval()


@torch.no_grad()
def chat_generate(tok, model, conversations: list[list[dict]], max_new_tokens: int = 320, batch: int = 8) -> list[str]:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(conversations), batch):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tok.pad_token_id)
        out.extend(t.strip() for t in tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out
