"""Upload the app, code, corpus and trained artifacts to a Hugging Face Space.

Usage (after training):
    huggingface-cli login            # or set HF_TOKEN
    python scripts/deploy_space.py --space <your-hf-username>/farumasi-pharmacist-qa
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPACE_README = """---
title: FARUMASI Pharmacist Assistant
emoji: 💊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

Domain-specific medicine question answering for community pharmacy in Rwanda.
Source code, data pipeline and experiments: https://github.com/Emmanuel-NS/FARUMASI-AI
"""

SPACE_REQUIREMENTS = """torch
spaces
transformers==5.19.0
tokenizers==0.23.2
sentence-transformers==6.1.0
peft==0.21.2
accelerate==1.15.0
scikit-learn>=1.4
pandas>=2.1
rank-bm25>=0.2.2
joblib>=1.3
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. username/farumasi-pharmacist-qa")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        shutil.copy(ROOT / "app.py", dst / "app.py")
        shutil.copytree(ROOT / "pharmqa", dst / "pharmqa", ignore=shutil.ignore_patterns("__pycache__"))
        (dst / "data" / "processed").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "processed" / "corpus.jsonl", dst / "data" / "processed" / "corpus.jsonl")
        art = ROOT / "artifacts"
        for item in ("intent_distilbert", "retriever_minilm_ft", "generator_flant5_lora",
                     "corpus_embeddings.npy", "intent_tfidf.joblib"):
            src = art / item
            if src.is_dir():
                shutil.copytree(src, dst / "artifacts" / item, ignore=shutil.ignore_patterns("checkpoints"))
            elif src.exists():
                (dst / "artifacts").mkdir(exist_ok=True)
                shutil.copy(src, dst / "artifacts" / item)
            else:
                print(f"warning: {item} missing; the app will fall back to a baseline component")
        (dst / "README.md").write_text(SPACE_README, encoding="utf-8")
        (dst / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")

        api = HfApi()
        if not api.repo_exists(args.space, repo_type="space"):
            try:
                api.create_repo(args.space, repo_type="space", space_sdk="gradio")
            except Exception as exc:  # noqa: BLE001 - HF returns 402 for scripted Space creation on some accounts
                raise SystemExit(
                    f"Could not create the Space ({exc}).\nCreate it in the browser instead: "
                    "https://huggingface.co/new-space (SDK: Gradio, template: Blank, hardware: ZeroGPU - free), "
                    f"name it '{args.space.split('/')[-1]}', then rerun this script to upload."
                ) from exc
        api.upload_folder(folder_path=str(dst), repo_id=args.space, repo_type="space",
                          commit_message="Deploy FARUMASI Pharmacist Assistant")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
