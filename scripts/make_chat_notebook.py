"""Write notebooks/train_chat_reader_colab.ipynb (Qwen2.5-1.5B-Instruct LoRA chat reader)."""
import json
from pathlib import Path

REPO = "https://github.com/Emmanuel-NS/FARUMASI-AI"
NB = "notebooks/train_chat_reader_colab.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


cells = [
    md(f'<a href="https://colab.research.google.com/github/Emmanuel-NS/FARUMASI-AI/blob/main/{NB}" target="_parent">'
       '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'),
    md("# Chat reader: Qwen2.5-1.5B-Instruct + LoRA\n\n"
       "Runtime → Change runtime type → **T4 GPU**. Run the cells in order (about 1.5–2 hours in total).\n\n"
       "Fine-tunes Qwen2.5-1.5B-Instruct with LoRA to write pharmacist-style answers grounded on the retrieved "
       "passages (direct first sentence, then details, and \"my sources don't cover this\" when the passages do not "
       "answer). Then compares it with the zero-shot model on the same test questions as the Flan-T5 reader."),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
         f'REPO_URL = "{REPO}.git"\n'
         "%cd /content\n"
         "!rm -rf repo && git clone -q $REPO_URL repo\n"
         "%cd /content/repo\n"
         "!pip install -q -r requirements.txt"),
    md("## 1. Data (same corpus and splits as the main notebook)"),
    code("!python -m pharmqa.data.download\n!python -m pharmqa.data.prepare"),
    md("## 2. LoRA fine-tuning (about 60–80 min on a T4)\n\n"
       "3,000 answer examples + up to 400 multi-section overviews + about 8% \"no answer\" examples, 1 epoch. "
       "If Colab disconnects, re-run the cell with `--resume` added."),
    code("!python -m pharmqa.generation.train_chat_lora --epochs 1 --max_train 3000 --batch 2 --grad_accum 8"),
    md("## 3. Evaluation (about 20–30 min)\n\n"
       "Gold context = the right passage + 2 hard negatives (same condition as the Flan-T5 table). "
       "Retrieved context falls back to BM25 here because the fine-tuned retriever is not trained in this notebook."),
    code('for system in ["zeroshot", "lora"]:\n'
         '    !python -m pharmqa.generation.evaluate_chat --system {system} --context gold --n 200'),
    code('!python -m pharmqa.generation.evaluate_chat --system lora --context retrieved --n 200 --tag _bm25'),
    md("## 4. Download the adapter and results\n\n"
       "Unzip `chat_reader.zip` into the project folder on your computer (it creates `artifacts/generator_qwen_lora/` "
       "and `results/chat_*`)."),
    code("!rm -rf artifacts/generator_qwen_lora/checkpoints\n"
         "!zip -qr chat_reader.zip artifacts/generator_qwen_lora results/chat_*\n"
         "from google.colab import files\n"
         'files.download("chat_reader.zip")'),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
Path(NB).write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB)
