"""Collect every results/*.json into results/summary.md (tables used in the report and README).

Usage:  python -m pharmqa.report_tables
"""
from __future__ import annotations

import json

import pandas as pd

from pharmqa.config import PROCESSED, RESULTS


def _load(prefix: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{prefix}*.json"))]


def main() -> None:
    lines = ["# Experiment summary", ""]

    stats = PROCESSED / "dataset_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        lines += ["## Dataset", "", f"- Corpus passages: {s['corpus_passages']} ({s['corpus_by_source']})",
                  f"- QA pairs: {s['qa_total']} ({s['qa_by_source']})"]
        lines += [f"- {k}: {v['n']} questions, {v['focus_entities']} distinct medicines/conditions" for k, v in s["splits"].items()]
        lines.append("")

    intent = _load("intent_")
    intent = [r for r in intent if "test" in r]
    if intent:
        rows = [{
            "model": r["model"], "test acc": r["test"]["accuracy"], "test macro-F1": r["test"]["macro_f1"],
            "MedicationQA acc": r["test_accuracy_by_source"].get("MedicationQA"),
            "challenge acc": r["challenge"]["accuracy"], "challenge macro-F1": r["challenge"]["macro_f1"],
        } for r in intent]
        lines += ["## Intent classification", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    retr = _load("retrieval_")
    if retr:
        rows = [{"model": r["model"], **{k: r[k] for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "topic@1") if k in r}}
                for r in retr]
        lines += ["## Retrieval (test split)", "", pd.DataFrame(rows).to_markdown(index=False), ""]
        by_src = [{"model": r["model"], "source": s, **v} for r in retr for s, v in r["by_source"].items()]
        lines += ["### Retrieval by data source", "", pd.DataFrame(by_src).to_markdown(index=False), ""]

    gen = _load("gen_")
    if gen:
        rows = [{"system": g["system"], "context": g["context"], "n": g["n"], "ROUGE-L": g["rougeL"],
                 "BERTScore-F1": g["bertscore_f1"], "avg words": g["avg_words"]} for g in gen]
        lines += ["## Answer generation", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    safety = RESULTS / "safety_eval.json"
    if safety.exists():
        s = json.loads(safety.read_text())
        lines += ["## Safety risk detector (challenge set)", "", f"Accuracy: {s['accuracy']}", "",
                  pd.DataFrame(s["per_class"]).T.round(3).to_markdown(), ""]

    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
