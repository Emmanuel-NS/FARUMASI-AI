"""Shared evaluation helpers that write results/ files used by the report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from pharmqa.config import INTENT_LABELS, PROCESSED, RESULTS, ROOT


def load_intent(split: str) -> pd.DataFrame:
    return pd.read_json(PROCESSED / f"intent_{split}.jsonl", lines=True)


def load_challenge() -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / "challenge" / "challenge_set.jsonl", lines=True)


def classification_summary(y_true, y_pred, labels=INTENT_LABELS) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def evaluate_intent_model(name: str, predict) -> dict:
    """predict: list[str] -> list[str]. Evaluates on the test split, per source, and the challenge set."""
    test = load_intent("test")
    test["pred"] = predict(test["text"].tolist())
    chal = load_challenge()
    chal["pred"] = predict(chal["text"].tolist())

    out = {
        "model": name,
        "test": classification_summary(test["label"], test["pred"]),
        "test_accuracy_by_source": {s: round(float((g["label"] == g["pred"]).mean()), 4) for s, g in test.groupby("source")},
        "challenge": classification_summary(chal["intent"], chal["pred"]),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"intent_{name}.json").write_text(json.dumps(out, indent=2))
    errors = pd.concat([
        test[test["label"] != test["pred"]].assign(set="test")[["set", "source", "text", "label", "pred"]],
        chal[chal["intent"] != chal["pred"]].assign(set="challenge", source="challenge").rename(columns={"intent": "label"})[["set", "source", "text", "label", "pred"]],
    ])
    errors.to_csv(RESULTS / f"intent_{name}_errors.csv", index=False)
    print(f"[{name}] test acc={out['test']['accuracy']} macroF1={out['test']['macro_f1']} | "
          f"challenge acc={out['challenge']['accuracy']} | by source={out['test_accuracy_by_source']}")
    return out


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
