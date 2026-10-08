"""Evaluate the rule-based risk detector on the hand-labelled challenge set.

Note: the rules were refined while inspecting this set (two emergency phrasings were added),
so this score is optimistic and is reported as a development check, not a held-out result.

Usage:  python -m pharmqa.safety_eval
"""
import json

from sklearn.metrics import accuracy_score, classification_report

from pharmqa.config import RESULTS
from pharmqa.metrics import load_challenge
from pharmqa.safety import assess

LEVELS = ["routine", "prescription", "special_group", "emergency"]


def main() -> None:
    df = load_challenge()
    df["pred"] = [assess(t).level for t in df["text"]]
    out = {
        "accuracy": round(accuracy_score(df["risk"], df["pred"]), 4),
        "per_class": classification_report(df["risk"], df["pred"], labels=LEVELS, output_dict=True, zero_division=0),
        "errors": df[df["risk"] != df["pred"]][["text", "risk", "pred"]].to_dict("records"),
    }
    out["per_class"] = {k: v for k, v in out["per_class"].items() if k in LEVELS}
    (RESULTS / "safety_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("accuracy", "errors")}, indent=2))


if __name__ == "__main__":
    main()
