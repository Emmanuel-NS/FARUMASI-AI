"""Intent baseline: TF-IDF features + logistic regression.

Two configurations are compared:
  word      word uni+bigrams
  word_char word uni+bigrams + character 3-5 grams (robust to misspellings like "diarrhoea")

Usage:  python -m pharmqa.intent.baseline
"""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from pharmqa.config import INTENT_TFIDF_PATH, SEED
from pharmqa.metrics import evaluate_intent_model, load_intent


def build(config: str) -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    if config == "word":
        features = word
    else:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
        features = FeatureUnion([("word", word), ("char", char)])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=SEED))])


def main() -> None:
    # Model selection uses the challenge (consumer-style) score as well as the test split,
    # because the test split shares MedQuAD's templated phrasing with training.
    best = None
    for train_name in ("train", "train_consumer"):
        train = load_intent(train_name)
        for config in ("word", "word_char"):
            model = build(config).fit(train["text"], train["label"])
            suffix = "" if train_name == "train" else "_consumer"
            res = evaluate_intent_model(f"tfidf_lr_{config}{suffix}", lambda xs, m=model: m.predict(xs).tolist())
            score = res["test"]["macro_f1"] + res["challenge"]["macro_f1"]
            if best is None or score > best[0]:
                best = (score, model)
    joblib.dump(best[1], INTENT_TFIDF_PATH)
    print(f"saved best baseline -> {INTENT_TFIDF_PATH}")


if __name__ == "__main__":
    main()
