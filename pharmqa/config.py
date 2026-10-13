"""Central paths and constants shared by every stage of the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

for _d in (RAW, PROCESSED, ARTIFACTS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Intent labels for the domain/scope classifier.
INTENT_LABELS = ["medicine_info", "condition_advice", "out_of_domain"]

# Base models (all from the Hugging Face Hub).
INTENT_BASE_MODEL = "distilbert-base-uncased"
RETRIEVER_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_BASE_MODEL = "google/flan-t5-base"

# Artifact locations produced by training.
INTENT_TFIDF_PATH = ARTIFACTS / "intent_tfidf.joblib"
INTENT_MODEL_DIR = ARTIFACTS / "intent_distilbert"
RETRIEVER_MODEL_DIR = ARTIFACTS / "retriever_minilm_ft"
GENERATOR_ADAPTER_DIR = ARTIFACTS / "generator_flant5_lora"
CORPUS_EMB_PATH = ARTIFACTS / "corpus_embeddings.npy"

# Corrected openFDA labels added to the app's corpus at start-up (scripts/fetch_supplementary.py).
SUPPLEMENTARY_PATH = DATA / "external" / "supplementary_labels.jsonl"
SUPPLEMENTARY_EMB_PATH = ARTIFACTS / "supplementary_embeddings.npy"
