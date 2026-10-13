"""End-to-end assistant used by the web app.

    question
      -> risk detector (rules)            -> safety warnings to show
      -> intent classifier (DistilBERT)   -> out_of_domain? politely decline
      -> retriever (fine-tuned MiniLM)    -> top passages from trusted sources
      -> generator (Flan-T5 + LoRA)       -> short answer grounded in the top passage
      -> answer + warnings + sources + disclaimer

Every component falls back to its baseline if the trained artifact is missing, so the app
still runs (with lower quality) before training on Colab has finished.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from pharmqa import safety
from pharmqa.config import (CORPUS_EMB_PATH, GENERATOR_ADAPTER_DIR, GENERATOR_BASE_MODEL, INTENT_LABELS,
                            INTENT_MODEL_DIR, INTENT_TFIDF_PATH, RETRIEVER_BASE_MODEL, RETRIEVER_MODEL_DIR,
                            SUPPLEMENTARY_EMB_PATH, SUPPLEMENTARY_PATH)
from pharmqa.generation.common import build_prompt, extractive_answer
from pharmqa.query import (QueryAnalyzer, clean_passage, drug_of, expand, is_stub, resolve_follow_up,
                           section_of, small_talk)
from pharmqa.retrieval.common import load_corpus

OOD_REPLY = (
    "I'm a pharmacy assistant, so I can only help with questions about medicines, symptoms and minor "
    "health problems. For anything else, please use another service."
)
LOW_CONFIDENCE_REPLY = (
    "I couldn't find reliable information about this in my medicine and health sources. If it is a health "
    "question, please ask a pharmacist or doctor directly."
)
MIN_RETRIEVAL_SIM = 0.35  # below this cosine similarity the top passage is unlikely to be relevant
CANDIDATES = 50           # dense hits re-ranked by the medicine / section the question names
DRUG_BOOST, SECTION_BOOST = 0.15, 0.08


@dataclass
class Result:
    question: str
    intent: str
    intent_confidence: float
    risk_level: str
    answer: str
    warnings: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    components: dict = field(default_factory=dict)
    interpreted: str = ""  # the question after attaching the subject of earlier turns


class PharmacistAssistant:
    def __init__(self, use_generator: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.corpus = load_corpus()
        self._load_intent()
        self._load_retriever()
        self._add_supplementary()
        self.titles = self.corpus["title"].fillna("").tolist()
        self.drugs = [drug_of(t, s) for t, s in zip(self.titles, self.corpus["source"])]
        self.stub = np.array([is_stub(i, t) for i, t in zip(self.corpus["id"], self.corpus["text"].fillna(""))])
        self.analyzer = QueryAnalyzer(self.drugs)
        self.gen = self._load_generator() if use_generator else None

    # Components -----------------------------------------------------------------
    def _load_intent(self) -> None:
        if INTENT_MODEL_DIR.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.intent_tok = AutoTokenizer.from_pretrained(str(INTENT_MODEL_DIR))
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL_DIR)).to(self.device).eval()
            self.intent_kind = "distilbert"
        else:
            import joblib
            self.intent_model = joblib.load(INTENT_TFIDF_PATH)
            self.intent_kind = "tfidf_lr"

    def _load_retriever(self) -> None:
        from sentence_transformers import SentenceTransformer
        finetuned = RETRIEVER_MODEL_DIR.exists() and CORPUS_EMB_PATH.exists()
        self.retriever = SentenceTransformer(str(RETRIEVER_MODEL_DIR) if finetuned else RETRIEVER_BASE_MODEL, device=self.device)
        if finetuned:
            self.emb = np.load(CORPUS_EMB_PATH)
        else:
            self.emb = self.retriever.encode(self.corpus["full"].tolist(), batch_size=128, normalize_embeddings=True)
        self.retriever_kind = "minilm_finetuned" if finetuned else "minilm_zeroshot"

    def _add_supplementary(self) -> None:
        """Swap in corrected openFDA labels for medicines whose original label was the wrong product."""
        if not SUPPLEMENTARY_PATH.exists():
            return
        extra = pd.read_json(SUPPLEMENTARY_PATH, lines=True)
        extra["full"] = extra["title"] + ". " + extra["text"]
        replaced = set(extra["title"].map(drug_of))
        keep = ~(self.corpus["source"].eq("openFDA") & self.corpus["title"].fillna("").map(drug_of).isin(replaced))
        # Precomputed by scripts/fetch_supplementary.py: on ZeroGPU any model compute at start-up breaks GPU workers.
        if SUPPLEMENTARY_EMB_PATH.exists() and self.retriever_kind == "minilm_finetuned":
            extra_emb = np.load(SUPPLEMENTARY_EMB_PATH)
        else:
            extra_emb = self.retriever.encode(extra["full"].tolist(), batch_size=64, normalize_embeddings=True)
        self.corpus = pd.concat([self.corpus[keep], extra], ignore_index=True)
        self.emb = np.vstack([self.emb[keep.to_numpy()], extra_emb])

    def _load_generator(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GENERATOR_BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_BASE_MODEL)
        kind = "flan_t5_zeroshot"
        if GENERATOR_ADAPTER_DIR.exists():
            from pharmqa.compat import alias_renamed_seq2seq_methods, hide_incompatible_torchao
            hide_incompatible_torchao()
            alias_renamed_seq2seq_methods(model)
            from peft import PeftModel
            # Load adapter weights on CPU: on ZeroGPU no GPU is attached at startup, only inside spaces.GPU calls.
            model = PeftModel.from_pretrained(model, str(GENERATOR_ADAPTER_DIR), torch_device="cpu").merge_and_unload()
            kind = "flan_t5_lora"
        return tok, model.to(self.device).eval(), kind

    # Steps -----------------------------------------------------------------------
    @torch.no_grad()
    def classify(self, text: str) -> tuple[str, float]:
        if self.intent_kind == "distilbert":
            enc = self.intent_tok(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
            probs = self.intent_model(**enc).logits.softmax(-1)[0].cpu().numpy()
            return INTENT_LABELS[int(probs.argmax())], float(probs.max())
        probs = self.intent_model.predict_proba([text])[0]
        classes = list(self.intent_model.classes_)
        return classes[int(probs.argmax())], float(probs.max())

    def retrieve(self, text: str, k: int = 3) -> list[dict]:
        """Dense retrieval, then passages about the medicine and section the question names move up.

        `score` stays the raw cosine similarity (used for the confidence threshold); `on_topic` marks
        passages about a medicine named in the question.
        """
        expanded = expand(text)
        focus = self.analyzer.focus(expanded)
        q = self.retriever.encode([expanded], normalize_embeddings=True)[0]
        sims = self.emb @ q
        cand = np.argsort(-sims)[:CANDIDATES]
        if focus.drugs:
            named = [i for i, d in enumerate(self.drugs) if d in focus.drugs]
            cand = np.unique(np.concatenate([cand, np.array(named, dtype=int)]))
        cand = cand[~self.stub[cand]]

        def boosted(i: int) -> float:
            on_drug = self.drugs[i] in focus.drugs
            return sims[i] + DRUG_BOOST * on_drug + SECTION_BOOST * (on_drug and section_of(self.titles[i]) in focus.sections)

        top = sorted(cand, key=boosted, reverse=True)[:k]
        cols = [c for c in ("id", "title", "text", "source", "full", "url") if c in self.corpus]
        return [{**self.corpus.iloc[i][cols].to_dict(), "score": float(sims[i]),
                 "on_topic": self.drugs[i] in focus.drugs} for i in top]

    @torch.no_grad()
    def generate(self, question: str, passages: list[str]) -> str:
        passages = [clean_passage(p) for p in passages]
        if self.gen is None:
            return extractive_answer(passages[0])
        tok, model, _ = self.gen
        enc = tok(build_prompt(question, passages), return_tensors="pt", truncation=True, max_length=512).to(self.device)
        ids = model.generate(**enc, max_new_tokens=128, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
        return tok.decode(ids[0], skip_special_tokens=True)

    # Public API ------------------------------------------------------------------
    def resolve(self, question: str, earlier: list[str]) -> str:
        """Carry the subject of earlier user turns into follow-ups ("what do they do?")."""
        context = None
        for msg in earlier:
            if not small_talk(msg):
                context = resolve_follow_up(msg, context, self.analyzer.has_own_topic)
        return resolve_follow_up(question, context, self.analyzer.has_own_topic)

    def answer(self, question: str, earlier: list[str] | tuple = ()) -> Result:
        """`earlier` = the user's previous messages in this conversation, oldest first."""
        components = {"intent_model": self.intent_kind, "retriever": self.retriever_kind,
                      "generator": self.gen[2] if self.gen else "extractive"}
        chat_reply = small_talk(question)
        if chat_reply and safety.assess(question).level == "routine":
            return Result(question, "small_talk", 1.0, "routine", chat_reply, components=components, interpreted=question)
        q = self.resolve(question, list(earlier))
        risk = safety.assess(q)
        intent, conf = self.classify(q)

        # Emergencies always get the safety message, even if the classifier is unsure.
        if intent == "out_of_domain" and risk.level != "emergency":
            return Result(question, intent, conf, risk.level, OOD_REPLY, components=components, interpreted=q)

        hits = self.retrieve(q)
        if hits[0]["score"] < MIN_RETRIEVAL_SIM:
            answer = LOW_CONFIDENCE_REPLY
        else:
            # When the question names a medicine, read only passages about it, so the reader cannot copy
            # a neighbouring drug's label (e.g. metoclopramide when asked about metformin).
            context = [h for h in hits if h["on_topic"]] or hits
            answer = safety.localize(self.generate(q, [h["full"] for h in context]))
        sources = [{"title": h["title"], "source": h["source"], "score": round(h["score"], 3), "text": h["text"],
                    "url": h.get("url") if isinstance(h.get("url"), str) else None} for h in hits]
        return Result(question, intent, conf, risk.level, answer, risk.warnings() + [safety.DISCLAIMER], sources,
                      components, interpreted=q)
