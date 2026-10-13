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
from pharmqa.query import (QueryAnalyzer, asks_section, clean_passage, drug_of, expand, is_follow_up, is_stub,
                           resolve_follow_up, section_of, small_talk, wants_detail)
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
# Sections combined into a fuller answer when the user asks for more detail, in order; each "tell me
# more" adds the next sections not shown yet.
OVERVIEW = [
    ("What it is for", {"Uses", "Purpose"}),
    ("How to take it", {"Dosage", "How to take", "Directions"}),
    ("Who should not take it", {"Do not use", "Contraindications", "Before you take"}),
    ("Possible side effects", {"Side effects", "When using"}),
    ("Examples", {"Examples"}),
    ("Warnings", {"Warnings", "Boxed warning"}),
    ("Ask a doctor or pharmacist first if", {"Ask a doctor or pharmacist", "Ask a doctor"}),
    ("Other medicines", {"Interactions"}),
    ("Pregnancy and breastfeeding", {"Pregnancy", "Pregnancy and breastfeeding"}),
    ("If you take too much", {"Overdose"}),
    ("Storage", {"Storage"}),
]
OVERVIEW_PARTS = 5
NOTHING_MORE_REPLY = (
    "That is everything my sources say about {drug}. For anything else (for example which brands are sold "
    "near you, or whether it suits you personally), please ask a pharmacist."
)


def _already_said(answer: str, previous_answers) -> bool:
    key = " ".join(answer.lower().split())[:80]
    return bool(key) and any(key in " ".join(p.lower().split()) for p in previous_answers)


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

    def _rank(self, text: str):
        """Dense retrieval, then passages about the medicine and section the question names move up."""
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

        return focus, sims, sorted(cand, key=boosted, reverse=True)

    def _hit(self, i: int, sims, focus) -> dict:
        """`score` stays the raw cosine similarity (used for the confidence threshold); `on_topic` marks
        passages about a medicine named in the question."""
        cols = [c for c in ("id", "title", "text", "source", "full", "url") if c in self.corpus]
        return {**self.corpus.iloc[i][cols].to_dict(), "score": float(sims[i]), "on_topic": self.drugs[i] in focus.drugs}

    def retrieve(self, text: str, k: int = 3) -> list[dict]:
        focus, sims, order = self._rank(text)
        return [self._hit(i, sims, focus) for i in order[:k]]

    def overview(self, text: str, previous_answers=()) -> tuple[str, list[dict]] | None:
        """A fuller answer for "tell me more": one short answer per leaflet section, skipping sections
        already given earlier in the conversation."""
        focus, sims, order = self._rank(text)
        if not focus.drugs:
            return None
        parts, hits, drug = [], [], None
        for heading, sections in OVERVIEW:
            if len(parts) == OVERVIEW_PARTS:
                break
            i = next((i for i in order if self.drugs[i] in focus.drugs and section_of(self.titles[i]) in sections), None)
            if i is None:
                continue
            hit, drug = self._hit(i, sims, focus), self.drugs[i]
            body = clean_passage(hit["full"]).partition(". ")[2] if heading == "Examples" else self.generate(f"{heading} ({drug})?", [hit["full"]])
            body = safety.localize(body)
            if _already_said(body, previous_answers):
                continue
            parts.append(f"**{heading}:** {body}")
            hits.append(hit)
        if not parts:
            return (NOTHING_MORE_REPLY.format(drug=drug), []) if drug else None
        return "\n\n".join(parts), hits

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
        """Attach the conversation's subject (the last user message that was not a follow-up, greeting or
        complaint) to a follow-up such as "what do they do?"."""
        topic = None
        for msg in earlier:
            if not small_talk(msg) and not is_follow_up(msg, self.analyzer.has_own_topic):
                topic = msg
        return resolve_follow_up(question, topic, self.analyzer.has_own_topic)

    def answer(self, question: str, earlier: list[str] | tuple = (), previous_answers: list[str] | tuple = ()) -> Result:
        """`earlier` / `previous_answers` = the user's and the assistant's previous messages, oldest first."""
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

        overview = self.overview(q, previous_answers) if wants_detail(question) else None
        if overview:
            answer, hits = overview
        else:
            hits = self.retrieve(q)
            if hits[0]["score"] < MIN_RETRIEVAL_SIM and not hits[0]["on_topic"]:
                answer = LOW_CONFIDENCE_REPLY
            else:
                # When the question names a medicine, read only passages about it, so the reader cannot copy
                # a neighbouring drug's label (e.g. metoclopramide when asked about metformin).
                context = [h for h in hits if h["on_topic"]] or hits
                examples = next((h for h in context if section_of(h["title"]) == "Examples"), None)
                if examples and asks_section(question, "Examples"):
                    answer = examples["text"]  # a brand-name list: the reader would only shorten or skip it
                else:
                    answer = safety.localize(self.generate(q, [h["full"] for h in context]))
                # A vague follow-up that brings back an earlier answer gets the sections not shown yet.
                if (_already_said(answer, previous_answers) and not asks_section(question)
                        and (overview := self.overview(q, previous_answers))):
                    answer, hits = overview
        sources = [{"title": h["title"], "source": h["source"], "score": round(h["score"], 3), "text": h["text"],
                    "url": h.get("url") if isinstance(h.get("url"), str) else None} for h in hits]
        return Result(question, intent, conf, risk.level, answer, risk.warnings() + [safety.DISCLAIMER], sources,
                      components, interpreted=q)
