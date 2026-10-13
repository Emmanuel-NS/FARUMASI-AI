"""Print a few chat-reader training examples (answer / overview / no-answer) for inspection."""
import random
from collections import Counter

from pharmqa.config import SEED
from pharmqa.generation.chat_common import NO_ANSWER
from pharmqa.generation.train_chat_lora import build_examples
from pharmqa.retrieval.common import BM25Retriever, load_corpus

corpus = load_corpus()
examples = build_examples("train", 300, corpus, BM25Retriever(corpus), random.Random(SEED), 30)
kind = lambda m, t: "no_answer" if t == NO_ANSWER else "overview" if "overview" in m[1]["content"] else "answer"  # noqa: E731
print(Counter(kind(m, t) for m, t in examples))
shown = set()
for messages, target in examples:
    k = kind(messages, target)
    if k in shown:
        continue
    shown.add(k)
    print(f"\n===== {k} =====\n{messages[1]['content'][:900]}\n----- target -----\n{target}")
