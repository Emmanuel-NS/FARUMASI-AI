"""Retrieval baseline: BM25 (Robertson & Zaragoza, 2009) over title + passage text.

Usage:  python -m pharmqa.retrieval.bm25
"""
from tqdm import tqdm

from pharmqa.retrieval.common import BM25Retriever, load_corpus, load_qa, save_retrieval


def main() -> None:
    corpus = load_corpus()
    qa = load_qa("test")
    bm25 = BM25Retriever(corpus)
    ranked = [[pid for pid, _ in bm25.search(q, 10)] for q in tqdm(qa["question"], desc="BM25")]
    save_retrieval("bm25", qa, ranked, corpus)


if __name__ == "__main__":
    main()
