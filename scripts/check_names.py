"""Check the unknown-product-name rule: patient messages, plus how often real test questions trigger it."""
from pharmqa.query import NameChecker, drug_of
from pharmqa.retrieval.common import load_corpus, load_qa

corpus = load_corpus()
checker = NameChecker(corpus["full"], [drug_of(t, s) for t, s in zip(corpus["title"].fillna(""), corpus["source"])])
for q in ["what do you know about kopavitamins", "do you know supliment called Kopavitamin?", "what do you know about quarteme",
          "after sex pills", "what is that", "what do you know about coartem", "can i take panadoll while pregnant",
          "i have diarhea what can i take", "what is amoxcilin used for", "tell me about zinc", "is ibuprofen safe for kids"]:
    print(f"{q!r:50} -> {checker.check(q)}")
qa = load_qa("test").sample(600, random_state=0)
flagged = [(q, checker.check(q)) for q in qa["question"]]
flagged = [f for f in flagged if f[1]]
print(f"\n{len(flagged)}/600 test questions flagged")
for q, r in flagged[:15]:
    print(" ", q, "->", r)
