from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

FAQS = [
    ("What is Smart FAQ Bot?", "It is a tiny RAG chatbot."),
    ("How do I use it?", "Type a question and press Enter; type exit to quit."),
    ("What powers retrieval?", "FAISS stores FAQ vectors and finds the closest matches."),
    ("What embedding model is used?", "A local TF-IDF embedder keeps the demo offline and fast."),
    ("Can it work without internet?", "Yes, it does not call external APIs."),
    ("How does it answer?", "It combines the query with retrieved FAQs and returns a concise summary."),
    ("What domain is this demo about?", "It can be adapted to any product, course, or support FAQ set."),
    ("Why use RAG here?", "Retrieval grounds the answer in known FAQs and reduces hallucinations."),
    ("Can I change the FAQs?", "Yes, just edit the FAQS list in this file."),
    ("Does it show similarity?", "Yes, the response includes a match score."),
]

class TfIdfEmbeddings(Embeddings):
    def __init__(self): self.v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    def fit(self, texts): self.v.fit(texts)
    def embed_documents(self, texts): return self.v.transform(texts).astype(np.float32).toarray().tolist()
    def embed_query(self, text): return self.v.transform([text]).astype(np.float32).toarray()[0].tolist()

def build_store():
    e = TfIdfEmbeddings(); e.fit([q for q, _ in FAQS])
    d = [Document(page_content=q, metadata={"a": a}) for q, a in FAQS]
    s = FAISS.from_documents(d, e); s._e = e; return s

def answer(s, q):
    h = s.similarity_search(q, k=2); e = s._e
    sim = max(0.0, float(cosine_similarity(np.array([e.embed_query(q)]), np.array([e.embed_query(h[0].page_content)]))[0][0]))
    return f"Matched FAQ: {h[0].page_content}\nAnswer: {h[0].metadata['a']}\nRelated: {h[1].page_content if len(h)>1 else ''}\nSimilarity score: {sim:.4f}"

def main():
    s = build_store(); print("Smart FAQ Bot ready. Type a question, or 'exit' to quit.")
    while (q := input("\nYou: ").strip()).lower() != "exit":
        if q: print(answer(s, q))

if __name__ == "__main__": main()
