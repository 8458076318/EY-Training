"""Semantic search with evaluation metrics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

CORPUS = [
    "How to hard-reset your iPhone 13 if the touch screen is completely frozen or unresponsive.",
    "Troubleshooting guide for iOS updates failing on newer Apple mobile devices.",
    "The new Samsung Galaxy S26 Ultra features an advanced generative AI camera system.",
    "Steps to recover a lost Google Pixel account recovery phrase or authentication token.",
    "Fixing Wi-Fi connectivity drops and network configuration errors on Apple Macbook laptops.",
    "Why is my smartphone battery draining so quickly? Top power optimization tips.",
]

GROUND_TRUTH = {
    "iPhone frozen screen": {0},
    "Apple mobile device issues": {1, 4},
    "pixels in camera": {2, 3},
    "Macbook laptop battery optimization": {4, 5},
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


TOKENIZED_CORPUS = [tokenize(doc) for doc in CORPUS]
BM25 = BM25Okapi(TOKENIZED_CORPUS) if BM25Okapi is not None else None

TERM_FREQUENCIES = [Counter(tokens) for tokens in TOKENIZED_CORPUS]
DOCUMENT_FREQUENCIES = defaultdict(int)
for tokens in TOKENIZED_CORPUS:
    for token in set(tokens):
        DOCUMENT_FREQUENCIES[token] += 1


def _cosine_score(query_tokens: Sequence[str], doc_index: int) -> float:
    query_counts = Counter(query_tokens)
    doc_counts = TERM_FREQUENCIES[doc_index]
    shared_terms = set(query_counts) & set(doc_counts)
    if not shared_terms:
        return 0.0

    dot_product = sum(query_counts[token] * doc_counts[token] for token in shared_terms)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    doc_norm = math.sqrt(sum(value * value for value in doc_counts.values()))
    return dot_product / (query_norm * doc_norm) if query_norm and doc_norm else 0.0


def sparse_search(query: str, top_n: int = 5) -> List[Tuple[int, float]]:
    tokens = tokenize(query)
    if BM25 is not None:
        scores = BM25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_n] if score > 0]

    ranked = sorted(
        ((doc_id, _cosine_score(tokens, doc_id)) for doc_id in range(len(CORPUS))),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(doc_id, score) for doc_id, score in ranked[:top_n] if score > 0]


def dense_search(query: str, top_n: int = 5) -> List[Tuple[int, float]]:
    tokens = tokenize(query)
    ranked = sorted(
        ((doc_id, _cosine_score(tokens, doc_id)) for doc_id in range(len(CORPUS))),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(doc_id, score) for doc_id, score in ranked[:top_n] if score > 0]


def reciprocal_rank_fusion(
    sparse_results: Sequence[Tuple[int, float]],
    dense_results: Sequence[Tuple[int, float]],
    k: int = 60,
    top_n: int = 5,
) -> List[Tuple[int, float]]:
    fused_scores: Dict[int, float] = defaultdict(float)

    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        fused_scores[doc_id] += 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        fused_scores[doc_id] += 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def hybrid_search(query: str, top_n: int = 5) -> List[Tuple[int, float]]:
    sparse_results = sparse_search(query, top_n=10)
    dense_results = dense_search(query, top_n=10)
    return reciprocal_rank_fusion(sparse_results, dense_results, top_n=top_n)


def reciprocal_rank(ranked_docs: Sequence[int], relevant_docs: set[int]) -> float:
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs: Sequence[int], relevant_docs: set[int], k: int = 5) -> float:
    def dcg(items: Sequence[int]) -> float:
        score = 0.0
        for index, doc_id in enumerate(items[:k], start=1):
            relevance = 1.0 if doc_id in relevant_docs else 0.0
            if relevance:
                score += relevance / math.log2(index + 1)
        return score

    ideal = min(len(relevant_docs), k)
    if ideal == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1))
    return dcg(ranked_docs) / ideal_dcg if ideal_dcg else 0.0


def evaluate_search(test_set: Dict[str, set[int]], top_n: int = 5) -> Tuple[float, float]:
    reciprocal_ranks = []
    ndcg_scores = []

    for query, relevant_docs in test_set.items():
        ranked_docs = [doc_id for doc_id, _ in hybrid_search(query, top_n=top_n)]
        reciprocal_ranks.append(reciprocal_rank(ranked_docs, relevant_docs))
        ndcg_scores.append(ndcg_at_k(ranked_docs, relevant_docs, k=top_n))

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    return mrr, mean_ndcg


def print_search_results(query: str, top_n: int = 3) -> None:
    print(f"\nQuery: {query}")
    for rank, (doc_id, score) in enumerate(hybrid_search(query, top_n=top_n), start=1):
        print(f"{rank}. Doc {doc_id} | Score: {score:.4f} | {CORPUS[doc_id]}")


if __name__ == "__main__":
    for sample_query in GROUND_TRUTH:
        print_search_results(sample_query, top_n=3)

    mrr, ndcg = evaluate_search(GROUND_TRUTH, top_n=5)
    print(f"\nMRR: {mrr:.4f}")
    print(f"NDCG@5: {ndcg:.4f}")
