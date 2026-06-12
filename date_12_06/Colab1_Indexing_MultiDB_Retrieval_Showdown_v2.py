from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]


def _load_key_from_dotenv(name: str) -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip().strip('"') or _load_key_from_dotenv("COHERE_API_KEY")

EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIM = 1024
TOP_K = 5


RAW_ARTICLES = [
    {
        "title": "Artificial Intelligence",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "content": (
            "Artificial intelligence is the field of building systems that perform "
            "tasks associated with human intelligence. It includes perception, "
            "language, reasoning, planning, and generation. Modern AI systems use "
            "machine learning and large language models."
        ),
    },
    {
        "title": "Machine Learning",
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "content": (
            "Machine learning is a branch of AI where algorithms learn patterns "
            "from data. Supervised, unsupervised, and reinforcement learning are "
            "common paradigms. Training and evaluation rely on generalization."
        ),
    },
    {
        "title": "Retrieval-Augmented Generation",
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "content": (
            "Retrieval-augmented generation combines retrieval with generation. "
            "Documents are chunked, embedded, and searched before the model "
            "creates a grounded answer using the retrieved context."
        ),
    },
    {
        "title": "FAISS",
        "url": "https://github.com/facebookresearch/faiss",
        "content": (
            "FAISS is a library for efficient similarity search and clustering of "
            "dense vectors. It supports exact and approximate nearest-neighbor "
            "search over embeddings in memory."
        ),
    },
    {
        "title": "Pinecone",
        "url": "https://www.pinecone.io/",
        "content": (
            "Pinecone is a managed vector database for semantic search and retrieval. "
            "It supports serverless deployments, metadata filtering, and cosine "
            "similarity over stored embeddings."
        ),
    },
    {
        "title": "Azure AI Search",
        "url": "https://learn.microsoft.com/azure/search/",
        "content": (
            "Azure AI Search supports hybrid retrieval by combining keyword BM25 "
            "matching with vector search. It is commonly used for enterprise search "
            "and retrieval-augmented generation pipelines."
        ),
    },
    {
        "title": "CRISPR Gene Editing",
        "url": "https://en.wikipedia.org/wiki/CRISPR",
        "content": (
            "CRISPR-Cas9 is a gene-editing technique that uses a guide RNA to locate "
            "a DNA sequence. The Cas9 enzyme cuts the DNA so genes can be modified."
        ),
    },
    {
        "title": "COVID-19 Pandemic",
        "url": "https://en.wikipedia.org/wiki/COVID-19_pandemic",
        "content": (
            "The COVID-19 pandemic was caused by the spread of the SARS-CoV-2 virus. "
            "It affected healthcare systems, supply chains, travel, and public policy "
            "worldwide."
        ),
    },
    {
        "title": "Electric Vehicles",
        "url": "https://en.wikipedia.org/wiki/Electric_vehicle",
        "content": (
            "Electric vehicles use batteries and electric motors instead of internal "
            "combustion engines. Regenerative braking recovers energy during deceleration."
        ),
    },
    {
        "title": "Transformers",
        "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "content": (
            "Transformer models rely on self-attention to process sequences in parallel. "
            "They are foundational for language models, retrieval, and many multimodal systems."
        ),
    },
    {
        "title": "Climate Change",
        "url": "https://en.wikipedia.org/wiki/Climate_change",
        "content": (
            "Climate change is driven by rising greenhouse gas concentrations. It can "
            "increase heatwaves, sea-level rise, extreme weather, and ecosystem disruption."
        ),
    },
    {
        "title": "Bitcoin",
        "url": "https://en.wikipedia.org/wiki/Bitcoin",
        "content": (
            "Bitcoin is a decentralized digital currency secured by proof of work. "
            "Mining validates transactions and creates new blocks on the blockchain."
        ),
    },
    {
        "title": "HNSW",
        "url": "https://arxiv.org/abs/1603.09320",
        "content": (
            "Hierarchical Navigable Small World graphs are a fast approximate nearest "
            "neighbor structure. They are widely used for vector search at scale."
        ),
    },
    {
        "title": "BM25",
        "url": "https://en.wikipedia.org/wiki/Okapi_BM25",
        "content": (
            "BM25 is a sparse lexical ranking function based on term frequency and "
            "document length normalization. It remains a strong baseline for keyword search."
        ),
    },
    {
        "title": "Quantum Computing",
        "url": "https://en.wikipedia.org/wiki/Quantum_computing",
        "content": (
            "Quantum computing uses qubits, superposition, and entanglement to solve "
            "certain problems differently from classical computers. Error correction is a major challenge."
        ),
    },
    {
        "title": "Supply Chain",
        "url": "https://en.wikipedia.org/wiki/Supply_chain",
        "content": (
            "A supply chain moves goods from suppliers to customers through production, "
            "storage, and transport. Disruptions can create bullwhip effects and shortages."
        ),
    },
    {
        "title": "Renewable Energy",
        "url": "https://en.wikipedia.org/wiki/Renewable_energy",
        "content": (
            "Renewable energy sources such as solar and wind reduce fossil-fuel dependence. "
            "Intermittency can be managed with storage, grid balancing, and forecasting."
        ),
    },
    {
        "title": "Cybersecurity",
        "url": "https://en.wikipedia.org/wiki/Computer_security",
        "content": (
            "Cybersecurity protects systems, networks, and data from unauthorized access. "
            "Common threats include phishing, malware, ransomware, and cloud misconfiguration."
        ),
    },
    {
        "title": "NIST Cybersecurity Framework",
        "url": "https://www.nist.gov/cyberframework",
        "content": (
            "The NIST Cybersecurity Framework helps organizations manage risk through "
            "identify, protect, detect, respond, and recover functions."
        ),
    },
    {
        "title": "Word Embeddings",
        "url": "https://en.wikipedia.org/wiki/Word_embedding",
        "content": (
            "Word embeddings represent words as dense vectors so similar terms are near "
            "each other in vector space. They are the core representation used by many retrieval systems."
        ),
    },
]


BENCHMARK_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does HNSW approximate nearest neighbour search work?",
    "What are the differences between BM25 and vector search?",
    "How does CRISPR-Cas9 cut DNA?",
    "What caused the COVID-19 pandemic?",
    "How do electric vehicles use regenerative braking?",
    "What is the transformer attention mechanism?",
    "What are the risks of climate change?",
    "How does Proof of Work consensus work in Bitcoin?",
    "What is FAISS and when should I use it?",
    "How does Pinecone handle metadata filtering?",
    "What is hybrid search combining BM25 and vectors?",
    "How does Azure AI Search differ from Pinecone?",
    "What are embedding models used for in RAG?",
    "How does quantum computing threaten RSA encryption?",
    "What is the supply chain bullwhip effect?",
    "How does renewable energy handle intermittency?",
    "What cybersecurity threats target cloud infrastructure?",
    "What is the NIST cybersecurity framework?",
    "How do word embeddings capture semantic meaning?",
]


class CohereEmbeddingsAdapter:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL) -> None:
        import cohere

        self._client = cohere.Client(api_key, timeout=5)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
            embedding_types=["float"],
            truncate="END",
        )
        embeddings = response.embeddings.float_
        if embeddings is None:
            raise RuntimeError("Cohere embed returned no float embeddings.")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="search_query",
            embedding_types=["float"],
            truncate="END",
        )
        embeddings = response.embeddings.float_
        if not embeddings:
            raise RuntimeError("Cohere embed returned no float embeddings.")
        return embeddings[0]

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_embedder() -> Any:
    if COHERE_API_KEY:
        return CohereEmbeddingsAdapter(COHERE_API_KEY)
    raise RuntimeError("COHERE_API_KEY is not set in the project .env file.")


def build_documents() -> tuple[list[Document], list[str], list[dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs: list[Document] = []
    for article in RAW_ARTICLES:
        chunks = splitter.create_documents(
            texts=[article["content"]],
            metadatas=[{"title": article["title"], "url": article["url"]}],
        )
        all_docs.extend(chunks)

    all_docs = all_docs[:500]
    texts = [doc.page_content for doc in all_docs]
    metas = [doc.metadata for doc in all_docs]
    return all_docs, texts, metas


def build_faiss_store(all_docs: list[Document], embedder: Any) -> tuple[Any, list[list[float]], Any]:
    print("Embedding documents for FAISS...")
    t0 = time.perf_counter()
    all_vectors = embedder.embed_documents([doc.page_content for doc in all_docs])
    elapsed = time.perf_counter() - t0
    print(f"Embedded {len(all_vectors)} chunks in {elapsed:.1f}s")
    print(f"Vector dimension: {len(all_vectors[0])}")

    raw_index = np.asarray(all_vectors, dtype=np.float32)
    if raw_index.shape[1] != EMBEDDING_DIM:
        print(
            f"WARNING: Embedding dimension {raw_index.shape[1]} differs from configured "
            f"{EMBEDDING_DIM}; using the observed dimension for downstream local stores."
        )

    faiss_store = FAISS.from_documents(all_docs, embedder)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": TOP_K})
    print("LangChain FAISS vectorstore ready")
    return faiss_store, all_vectors, faiss_retriever


def benchmark_retriever(name: str, retriever: Any, queries: list[str]) -> tuple[list[float], float, float]:
    latencies: list[float] = []
    for query in tqdm(queries, desc=f"{name} benchmark"):
        t0 = time.perf_counter()
        retriever.invoke(query)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(f"{name}  p50={p50:.1f}ms  p95={p95:.1f}ms")
    return latencies, p50, p95


def run_rerank_extension(faiss_store: Any) -> None:
    if not COHERE_API_KEY:
        print("COHERE_API_KEY is empty -- skipping rerank extension.")
        return

    try:
        import cohere
    except Exception as exc:  # pragma: no cover - runtime specific
        print(f"WARNING: Cohere package not usable, skipping rerank extension: {exc}")
        return

    co = cohere.Client(COHERE_API_KEY, timeout=5)
    eval_set = [
        ("How does FAISS handle approximate nearest neighbour search?", "FAISS"),
        ("What is the difference between BM25 and dense retrieval?", "BM25"),
        ("How does CRISPR cut DNA at a specific location?", "CRISPR"),
        ("What triggered the COVID-19 pandemic?", "COVID-19"),
        ("How do electric vehicles recover energy during braking?", "Electric Vehicles"),
    ]

    def compute_mrr(docs: list[Document], relevant_title: str) -> float:
        for rank, doc in enumerate(docs, 1):
            if relevant_title.lower() in doc.metadata.get("title", "").lower():
                return 1.0 / rank
        return 0.0

    def rerank(query: str, top_k_retrieve: int = 10, top_k_final: int = 10) -> list[Document]:
        docs = faiss_store.similarity_search(query, k=top_k_retrieve)
        passages = [doc.page_content for doc in docs]
        reranked = co.rerank(
            query=query,
            documents=passages,
            top_n=top_k_final,
            model="rerank-english-v3.0",
        )
        return [docs[result.index] for result in reranked.results]

    before_mrr: list[float] = []
    after_mrr: list[float] = []
    for query, rel in eval_set:
        before_mrr.append(compute_mrr(faiss_store.similarity_search(query, k=10), rel))
        after_mrr.append(compute_mrr(rerank(query), rel))

    print(f"MRR@10 Before re-rank : {np.mean(before_mrr):.3f}")
    print(f"MRR@10 After  re-rank : {np.mean(after_mrr):.3f}")
    print(f"Improvement          : +{(np.mean(after_mrr) - np.mean(before_mrr)) * 100:.1f}pp")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Before Re-rank", "After Cohere Re-rank"],
        [np.mean(before_mrr), np.mean(after_mrr)],
        color=["#DD8452", "#55A868"],
        edgecolor="white",
    )
    ax.set_title("MRR@10: FAISS -> Cohere Rerank", fontweight="bold")
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("mrr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved mrr_comparison.png")


def main() -> None:
    print("Configuration loaded!")
    print(f"COHERE_API_KEY {'set' if COHERE_API_KEY else 'missing'}")

    embedder = get_embedder()
    all_docs, texts, metas = build_documents()

    print(f"Total chunks: {len(all_docs)}")
    print(f"Avg chunk size: {sum(len(text) for text in texts) // len(texts)} chars")
    print(f"\nSample chunk from '{all_docs[0].metadata['title']}':")
    print(all_docs[0].page_content[:200] + "...")

    faiss_store, all_vectors, faiss_retriever = build_faiss_store(all_docs, embedder)

    sample_q = "How does vector similarity search work?"
    print(f"\nQuery: '{sample_q}'")
    for i, result in enumerate(faiss_retriever.invoke(sample_q)):
        print(f"  [{i + 1}] [{result.metadata['title']}] {result.page_content[:100]}...")

    faiss_latencies, faiss_p50, faiss_p95 = benchmark_retriever(
        "FAISS",
        faiss_retriever,
        BENCHMARK_QUERIES,
    )

    pinecone_latencies: list[float] = []
    pinecone_p50 = np.nan
    pinecone_p95 = np.nan
    print(
        "Pinecone section skipped: set PINECONE_API_KEY and restore the "
        "notebook's Pinecone cells to run the cloud benchmark."
    )

    azure_latencies: list[float] = []
    azure_p50 = np.nan
    azure_p95 = np.nan
    print(
        "Azure AI Search section skipped: set AZURE_SEARCH_ENDPOINT and "
        "AZURE_SEARCH_API_KEY and restore the notebook's Azure cells to run "
        "the cloud benchmark."
    )

    summary = pd.DataFrame(
        {
            "Vector DB": ["FAISS (in-memory)", "Pinecone Serverless", "Azure AI Search (Hybrid)"],
            "p50 (ms)": [round(faiss_p50, 1), round(pinecone_p50, 1), round(azure_p50, 1)],
            "p95 (ms)": [round(faiss_p95, 1), round(pinecone_p95, 1), round(azure_p95, 1)],
            "Search Type": ["Exact k-NN (L2)", "Needs cloud config", "Needs cloud config"],
            "Infra": ["In-process", "Cloud", "Cloud"],
            "Cost": ["Free", "External", "External"],
        }
    )
    print("=" * 76)
    print("LATENCY BENCHMARK -- 20 queries, embedding time included")
    print("=" * 76)
    print(summary.to_string(index=False))
    print("=" * 76)

    dbs = ["FAISS"]
    colors = ["#4C72B0"]
    p50s = [faiss_p50]
    p95s = [faiss_p95]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Vector DB Latency Showdown: FAISS vs Pinecone vs Azure AI Search",
        fontsize=13,
        fontweight="bold",
    )
    for ax, vals, label in [(ax1, p50s, "p50 Latency (ms)"), (ax2, p95s, "p95 Latency (ms)")]:
        bars = ax.bar(dbs, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(0, max(vals) * 1.4 if max(vals) > 0 else 1)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.5,
                f"{v:.0f}ms",
                ha="center",
                fontweight="bold",
                fontsize=11,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("latency_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved latency_bar.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        [faiss_latencies],
        tick_labels=["FAISS"],
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.5),
    )
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][0].set_alpha(0.85)

    ax.set_title("Query Latency Distribution (20 queries)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved latency_distribution.png")

    run_rerank_extension(faiss_store)


if __name__ == "__main__":
    main()
