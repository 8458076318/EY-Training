"""Financial RAG system converted from the Colab notebook.

This script keeps the notebook's learning flow but makes it runnable outside
Colab by:
- loading local environment variables from a .env file
- removing notebook-only install cells
- wrapping the demo into functions and a main entry point

The demo uses inline Apple 10-K excerpts, FAISS, HuggingFace embeddings, Azure
OpenAI, and optional RAGAS evaluation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load environment variables from the repo-root .env file."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is not None and isinstance(value, str):
        value = value.strip()
    return value or None


def get_env_any(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = get_env(name)
        if value:
            return value
    return default


def get_azure_api_version() -> str:
    """Return the Azure OpenAI API version to use for this script."""
    explicit = get_env("AZURE_OPENAI_API_VERSION")
    if explicit:
        return explicit

    legacy = get_env("AZURE_API_VERSION")
    if legacy and legacy != "2024-06-01":
        print(
            f"Note: ignoring AZURE_API_VERSION={legacy!r} for this script and "
            "using 2024-06-01 instead."
        )
    return "2024-06-01"


def import_langchain_primitives():
    """Import LangChain classes with fallbacks for older/newer package layouts."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    try:
        from langchain_core.documents import Document
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.schema import Document

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.prompts import ChatPromptTemplate

    try:
        from langchain_core.runnables import RunnablePassthrough
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.schema.runnable import RunnablePassthrough

    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.schema.runnable import RunnableLambda

    try:
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain.schema.output_parser import StrOutputParser

    return RecursiveCharacterTextSplitter, Document, ChatPromptTemplate, RunnablePassthrough, RunnableLambda, StrOutputParser


def import_vectorstore_primitives():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain_community.embeddings import HuggingFaceEmbeddings
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Missing vector store dependencies. Install langchain-community, "
            "faiss-cpu, and sentence-transformers."
        ) from exc

    return HuggingFaceEmbeddings, FAISS


SAMPLE_DOCS = [
    {
        "source": "Apple_10K_2023_Risk",
        "text": """RISK FACTORS
Apple's operations and financial results are subject to various risks and uncertainties.
Global and regional economic conditions, including conditions resulting from financial and credit market fluctuations,
can adversely affect demand for Apple's products and services.
Apple faces intense competition in all of its business areas from well-established companies with significant
resources, as well as from new market entrants.
Apple depends on the performance of distributors, carriers, wholesalers and other resellers.
The Company's fiscal year 2023 revenue was $383.3 billion, compared to $394.3 billion in fiscal 2022,
a decrease of approximately 2.8 percent.
The Company's net income for fiscal 2023 was $97.0 billion, or $6.13 diluted earnings per share,
compared to $99.8 billion, or $6.11 diluted earnings per share, in fiscal 2022.
Apple's gross margin percentage was 44.1% in fiscal 2023, compared to 43.3% in fiscal 2022.
Services revenue reached an all-time high of $85.2 billion in fiscal 2023, up 9 percent year over year.""",
    },
    {
        "source": "Apple_10K_2023_Products",
        "text": """PRODUCTS AND SERVICES
Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories.
iPhone is Apple's line of smartphones based on its iOS operating system.
iPhone net sales were $200.6 billion in fiscal 2023, representing approximately 52% of total revenue.
Mac net sales were $29.4 billion in fiscal 2023, down from $40.2 billion in fiscal 2022.
iPad net sales were $28.3 billion in fiscal 2023.
Wearables, Home and Accessories net sales were $39.8 billion in fiscal 2023.
Apple's Services segment includes advertising, AppleCare, cloud, digital content, payment and other services.
The App Store, Apple Music, Apple TV+, Apple Arcade, iCloud and Apple Pay are key Services offerings.
The Company had approximately 2.2 billion active devices at the end of fiscal year 2023.""",
    },
    {
        "source": "Apple_10K_2023_Liquidity",
        "text": """LIQUIDITY AND CAPITAL RESOURCES
The Company believes its existing balances of cash, cash equivalents and unrestricted marketable securities,
together with cash generated by operations, will be sufficient to satisfy its expected cash needs.
Cash and cash equivalents as of September 30, 2023 were $29.965 billion.
Total marketable securities were $100.544 billion, consisting of current marketable securities of $31.590 billion
and non-current marketable securities of $100.544 billion.
During fiscal 2023, the Company returned over $77 billion to shareholders,
including $15.1 billion in dividends and dividend equivalents and $62.2 billion through repurchases of 471 million shares.
Capital expenditures were $10.959 billion in fiscal 2023.
The Company's long-term debt as of September 30, 2023 was $95.281 billion.""",
    },
]


def create_chunks(
    docs: list,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """Convert raw documents to LangChain Document chunks."""
    RecursiveCharacterTextSplitter, Document, _, _, _, _ = import_langchain_primitives()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    lc_docs = [Document(page_content=d["text"], metadata={"source": d["source"]}) for d in docs]
    return splitter.split_documents(lc_docs)


def build_embeddings():
    HuggingFaceEmbeddings, _ = import_vectorstore_primitives()
    print("Loading HuggingFace embedding model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def build_vectorstore(chunks):
    _, FAISS = import_vectorstore_primitives()
    embeddings = build_embeddings()
    print("Building FAISS index...")
    t0 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t0
    print(f"FAISS index built in {elapsed:.2f}s")
    print(f"Vectors: {vectorstore.index.ntotal}")
    print(f"Dimension: {vectorstore.index.d}")
    return vectorstore, embeddings


def build_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },
    )
    test_query = "What was Apple revenue in 2023?"
    results = retriever.invoke(test_query)
    print(f'Query: "{test_query}"')
    print(f"Retrieved {len(results)} chunks:")
    for i, r in enumerate(results):
        print(f'  [{i + 1}] {r.metadata["source"]}: {r.page_content[:100]}...')
    return retriever


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )


def import_hybrid_primitives():
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Missing BM25 retriever dependency. Install rank_bm25 and langchain_community."
        ) from exc

    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:  # pragma: no cover - compatibility fallback
        from langchain_community.retrievers import EnsembleRetriever

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Missing cross-encoder dependency. Install sentence-transformers."
        ) from exc

    return BM25Retriever, EnsembleRetriever, CrossEncoder


def build_bm25_retriever(chunks):
    BM25Retriever, _, _ = import_hybrid_primitives()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    return bm25_retriever


def build_hybrid_retriever(dense_retriever, bm25_retriever):
    _, EnsembleRetriever, _ = import_hybrid_primitives()
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


def build_cross_encoder():
    _, _, CrossEncoder = import_hybrid_primitives()
    print("Loading cross-encoder reranker...")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, docs, cross_encoder, top_n: int = 4):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def make_context_runnable(retriever, reranker=None, top_n: int = 4):
    _, _, _, _, RunnableLambda, _ = import_langchain_primitives()

    def _context(query: str) -> str:
        docs = retriever.invoke(query)
        if reranker is not None:
            docs = rerank_documents(query, docs, reranker, top_n=top_n)
        else:
            docs = docs[:top_n]
        return format_docs(docs)

    return RunnableLambda(_context)


def build_rag_chain(retriever):
    _, _, ChatPromptTemplate, RunnablePassthrough, _, StrOutputParser = import_langchain_primitives()
    from langchain_openai import AzureChatOpenAI

    azure_endpoint = get_env_any("AZURE_OPENAI_ENDPOINT")
    azure_api_key = get_env_any("AZURE_OPENAI_KEY", "AZURE_API_KEY")
    azure_deployment = get_env_any("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = get_azure_api_version()

    if not azure_endpoint or not azure_api_key or not azure_deployment:
        raise RuntimeError(
            "Missing Azure OpenAI settings. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_KEY, and AZURE_OPENAI_DEPLOYMENT in your .env or shell."
        )

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_version=azure_api_version,
        openai_api_key=azure_api_key,
        temperature=0,
        max_tokens=512,
        timeout=30,
    )

    rag_prompt = ChatPromptTemplate.from_template(
        """You are FinSight, an AI research analyst for a Tier-1 investment bank.
Answer the analyst's question ONLY using the provided context.
If the context does not contain the answer, say: "Insufficient information in the retrieved context."
Always cite the specific source document at the end of your answer.

CONTEXT:
{context}

ANALYST QUESTION: {question}

ANSWER (cite source):
"""
    )

    context_runnable = make_context_runnable(retriever, top_n=4)
    rag_chain = (
        {"context": context_runnable, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain built")
    return rag_chain, llm


def build_rag_chain_with_context(context_runnable, llm):
    _, _, ChatPromptTemplate, RunnablePassthrough, _, StrOutputParser = import_langchain_primitives()
    rag_prompt = _build_prompt_template(ChatPromptTemplate)
    return (
        {"context": context_runnable, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


TEST_QUERIES = [
    "What was Apple's total revenue in fiscal year 2023?",
    "How much cash did Apple have at the end of fiscal 2023?",
    "What percentage of Apple's revenue came from iPhone in 2023?",
    "How much did Apple return to shareholders in fiscal 2023?",
    "What is Apple's gross margin for fiscal 2023?",
]


GROUND_TRUTHS = [
    "Apple's total revenue in fiscal 2023 was $383.3 billion.",
    "Apple had $29.965 billion in cash and cash equivalents at end of fiscal 2023.",
    "iPhone represented approximately 52% of Apple's total revenue in fiscal 2023.",
    "Apple returned over $77 billion to shareholders in fiscal 2023.",
    "Apple's gross margin was 44.1% in fiscal 2023.",
]


def run_queries(rag_chain):
    results_log = []
    for query in TEST_QUERIES:
        t0 = time.time()
        try:
            answer = rag_chain.invoke(query)
        except Exception as exc:
            raise RuntimeError(
                "Azure OpenAI request failed. The most likely causes are a bad "
                "endpoint, deployment name, or API version. Check the values in "
                "C:\\Training\\AI-ML-Training-Projects\\.env, especially "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, and "
                "AZURE_OPENAI_API_VERSION."
            ) from exc
        latency = time.time() - t0
        results_log.append({"query": query, "answer": answer, "latency_s": round(latency, 2)})
        print(f"Question: {query}")
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {answer[:200]}...")
        print()
    return results_log


def run_ragas_evaluation(retriever, results_log, llm):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from langchain_openai import AzureOpenAIEmbeddings
    except ImportError:
        print("RAGAS dependencies are not installed. Skipping evaluation.")
        return None

    azure_endpoint = get_env_any("AZURE_OPENAI_ENDPOINT")
    azure_api_key = get_env_any("AZURE_OPENAI_KEY", "AZURE_API_KEY")
    azure_api_version = get_azure_api_version()
    azure_embedding_deployment = get_env_any("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", default="text-embedding-ada-002")

    contexts_used = []
    for q in TEST_QUERIES:
        docs = retriever.invoke(q)
        contexts_used.append([d.page_content for d in docs])

    eval_dataset = Dataset.from_dict(
        {
            "question": TEST_QUERIES,
            "answer": [r["answer"] for r in results_log],
            "contexts": contexts_used,
            "ground_truth": GROUND_TRUTHS,
        }
    )

    az_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_embedding_deployment,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )

    print("Running RAGAS evaluation...")
    ragas_results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=az_embeddings,
    )

    print("RAGAS Evaluation Results:")
    print(ragas_results)
    return ragas_results


def extract_metric_value(ragas_result, metric_name: str):
    if ragas_result is None:
        return None

    if hasattr(ragas_result, "to_pandas"):
        try:
            df = ragas_result.to_pandas()
            if metric_name in getattr(df, "columns", []):
                return float(df[metric_name].mean())
        except Exception:
            pass

    if isinstance(ragas_result, dict) and metric_name in ragas_result:
        try:
            return float(ragas_result[metric_name])
        except Exception:
            return None

    if hasattr(ragas_result, metric_name):
        try:
            return float(getattr(ragas_result, metric_name))
        except Exception:
            return None

    return None


def collect_contexts_for_queries(retriever, queries, reranker=None, top_n: int = 4):
    contexts = []
    for query in queries:
        docs = retriever.invoke(query)
        if reranker is not None:
            docs = rerank_documents(query, docs, reranker, top_n=top_n)
        else:
            docs = docs[:top_n]
        contexts.append([doc.page_content for doc in docs])
    return contexts


def build_strategy_results(strategy_name, retriever, llm, reranker=None, queries=None):
    queries = queries or TEST_QUERIES
    context_runnable = make_context_runnable(retriever, reranker=reranker, top_n=4)
    chain = build_rag_chain_with_context(context_runnable, llm)

    rows = []
    for query in queries:
        t0 = time.time()
        answer = chain.invoke(query)
        latency = time.time() - t0
        rows.append(
            {
                "strategy": strategy_name,
                "query": query,
                "answer": answer,
                "latency_s": round(latency, 2),
            }
        )

    ragas_result = None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from langchain_openai import AzureOpenAIEmbeddings

        azure_endpoint = get_env_any("AZURE_OPENAI_ENDPOINT")
        azure_api_key = get_env_any("AZURE_OPENAI_KEY", "AZURE_API_KEY")
        azure_api_version = get_azure_api_version()
        azure_embedding_deployment = get_env_any("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", default="text-embedding-ada-002")

        eval_dataset = Dataset.from_dict(
            {
                "question": queries,
                "answer": [row["answer"] for row in rows],
                "contexts": collect_contexts_for_queries(retriever, queries, reranker=reranker, top_n=4),
                "ground_truth": GROUND_TRUTHS[: len(queries)],
            }
        )

        az_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_embedding_deployment,
            openai_api_key=azure_api_key,
            openai_api_version=azure_api_version,
        )

        ragas_result = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=az_embeddings,
        )
    except ImportError:
        ragas_result = None

    faithfulness_score = extract_metric_value(ragas_result, "faithfulness")
    avg_latency = round(sum(row["latency_s"] for row in rows) / len(rows), 2) if rows else None

    return {
        "strategy": strategy_name,
        "rows": rows,
        "avg_latency_s": avg_latency,
        "faithfulness": faithfulness_score,
        "ragas_result": ragas_result,
    }


def save_faithfulness_latency_plot(strategy_results, output_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return None

    points = [
        r for r in strategy_results
        if r.get("faithfulness") is not None and r.get("avg_latency_s") is not None
    ]
    if not points:
        print("No faithfulness values were available, so the plot was skipped.")
        return None

    plt.figure(figsize=(8, 5))
    for row in points:
        plt.scatter(row["avg_latency_s"], row["faithfulness"], s=90)
        plt.text(
            row["avg_latency_s"] + 0.01,
            row["faithfulness"] + 0.002,
            row["strategy"],
            fontsize=9,
        )

    plt.axhline(0.85, color="gray", linestyle="--", linewidth=1, label="FinSight target 0.85")
    plt.axhline(0.88, color="green", linestyle=":", linewidth=1, label="Extension target 0.88")
    plt.xlabel("Average latency (s)")
    plt.ylabel("Faithfulness")
    plt.title("Faithfulness vs Latency: Dense vs Hybrid vs Hybrid + Rerank")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved faithfulness/latency plot to {output_path}")
    return output_path


def run_hybrid_retrieval_extension(chunks, dense_retriever, llm):
    try:
        bm25_retriever = build_bm25_retriever(chunks)
        hybrid_retriever = build_hybrid_retriever(dense_retriever, bm25_retriever)
        cross_encoder = build_cross_encoder()
    except RuntimeError as exc:
        print(str(exc))
        print("Skipping hybrid retrieval extension because an optional dependency is missing.")
        return None

    strategy_results = []
    dense_results = build_strategy_results("dense", dense_retriever, llm)
    hybrid_results = build_strategy_results("hybrid", hybrid_retriever, llm)
    hybrid_rerank_results = build_strategy_results(
        "hybrid+rerank",
        hybrid_retriever,
        llm,
        reranker=cross_encoder,
    )

    strategy_results.extend([dense_results, hybrid_results, hybrid_rerank_results])

    print("\nExtension comparison:")
    for row in strategy_results:
        faithfulness = row["faithfulness"]
        faithfulness_text = f"{faithfulness:.3f}" if faithfulness is not None else "n/a"
        latency_text = f"{row['avg_latency_s']:.2f}s" if row["avg_latency_s"] is not None else "n/a"
        print(f"  - {row['strategy']}: faithfulness={faithfulness_text}, avg latency={latency_text}")

    plot_path = Path(__file__).with_name("faithfulness_latency_comparison.png")
    save_faithfulness_latency_plot(strategy_results, plot_path)

    print("\nReflection prompts:")
    print("1. Compare the faithfulness scores against the FinSight target of >= 0.85.")
    print("2. Identify which strategy best balances faithfulness and latency.")
    print("3. Note any question types that still fail even with BM25 + reranking.")
    print("4. For 10,000 PDFs/day, move from local FAISS to a scalable vector store, batch ingestion, and async evaluation.")
    return strategy_results


def run_chunk_size_experiment(embedding_model, llm):
    import pandas as pd

    _, FAISS = import_vectorstore_primitives()
    _, _, _, RunnablePassthrough, _, StrOutputParser = import_langchain_primitives()

    chunk_sizes_to_test = [256, 512, 1024]
    experiment_results = []

    for cs in chunk_sizes_to_test:
        print(f"Testing chunk_size={cs}...")
        exp_chunks = create_chunks(SAMPLE_DOCS, chunk_size=cs, chunk_overlap=cs // 8)
        exp_vs = FAISS.from_documents(exp_chunks, embedding_model)
        exp_retriever = exp_vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

        rag_prompt = _build_prompt_template()
        exp_chain = (
            {"context": exp_retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        latencies: List[float] = []
        for q in TEST_QUERIES[:3]:
            t0 = time.time()
            _ = exp_chain.invoke(q)
            latencies.append(time.time() - t0)

        avg_latency = round(sum(latencies) / len(latencies), 2)
        experiment_results.append(
            {
                "chunk_size": cs,
                "n_chunks": len(exp_chunks),
                "avg_latency_s": avg_latency,
            }
        )
        print(f"Chunks: {len(exp_chunks)}, Avg latency: {avg_latency}s")

    df = pd.DataFrame(experiment_results)
    print("Chunk Size Comparison:")
    print(df.to_string(index=False))
    print("Tip: Run full RAGAS evaluation per chunk_size to see faithfulness trade-offs")
    return df


def _build_prompt_template(ChatPromptTemplate=None):
    if ChatPromptTemplate is None:
        _, _, ChatPromptTemplate, _, _, _ = import_langchain_primitives()
    return ChatPromptTemplate.from_template(
        """You are FinSight, an AI research analyst for a Tier-1 investment bank.
Answer the analyst's question ONLY using the provided context.
If the context does not contain the answer, say: "Insufficient information in the retrieved context."
Always cite the specific source document at the end of your answer.

CONTEXT:
{context}

ANALYST QUESTION: {question}

ANSWER (cite source):
"""
    )


def print_summary_and_extension_note():
    print("Summary and reflection prompts:")
    print("1. What faithfulness score did you achieve? How does it compare to the FinSight target of >= 0.85?")
    print("2. Which chunk size gave the best trade-off between faithfulness and latency?")
    print("3. What types of queries failed? Why might the RAG pipeline struggle with them?")
    print("4. How would you adapt this pipeline for 10,000 PDFs ingested daily?")
    print()
    print("Extension task: hybrid retrieval, BM25, cross-encoder reranking, and plotting are implemented below.")
    print("="*50)

def main() -> None:
    load_project_env()

    print(f"Loaded {len(SAMPLE_DOCS)} documents")
    for doc in SAMPLE_DOCS:
        print(f'  - {doc["source"]}: {len(doc["text"])} characters')

    chunks = create_chunks(SAMPLE_DOCS, chunk_size=512, chunk_overlap=64)
    print(f"Created {len(chunks)} chunks (size=512, overlap=64)")
    print("Sample chunk:")
    print(f'  Source: {chunks[0].metadata["source"]}')
    print(f"  Content: {chunks[0].page_content[:200]}...")

    vectorstore, embedding_model = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore)
    rag_chain, llm = build_rag_chain(retriever)
    results_log = run_queries(rag_chain)
    run_ragas_evaluation(retriever, results_log, llm)
    run_chunk_size_experiment(embedding_model, llm)
    run_hybrid_retrieval_extension(chunks, retriever, llm)
    print_summary_and_extension_note()


if __name__ == "__main__":
    main()
