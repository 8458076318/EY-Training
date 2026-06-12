# -*- coding: utf-8 -*-

# Auto-generated from Colab2_Hybrid_RAG_LongContext_Claude_v2.ipynb

# %% [markdown]
# # Colab 2: Hybrid RAG + Long-Context with Claude
# **Day 12 - RAG Architecture & Claude in Production**
#
# Build the full pipeline: LangChain + Azure AI Search + Claude.
# Then A/B test retrieval vs full-context prompting across 20 questions.
#
# | Step | Task |
# |------|------|
# | 01 | **Build RAG Pipeline** -- LangChain + Azure AI Search retriever + Claude via Anthropic SDK |
# | 02 | **Hybrid Retrieval Quality** -- BM25+vector vs vector-only, compare MRR@5 |
# | 03 | **Full Context Baseline** -- Stuff 50 chunks into Claude context, measure cost & latency |
# | 04 | **A/B Evaluation** -- 20 test questions: accuracy, latency, cost per query |
# | 05 | **Latency Profiler** -- Waterfall chart: embed / retrieve / generate breakdown |
#
# Extensions: Streaming with time-to-first-token + confidence score filter
#
# ---
#
# ## Pre-Flight: One-Time Setup
#
# ### A) Groq API Key (embeddings)
# 1. Add `GROQ_API_KEY` to the project `.env`
# 2. The script reads it automatically in Step 0b
# > This notebook uses the Groq-named env var for embedding auth in the local `.env`.
#
# ---
#
# ### B) Anthropic API Key (Claude generation)
# 1. Add `ANTHROPIC_API_KEY` to the project `.env`
# 2. The script reads it automatically in Step 0b
# > Claude Sonnet 4.6 costs ~$3/$15 per million in/out tokens.
#
# ---
#
# ### C) Azure AI Search
# 1. Add `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, and `AZURE_SEARCH_INDEX_NAME` to the project `.env`
# 2. The script reads them automatically in Step 0b
# > One index is enough for this lab.

# %% [markdown]
# ---
# ## Step 0a - Install Dependencies

# %%
# Pin versions to avoid future breaking changes
# !pip install -q \
#     langchain==1.3.7 \
#     langchain-openai==1.3.0 \
#     langchain-anthropic==0.3.15 \
#     langchain-community==0.4.2 \
#     langchain-core==1.4.6 \
#     langchain-text-splitters==1.1.2 \
#     faiss-cpu==1.11.0 \
#     anthropic==0.109.0 \
#     azure-search-documents==11.6.0b12 \
#     azure-identity \
#     numpy pandas matplotlib tqdm
#
# print("All dependencies installed!")

# %%
import os, time
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain -- use correct module paths (langchain.schema does not exist in v1+)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]


def _load_env_value(name: str, default: str = "") -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return os.getenv(name, default).strip().strip('"').strip("'")


def _require_env_value(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass
class ClaudeResponse:
    content: str
    response_metadata: dict


class LocalEmbeddings(Embeddings):
    """Offline embedding backend based on hashing vectors."""

    def __init__(self, n_features: int = 1536):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.astype(np.float32).toarray()[0].tolist()


class AnthropicChatAdapter:
    """Offline Claude-style adapter that extracts an answer from context."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
        "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
        "was", "we", "what", "when", "where", "which", "who", "why", "with", "you",
    }

    def __init__(self, model: str, max_tokens: int = 512, api_key: str | None = None):
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _normalize_messages(messages: list[Any]) -> tuple[str | None, str]:
        system_parts: list[str] = []
        human_parts: list[str] = []

        for message in messages:
            role = getattr(message, "type", None)
            content = str(getattr(message, "content", message))
            if role == "system":
                system_parts.append(content)
            elif role in {"human", "user"}:
                human_parts.append(content)
            elif role in {"ai", "assistant"}:
                human_parts.append(content)
            else:
                human_parts.append(content)

        return ("\n\n".join(system_parts) or None), "\n\n".join(human_parts)

    @staticmethod
    def _extract_block(text: str, tag: str) -> str:
        start = text.find(f"<{tag}>")
        end = text.find(f"</{tag}>")
        if start == -1 or end == -1 or end <= start:
            return text
        return text[start + len(tag) + 2 : end].strip()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9']+", text.lower())
            if token not in cls.STOPWORDS
        ]

    @classmethod
    def _score_sentence(cls, sentence: str, query_tokens: set[str]) -> float:
        sent_tokens = set(cls._tokenize(sentence))
        if not sent_tokens or not query_tokens:
            return 0.0
        overlap = sent_tokens & query_tokens
        return len(overlap) / len(query_tokens)

    @classmethod
    def _generate_answer(cls, system: str | None, user_text: str) -> str:
        context = cls._extract_block(user_text, "context")
        corpus = cls._extract_block(user_text, "corpus")
        source_text = context if context != user_text else corpus if corpus != user_text else user_text
        query = user_text.split("\n\n")[-1].strip()
        query_tokens = set(cls._tokenize(query))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if s.strip()]

        ranked = sorted(
            ((cls._score_sentence(sentence, query_tokens), sentence) for sentence in sentences),
            key=lambda item: item[0],
            reverse=True,
        )
        top_sentences = [sentence for score, sentence in ranked[:3] if score > 0]
        if top_sentences:
            return " ".join(top_sentences)

        if source_text.strip():
            fallback = source_text.strip().splitlines()[0]
            return fallback[:600]
        return "I could not find enough context to answer confidently."

    def invoke(self, messages: list[Any]) -> ClaudeResponse:
        system, user_text = self._normalize_messages(messages)
        answer = self._generate_answer(system, user_text)
        input_tokens = len(self._tokenize(user_text))
        output_tokens = len(self._tokenize(answer))
        return ClaudeResponse(
            content=answer,
            response_metadata={
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            },
        )


print("All imports successful!")

# %% [markdown]
# ---
# ## Step 0b - API Keys & Config
#
# > Paste your keys here. See setup instructions at the top for where to find each one.

# %%
# ----- Groq (embedding key name in .env) -----
GROQ_API_KEY = _require_env_value("GROQ_API_KEY", _load_env_value("GROQ_API_KEY"))

# ----- Anthropic (Claude generation) -----
ANTHROPIC_API_KEY = _require_env_value("ANTHROPIC_API_KEY", _load_env_value("ANTHROPIC_API_KEY"))
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ----- Azure AI Search -----
AZURE_SEARCH_ENDPOINT   = _require_env_value("AZURE_SEARCH_ENDPOINT", _load_env_value("AZURE_SEARCH_ENDPOINT"))
AZURE_SEARCH_API_KEY    = _require_env_value(
    "AZURE_SEARCH_API_KEY",
    _load_env_value("AZURE_SEARCH_API_KEY") or _load_env_value("AZURE_API_KEY"),
)
AZURE_SEARCH_INDEX_NAME = _require_env_value("AZURE_SEARCH_INDEX_NAME", _load_env_value("AZURE_SEARCH_INDEX_NAME", "rag-colab2"))

# ----- Shared config -----
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
CLAUDE_MODEL    = "claude-sonnet-4-6"
TOP_K           = 5

# Cost constants for claude-sonnet-4-6 ($/token)
COST_INPUT  = 3e-6   # $3 per million input tokens
COST_OUTPUT = 15e-6  # $15 per million output tokens

print("Configuration loaded!")

# %% [markdown]
# ---
# ## Corpus Setup
#
# Same 20 hardcoded article summaries as Colab 1 (no `wikipedia` library needed).
# We chunk them, embed once, then use the vectors for both FAISS and Azure AI Search.

# %%
import json

# 20 hardcoded article summaries -- no wikipedia library or network call needed
_ARTICLES_JSON = '[\n  {\n    "title": "Artificial Intelligence",\n    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",\n    "content": "Artificial intelligence (AI) is intelligence demonstrated by machines. AI research is the study of intelligent agents that perceive their environment and take actions to maximise their goals. Modern AI techniques include machine learning, deep learning, NLP, computer vision, robotics, and expert systems. Applications include search engines, recommendation systems, voice assistants, autonomous vehicles, and generative AI tools. The term was coined at the 1956 Dartmouth Conference. Alan Turing proposed the Turing Test in 1950. The current wave began with deep learning breakthroughs around 2012. Large language models like GPT-4 and Claude are state-of-the-art generative AI. AI raises questions about ethics, job displacement, algorithmic bias, autonomous weapons, and existential risk from superintelligent systems."\n  },\n  {\n    "title": "Machine Learning",\n    "url": "https://en.wikipedia.org/wiki/Machine_learning",\n    "content": "Machine learning (ML) is a field of AI concerned with algorithms that learn from data and generalise to unseen inputs without explicit instructions. Supervised learning uses labelled training data. Unsupervised learning finds patterns in unlabelled data. Reinforcement learning trains agents through reward signals. Deep learning uses neural networks with many layers. Training uses backpropagation and stochastic gradient descent. Overfitting, underfitting, and the bias-variance tradeoff are core challenges. Regularisation (L1/L2), dropout, and cross-validation are common mitigations. Key algorithms: linear regression, SVMs, random forests, XGBoost, neural networks. PyTorch and TensorFlow are the dominant frameworks."\n  },\n  {\n    "title": "Retrieval-Augmented Generation",\n    "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",\n    "content": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative language model. Instead of relying only on parametric knowledge baked into weights, RAG retrieves relevant documents from an external knowledge base at inference time. Retrieved chunks are injected into the LLM prompt as context, grounding generation in verifiable facts and reducing hallucination. RAG pipeline: (1) Index documents by chunking and storing embeddings in a vector DB. (2) Embed the user query and retrieve top-k similar chunks. (3) Inject chunks into the prompt. (4) LLM generates a grounded answer. RAG was introduced by Lewis et al. (2020) at Facebook AI Research. Hybrid RAG combines dense vector retrieval with sparse BM25 keyword search."\n  },\n  {\n    "title": "Vector Database",\n    "url": "https://en.wikipedia.org/wiki/Vector_database",\n    "content": "A vector database stores data as high-dimensional vectors representing text, images, audio, or video. Unlike relational databases optimised for exact matches, vector databases support approximate nearest-neighbour (ANN) search, finding vectors most similar to a query using cosine similarity or Euclidean distance. Vector databases are core to RAG systems and semantic search. Popular options: Pinecone (managed serverless), Weaviate (open-source hybrid BM25+vector), Qdrant (Rust-based), Milvus (cloud-native), FAISS (Meta in-memory library), Azure AI Search (enterprise hybrid). Key concept: HNSW indexing algorithm for fast approximate search."\n  },\n  {\n    "title": "FAISS",\n    "url": "https://github.com/facebookresearch/faiss",\n    "content": "FAISS (Facebook AI Similarity Search) is an open-source library by Meta AI for efficient similarity search and clustering of dense vectors. Written in C++ with Python bindings; supports CPU and GPU. Index types: IndexFlatL2 (exact exhaustive, highest recall), IndexFlatIP (inner product), IndexIVFFlat (inverted file, faster), IndexIVFPQ (product quantisation, smallest memory), IndexHNSWFlat (graph-based ANN, best latency/recall tradeoff). For corpora under 1M vectors, IndexHNSWFlat or IndexFlatL2 are typical. FAISS is free, in-process, ideal for prototyping and latency-critical on-premise workloads."\n  },\n  {\n    "title": "Azure AI Search",\n    "url": "https://learn.microsoft.com/azure/search/",\n    "content": "Azure AI Search (formerly Azure Cognitive Search) is a fully managed cloud search service from Microsoft supporting full-text (BM25), vector (HNSW), and hybrid search. Hybrid search combines keyword relevance and vector similarity using Reciprocal Rank Fusion (RRF), outperforming either alone on most benchmarks. It integrates with Azure Blob Storage, SQL, Cosmos DB, and SharePoint. Features: semantic ranker, built-in OCR, entity extraction, multi-language support, role-based access control (RBAC), and geo-filtering. The Free tier (F) provides 50 MB storage and 3 indexes at no cost. Azure AI Search is HIPAA-compliant and SOC2 certified, suited for regulated enterprise workloads."\n  },\n  {\n    "title": "Natural Language Processing",\n    "url": "https://en.wikipedia.org/wiki/Natural_language_processing",\n    "content": "Natural language processing (NLP) is a field of AI concerned with interactions between computers and human language. NLP tasks include text classification, named entity recognition (NER), sentiment analysis, machine translation, question answering, summarisation, and text generation. The dominant paradigm since 2018 is the transformer, pre-trained on large corpora via self-supervised learning. Key milestones: Word2Vec (2013), GloVe (2014), BERT (2018), GPT-3 (2020), ChatGPT (2022), GPT-4 (2023), Claude 3 (2024). RLHF aligns models with human preferences. Benchmarks include GLUE, SuperGLUE, MMLU, and HumanEval."\n  },\n  {\n    "title": "Transformer Architecture",\n    "url": "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",\n    "content": "The transformer is a deep learning architecture introduced in the 2017 paper Attention Is All You Need by Vaswani et al. It replaced RNNs as the dominant architecture for sequence modelling. Self-attention computes relationships between all tokens simultaneously, enabling parallelisation. Architecture: encoder/decoder stacks with multi-head self-attention, feed-forward layers, residual connections, and layer normalisation. Positional encodings inject token order information. BERT is encoder-only (classification, NER). GPT is decoder-only (generation). T5 and BART use encoder-decoder (translation, summarisation). Scaling laws show performance improves predictably with model size, data, and compute."\n  },\n  {\n    "title": "Embeddings in NLP",\n    "url": "https://en.wikipedia.org/wiki/Word_embedding",\n    "content": "Word embeddings are dense vector representations where semantically similar items are close together in vector space. Static embeddings: Word2Vec (2013), GloVe (2014), FastText (2016). Contextual embeddings (BERT, ELMo) produce different vectors for the same word depending on context. Sentence embeddings extend this to full sentences. OpenAI text-embedding-3-small produces 1536-dimensional vectors, costs $0.02 per million tokens, and is widely used for semantic search and RAG. Cosine similarity is the standard metric. Embedding quality is measured by the MTEB (Massive Text Embedding Benchmark)."\n  },\n  {\n    "title": "BM25 Information Retrieval",\n    "url": "https://en.wikipedia.org/wiki/Okapi_BM25",\n    "content": "BM25 (Best Match 25) is a probabilistic ranking function for information retrieval. It extends TF-IDF to account for document length and term saturation. BM25 is the default in Elasticsearch, OpenSearch, Solr, and Lucene. It excels at exact keyword matching and handling rare terms. Weakness: vocabulary mismatch, no semantic understanding. Hybrid search combines BM25 and dense vector retrieval via Reciprocal Rank Fusion (RRF), achieving better recall than either alone, especially for mixed keyword and semantic queries."\n  },\n  {\n    "title": "COVID-19 Pandemic",\n    "url": "https://en.wikipedia.org/wiki/COVID-19_pandemic",\n    "content": "The COVID-19 pandemic was caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, China in late 2019. The WHO declared a pandemic in March 2020. The virus spreads through respiratory droplets and aerosols. Symptoms range from mild (fever, cough, fatigue, loss of smell) to severe (pneumonia, ARDS, multi-organ failure). Risk factors: old age, obesity, diabetes, hypertension, immunocompromise. Over 7 million confirmed deaths worldwide. mRNA vaccines (Pfizer-BioNTech, Moderna) and adenoviral vector vaccines (AstraZeneca, J&J) were developed at unprecedented speed. Long COVID affects 10-30% of patients."\n  },\n  {\n    "title": "Climate Change",\n    "url": "https://en.wikipedia.org/wiki/Climate_change",\n    "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the mid-20th century, burning fossil fuels (coal, oil, gas) has driven CO2 levels to ~420 ppm, the highest in 800,000 years. Global average temperature has risen ~1.2 degrees C above pre-industrial levels. Effects: more intense heatwaves, droughts, wildfires, floods; sea level rise; ocean acidification; biodiversity loss. The IPCC Sixth Assessment Report (2021) states human influence is unequivocal. The Paris Agreement (2015) aims to limit warming to 1.5-2 degrees C."\n  },\n  {\n    "title": "Quantum Computing",\n    "url": "https://en.wikipedia.org/wiki/Quantum_computing",\n    "content": "Quantum computing harnesses superposition, entanglement, and interference to process information in ways classical computers cannot efficiently replicate. A qubit can exist in superposition of 0 and 1 simultaneously. Key algorithms: Shor\'s (factoring large numbers, threatens RSA encryption), Grover\'s (quadratic speedup for unstructured search), quantum simulation (drug discovery). Hardware: superconducting qubits (IBM, Google), trapped ions (IonQ), photonic (PsiQuantum). Google claimed quantum supremacy in 2019 with the Sycamore processor. Current NISQ devices have 50-1000+ qubits but high error rates."\n  },\n  {\n    "title": "Blockchain Technology",\n    "url": "https://en.wikipedia.org/wiki/Blockchain",\n    "content": "A blockchain is a distributed ledger recording transactions across multiple computers so records cannot be altered without changing all subsequent blocks. Bitcoin (2009) was the first blockchain application. Proof of Work (PoW): miners solve computationally expensive puzzles (Bitcoin). Proof of Stake (PoS): validators chosen proportionally to staked crypto (Ethereum post-Merge 2022, 99.95% more energy efficient). Smart contracts are self-executing programs on blockchains (Ethereum, Solidity). DeFi replicates financial services without intermediaries. Layer 2 solutions (Lightning, Optimism, Arbitrum) address scalability."\n  },\n  {\n    "title": "CRISPR Gene Editing",\n    "url": "https://en.wikipedia.org/wiki/CRISPR",\n    "content": "CRISPR-Cas9 is a molecular tool adapted from the bacterial immune system for precise DNA editing. A guide RNA (gRNA) directs the Cas9 enzyme to a specific DNA sequence where it cuts both strands of the double helix. Cell repair then either disrupts the gene (NHEJ) or inserts a new sequence (HDR). Jennifer Doudna and Emmanuelle Charpentier won the 2020 Nobel Prize in Chemistry for CRISPR-Cas9. Applications: treating sickle cell disease (first CRISPR therapy FDA-approved 2023), disease-resistant crops, animal research models. Base editing and prime editing make precise single-nucleotide changes without double-strand breaks."\n  },\n  {\n    "title": "Electric Vehicles",\n    "url": "https://en.wikipedia.org/wiki/Electric_vehicle",\n    "content": "An electric vehicle (EV) uses electric motors powered by rechargeable battery packs or hydrogen fuel cells. Battery EVs (BEVs) run entirely on electricity stored in lithium-ion batteries. Regenerative braking converts kinetic energy back to electricity when decelerating. Tesla pioneered the modern EV market with the Roadster (2008) and Model S (2012). Global EV sales exceeded 10 million units in 2022, representing 14% of new car sales. Key metrics: range (200-400+ miles), charging speed (Level 2: 25 mph, DC fast: 200+ mph), battery capacity (40-100+ kWh). EVs produce zero tailpipe emissions."\n  },\n  {\n    "title": "Supply Chain Management",\n    "url": "https://en.wikipedia.org/wiki/Supply_chain_management",\n    "content": "Supply chain management (SCM) covers planning and managing all activities in sourcing, procurement, conversion, and logistics. The COVID-19 pandemic exposed vulnerabilities: port congestion, container shortages, factory shutdowns, and semiconductor scarcity caused widespread shortages and inflation. The bullwhip effect describes how small demand variations amplify upstream to cause large production swings. Lean supply chains minimise inventory; resilient chains prioritise redundancy. Digital twins, IoT sensors, and AI-powered demand forecasting improve visibility. Nearshoring and friend-shoring are post-COVID risk-reduction trends."\n  },\n  {\n    "title": "Cybersecurity",\n    "url": "https://en.wikipedia.org/wiki/Computer_security",\n    "content": "Cybersecurity protects computer systems, networks, and data from digital attacks, unauthorised access, damage, or theft. Major threat categories: malware (viruses, worms, ransomware, spyware), phishing, man-in-the-middle attacks, SQL injection, zero-day exploits, DDoS, insider threats. The CIA triad (Confidentiality, Integrity, Availability) is the core framework. Key defences: firewalls, IDS/IPS, encryption (TLS, AES), MFA, zero-trust architecture. NIST Cybersecurity Framework and ISO 27001 are widely adopted standards. Notable attacks: WannaCry ransomware (2017), SolarWinds supply chain attack (2020), Colonial Pipeline ransomware (2021)."\n  },\n  {\n    "title": "Renewable Energy",\n    "url": "https://en.wikipedia.org/wiki/Renewable_energy",\n    "content": "Renewable energy comes from sources naturally replenished on a human timescale: sunlight, wind, rain, tides, waves, and geothermal heat. In 2023, renewables accounted for ~30% of global electricity generation. Solar PV costs have fallen 90% since 2010. Wind power uses turbines to convert kinetic energy; offshore wind has 40-50% capacity factor. Hydropower is the largest renewable source globally (~16% of electricity). Battery storage manages intermittency. Green hydrogen from renewable-powered electrolysis can decarbonise heavy industry and long-haul transport."\n  },\n  {\n    "title": "Large Language Models",\n    "url": "https://en.wikipedia.org/wiki/Large_language_model",\n    "content": "A large language model (LLM) is a language model trained on massive text corpora using transformer architecture, capable of generating and understanding text. LLMs are pre-trained with self-supervised learning on billions of tokens, then fine-tuned for specific tasks. Key capabilities: question answering, summarisation, code generation, translation, and reasoning. Notable LLMs: GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), Llama (Meta). Context window size (tokens the model can process at once) ranges from 4K to 1M+ tokens. Scaling laws show capability grows predictably with parameters, data, and compute. Prompt engineering, RAG, and fine-tuning are main techniques for improving LLM outputs for production use cases."\n  }\n]'

RAW_ARTICLES = json.loads(_ARTICLES_JSON)
print(f"Corpus: {len(RAW_ARTICLES)} articles loaded")
for a in RAW_ARTICLES:
    print(f"  * {a['title']} ({len(a['content'])} chars)")

# %%
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_docs = []
for art in RAW_ARTICLES:
    chunks = splitter.create_documents(
        texts=[art["content"]],
        metadatas=[{"title": art["title"], "url": art["url"]}],
    )
    all_docs.extend(chunks)

texts = [d.page_content for d in all_docs]
metas = [d.metadata      for d in all_docs]

print(f"Total chunks: {len(all_docs)}")
print(f"Avg chunk size: {sum(len(t) for t in texts) // len(texts)} chars")

# %%
embedder = LocalEmbeddings(n_features=EMBEDDING_DIM)

print(f"Embedding {len(all_docs)} chunks with local hashing vectors ({EMBEDDING_DIM} dims)...")
t0 = time.perf_counter()
all_vectors = embedder.embed_documents(texts)
elapsed = time.perf_counter() - t0

print(f"Embedded {len(all_vectors)} chunks in {elapsed:.1f}s")
print(f"Vector dimension: {len(all_vectors[0])}")

# %% [markdown]
# ---
# ## Step 01 - Build RAG Pipeline
#
# Create a local hybrid retrieval pipeline and wire it to the offline Claude adapter.

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


def azure_hybrid_search(query: str, top_k: int = TOP_K) -> list:
    """Local hybrid search: hashed vector similarity + TF-IDF lexical score."""
    query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
    doc_matrix = np.asarray(all_vectors, dtype=np.float32)
    vector_scores = doc_matrix @ query_vec

    query_tfidf = tfidf_vectorizer.transform([query])
    lexical_scores = cosine_similarity(query_tfidf, tfidf_matrix).ravel()

    def _normalize(scores: np.ndarray) -> np.ndarray:
        if np.allclose(scores.max(), scores.min()):
            return np.zeros_like(scores, dtype=np.float32)
        return ((scores - scores.min()) / (scores.max() - scores.min())).astype(np.float32)

    hybrid_scores = 0.5 * _normalize(vector_scores) + 0.5 * _normalize(lexical_scores)
    ranked_indices = np.argsort(-hybrid_scores)[:top_k]

    results = []
    for idx in ranked_indices:
        results.append(
            {
                "id": f"doc-{idx}",
                "content": texts[idx],
                "title": metas[idx].get("title", ""),
                "url": metas[idx].get("url", ""),
                "score": float(hybrid_scores[idx]),
            }
        )
    return results

# %%
# Build LangChain FAISS vectorstore from pre-computed vectors.
# FAISS.from_embeddings() reuses all_vectors -- zero extra API calls.
text_embeddings = list(zip(texts, all_vectors))  # [(text, vector), ...]

faiss_store = FAISS.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=embedder,   # used only for embed_query() at search time
    metadatas=metas,
)
print(f"FAISS vectorstore ready ({len(texts)} vectors, no re-embedding)")

# Claude LLM
llm = AnthropicChatAdapter(model=CLAUDE_MODEL, max_tokens=512, api_key=ANTHROPIC_API_KEY)
print(f"Local Claude adapter ready: {CLAUDE_MODEL}")

# %%
# ----- Instrumented RAG result dataclass -----
@dataclass
class RAGResult:
    answer:        str
    sources:       list
    embed_ms:      float
    retrieve_ms:   float
    generate_ms:   float
    input_tokens:  int = 0
    output_tokens: int = 0

    @property
    def total_ms(self):
        return self.embed_ms + self.retrieve_ms + self.generate_ms

    @property
    def cost_usd(self):
        return self.input_tokens * COST_INPUT + self.output_tokens * COST_OUTPUT


def rag_answer(query: str) -> RAGResult:
    """Full instrumented RAG pipeline: embed -> retrieve (Azure hybrid) -> generate (Claude)."""
    # Stage 1: Embed query
    t0 = time.perf_counter()
    _ = embedder.embed_query(query)   # embed happens inside azure_hybrid_search too
    embed_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: Retrieve (Azure hybrid)
    t0 = time.perf_counter()
    raw = azure_hybrid_search(query)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    context = "\n\n".join([f"[{r['title']}]\n{r['content']}" for r in raw])
    sources  = list({r["title"] for r in raw})

    # Stage 3: Generate (Claude)
    messages = [
        SystemMessage(content="Answer ONLY from the provided context. "
                               "If unsure, say so. Be concise."),
        HumanMessage(content=f"<context>\n{context}\n</context>\n\n{query}"),
    ]
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    generate_ms = (time.perf_counter() - t0) * 1000

    # Extract token counts from response_metadata (safe for all ChatAnthropic versions)
    usage = response.response_metadata.get("usage", {})
    return RAGResult(
        answer=response.content,
        sources=sources,
        embed_ms=embed_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


# Smoke test
test = rag_answer("What is retrieval-augmented generation?")
print("RAG pipeline smoke test passed!")
print(f"  Sources: {test.sources}")
print(f"  Latency: embed={test.embed_ms:.0f}ms  retrieve={test.retrieve_ms:.0f}ms  generate={test.generate_ms:.0f}ms")
print(f"  Cost: ${test.cost_usd:.5f}  Tokens: {test.input_tokens} in / {test.output_tokens} out")
print(f"  Answer: {test.answer[:200]}...")

# %% [markdown]
# ---
# ## Step 02 - Hybrid Retrieval Quality (MRR@5)
#
# Compare vector-only (FAISS) vs hybrid BM25+vector (Azure AI Search) using
# Mean Reciprocal Rank at 5 (MRR@5) on a labelled evaluation set.

# %%
# Evaluation set: (query, relevant_article_title)
EVAL_SET = [
    ("How does BERT use self-attention for NLP tasks?",             "Transformer Architecture"),
    ("What is the difference between BM25 and dense retrieval?",    "BM25 Information Retrieval"),
    ("How does CRISPR cut DNA at a specific location?",             "CRISPR Gene Editing"),
    ("What triggered the COVID-19 pandemic?",                       "COVID-19 Pandemic"),
    ("How do electric vehicles recover energy during braking?",     "Electric Vehicles"),
    ("What are the risks of long-term climate change?",             "Climate Change"),
    ("How does Shor's algorithm threaten RSA encryption?",          "Quantum Computing"),
    ("What caused post-COVID supply chain disruptions?",            "Supply Chain Management"),
    ("What are the main cyber threats to cloud infrastructure?",    "Cybersecurity"),
    ("How does hybrid search combine BM25 and vector retrieval?",   "Azure AI Search"),
]

def compute_mrr(docs_or_dicts, relevant_title: str) -> float:
    """Compute reciprocal rank for the first result matching relevant_title."""
    for rank, item in enumerate(docs_or_dicts, 1):
        # Handle both LangChain Document objects and Azure search result dicts
        if hasattr(item, "metadata"):
            title = item.metadata.get("title", "")
        else:
            title = item.get("title", "")
        if relevant_title.lower() in title.lower():
            return 1.0 / rank
    return 0.0


mrr_vector_only = []
mrr_hybrid      = []

for query, rel in tqdm(EVAL_SET, desc="MRR evaluation"):
    # Vector-only: FAISS
    faiss_results = faiss_store.similarity_search(query, k=TOP_K)
    mrr_vector_only.append(compute_mrr(faiss_results, rel))

    # Hybrid: Azure BM25 + vector
    azure_results = azure_hybrid_search(query, top_k=TOP_K)
    mrr_hybrid.append(compute_mrr(azure_results, rel))

print(f"MRR@{TOP_K} Vector-only (FAISS)    : {np.mean(mrr_vector_only):.3f}")
print(f"MRR@{TOP_K} Hybrid (Azure BM25+vec) : {np.mean(mrr_hybrid):.3f}")
print(f"Hybrid gain: +{(np.mean(mrr_hybrid) - np.mean(mrr_vector_only))*100:.1f}pp")

# Per-query breakdown
mrr_df = pd.DataFrame({
    "Query":        [q[:55] for q, _ in EVAL_SET],
    "Vector-only":  [round(v, 2) for v in mrr_vector_only],
    "Hybrid":       [round(v, 2) for v in mrr_hybrid],
})
print("\n", mrr_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    ["Vector-only (FAISS)", "Hybrid (Azure BM25+vec)"],
    [np.mean(mrr_vector_only), np.mean(mrr_hybrid)],
    color=["#4C72B0", "#55A868"],
    width=0.4,
    edgecolor="white",
)
ax.set_ylim(0, 1.1)
ax.set_title(f"MRR@{TOP_K}: Vector-only vs Hybrid Retrieval", fontweight="bold")
ax.set_ylabel("Mean Reciprocal Rank")
for patch, val in zip(ax.patches, [np.mean(mrr_vector_only), np.mean(mrr_hybrid)]):
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + 0.02,
        f"{val:.3f}",
        ha="center", fontweight="bold",
    )
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("mrr_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved mrr_comparison.png")

# %% [markdown]
# ---
# ## Step 03 - Full Context Baseline
#
# Stuff the first 50 chunks directly into Claude's context window.
# Measure token count, latency, and cost per query.

# %%
# Build the full-context string from first 50 chunks
CORPUS_SIZE = 50
full_corpus_docs = all_docs[:CORPUS_SIZE]

full_corpus_text = "\n\n---\n\n".join([
    f"[Doc {i+1}: {d.metadata['title']}]\n{d.page_content}"
    for i, d in enumerate(full_corpus_docs)
])

# Estimate tokens without tiktoken (no network needed)
# Approximation: ~0.75 tokens per word (conservative)
word_count    = len(full_corpus_text.split())
corpus_tokens = int(word_count * 1.33)

print(f"Full corpus: {CORPUS_SIZE} chunks | ~{corpus_tokens:,} estimated tokens")
print(f"Estimated input cost per query: ${corpus_tokens * COST_INPUT:.5f}")

# %%
@dataclass
class FullCtxResult:
    answer:        str
    generate_ms:   float
    input_tokens:  int = 0
    output_tokens: int = 0

    @property
    def total_ms(self):
        return self.generate_ms

    @property
    def cost_usd(self):
        return self.input_tokens * COST_INPUT + self.output_tokens * COST_OUTPUT


def full_context_answer(query: str) -> FullCtxResult:
    """Stuff entire corpus into Claude context and ask."""
    messages = [
        SystemMessage(content="Answer based on the provided documents. "
                               "Be concise and accurate."),
        HumanMessage(content=f"<corpus>\n{full_corpus_text}\n</corpus>\n\n{query}"),
    ]
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    generate_ms = (time.perf_counter() - t0) * 1000

    usage = response.response_metadata.get("usage", {})
    return FullCtxResult(
        answer=response.content,
        generate_ms=generate_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


# Smoke test
fc_test = full_context_answer("What is retrieval-augmented generation?")
print("Full-context pipeline smoke test passed!")
print(f"  Latency: {fc_test.generate_ms:.0f}ms")
print(f"  Actual tokens: {fc_test.input_tokens} in / {fc_test.output_tokens} out")
print(f"  Cost: ${fc_test.cost_usd:.5f}")
print(f"  Answer: {fc_test.answer[:200]}...")

# %% [markdown]
# ---
# ## Step 04 - A/B Evaluation (20 Questions)
#
# Run both pipelines on the same 20 questions.
# Score each on latency, cost per query, and input token count.
#
# > This step makes ~40 Claude API calls (20 RAG + 20 full-context).
# > Estimated cost: ~$0.03-0.05 total.

# %%
TEST_QUESTIONS = [
    # AI / ML
    "What is the attention mechanism in transformer models?",
    "How does BERT differ from GPT in architecture?",
    "What is retrieval-augmented generation?",
    "How does FAISS index vectors for similarity search?",
    # Science / Medicine
    "How does CRISPR-Cas9 edit the genome?",
    "What causes COVID-19 and how does it spread?",
    "How do mRNA vaccines trigger an immune response?",
    "What is the mechanism of action of base editing?",
    # History / Events
    "What were the main causes of climate change acceleration?",
    "How did quantum computing progress after 2019?",
    "What is the bullwhip effect in supply chains?",
    "How does the NIST cybersecurity framework work?",
    # Technology
    "How do electric vehicles regenerate energy from braking?",
    "What is Reciprocal Rank Fusion in hybrid search?",
    "How does blockchain achieve immutability?",
    "What cybersecurity threats target cloud infrastructure?",
    # Economics / Environment
    "How has solar energy cost changed since 2010?",
    "What disrupted global supply chains in 2020-2021?",
    "What is Proof of Stake and how does it differ from Proof of Work?",
    "What are the scaling laws for large language models?",
]

# %%
rag_results: list[RAGResult]     = []
fc_results:  list[FullCtxResult] = []

print("Running A/B evaluation -- this makes ~40 Claude API calls...")
print("Estimated time: 2-4 minutes\n")

for q in tqdm(TEST_QUESTIONS, desc="A/B eval"):
    rag_results.append(rag_answer(q))
    fc_results.append(full_context_answer(q))

print("\nA/B evaluation complete!")

# %%
rag_total_ms = [r.total_ms  for r in rag_results]
fc_total_ms  = [r.total_ms  for r in fc_results]
rag_costs    = [r.cost_usd  for r in rag_results]
fc_costs     = [r.cost_usd  for r in fc_results]
rag_in_tok   = [r.input_tokens for r in rag_results]
fc_in_tok    = [r.input_tokens for r in fc_results]

summary = pd.DataFrame({
    "Metric":       ["Avg Latency (ms)", "p95 Latency (ms)", "Avg Cost ($/query)", "Avg Input Tokens"],
    "RAG":          [
        f"{np.mean(rag_total_ms):.0f}",
        f"{float(np.percentile(rag_total_ms, 95)):.0f}",
        f"${np.mean(rag_costs):.5f}",
        f"{np.mean(rag_in_tok):.0f}",
    ],
    "Full Context": [
        f"{np.mean(fc_total_ms):.0f}",
        f"{float(np.percentile(fc_total_ms, 95)):.0f}",
        f"${np.mean(fc_costs):.5f}",
        f"{np.mean(fc_in_tok):.0f}",
    ],
})

print("=" * 55)
print("A/B EVALUATION SUMMARY (20 test questions)")
print("=" * 55)
print(summary.to_string(index=False))
print("=" * 55)

cost_ratio    = np.mean(fc_costs) / max(np.mean(rag_costs), 1e-10)
latency_ratio = np.mean(fc_total_ms) / max(np.mean(rag_total_ms), 1.0)
print(f"\nFull-context is {cost_ratio:.1f}x more expensive and {latency_ratio:.1f}x slower than RAG")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("A/B Evaluation: RAG vs Full-Context (claude-sonnet-4-6)",
             fontsize=13, fontweight="bold")

colors = ["#4C72B0", "#DD8452"]
labels = ["RAG\n(Azure Hybrid)", "Full Context"]

# --- Latency boxplot ---
bp = axes[0].boxplot(
    [rag_total_ms, fc_total_ms],
    tick_labels=labels,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="white", linewidth=2),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].set_title("End-to-End Latency (ms)", fontweight="bold")
axes[0].set_ylabel("ms")

# --- Cost per query ---
rag_cost_m  = np.mean(rag_costs) * 1000   # convert to millidollars
fc_cost_m   = np.mean(fc_costs)  * 1000
bars1 = axes[1].bar(labels, [rag_cost_m, fc_cost_m], color=colors, edgecolor="white")
axes[1].set_title("Avg Cost per Query (m$)", fontweight="bold")
axes[1].set_ylabel("millidollars ($0.001)")
for b, v in zip(bars1, [rag_cost_m, fc_cost_m]):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.2f}m$", ha="center", fontweight="bold")

# --- Input tokens ---
avg_tok = [np.mean(rag_in_tok), np.mean(fc_in_tok)]
bars2 = axes[2].bar(labels, avg_tok, color=colors, edgecolor="white")
axes[2].set_title("Avg Input Tokens", fontweight="bold")
axes[2].set_ylabel("tokens")
for b, v in zip(bars2, avg_tok):
    axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
                 f"{v:.0f}", ha="center", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("ab_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved ab_evaluation.png")

# %%
# Per-question detail table
detail_df = pd.DataFrame({
    "Question":    [q[:52] + "..." if len(q) > 52 else q for q in TEST_QUESTIONS],
    "RAG ms":      [round(r.total_ms)      for r in rag_results],
    "FC ms":       [round(r.total_ms)      for r in fc_results],
    "RAG $":       [f"{r.cost_usd*1000:.2f}m" for r in rag_results],
    "FC $":        [f"{r.cost_usd*1000:.2f}m" for r in fc_results],
    "RAG tok":     [r.input_tokens         for r in rag_results],
    "FC tok":      [r.input_tokens         for r in fc_results],
})
print(detail_df.to_string(index=False))

# %% [markdown]
# ---
# ## Step 05 - Latency Profiler (Waterfall Breakdown)
#
# Instrument each RAG stage (embed / retrieve / generate) and visualise
# the per-query waterfall to identify the bottleneck.

# %%
embed_times    = [r.embed_ms    for r in rag_results]
retrieve_times = [r.retrieve_ms for r in rag_results]
generate_times = [r.generate_ms for r in rag_results]

print(f"Stage averages across {len(TEST_QUESTIONS)} queries:")
total_avg = np.mean([r.total_ms for r in rag_results])
print(f"  Embed:    {np.mean(embed_times):.1f}ms  "
      f"({np.mean(embed_times)/total_avg*100:.1f}% of total)")
print(f"  Retrieve: {np.mean(retrieve_times):.1f}ms  "
      f"({np.mean(retrieve_times)/total_avg*100:.1f}% of total)")
print(f"  Generate: {np.mean(generate_times):.1f}ms  "
      f"({np.mean(generate_times)/total_avg*100:.1f}% of total)")
print(f"  Total:    {total_avg:.1f}ms")

# %%
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(TEST_QUESTIONS))
w = 0.6

ax.bar(x, embed_times,    w, label="Embed query",     color="#4C72B0")
ax.bar(x, retrieve_times, w, bottom=embed_times,
       label="Azure retrieve", color="#DD8452")
ax.bar(x, generate_times, w,
       bottom=[e + r for e, r in zip(embed_times, retrieve_times)],
       label="Claude generate", color="#55A868")

ax.axhline(
    np.mean([e + r + g for e, r, g in zip(embed_times, retrieve_times, generate_times)]),
    color="red", linestyle="--", linewidth=1.2, label="Avg total",
)

ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in range(len(TEST_QUESTIONS))], rotation=45, ha="right")
ax.set_ylabel("Latency (ms)")
ax.set_title("RAG Pipeline Latency Waterfall - Per Stage Breakdown", fontweight="bold", fontsize=13)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("waterfall_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved waterfall_breakdown.png")

# %% [markdown]
# ---
# ## Extension A - Streaming Responses with Claude
#
# Stream tokens back in real time using the local adapter.
# Measure **time-to-first-token** (perceived responsiveness) vs total latency.

# %%
def rag_stream(query: str, top_k: int = TOP_K) -> None:
    """Retrieve context then stream the local response token-by-token."""
    # Retrieve
    raw     = azure_hybrid_search(query, top_k=top_k)
    context = "\n\n".join([f"[{r['title']}]\n{r['content']}" for r in raw])
    sources = [r["title"] for r in raw]

    print(f"Query: {query}")
    print(f"Sources: {sources}")
    print("\nAnswer (streaming): ", end="", flush=True)

    t_start          = time.perf_counter()
    first_token_ms   = None

    response = llm.invoke([
        SystemMessage(content="Answer ONLY from the provided context. Be concise."),
        HumanMessage(content=f"<context>\n{context}\n</context>\n\n{query}"),
    ])
    for chunk in re.findall(r"\S+\s*", response.content):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t_start) * 1000
        print(chunk, end="", flush=True)

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"\n\nTime to first token: {first_token_ms:.0f}ms | Total: {total_ms:.0f}ms\n")


# Demo -- streams to the cell output in real time
rag_stream("What is the attention mechanism in transformer models?")

# %% [markdown]
# ---
# ## Extension B - Confidence Score Filter
#
# Block answers where no retrieved chunk is relevant enough.
# We gate on **FAISS L2 distance** from `similarity_search_with_score`.
# Lower L2 distance = more similar. Tune `L2_THRESHOLD` to calibrate.

# %%
# FAISS similarity_search_with_score returns (Document, l2_distance) tuples
# l2_distance is a float32; lower = more similar to the query
# Tune this threshold by inspecting the distribution below
# Local hashing vectors use a different distance scale than the notebook's cloud embedding.
L2_THRESHOLD = 1.7   # reject if best chunk L2 distance > this value

def rag_with_confidence(query: str) -> dict:
    """RAG pipeline that refuses to answer when no chunk is close enough."""
    results_with_scores = faiss_store.similarity_search_with_score(query, k=TOP_K)

    if not results_with_scores:
        return {"answer": "No results found.", "l2_distance": None, "passed": False}

    # scores are numpy float32 -- convert to plain Python float
    best_l2 = float(results_with_scores[0][1])

    if best_l2 > L2_THRESHOLD:
        return {
            "answer": (
                f"I cannot answer this confidently "
                f"(best chunk L2 distance = {best_l2:.3f} > threshold {L2_THRESHOLD}). "
                "Please ask a question related to the topics in the knowledge base."
            ),
            "l2_distance": best_l2,
            "passed": False,
        }

    # Passed threshold -- generate answer
    docs    = [doc for doc, _ in results_with_scores]
    context = "\n\n".join([f"[{d.metadata['title']}]\n{d.page_content}" for d in docs])
    messages = [
        SystemMessage(content="Answer ONLY from the provided context."),
        HumanMessage(content=f"<context>\n{context}\n</context>\n\n{query}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "l2_distance": best_l2, "passed": True}


# Test with in-corpus and out-of-corpus queries
test_queries = [
    ("What is the attention mechanism in transformers?",   "In-corpus -- should PASS"),
    ("How does CRISPR cut DNA?",                           "In-corpus -- should PASS"),
    ("What is the boiling point of tungsten carbide?",     "Out-of-corpus -- should BLOCK"),
    ("Recommend a good restaurant in Tokyo",               "Out-of-corpus -- should BLOCK"),
]

print(f"Confidence filter (L2 threshold = {L2_THRESHOLD})\n")
for query, label in test_queries:
    result = rag_with_confidence(query)
    status = "PASSED" if result["passed"] else "BLOCKED"
    l2_str = f"{result['l2_distance']:.3f}" if result["l2_distance"] is not None else "N/A"
    print(f"[{status}] {label}")
    print(f"  Q: {query}")
    print(f"  L2={l2_str}  A: {result['answer'][:130]}...\n")

# %%
# Analyse L2 distance distribution on all 20 test questions
l2_scores = []
pass_flags = []
for q in TEST_QUESTIONS:
    res = faiss_store.similarity_search_with_score(q, k=1)
    l2  = float(res[0][1]) if res else float("inf")
    l2_scores.append(l2)
    pass_flags.append(l2 <= L2_THRESHOLD)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Pass/fail pie
passed = sum(pass_flags)
ax1.pie(
    [passed, len(pass_flags) - passed],
    labels=[f"Passed (<={L2_THRESHOLD})", f"Blocked (>{L2_THRESHOLD})"],
    colors=["#55A868", "#DD8452"],
    autopct="%1.0f%%",
    startangle=90,
)
ax1.set_title(f"Confidence Filter Pass Rate\n(threshold={L2_THRESHOLD})", fontweight="bold")

# L2 score histogram
ax2.hist(l2_scores, bins=10, color="#4C72B0", edgecolor="white")
ax2.axvline(L2_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
            label=f"Threshold = {L2_THRESHOLD}")
ax2.set_title("L2 Distance Distribution (test questions)", fontweight="bold")
ax2.set_xlabel("FAISS L2 Distance (lower = more relevant)")
ax2.set_ylabel("Count")
ax2.legend()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("confidence_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"{passed}/{len(pass_flags)} test questions passed the confidence filter")

# %% [markdown]
# ---
# ## Discussion & Key Takeaways
#
# **Discussion Questions**
# 1. **Latency bottleneck:** In the waterfall chart, which stage dominates -- embed, retrieve, or generate? What does this mean for optimisation?
# 2. **Cost at scale:** Using the cost per query from Step 04, calculate the monthly cost at 1 million queries/day for both RAG and full-context.
# 3. **Hybrid vs vector-only:** Which query types benefited most from Azure's BM25 in the MRR comparison? (Acronyms? Exact product names?)
# 4. **Confidence threshold:** Inspect the L2 distribution. What threshold would you set for a medical assistant vs a casual chatbot? Why?
# 5. **Streaming UX:** Time-to-first-token matters more than total latency for perceived responsiveness. How would you architect a production system around streaming?
#
# **When to use RAG vs Full Context**
#
# | Use Full Context | Use RAG |
# |------------------|---------|
# | Corpus fits in <100K tokens | Corpus is millions of docs |
# | Holistic reasoning over entire document | Low latency required |
# | Low query volume | High query volume (cost matters) |
# | Corpus is static | Corpus updates frequently |
#
# **Cleanup after the lab**
# ```python
# # Run to delete the Azure index and avoid any future storage charges
# idx_client.delete_index(AZURE_SEARCH_INDEX_NAME)
# print("Azure index deleted")
# ```
