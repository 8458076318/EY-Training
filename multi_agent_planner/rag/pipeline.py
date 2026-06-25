"""
Full RAG pipeline: retrieve → rerank → generate → hallucination check.
"""
import json
import logging
from agents.openai_agent import OpenAIAgent
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.hallucination import HallucinationDetector

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = OpenAIAgent()
        self.detector = HallucinationDetector()

    async def query(self, question: str) -> dict:
        # 1. Retrieve
        raw_chunks = await self.retriever.retrieve(question, top_k=10)

        # 2. Re-rank
        ranked = await self.reranker.rerank(question, raw_chunks)
        top_chunks = ranked[:5]

        # 3. Generate answer
        context_text = "\n".join(str(c) for c in top_chunks)
        prompt = f"Using the context below, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {question}\nReturn JSON: {{answer: str, sources: []}}"
        gen_result = await self.generator.run(prompt)
        answer_data = json.loads(gen_result["result"])

        # 4. Hallucination check
        check = await self.detector.check(answer_data.get("answer", ""), [str(c) for c in top_chunks])

        return {
            "question": question,
            "answer": answer_data.get("answer"),
            "sources": answer_data.get("sources", []),
            "hallucination_check": check,
            "chunks_retrieved": len(raw_chunks),
        }
