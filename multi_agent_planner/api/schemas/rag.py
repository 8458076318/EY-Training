from pydantic import BaseModel


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    question: str
    answer: str | None
    sources: list
    hallucination_check: dict
    chunks_retrieved: int
