from fastapi import APIRouter, HTTPException
from api.schemas.rag import RAGRequest, RAGResponse
from rag.pipeline import RAGPipeline
import logging

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Assistant"])
logger = logging.getLogger(__name__)
pipeline = RAGPipeline()


@router.post("/query", response_model=RAGResponse)
async def rag_query(req: RAGRequest):
    try:
        return await pipeline.query(req.question)
    except Exception as e:
        logger.error("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
