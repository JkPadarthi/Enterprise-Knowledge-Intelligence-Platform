"""Question-answering endpoint (Phase 1 — vector RAG)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.qa import QAOrchestrator
from api.deps import get_settings_dep, get_vector_store
from config.settings import Settings
from llm import get_llm_client
from models.schema import QARequest, QAResponse
from vector.chroma_store import ChromaStore

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("")
async def answer_question(
    req: QARequest,
    settings: Settings = Depends(get_settings_dep),
    store: ChromaStore = Depends(get_vector_store),
) -> QAResponse:
    """Answer a question over indexed documents with vector retrieval + citations."""
    llm = get_llm_client("qa", settings)
    orchestrator = QAOrchestrator(settings, llm=llm, vector_store=store)
    result = await orchestrator.answer(req.question, doc_id=req.doc_id, top_k=req.top_k)
    return QAResponse(
        answer=result["qa_answer"],
        citations=result["citations"],
        model=settings.qa_model,
    )
