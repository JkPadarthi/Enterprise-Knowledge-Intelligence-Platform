"""Summary endpoint (Phase 4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.summarizer import SummaryAgent
from api.deps import get_settings_dep, get_vector_store
from config.settings import Settings
from llm import get_llm_client
from models.schema import AgentState
from vector.chroma_store import ChromaStore

router = APIRouter(prefix="/documents", tags=["summary"])


@router.get("/{doc_id}/summary")
async def get_summary(
    doc_id: str,
    settings: Settings = Depends(get_settings_dep),
    vector_store: ChromaStore = Depends(get_vector_store),
) -> dict:
    """Return extractive and abstractive summary for a document."""
    # Get LLM for summarization
    llm = get_llm_client("summary", settings)
    
    # Create summarizer agent
    agent = SummaryAgent(settings, llm=llm, embedder=None)
    
    # Get document chunks from vector store
    result = vector_store.get_by_doc(doc_id)
    documents = result.get("documents", [])
    
    # Flatten list of lists and join with newlines
    if documents:
        # Flatten the list of lists
        flat_chunks = [chunk for sublist in documents for chunk in sublist]
        text = "\n".join(flat_chunks)
    else:
        text = ""
    
    # If no chunks, return empty summary
    if not text:
        return {"summary": {"abstractive": "", "extractive": ""}}
    
    # Create agent state and run summarizer
    state = AgentState(doc_id=doc_id, raw_text=text)
    result = await agent.run(state)
    
    # Return the summary dict
    return {"summary": result.get("summary", {"abstractive": "", "extractive": ""})}