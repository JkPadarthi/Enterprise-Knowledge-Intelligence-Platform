"""Tests for the QA orchestrator (basic vector RAG)."""

from __future__ import annotations

import pytest

from agents.qa import QAOrchestrator
from models.schema import Chunk


@pytest.mark.asyncio
async def test_qa_returns_answer_and_citations(embedder, chroma_store, settings, mock_llm):
    texts = [
        "The GraphRAG engine uses Neo4j and ChromaDB.",
        "LangGraph orchestrates the agents.",
    ]
    chunks = [
        Chunk(id="d1::c0", doc_id="d1", text=texts[0], index=0),
        Chunk(id="d1::c1", doc_id="d1", text=texts[1], index=1),
    ]
    chroma_store.add_chunks(chunks, embedder.encode(texts).tolist())

    orch = QAOrchestrator(settings, llm=mock_llm, embedder=embedder, vector_store=chroma_store)
    result = await orch.answer("What does the engine use?", doc_id="d1", top_k=2)

    assert result["qa_answer"] == "The answer is 42."
    assert len(result["citations"]) == 2
    assert result["citations"][0].chunk_id.startswith("d1::c")
