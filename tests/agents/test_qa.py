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


class FakeGraphStore:
    """Fake graph store that returns a fixed result for query_graph."""
    async def query_graph(self, cypher: str, params: dict | None = None):
        # Return a single row with subject, relation, object
        return [{"subject": "Acme Corp", "relation": "HEADQUARTERED_IN", "object": "Paris"}]


class FakeVectorStore:
    """Fake vector store that returns a fixed result for query."""
    def query(self, query_embedding, top_k: int, where: dict | None = None):
        # Return a dict in the format expected by the QA orchestrator
        return {
            "ids": [["c0"]],
            "documents": [["Acme is in Paris."]],
            "metadatas": [[{"doc_id": "d1"}]],
            "distances": [[0.1]],
        }


class MessageCapturingMockBackend:
    """Mock LLM backend that captures the messages passed to acomplete."""
    def __init__(self, text_response: str = "Acme is in Paris."):
        self.text_response = text_response
        self.last_messages = None

    async def acomplete(self, messages, **kwargs):
        self.last_messages = messages
        return self.text_response


@pytest.mark.asyncio
async def test_hybrid_qa_returns_graph_citations(
    embedder, settings, chroma_store
):
    # Arrange
    fake_graph = FakeGraphStore()
    mock_llm = MessageCapturingMockBackend(text_response="Acme is in Paris.")
    orch = QAOrchestrator(
        settings,
        llm=mock_llm,
        embedder=embedder,
        vector_store=chroma_store,
        graph_store=fake_graph,
    )

    # Act
    result = await orch.answer(
        "Where is Acme headquartered?", doc_id="d1", top_k=2, graph_store=fake_graph
    )

    # Assert
    # Check that at least one citation is from the graph
    graph_citations = [c for c in result["citations"] if c.source == "graph"]
    assert len(graph_citations) > 0
    assert graph_citations[0].node_ref == "Acme Corp"

    # Check that the LLM prompt contains the graph facts and the heading
    user_message = None
    for msg in mock_llm.last_messages:
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    assert user_message is not None
    assert "## Knowledge graph" in user_message
    assert "Acme Corp HEADQUARTERED_IN Paris" in user_message


@pytest.mark.asyncio
async def test_qa_vector_only_without_graph_store(
    embedder, settings, chroma_store
):
    # Arrange
    fake_vector = FakeVectorStore()
    mock_llm = MessageCapturingMockBackend(text_response="Answer from vector.")
    orch = QAOrchestrator(
        settings,
        llm=mock_llm,
        embedder=embedder,
        vector_store=fake_vector,
        graph_store=None,  # No graph store
    )

    # Act
    result = await orch.answer(
        "What is in the document?", doc_id="d1", top_k=2, graph_store=None
    )

    # Assert
    # All citations should be from vector
    for citation in result["citations"]:
        assert citation.source == "vector"

    # Check that the LLM prompt does NOT contain the graph heading
    user_message = None
    for msg in mock_llm.last_messages:
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    assert user_message is not None
    assert "## Knowledge graph" not in user_message