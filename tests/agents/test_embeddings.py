"""Tests for the embedding agent and the chunker."""

from __future__ import annotations

import pytest

from agents.embeddings import EmbeddingAgent, chunk_text
from models.schema import AgentState


def test_chunk_text_splits_with_overlap():
    chunks = chunk_text("a" * 250, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 3
    assert all(c for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("   ", 100, 20) == []


@pytest.mark.asyncio
async def test_embeddings_stores_chunks(embedder, chroma_store, settings):
    agent = EmbeddingAgent(settings, embedder=embedder, vector_store=chroma_store)
    state = AgentState(doc_id="d1", raw_text="Sentence one. Sentence two. Sentence three. " * 10)
    result = await agent.run(state)
    assert len(result["chunk_ids"]) > 0
    assert chroma_store.count() == len(result["chunk_ids"])
