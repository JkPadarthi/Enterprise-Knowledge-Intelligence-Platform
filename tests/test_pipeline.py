"""End-to-end ingestion pipeline test."""

from __future__ import annotations

from unittest.mock import patch

import fitz
import pytest

from models.schema import AgentState
from orchestration.pipeline import run_ingest


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text)
    data = doc.tobytes()
    doc.close()
    return data


async def _ner_run(state, **deps):
    # Lightweight stand-in for the (model-downloading) NERAgent so this offline
    # smoke test exercises the orchestration graph without fetching GLiNER.
    return {"entities": [], "relationships": []}


@pytest.mark.asyncio
async def test_run_ingest_indexes(embedder, chroma_store, settings):
    pdf = _make_pdf("GraphRAG ingests PDFs and builds a knowledge graph. " * 20)
    with patch("orchestration.pipeline.NERAgent") as MockNER:
        MockNER.return_value.run = _ner_run
        state = await run_ingest(
            pdf, "doc.pdf", "doc1", settings, embedder=embedder, vector_store=chroma_store
        )
    assert isinstance(state, AgentState)
    assert state.num_pages == 1
    assert len(state.chunk_ids) > 0
    assert chroma_store.count() == len(state.chunk_ids)
