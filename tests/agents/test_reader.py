"""Tests for the reader agent (PDF ingestion)."""

from __future__ import annotations

import fitz
import pytest

from agents.reader import ReaderAgent
from models.schema import AgentState


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_reader_extracts_text():
    agent = ReaderAgent()
    state = AgentState(file_bytes=_make_pdf("Hello World from the GraphRAG engine."))
    result = await agent.run(state)
    assert result["num_pages"] == 1
    assert "Hello World" in result["raw_text"]


@pytest.mark.asyncio
async def test_reader_rejects_non_pdf():
    agent = ReaderAgent()
    with pytest.raises(ValueError):
        await agent.run(AgentState(file_bytes=b"not a pdf"))
