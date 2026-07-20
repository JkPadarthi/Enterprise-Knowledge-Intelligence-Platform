"""Tests for the summarization agent (Phase 4)."""

from __future__ import annotations

import pytest

from agents.summarizer import SummaryAgent
from llm.mock import MockBackend
from models.schema import AgentState


@pytest.mark.asyncio
async def test_summarizer_both_keys(embedder, mock_llm):
    """SummaryAgent returns both abstractive and extractive summaries for non-empty text."""
    # Arrange
    agent = SummaryAgent(llm=mock_llm, embedder=embedder)
    state = AgentState(doc_id="d", raw_text="One. Two. Three.")

    # Act
    result = await agent.run(state)

    # Assert
    assert "summary" in result
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert "abstractive" in summary
    assert "extractive" in summary
    assert summary["abstractive"] == mock_llm.text_response
    assert isinstance(summary["extractive"], str)
    assert len(summary["extractive"]) > 0
    # The extractive summary should contain at least one of the sentences
    assert any(sentence in summary["extractive"] for sentence in ["One.", "Two.", "Three."])


@pytest.mark.asyncio
async def test_summarizer_empty(embedder, mock_llm):
    """SummaryAgent returns empty dict for empty or whitespace-only text."""
    # Arrange
    agent = SummaryAgent(llm=mock_llm, embedder=embedder)
    state = AgentState(raw_text="   ")  # only spaces

    # Act
    result = await agent.run(state)

    # Assert
    assert result == {}