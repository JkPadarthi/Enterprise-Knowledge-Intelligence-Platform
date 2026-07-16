"""Tests for the language detection agent."""

from __future__ import annotations

import pytest

from agents.language_detect import LanguageDetectionAgent
from config.settings import Settings
from models.schema import AgentState


@pytest.mark.asyncio
async def test_language_detect_happy():
    """Should detect French text and set language and confidence."""
    settings = Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir="/tmp")
    agent = LanguageDetectionAgent(settings)
    state = AgentState(raw_text="Bonjour le monde entier")
    result = await agent.run(state)
    # Expect a dict with language and language_confidence
    assert "language" in result
    assert "language_confidence" in result
    # langdetect returns a two-letter code for French
    assert result["language"] == "fr"
    # Confidence should be a float between 0 and 1
    assert 0.0 <= result["language_confidence"] <= 1.0
    # Actually, for clear French it should be high
    assert result["language_confidence"] > 0.5


@pytest.mark.asyncio
async def test_language_detect_empty():
    """Should return empty dict for blank or whitespace-only text."""
    settings = Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir="/tmp")
    agent = LanguageDetectionAgent(settings)
    # Test empty string
    state = AgentState(raw_text="")
    result = await agent.run(state)
    assert result == {}
    # Test whitespace only
    state = AgentState(raw_text="   \t\n   ")
    result = await agent.run(state)
    assert result == {}