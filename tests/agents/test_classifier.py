"""Tests for the ClassificationAgent."""

from __future__ import annotations

import asyncio
import logging

import pytest

from agents.classifier import ClassificationAgent
from config.settings import Settings
from models.schema import AgentState
from tests.conftest import FakeClassifierPipe


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_happy_path(settings) -> None:
    """Classification returns a valid label and scores sum to ~1.0."""
    agent = ClassificationAgent(settings)
    agent._pipe = FakeClassifierPipe()  # inject fake
    state = AgentState(raw_text="Invoice for services rendered")
    result = asyncio.run(agent.run(state))

    assert result == {
        "doc_type": state.doc_type,
        "doc_type_scores": state.doc_type_scores,
    }
    # doc_type must be one of the labels
    assert state.doc_type in settings.doc_type_label_list
    # scores dict keys must be subset of labels
    assert set(state.doc_type_scores.keys()) == set(settings.doc_type_label_list)
    # scores sum to approximately 1.0
    assert sum(state.doc_type_scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_empty_input(settings) -> None:
    """Empty or whitespace-only input returns empty dict."""
    agent = ClassificationAgent(settings)
    agent._pipe = FakeClassifierPipe()
    for raw in ["", "   ", "\t\n  "]:
        state = AgentState(raw_text=raw)
        result = asyncio.run(agent.run(state))
        assert result == {}


def test_truncation(settings) -> None:
    """Input longer than the token limit is truncated before classification."""
    agent = ClassificationAgent(settings)
    fake = FakeClassifierPipe()
    agent._pipe = fake
    long_text = "x" * 5000
    state = AgentState(raw_text=long_text)
    asyncio.run(agent.run(state))
    # Fake pipe has no tokenizer, so _truncate falls back to a 1000-char cap.
    assert len(fake.last_text) <= 1000