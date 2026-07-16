"""Tests for the sentiment analysis agent."""

from __future__ import annotations

import asyncio

from agents.sentiment import SentimentAgent
from config.settings import Settings
from models.schema import AgentState
from tests.conftest import FakeSentimentPipe


def test_sentiment_happy():
    agent = SentimentAgent(Settings())
    agent._pipe = FakeSentimentPipe()
    state = AgentState(raw_text="I love this product")
    result = asyncio.run(agent.run(state))
    assert result["sentiment_label"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert 0.0 <= result["sentiment_score"] <= 1.0


def test_sentiment_empty():
    agent = SentimentAgent(Settings())
    agent._pipe = FakeSentimentPipe()
    result = asyncio.run(agent.run(AgentState(raw_text="   ")))
    assert result == {}


def test_sentiment_truncation():
    agent = SentimentAgent(Settings())
    agent._pipe = FakeSentimentPipe()
    long_text = "word " * 5000
    asyncio.run(agent.run(AgentState(raw_text=long_text)))
    # Fake pipe has no tokenizer, so _truncate falls back to a 1000-char cap.
    assert len(agent._pipe.last_text) <= 1000
