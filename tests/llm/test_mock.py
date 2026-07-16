"""Tests for the mock LLM backend."""

from __future__ import annotations

import pytest

from llm.mock import MockBackend


@pytest.mark.asyncio
async def test_mock_returns_text():
    mock = MockBackend(text_response="hi")
    assert await mock.acomplete([{"role": "user", "content": "x"}]) == "hi"


@pytest.mark.asyncio
async def test_mock_returns_json():
    mock = MockBackend(json_response={"relations": []})
    assert await mock.acomplete_json([{"role": "user", "content": "x"}]) == {"relations": []}
