import asyncio
from unittest.mock import AsyncMock

import pytest

from agents.translator import TranslationAgent
from config.settings import Settings
from llm.mock import MockBackend
from models.schema import AgentState


@pytest.mark.asyncio
async def test_translator_happy_path_non_english():
    """Happy path: non-English text gets translated."""
    # Arrange
    mock_backend = MockBackend(text_response="hello world")
    agent = TranslationAgent(Settings(), llm=mock_backend)
    state = AgentState(raw_text="Bonjour", language="fr")

    # Act
    result = await agent.run(state)

    # Assert
    assert result == {"translated_text": "hello world"}
    assert state.translated_text == "hello world"


@pytest.mark.asyncio
async def test_translator_skip_when_target_language():
    """Skip translation when language is already target (English)."""
    # Arrange
    # We'll use a mock that tracks if acomplete was called.
    class CallTrackingMock(MockBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.acomplete_calls = 0

        async def acomplete(self, messages, **kwargs):
            self.acomplete_calls += 1
            # Return something that would be wrong if called
            return "SHOULD_NOT_BE_CALLED"

    mock_backend = CallTrackingMock(text_response="unused")
    agent = TranslationAgent(Settings(), llm=mock_backend)
    state = AgentState(raw_text="Hello", language="en")  # target language is 'en'

    # Act
    result = await agent.run(state)

    # Assert
    assert result == {}
    assert state.translated_text is None  # Should remain None
    assert mock_backend.acomplete_calls == 0  # LLM not called


@pytest.mark.asyncio
async def test_translator_skip_when_empty_text():
    """Skip translation when raw_text is empty or whitespace."""
    # Arrange
    mock_backend = MockBackend(text_response="SHOULD_NOT_BE_CALLED")
    agent = TranslationAgent(Settings(), llm=mock_backend)
    state = AgentState(raw_text="   ", language="fr")  # whitespace only

    # Act
    result = await agent.run(state)

    # Assert
    assert result == {}
    assert state.translated_text is None
    # We can also check that the mock wasn't called if we want, but the contract
    # only requires that we return {} and not call the LLM. We'll check the call count.
    # However, the mock doesn't track calls. Let's make a tracking mock for this test too.
    # But to keep the test simple, we'll just trust that we returned {} and didn't call.
    # We'll do a separate test with a tracking mock if needed, but the spec says to
    # assert the LLM call count is 0 for the English skip case. For empty text, we
    # can do similarly.

    # Let's redo with a tracking mock for completeness.
    class TrackingMock(MockBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.acomplete_calls = 0

        async def acomplete(self, messages, **kwargs):
            self.acomplete_calls += 1
            return "SHOULD_NOT_BE_CALLED"

    tracking_mock = TrackingMock(text_response="unused")
    agent2 = TranslationAgent(Settings(), llm=tracking_mock)
    state2 = AgentState(raw_text="   ", language="fr")
    result2 = await agent2.run(state2)
    assert result2 == {}
    assert tracking_mock.acomplete_calls == 0