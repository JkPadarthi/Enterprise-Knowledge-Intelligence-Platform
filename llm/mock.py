"""Deterministic mock backend used for unit tests and offline dry-runs."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.client import LLMClient, Message

logger = logging.getLogger(__name__)


class MockBackend(LLMClient):
    """Returns canned responses so agents can be tested without network or keys."""

    def __init__(
        self,
        model: str = "mock",
        *,
        text_response: str = "mock completion",
        json_response: Any | None = None,
    ) -> None:
        super().__init__(model)
        self.text_response = text_response
        self.json_response = json_response

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        return self.text_response

    async def acomplete_json(self, messages: list[Message], **kwargs: Any) -> Any:
        if self.json_response is not None:
            return self.json_response
        try:
            return json.loads(self.text_response)
        except json.JSONDecodeError:
            logger.warning(
                "MockBackend.json_response is None and text_response is not valid JSON; "
                "returning {} as a safe default. text_response=%r",
                self.text_response,
            )
            return {}
