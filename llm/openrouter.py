"""OpenRouter backend (OpenAI-compatible chat completions API)."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from llm.client import LLMClient, Message

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ALLOWED_KWARGS = {"temperature", "max_tokens", "top_p", "stop"}


class OpenRouterBackend(LLMClient):
    """LLM backend that routes through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str, *, timeout: float = 120.0) -> None:
        super().__init__(model)
        self._client = AsyncOpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=api_key or "dummy",
            timeout=timeout,
        )

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        filtered = {k: v for k, v in kwargs.items() if k in _ALLOWED_KWARGS}
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **filtered,
        )
        return response.choices[0].message.content or ""
