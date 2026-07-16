"""Ollama backend for fully offline / local-LLM development."""

from __future__ import annotations

from typing import Any

from llm.client import LLMClient, Message


class OllamaBackend(LLMClient):
    """LLM backend that talks to a local Ollama server.

    Accepts either an ``ollama.AsyncClient`` instance or a base URL; the heavy
    ``ollama`` import is deferred to construction so the dependency is only required
    when this backend is actually selected.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        super().__init__(model)
        from ollama import AsyncClient

        self._client = AsyncClient(host=base_url)

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        response = await self._client.chat(model=self.model, messages=messages)
        if isinstance(response, dict):
            return response["message"]["content"] or ""
        return response.message.content or ""
