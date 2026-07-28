"""Ollama backend for fully offline / local-LLM development."""

from __future__ import annotations

import asyncio
from typing import Any

from llm.client import LLMClient, Message


class OllamaBackend(LLMClient):
    """LLM backend that talks to a local Ollama server.

    Accepts either an ``ollama.AsyncClient`` instance or a base URL; the heavy
    ``ollama`` import is deferred to construction so the dependency is only required
    when this backend is actually selected.
    """

    def __init__(
        self, model: str, base_url: str = "http://localhost:11434", *, timeout: float = 120.0
    ) -> None:
        super().__init__(model)
        from ollama import AsyncClient

        self._client = AsyncClient(host=base_url)
        self.timeout = timeout

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        try:
            response = await asyncio.wait_for(
                self._client.chat(model=self.model, messages=messages),
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s (model={self.model!r})"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed (model={self.model!r}, "
                f"base_url={self._client.host!r}): {exc}"
            ) from exc
        if isinstance(response, dict):
            return response["message"]["content"] or ""
        return response.message.content or ""
