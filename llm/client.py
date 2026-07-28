"""Provider-agnostic LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Message = dict[str, str]


class LLMParseError(Exception):
    """Raised when an LLM completion cannot be parsed as JSON."""


class LLMClient(ABC):
    """Common interface for every LLM backend.

    Implementations must provide :meth:`acomplete` (async). The sync
    :meth:`complete` wrapper exists for non-event-loop contexts (e.g. a CLI);
    callers inside an async runtime should use :meth:`acomplete` directly.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def acomplete(self, messages: list[Message], **kwargs: Any) -> str:
        """Return the assistant completion text for ``messages``."""
        raise NotImplementedError

    async def acomplete_json(self, messages: list[Message], **kwargs: Any) -> Any:
        """Return the completion parsed as JSON (used for structured extraction)."""
        import json
        import re

        raw = await self.acomplete(messages, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Some models wrap JSON in a ```json ... ``` markdown fence; strip it and retry once.
            fenced = re.search(
                r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE
            )
            if fenced:
                try:
                    return json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    pass
            truncated = raw if len(raw) <= 200 else raw[:200] + "..."
            raise LLMParseError(
                "Failed to parse LLM response as JSON. "
                f"Raw response (truncated): {truncated!r}"
            ) from None

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        """Synchronous wrapper around :meth:`acomplete`.

        Raises if called from within a running event loop; use ``acomplete`` there.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acomplete(messages, **kwargs))
        raise RuntimeError("complete() cannot run inside an event loop; use acomplete()")
