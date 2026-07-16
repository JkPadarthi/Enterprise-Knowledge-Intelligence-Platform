"""Provider-agnostic LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Message = dict[str, str]


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

        raw = await self.acomplete(messages, **kwargs)
        return json.loads(raw)

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
