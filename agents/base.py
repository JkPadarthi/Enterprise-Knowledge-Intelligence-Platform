"""Base class shared by every pipeline agent.

Provides structured logging (prefixed with the agent name) and a retry wrapper so
individual agents stay small and consistent. Agents are pure: ``run(state, **deps)``
reads the shared state and returns a *partial* dict of fields it updated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models.schema import AgentState


class BaseAgent:
    """Common logging + retry behaviour for all agents."""

    name: str = "base"

    def __init__(self, settings=None, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(f"graphrag.agents.{self.name}")

    def _log(self, message: str, level: int = logging.INFO, *args: Any) -> None:
        if args:
            self.logger.log(level, "[%s] " + message, self.name, *args)
        else:
            self.logger.log(level, "[%s] %s", self.name, message)

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        """Process ``state`` and return the fields this agent updated."""
        raise NotImplementedError

    async def run_with_retry(self, state: AgentState, retries: int = 3, **deps: Any) -> dict[str, Any]:
        """Run :meth:`run` with exponential backoff up to ``retries`` attempts."""
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return await self.run(state, **deps)
            except Exception as exc:  # noqa: BLE001 - we want to retry anything
                last_exc = exc
                self._log("attempt %d failed: %s", logging.ERROR, attempt, exc)
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 10))
        raise RuntimeError(f"{self.name} failed after {retries} attempts: {last_exc}") from last_exc
