"""Summarization agent (Phase 4 — not yet implemented)."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from models.schema import AgentState


class SummaryAgent(BaseAgent):
    """Produces extractive (TextRank) + abstractive (LLM) summaries."""

    name = "summarizer"

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Phase 4: build an extractive summary (e.g. sumy LexRank) and an abstractive "
            "summary via LLMClient, then set state.summary = {abstractive, extractive}."
        )
