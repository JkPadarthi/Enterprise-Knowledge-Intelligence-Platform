"""NER + relation extraction agent (Phase 3 — not yet implemented)."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from models.schema import AgentState


class NERAgent(BaseAgent):
    """Extracts entities (GLiNER) and relations (LLM prompt) into ``state.*``."""

    name = "ner"

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Phase 3: extract entities with GLiNER (labels from settings.ner_label_list) "
            "and relations via LLMClient.acomplete_json, then set state.entities / "
            "state.relationships."
        )
