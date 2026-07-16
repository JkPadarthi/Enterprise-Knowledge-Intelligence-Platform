"""Knowledge-graph build agent (Phase 3 — not yet implemented)."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from models.schema import AgentState


class KnowledgeGraphAgent(BaseAgent):
    """Writes extracted entities/relations into Neo4j via ``Neo4jStore``."""

    name = "graph_builder"

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "Phase 3: upsert state.entities / state.relationships into Neo4j with "
            "idempotent MERGE (via Neo4jStore) and set state.graph_written."
        )
