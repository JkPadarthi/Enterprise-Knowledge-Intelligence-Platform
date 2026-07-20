"""Knowledge-graph build agent (Phase 3).

Upserts the entities/relations produced by :class:`NERAgent` into a graph store.
The store is injectable (defaults to ``Neo4jStore``) so the agent is testable with
a fake that records writes. Writes are idempotent (keyed by ``doc_id``) so
re-ingesting the same document does not duplicate nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from config.settings import Settings
from models.schema import AgentState


class KnowledgeGraphAgent(BaseAgent):
    """Writes extracted entities/relations into the graph store (Neo4j by default)."""

    name = "graph_builder"

    def __init__(
        self,
        settings: Settings | None = None,
        logger: Any | None = None,
        graph_store: Any | None = None,
    ) -> None:
        super().__init__(settings, logger)
        self._graph_store = graph_store

    def _get_store(self, graph_store: Any | None) -> Any:
        if graph_store is not None:
            return graph_store
        if self._graph_store is not None:
            return self._graph_store
        from graph.neo4j_store import Neo4jStore

        return Neo4jStore(self.settings)

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        if not state.entities and not state.relationships:
            self._log("no entities/relations to write; skipping", level=logging.INFO)
            return {"graph_written": False}

        store = self._get_store(deps.get("graph_store"))
        # The Neo4j driver is created lazily; open it if it isn't already
        # connected (e.g. when the store is default-constructed here rather
        # than injected already-open from the API dependency).
        opened_here = False
        if getattr(store, "_driver", None) is None and hasattr(store, "connect"):
            await store.connect()
            opened_here = True

        doc_id = state.doc_id

        entities = [e.model_dump() for e in state.entities]
        relations = [r.model_dump() for r in state.relationships]

        try:
            # Idempotent upsert: replace this document's subgraph, then write entities/relations.
            if hasattr(store, "replace_document_graph"):
                await store.replace_document_graph(doc_id, entities, relations)
            else:
                await store.upsert_entities(entities)
                await store.upsert_relationships(relations)
        finally:
            if opened_here and hasattr(store, "close"):
                await store.close()

        self._log(
            "wrote %d entities, %d relations for %s",
            logging.INFO,
            len(entities),
            len(relations),
            doc_id,
        )
        return {"graph_written": True}
