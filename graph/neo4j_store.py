"""Neo4j knowledge-graph store (Phase 3).

The connection plumbing is implemented; entity/relationship writes and graph
queries are intentionally left as explicit ``NotImplementedError`` stubs so the
import graph and the orchestration skeleton stay complete without shipping
half-finished graph logic. Phase 3 fills these in.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j import AsyncGraphDatabase

from config.settings import Settings


class Neo4jStore:
    """Async wrapper around the Neo4j Python driver (v5)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._driver: Optional[Any] = None

    async def connect(self) -> None:
        """Open the driver connection."""
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )

    async def close(self) -> None:
        """Close the driver connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def upsert_entities(self, *args: Any, **kwargs: Any) -> None:
        """Persist extracted entities into the graph. Implemented in Phase 3."""
        raise NotImplementedError("Neo4j entity writes are implemented in Phase 3.")

    async def upsert_relationships(self, *args: Any, **kwargs: Any) -> None:
        """Persist extracted relations into the graph. Implemented in Phase 3."""
        raise NotImplementedError("Neo4j relationship writes are implemented in Phase 3.")

    async def query_graph(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """Run a read-only Cypher query. Implemented in Phase 3."""
        raise NotImplementedError("Neo4j graph queries are implemented in Phase 3.")
