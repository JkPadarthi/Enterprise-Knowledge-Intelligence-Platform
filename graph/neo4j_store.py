"""Neo4j knowledge-graph store (Phase 3).

Connection plumbing plus the entity/relationship write and read methods are
implemented. Writes are doc-scoped (keyed by ``doc_id``) so a document's
subgraph can be deleted cleanly without touching other documents; graph *search*
via :meth:`query_graph` is global unless a caller passes a ``doc_id`` filter.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from neo4j import AsyncGraphDatabase

from config.settings import Settings

# Neo4j relationship type identifiers are restricted to this character class.
_REL_TYPE_RE = re.compile(r"^[A-Z0-9_]+$")


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

    @staticmethod
    def _safe_rel_type(relation: str) -> str:
        """Sanitize a relation name into a valid Cypher relationship type.

        Relationship *types* must be identifiers, not data, so they cannot be
        parameterized. We uppercase and strip to a safe character set; anything
        outside [A-Z0-9_] collapses to an underscore.
        """
        safe = re.sub(r"[^A-Za-z0-9_]", "_", relation).upper()
        return safe or "RELATED"

    async def upsert_entities(self, entities: list[dict]) -> None:
        """Merge each entity as a ``:Entity`` node keyed by (doc_id, text, label)."""
        if not entities:
            return
        safe = [
            {
                "id": e.get("id"),
                "doc_id": e.get("doc_id", ""),
                "text": e.get("text", ""),
                "label": e.get("label", "MISC"),
                "start": e.get("start"),
                "end": e.get("end"),
            }
            for e in entities
        ]

        async def _work(tx, entities_batch):
            await tx.run(
                """
                UNWIND $entities AS e
                MERGE (n:Entity {doc_id: e.doc_id, text: e.text, label: e.label})
                SET n.id = e.id, n.start = e.start, n.end = e.end
                """,
                entities=entities_batch,
            )

        async with self._driver.session() as session:
            await session.execute_write(_work, safe)

    async def upsert_relationships(self, relationships: list[dict]) -> None:
        """Merge each relation as a typed edge between its (doc-scoped) entities."""
        if not relationships:
            return
        for rel in relationships:
            rel_type = self._safe_rel_type(rel.get("relation", "RELATED"))
            params = {
                "doc_id": rel.get("doc_id", ""),
                "subject": rel.get("subject", ""),
                "object": rel.get("object", ""),
                "relation": rel.get("relation", ""),
                "id": rel.get("id"),
            }

            async def _work(tx, rel_type, params):
                await tx.run(
                    f"""
                    MATCH (s:Entity {{doc_id: $doc_id, text: $subject}})
                    MATCH (o:Entity {{doc_id: $doc_id, text: $object}})
                    MERGE (s)-[r:{rel_type} {{doc_id: $doc_id, relation: $relation, id: $id}}]->(o)
                    """,
                    params,
                )

            async with self._driver.session() as session:
                await session.execute_write(_work, rel_type, params)

    async def replace_document_graph(
        self, doc_id: str, entities: list[dict], relationships: list[dict]
    ) -> None:
        """Idempotently (re)write a document's subgraph.

        Deletes every node carrying this ``doc_id`` (and its relationships), then
        merges the supplied entities and relations. Re-ingesting the same document
        is therefore a clean replace with no duplicates.
        """

        async def _delete(tx, doc_id):
            await tx.run(
                """
                MATCH (n:Entity|Relation {doc_id: $doc_id})
                DETACH DELETE n
                """,
                doc_id=doc_id,
            )

        async with self._driver.session() as session:
            await session.execute_write(_delete, doc_id)
        await self.upsert_entities(entities)
        await self.upsert_relationships(relationships)

    async def query_graph(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """Run a read-only Cypher query and return the rows as dicts."""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            return [dict(record) async for record in result]
