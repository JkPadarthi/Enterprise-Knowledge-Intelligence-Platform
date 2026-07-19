"""Shared API dependencies and the in-memory document registry."""

from __future__ import annotations

from functools import lru_cache

from config.settings import Settings, get_settings
from graph.neo4j_store import Neo4jStore
from models.schema import DocumentMeta
from vector.chroma_store import ChromaStore

# In-memory document registry. Phase 3 persists document metadata to Neo4j;
# for Phase 1 this is sufficient to list/inspect ingested documents.
DOCUMENT_REGISTRY: dict[str, DocumentMeta] = {}


@lru_cache
def get_vector_store() -> ChromaStore:
    """Process-wide singleton ChromaDB store."""
    return ChromaStore(get_settings())


@lru_cache
def get_graph_store() -> Neo4jStore:
    """Process-wide singleton Neo4j store (driver created, not yet connected)."""
    return Neo4jStore(get_settings())


async def get_graph_store_connected() -> Neo4jStore:
    """Dependency that connects the Neo4j driver for the request and closes it after.

    ``Neo4jStore`` builds the driver in :meth:`connect`; the v5 async driver
    must be opened before any query and closed afterwards, so we manage that
    lifespan per request here rather than leaking a driver across the process.
    """
    store = get_graph_store()
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


def get_settings_dep() -> Settings:
    """FastAPI dependency returning the cached settings object."""
    return get_settings()