"""Shared API dependencies and the in-memory document registry."""

from __future__ import annotations

from functools import lru_cache

from config.settings import Settings, get_settings
from models.schema import DocumentMeta
from vector.chroma_store import ChromaStore

# In-memory document registry. Phase 3 persists document metadata to Neo4j;
# for Phase 1 this is sufficient to list/inspect ingested documents.
DOCUMENT_REGISTRY: dict[str, DocumentMeta] = {}


@lru_cache
def get_vector_store() -> ChromaStore:
    """Process-wide singleton ChromaDB store."""
    return ChromaStore(get_settings())


def get_settings_dep() -> Settings:
    """FastAPI dependency returning the cached settings object."""
    return get_settings()
