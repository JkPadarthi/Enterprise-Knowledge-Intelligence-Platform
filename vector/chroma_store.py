"""ChromaDB vector-store wrapper.

Supports both embedded persistent mode (default, no extra container) and HTTP
server mode (set ``CHROMA_HOST``). The collection stores document chunks with
flat metadata so retrieval can be filtered by ``doc_id`` and ranked by cosine
similarity.
"""

from __future__ import annotations

from typing import Optional

import chromadb

from config.settings import Settings
from models.schema import Chunk


class ChromaStore:
    """Thin async-friendly wrapper around a ChromaDB collection."""

    def __init__(self, settings: Settings, collection_name: str = "documents") -> None:
        self._settings = settings
        if settings.chroma_host:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        else:
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist ``chunks`` with their ``embeddings`` (parallel lists)."""
        if not chunks:
            return
        metadatas = [
            {**chunk.metadata, "doc_id": chunk.doc_id, "index": chunk.index}
            for chunk in chunks
        ]
        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Return the top-``top_k`` nearest chunks to ``query_embedding``."""
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def get_by_doc(self, doc_id: str) -> dict:
        """Return every stored chunk for a document."""
        return self._collection.get(where={"doc_id": doc_id})

    def count(self) -> int:
        """Number of stored chunks."""
        return self._collection.count()
