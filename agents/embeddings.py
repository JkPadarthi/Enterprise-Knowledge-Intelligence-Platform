"""Embedding agent — chunks text and indexes it into ChromaDB (Phase 1).

Also exposes :func:`chunk_text`, a dependency-free character-splitter with overlap
used by other phases. The embedder and vector store are injectable so the agent is
fully unit-testable without downloading models or touching disk.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from agents.base import BaseAgent
from config.settings import Settings
from models.schema import AgentState, Chunk


class Embedder(Protocol):
    """Minimal embedder protocol: map texts -> float matrix."""

    def encode(self, texts: list[str]) -> Any:  # noqa: D102
        ...


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ``chunk_size`` characters.

    Args:
        text: Source document text.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks (must be < chunk_size).

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(stripped):
        end = min(start + chunk_size, len(stripped))
        segment = stripped[start:end].strip()
        if segment:
            chunks.append(segment)
        if end == len(stripped):
            break
        start += step
    return chunks


class EmbeddingAgent(BaseAgent):
    """Chunks the (translated) document text, embeds it, and stores it in ChromaDB."""

    name = "embeddings"

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        vector_store: Any | None = None,
    ) -> None:
        super().__init__(settings)
        self._embedder = embedder
        self._vector_store = vector_store

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _get_store(self, vector_store: Any | None) -> Any:
        if vector_store is not None:
            return vector_store
        if self._vector_store is not None:
            return self._vector_store
        from vector.chroma_store import ChromaStore

        return ChromaStore(self.settings)

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        text = state.translated_text or state.raw_text
        if not text.strip():
            self._log("no text to embed; skipping", logging.WARNING)
            return {"chunk_ids": []}

        store = self._get_store(deps.get("vector_store"))
        texts = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        chunks = [
            Chunk(id=f"{state.doc_id}::c{i}", doc_id=state.doc_id, text=t, index=i)
            for i, t in enumerate(texts)
        ]

        loop = asyncio.get_running_loop()
        raw_embeddings = await loop.run_in_executor(
            None, lambda: self._get_embedder().encode(texts)
        )
        embeddings = raw_embeddings.tolist() if hasattr(raw_embeddings, "tolist") else list(raw_embeddings)

        await loop.run_in_executor(None, lambda: store.add_chunks(chunks, embeddings))
        self._log("embedded %d chunks", logging.INFO, len(chunks))
        return {"chunk_ids": [c.id for c in chunks]}
