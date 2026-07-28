"""Shared pytest fixtures for Phase 1 tests.

Provides a settings object, a deterministic fake embedder, an in-memory ChromaDB
store (ephemeral, no disk), and a mock LLM client — so agents run without network,
API keys, or model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import Settings
from models.schema import Chunk


class FakeEmbedder:
    """Deterministic embedder returning fixed-dimension random vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: list[str]):
        rng = np.random.default_rng(42)
        return rng.random((len(texts), self.dim))


class FakeClassifierPipe:
    """Deterministic fake zero-shot classifier for testing.

    Returns fixed scores: first label gets 0.7, remaining share 0.3 equally.
    """

    def __init__(self) -> None:
        self.last_text: str = ""

    def __call__(self, text: str, candidate_labels):
        self.last_text = text
        labels = list(candidate_labels)
        scores = [0.7] + [0.3 / (len(labels) - 1)] * (len(labels) - 1)
        return {"sequence": text, "labels": labels, "scores": scores}


class FakeChromaStore:
    """In-memory fake ChromaStore for testing without ChromaDB."""

    def __init__(self):
        self._chunks = {}
        self._embeddings = {}

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = emb

    def query(
        self, query_embedding: list[float], top_k: int = 5, where: dict = None
    ) -> dict:
        import numpy as np

        if not self._embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embedding)
        scores = []
        for cid, emb in self._embeddings.items():
            e = np.array(emb)
            if np.linalg.norm(e) > 0 and np.linalg.norm(q) > 0:
                sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e)))
            else:
                sim = 0.0
            scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        if where and "doc_id" in where:
            doc_id = where["doc_id"]
            top = [(cid, sim) for cid, sim in top if self._chunks[cid].doc_id == doc_id]

        ids = [[cid for cid, _ in top]]
        docs = [[self._chunks[cid].text for cid, _ in top]]
        metas = [[self._chunk_meta(cid) for cid, _ in top]]
        dists = [[1.0 - sim for _, sim in top]]

        return {"ids": ids, "documents": docs, "metadatas": metas, "distances": dists}

    def _chunk_meta(self, cid: str) -> dict:
        chunk = self._chunks[cid]
        return {**chunk.metadata, "doc_id": chunk.doc_id, "index": chunk.index}

    def get_by_doc(self, doc_id: str) -> dict:
        result = {"ids": [], "documents": [], "metadatas": []}
        for cid, chunk in self._chunks.items():
            if chunk.doc_id == doc_id:
                result["ids"].append(cid)
                result["documents"].append(chunk.text)
                result["metadatas"].append(self._chunk_meta(cid))
        return result

    def count(self) -> int:
        return len(self._chunks)


class FakeNeo4jStore:
    """In-memory fake Neo4jStore for testing without Neo4j."""

    def __init__(self):
        self._nodes = []
        self._edges = []
        self._connected = False

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False

    def upsert_nodes(self, nodes):
        self._nodes.extend(nodes)

    def upsert_edges(self, edges):
        self._edges.extend(edges)

    def get_subgraph(self, doc_id):
        return {"nodes": [], "edges": []}

    def get_document_metadata(self, doc_id):
        return {}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def chroma_store():
    return FakeChromaStore()


@pytest.fixture
def mock_llm():
    from llm.mock import MockBackend

    return MockBackend(model="mock", text_response="The answer is 42.")


class FakeSentimentPipe:
    """Deterministic mock sentiment pipeline returning a fixed label/score."""

    def __init__(self) -> None:
        self.last_text: str | None = None

    def __call__(self, text: str):
        self.last_text = text
        return [{"label": "POSITIVE", "score": 0.98}]
