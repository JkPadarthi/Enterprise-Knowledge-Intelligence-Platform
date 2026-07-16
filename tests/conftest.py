"""Shared pytest fixtures for Phase 1 tests.

Provides a settings object, a deterministic fake embedder, an in-memory ChromaDB
store (ephemeral, no disk), and a mock LLM client — so agents run without network,
API keys, or model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from chromadb import EphemeralClient

from config.settings import Settings
from vector.chroma_store import ChromaStore


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


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def chroma_store(settings) -> ChromaStore:
    import uuid

    store = ChromaStore(settings)
    # EphemeralClient is process-wide in this chromadb version, so use a unique
    # collection name per test to avoid cross-test contamination of counts.
    store._client = EphemeralClient()
    collection_name = f"test_{uuid.uuid4().hex}"
    store._collection = store._client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    return store


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