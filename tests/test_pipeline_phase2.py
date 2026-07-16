"""Tests for the Phase 2 ingestion graph with NLP agents."""

from __future__ import annotations

import asyncio
import fitz  # PyMuPDF
from unittest.mock import patch

import pytest

from chromadb import EphemeralClient
from config.settings import Settings
from orchestration.pipeline import build_ingest_graph, run_ingest
from vector.chroma_store import ChromaStore

# Reuse fixtures from conftest by importing them via pytest
# We'll define our own fakes similar to those in conftest for simplicity in this file,
# but we can also import them if we mark the test to use the fixtures.
# For simplicity, we'll define minimal fakes here.


class FakeEmbedder:
    """Deterministic embedder returning fixed-dimension random vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: list[str]):
        import numpy as np

        rng = np.random.default_rng(42)
        return rng.random((len(texts), self.dim))


class FakeClassifierPipe:
    """Deterministic fake zero-shot classifier for testing."""

    def __init__(self) -> None:
        self.last_text: str = ""

    def __call__(self, text: str, candidate_labels):
        self.last_text = text
        labels = list(candidate_labels)
        scores = [0.7] + [0.3 / (len(labels) - 1)] * (len(labels) - 1)
        return {"sequence": text, "labels": labels, "scores": scores}


class FakeSentimentPipe:
    """Deterministic mock sentiment pipeline returning a fixed label/score."""

    def __init__(self) -> None:
        self.last_text: str | None = None

    def __call__(self, text: str):
        self.last_text = text
        return [{"label": "POSITIVE", "score": 0.9}]


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


def test_build_ingest_graph_returns_compiled_graph(settings):
    """Smoke test: build_ingest_graph returns a compiled graph object."""
    graph = build_ingest_graph(settings)
    assert graph is not None
    # The compiled graph should have a `ainvoke` method
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_english_doc_skips_translator(settings, embedder, chroma_store):
    """English document should skip the translator (translated_text remains None)."""
    # Create a simple one-page PDF with English text.
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Invoice for services. I love this product.")
    pdf_bytes = doc.tobytes()
    doc.close()

    # Patch the agent classes at the pipeline module's binding so the graph uses
    # fakes (no real model loading). run() must stay async because the graph awaits it.
    async def _langdetect_run(state, **deps):
        return {"language": "en", "language_confidence": 1.0}

    async def _classifier_run(state, **deps):
        return {"doc_type": "invoice"}

    async def _sentiment_run(state, **deps):
        return {"sentiment_label": "POSITIVE", "sentiment_score": 0.9}

    with patch(
        "orchestration.pipeline.LanguageDetectionAgent"
    ) as MockLangDetect, patch(
        "orchestration.pipeline.ClassificationAgent"
    ) as MockClassifier, patch(
        "orchestration.pipeline.SentimentAgent"
    ) as MockSentiment:
        MockLangDetect.return_value.run = _langdetect_run
        MockClassifier.return_value.run = _classifier_run
        MockSentiment.return_value.run = _sentiment_run

        # Run the ingestion pipeline.
        state = await run_ingest(
            pdf_bytes,
            filename="doc.pdf",
            doc_id="d1",
            settings=settings,
            embedder=embedder,
            vector_store=chroma_store,
        )

        # Assertions
        # Language should be set to English.
        assert state.language == "en"
        # Translator should have been skipped, so translated_text should be None.
        assert state.translated_text is None
        # Classifier should have run.
        assert state.doc_type == "invoice"
        # Sentiment should have run.
        assert state.sentiment_label == "POSITIVE"
        assert state.sentiment_score == 0.9
        # Embeddings should have been generated (at least one chunk).
        assert len(state.chunk_ids) > 0