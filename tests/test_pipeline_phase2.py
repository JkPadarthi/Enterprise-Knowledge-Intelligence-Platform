"""Tests for the Phase 2 ingestion graph with NLP agents."""

from __future__ import annotations

from unittest.mock import patch

import fitz  # PyMuPDF
import pytest

from config.settings import Settings
from orchestration.pipeline import build_ingest_graph, run_ingest
from tests.conftest import FakeChromaStore, FakeEmbedder


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def chroma_store() -> FakeChromaStore:
    return FakeChromaStore()


def test_build_ingest_graph_returns_compiled_graph(settings):
    """Smoke test: build_ingest_graph returns a compiled graph object."""
    graph = build_ingest_graph(settings)
    assert graph is not None
    # The compiled graph should have an `ainvoke` method
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

    async def _ner_run(state, **deps):
        return {"entities": [], "relationships": []}

    with patch(
        "orchestration.pipeline.LanguageDetectionAgent"
    ) as MockLangDetect, patch(
        "orchestration.pipeline.ClassificationAgent"
    ) as MockClassifier, patch(
        "orchestration.pipeline.SentimentAgent"
    ) as MockSentiment, patch(
        "orchestration.pipeline.NERAgent"
    ) as MockNER:
        MockLangDetect.return_value.run = _langdetect_run
        MockClassifier.return_value.run = _classifier_run
        MockSentiment.return_value.run = _sentiment_run
        MockNER.return_value.run = _ner_run

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
        # NER should have run (stubbed) and embeddings generated (at least one chunk).
        assert state.entities == []
        assert len(state.chunk_ids) > 0
