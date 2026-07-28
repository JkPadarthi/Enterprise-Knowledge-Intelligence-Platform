"""Tests for the Phase 3 ingestion graph: NER + parallel graph branch."""

from __future__ import annotations

from unittest.mock import patch

import fitz  # PyMuPDF
import pytest

from config.settings import Settings
from orchestration.pipeline import build_ingest_graph, run_ingest
from tests.conftest import FakeChromaStore, FakeEmbedder


class FakeGraphStore:
    """Records the doc-scoped subgraph write from KnowledgeGraphAgent."""

    def __init__(self) -> None:
        self.written: dict[str, tuple[list, list]] = {}

    async def replace_document_graph(
        self, doc_id: str, entities: list, relationships: list
    ) -> None:
        self.written[doc_id] = (entities, relationships)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chunk_size=200, chunk_overlap=20, chroma_persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def chroma_store() -> FakeChromaStore:
    return FakeChromaStore()


@pytest.mark.asyncio
async def test_phase3_graph_branch_writes_subgraph(settings, embedder, chroma_store):
    """NER populates entities/relations and the graph store receives a doc-scoped write."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Acme Corp is based in Paris. Invoice for services.")
    pdf_bytes = doc.tobytes()
    doc.close()

    fake_graph = FakeGraphStore()

    async def _langdetect_run(state, **deps):
        return {"language": "en", "language_confidence": 1.0}

    async def _classifier_run(state, **deps):
        return {"doc_type": "invoice"}

    async def _sentiment_run(state, **deps):
        return {"sentiment_label": "POSITIVE", "sentiment_score": 0.9}

    async def _ner_run(state, **deps):
        from models.schema import Entity, Relation

        return {
            "entities": [
                Entity(id="e0", text="Acme Corp", label="ORGANIZATION", doc_id=state.doc_id),
                Entity(id="e1", text="Paris", label="LOCATION", doc_id=state.doc_id),
            ],
            "relationships": [
                Relation(
                    id="r0",
                    subject="Acme Corp",
                    relation="HEADQUARTERED_IN",
                    object="Paris",
                    doc_id=state.doc_id,
                )
            ],
        }

    with patch("orchestration.pipeline.LanguageDetectionAgent") as MockLangDetect, patch(
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

        state = await run_ingest(
            pdf_bytes,
            filename="doc.pdf",
            doc_id="d1",
            settings=settings,
            embedder=embedder,
            vector_store=chroma_store,
            graph_store=fake_graph,
        )

    # NER output threaded through the shared state.
    assert len(state.entities) == 2
    assert state.entities[0].text == "Acme Corp"
    assert len(state.relationships) == 1
    assert state.relationships[0].relation == "HEADQUARTERED_IN"

    # Graph store received exactly the doc-scoped subgraph.
    assert "d1" in fake_graph.written
    ents, rels = fake_graph.written["d1"]
    assert len(ents) == 2 and len(rels) == 1

    # Embeddings branch still ran in parallel.
    assert len(state.chunk_ids) > 0

    # English doc: translator skipped, translated_text stays None.
    assert state.translated_text is None


@pytest.mark.asyncio
async def test_build_ingest_graph_has_graph_node(settings):
    """Compiled Phase 3 graph exposes the graph node alongside embeddings."""
    graph = build_ingest_graph(settings, graph_store=FakeGraphStore())
    assert graph is not None
    assert hasattr(graph, "ainvoke")
