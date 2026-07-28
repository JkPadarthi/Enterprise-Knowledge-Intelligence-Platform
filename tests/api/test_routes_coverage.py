"""Coverage tests for the previously-untested API routes: /qa, /documents/{id}/graph,
/documents/{id}/summary, and the API-key auth gate.

Uses FastAPI's TestClient with dependency overrides (fake vector/graph stores, mock
LLM) so no network, keys, or running services are required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.main import app
from llm.mock import MockBackend
from models.schema import DocumentMeta
from tests.conftest import FakeEmbedder


class FakeVectorStore:
    """In-memory stand-in for ChromaStore used by the QA/summary routes."""

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}

    def add_chunks(self, chunks: list, embeddings: list) -> None:  # noqa: D102
        for chunk in chunks:
            self._docs.setdefault(chunk.doc_id, []).append(chunk.text)

    def query(
        self, query_embedding: list[float], top_k: int = 5, where: dict | None = None
    ) -> dict:
        doc_id = (where or {}).get("doc_id")
        default = ["The quarterly revenue increased by 12%."]
        docs = self._docs.get(doc_id, default) if doc_id else default
        ids = [f"{doc_id}::c0"] if doc_id else ["doc::c0"]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [[{"doc_id": doc_id or "doc"}]],
            "distances": [[0.1]],
        }

    def get_by_doc(self, doc_id: str) -> dict:
        return {"documents": self._docs.get(doc_id, [])}


class FakeGraphStore:
    """In-memory stand-in for Neo4jStore implementing the methods the routes call."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self._rows: list[dict] = rows if rows is not None else [
            {"subject": "Acme", "relation": "REPORTED", "object": "Growth"},
        ]

    async def connect(self) -> None:  # noqa: D102
        pass

    async def close(self) -> None:  # noqa: D102
        pass

    async def query_graph(self, cypher: str, params: dict | None = None) -> list[dict]:
        # The route issues two queries: an entity query (matches :Entity nodes) and a
        # relation query (matches -[r]-> edges). Return relation rows only for the
        # relation query; the entity query returns nothing in this fake.
        if "r]" in cypher or "r " in cypher or "-[r" in cypher:
            return self._rows
        return []


@pytest.fixture
def vector() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, vector: FakeVectorStore) -> TestClient:
    graph = FakeGraphStore()

    app.dependency_overrides.clear()
    app.dependency_overrides[deps.get_vector_store] = lambda: vector
    app.dependency_overrides[deps.get_graph_store_connected] = lambda: graph
    monkeypatch.setattr(
        "api.routes.query.get_llm_client",
        lambda *a, **k: MockBackend(model="mock", text_response="Yes, revenue grew."),
    )
    monkeypatch.setattr(
        "api.routes.summary.get_llm_client",
        lambda *a, **k: MockBackend(model="mock", text_response="Short summary."),
    )
    monkeypatch.setattr(
        "agents.qa.QAOrchestrator._get_embedder",
        lambda self: FakeEmbedder(),
    )
    monkeypatch.setattr(
        "agents.summarizer.SummaryAgent._get_embedder",
        lambda self: FakeEmbedder(),
    )

    # Reset the registry so tests are isolated.
    deps.DOCUMENT_REGISTRY.clear()
    return TestClient(app)


def _seed_doc(client: TestClient, doc_id: str = "doc1") -> None:
    deps.DOCUMENT_REGISTRY[doc_id] = DocumentMeta(id=doc_id, filename="doc.pdf", status="indexed")


def test_qa_returns_citations(client: TestClient):
    _seed_doc(client)
    resp = client.post("/qa", json={"question": "Did revenue grow?", "doc_id": "doc1", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert body["model"]


def test_graph_endpoint_returns_relations(client: TestClient):
    _seed_doc(client)
    resp = client.get("/documents/doc1/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc1"
    assert body["entities"] == []
    assert body["relations"][0]["subject"] == "Acme"
    assert body["relations"][0]["relation"] == "REPORTED"


def test_graph_endpoint_empty_when_no_data(client: TestClient):
    deps.DOCUMENT_REGISTRY["empty"] = DocumentMeta(id="empty", filename="e.pdf", status="indexed")
    app.dependency_overrides[deps.get_graph_store_connected] = lambda: FakeGraphStore(rows=[])
    resp = client.get("/documents/empty/graph")
    assert resp.status_code == 200
    assert resp.json()["graph_written"] is False


def test_summary_endpoint_returns_summary(client: TestClient, vector: FakeVectorStore):
    from models.schema import Chunk

    _seed_doc(client)
    chunk = Chunk(
        id="doc1::c0", doc_id="doc1", text="Revenue grew strongly. Profit followed.", index=0
    )
    vector.add_chunks([chunk], embeddings=[[0.1] * 8])
    resp = client.get("/documents/doc1/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert body["summary"]["abstractive"] == "Short summary."
    assert "Revenue grew strongly." in body["summary"]["extractive"]


def test_summary_endpoint_empty_doc(client: TestClient):
    deps.DOCUMENT_REGISTRY["nodata"] = DocumentMeta(id="nodata", filename="n.pdf", status="indexed")
    app.dependency_overrides[deps.get_vector_store] = lambda: FakeVectorStore()
    resp = client.get("/documents/nodata/summary")
    assert resp.status_code == 200
    assert resp.json()["summary"] == {"abstractive": "", "extractive": ""}


def test_auth_health_open_without_key(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_auth_protected_requires_key(monkeypatch: pytest.MonkeyPatch):
    from config.settings import Settings, get_settings

    secured = Settings(api_key="secret", chroma_persist_dir="/tmp/omo-test-chroma")
    monkeypatch.setattr("api.main.get_settings", lambda: secured)
    get_settings.cache_clear()

    local_client = TestClient(app)
    auth = {"Authorization": "Bearer secret"}

    # No key -> 401
    assert local_client.get("/documents").status_code == 401
    # Wrong key -> 401
    assert (
        local_client.get("/documents", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    # Correct key -> 200
    assert local_client.get("/documents", headers=auth).status_code == 200
    # Health stays open even with auth on
    assert local_client.get("/health").status_code == 200
