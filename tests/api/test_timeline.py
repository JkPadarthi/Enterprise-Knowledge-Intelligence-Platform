from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app
from models.schema import AgentState, ExecutionStep


def _minimal_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"%\xe2\xe3\xcf\xd3\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Length 44 >>\n"
        b"stream\n"
        b"BT\n"
        b"70 720 Td\n"
        b"(Hello World) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 5\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000053 00000 n \n"
        b"0000000102 00000 n \n"
        b"0000000118 00000 n \n"
        b"trailer\n"
        b"<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n"
        b"195\n"
        b"%%EOF"
    )


def _state_with_log() -> AgentState:
    state = AgentState(
        doc_id="tl",
        filename="tl.pdf",
        language="en",
        num_pages=1,
        chunk_ids=["tl::c0"],
    )
    state.execution_log = [
        ExecutionStep(order=1, agent="reader", started_at="2024-01-01T00:00:00+00:00",
                      ended_at="2024-01-01T00:00:01+00:00", duration_ms=10.0, status="success"),
        ExecutionStep(order=2, agent="language_detect", started_at="2024-01-01T00:00:01+00:00",
                      ended_at="2024-01-01T00:00:02+00:00", duration_ms=20.0, status="success"),
    ]
    return state


def test_upload_echoes_execution_log():
    """Phase 5: upload response includes the execution_log from run_ingest."""
    with patch("api.routes.ingest.run_ingest", return_value=_state_with_log()):
        client = TestClient(app)
        resp = client.post(
            "/documents/upload",
            files={"file": ("tl.pdf", _minimal_pdf(), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "execution_log" in data
        assert len(data["execution_log"]) == 2
        assert data["execution_log"][0]["agent"] == "reader"


def test_timeline_endpoint_returns_log():
    """Phase 5: GET /documents/{doc_id}/timeline returns the persisted execution_log."""
    with patch("api.routes.ingest.run_ingest", return_value=_state_with_log()):
        client = TestClient(app)
        doc_id = client.post(
            "/documents/upload",
            files={"file": ("tl.pdf", _minimal_pdf(), "application/pdf")},
        ).json()["doc_id"]

    resp = client.get(f"/documents/{doc_id}/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == doc_id
    assert len(body["execution_log"]) == 2
    assert [e["agent"] for e in body["execution_log"]] == ["reader", "language_detect"]


def test_timeline_unknown_doc_404():
    """Phase 5: timeline for an unknown doc_id returns 404."""
    client = TestClient(app)
    resp = client.get("/documents/does-not-exist/timeline")
    assert resp.status_code == 404
