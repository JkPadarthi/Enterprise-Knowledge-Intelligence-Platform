import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app
from models.schema import AgentState


def _minimal_pdf() -> bytes:
    # Minimal valid PDF 1.4 (ASCII-only bytes to keep the source parseable).
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


def test_upload_doc_phase2_metadata():
    """Phase 2: doc_type, sentiment_label, and sentiment_score appear in upload response."""
    with patch("api.routes.ingest.run_ingest") as mock_run_ingest:
        mock_run_ingest.return_value = AgentState(
            doc_id="test",
            filename="test.pdf",
            language="en",
            doc_type="invoice",
            sentiment_label="POSITIVE",
            sentiment_score=0.9,
            num_pages=1,
            chunk_ids=["test::c0"],
        )

        client = TestClient(app)
        resp = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", _minimal_pdf(), "application/pdf")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_type"] == "invoice"
        assert data["sentiment_label"] == "POSITIVE"
        assert data["sentiment_score"] == 0.9

        from api.deps import DOCUMENT_REGISTRY

        registry_id = data["doc_id"]
        meta = DOCUMENT_REGISTRY.get(registry_id)
        assert meta is not None
        assert meta.doc_type == "invoice"
        assert meta.sentiment_label == "POSITIVE"
        assert meta.sentiment_score == 0.9


def test_upload_doc_phase2_metadata_empty_strings():
    """Empty doc_type/sentiment_label become None in the response; non-PDF still 400."""
    with patch("api.routes.ingest.run_ingest") as mock_run_ingest:
        mock_run_ingest.return_value = AgentState(
            doc_id="test2",
            filename="test2.pdf",
            language="en",
            doc_type="",
            sentiment_label="",
            sentiment_score=0.0,
            num_pages=1,
            chunk_ids=["test2::c0"],
        )

        client = TestClient(app)
        resp = client.post(
            "/documents/upload",
            files={"file": ("test2.pdf", _minimal_pdf(), "application/pdf")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_type"] is None
        assert data["sentiment_label"] is None

    # Non-PDF must still be rejected with 400 (guard preserved).
    client = TestClient(app)
    bad = client.post(
        "/documents/upload",
        files={"file": ("test2.txt", b"not a pdf", "text/plain")},
    )
    assert bad.status_code == 400
