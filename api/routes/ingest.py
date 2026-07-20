"""Document ingestion endpoint (Phase 1)."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from api.deps import DOCUMENT_REGISTRY, get_settings_dep, get_vector_store
from config.settings import Settings
from models.schema import DocumentMeta
from orchestration.pipeline import run_ingest
from vector.chroma_store import ChromaStore

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings_dep),
    store: ChromaStore = Depends(get_vector_store),
) -> dict:
        """Accept a PDF, run the ingestion graph, and return indexing metadata."""
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported in Phase 1.")

        data = await file.read()
        doc_id = uuid.uuid4().hex

        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"{doc_id}.pdf").write_bytes(data)

        meta = DocumentMeta(id=doc_id, filename=file.filename or "upload.pdf", status="processing")
        DOCUMENT_REGISTRY[doc_id] = meta

        try:
            state = await run_ingest(
                data, file.filename or "upload.pdf", doc_id, settings, vector_store=store
            )
        except Exception as exc:  # noqa: BLE001
            meta.status = "failed"
            meta.error = str(exc)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

        meta.status = "indexed"
        meta.num_pages = state.num_pages
        meta.language = state.language
        meta.doc_type = state.doc_type or None
        meta.sentiment_label = state.sentiment_label or None
        meta.sentiment_score = state.sentiment_score
        meta.num_entities = len(state.entities) if state.entities else None
        meta.graph_written = state.graph_written
        meta.execution_log = [step.model_dump() for step in (state.execution_log or [])]
        return {
            "doc_id": doc_id,
            "filename": meta.filename,
            "status": meta.status,
            "num_pages": meta.num_pages,
            "chunk_ids": state.chunk_ids,
            "doc_type": meta.doc_type,
            "sentiment_label": meta.sentiment_label,
            "sentiment_score": meta.sentiment_score,
            "num_entities": meta.num_entities,
            "graph_written": meta.graph_written,
            "execution_log": meta.execution_log,
        }
