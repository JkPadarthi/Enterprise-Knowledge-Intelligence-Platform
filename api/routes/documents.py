"""Document listing / inspection endpoints (Phase 1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import DOCUMENT_REGISTRY, get_settings_dep
from config.settings import Settings
from models.schema import DocumentMeta

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(settings: Settings = Depends(get_settings_dep)) -> list[DocumentMeta]:
    """Return all ingested documents and their status."""
    return list(DOCUMENT_REGISTRY.values())


@router.get("/{doc_id}")
async def get_document(
    doc_id: str, settings: Settings = Depends(get_settings_dep)
) -> DocumentMeta:
    """Return metadata for a single document."""
    if doc_id not in DOCUMENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Document not found")
    return DOCUMENT_REGISTRY[doc_id]
