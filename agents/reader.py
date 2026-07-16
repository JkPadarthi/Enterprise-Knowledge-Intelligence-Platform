"""Reader agent — extracts text from an uploaded PDF (Phase 1)."""

from __future__ import annotations

import logging
from typing import Any

import fitz  # PyMuPDF

from agents.base import BaseAgent
from models.schema import AgentState


class ReaderAgent(BaseAgent):
    """Extracts per-page text from a PDF's raw bytes into the shared state."""

    name = "reader"

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        if not state.file_bytes:
            raise ValueError("ReaderAgent requires state.file_bytes (PDF content)")

        try:
            doc = fitz.open(stream=state.file_bytes, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {exc}") from exc

        try:
            pages = [page.get_text() for page in doc]
        finally:
            doc.close()

        raw_text = "\n".join(pages)
        self._log("extracted %d pages (%d chars)", logging.INFO, len(pages), len(raw_text))
        return {"raw_text": raw_text, "pages": pages, "num_pages": len(pages)}
