"""PDF reader backed by PyMuPDF (extracted from the original Phase 1 reader)."""

from __future__ import annotations

import fitz

from agents.readers.base import BaseReader, ReaderResult


class PDFReader(BaseReader):
    """Extracts per-page text from a PDF's raw bytes."""

    extensions = (".pdf",)

    def read(self, data: bytes, filename: str = "") -> ReaderResult:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to open PDF: {exc}") from exc

        try:
            pages = [page.get_text() for page in doc]
        finally:
            doc.close()

        raw_text = "\n".join(pages)
        return ReaderResult(raw_text=raw_text, pages=pages, num_pages=len(pages), format="pdf")
