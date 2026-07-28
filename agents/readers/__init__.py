"""Pluggable document readers (Future Enhancement: multi-format ingestion).

Each backend implements :class:`BaseReader` and declares the file extensions it
handles. :func:`get_reader` selects a backend by file extension and falls back to
the PDF reader for unknown / extension-less uploads (backward compatible with the
original Phase 1 PDF-only behavior).

Backends:
- ``PDFReader``       — PyMuPDF (required dependency)
- ``MarkdownReader``  — plain text / Markdown (standard library only)
- ``HTMLReader``      — HTML tag-stripping (standard library only)
- ``DocxReader``      — Microsoft Word ``.docx`` (optional ``python-docx``)
- ``PptxReader``      — PowerPoint ``.pptx`` (optional ``python-pptx``)

DOCX/PPTX are guarded behind optional imports: if the package is not installed
the reader raises a clear ``RuntimeError`` instead of a confusing import error.
"""

from __future__ import annotations

from agents.readers.base import BaseReader, ReaderResult
from agents.readers.docx import DocxReader
from agents.readers.html import HTMLReader
from agents.readers.markdown import MarkdownReader
from agents.readers.pdf import PDFReader
from agents.readers.pptx import PptxReader
from agents.readers.registry import ReaderRegistry, get_reader, register_reader, supported_extensions

__all__ = [
    "BaseReader",
    "DocxReader",
    "HTMLReader",
    "MarkdownReader",
    "PDFReader",
    "PptxReader",
    "ReaderRegistry",
    "ReaderResult",
    "get_reader",
    "register_reader",
    "supported_extensions",
]
