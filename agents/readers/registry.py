"""Reader registry and lookup by file extension."""

from __future__ import annotations

from agents.readers.base import BaseReader
from agents.readers.pdf import PDFReader

# Fallback reader for unknown / extension-less uploads (original Phase 1 behavior).
_DEFAULT_READER = PDFReader


def _extension(filename: str) -> str:
    """Return the lowercase extension (including the dot) for ``filename``."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


class ReaderRegistry:
    """Maps lowercase file extensions to reader instances."""

    def __init__(self) -> None:
        self._readers: dict[str, BaseReader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        from agents.readers.docx import DocxReader
        from agents.readers.html import HTMLReader
        from agents.readers.markdown import MarkdownReader
        from agents.readers.pptx import PptxReader

        for reader in (
            PDFReader(),
            MarkdownReader(),
            HTMLReader(),
            DocxReader(),
            PptxReader(),
        ):
            self.register(reader)

    def register(self, reader: BaseReader) -> None:
        """Register a reader for each of its declared extensions."""
        for ext in reader.extensions:
            self._readers[ext.lower()] = reader

    def get(self, extension: str) -> BaseReader:
        """Return the reader for ``extension`` (falling back to PDF)."""
        return self._readers.get(extension.lower(), _DEFAULT_READER())

    def supported_extensions(self) -> list[str]:
        """Return all registered extensions (sorted, excluding the empty key)."""
        return sorted(ext for ext in self._readers)


_REGISTRY = ReaderRegistry()


def get_reader(filename: str = "") -> BaseReader:
    """Return a reader appropriate for ``filename``'s extension."""
    return _REGISTRY.get(_extension(filename))


def register_reader(reader: BaseReader) -> None:
    """Register an additional/custom reader at runtime."""
    _REGISTRY.register(reader)


def supported_extensions() -> list[str]:
    """Return the list of file extensions the pipeline can ingest."""
    return _REGISTRY.supported_extensions()
