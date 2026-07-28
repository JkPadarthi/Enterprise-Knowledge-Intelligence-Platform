"""Markdown / plain-text reader (standard library only)."""

from __future__ import annotations

from agents.readers.base import BaseReader, ReaderResult


class MarkdownReader(BaseReader):
    """Reads Markdown and plain-text files as-is (no markup stripping required)."""

    extensions = (".md", ".markdown", ".mdx", ".txt")

    def read(self, data: bytes, filename: str = "") -> ReaderResult:
        text = self._decode(data)
        return ReaderResult(raw_text=text, pages=[text], num_pages=1, format="markdown")
