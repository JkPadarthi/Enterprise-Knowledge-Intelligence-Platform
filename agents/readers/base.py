"""Shared types for document readers.

A reader returns a :class:`ReaderResult`: the concatenated ``raw_text`` (consumed
downstream by the chunker / embedder), a ``pages`` list of text blocks (one per
page for paginated formats, a single element for flat formats), and ``num_pages``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReaderResult:
    """Normalized output of a document reader."""

    raw_text: str
    pages: list[str] = field(default_factory=list)
    num_pages: int = 0
    format: str = "unknown"

    def __post_init__(self) -> None:
        if not self.pages:
            self.pages = [self.raw_text] if self.raw_text else []
        if self.num_pages == 0:
            self.num_pages = len(self.pages)


class BaseReader:
    """Base class for document readers.

    Subclasses set :attr:`extensions` (lowercase, including the dot, e.g.
    ``(".pdf",)``) and implement :meth:`read`.
    """

    #: Lowercase extensions this reader handles.
    extensions: tuple[str, ...] = ()

    def read(self, data: bytes, filename: str = "") -> ReaderResult:
        """Parse ``data`` and return a :class:`ReaderResult`."""
        raise NotImplementedError

    @staticmethod
    def _decode(data: bytes) -> str:
        """Decode bytes to text, tolerating common encodings."""
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
