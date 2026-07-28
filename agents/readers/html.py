"""HTML reader (standard library only).

Strips markup and returns the visible text: script/style/head content is dropped,
block-level elements introduce line breaks, and HTML entities are decoded.
"""

from __future__ import annotations

from html.parser import HTMLParser

from agents.readers.base import BaseReader, ReaderResult

# Tags whose content is dropped entirely.
_SKIP_TAGS = {"script", "style", "head", "noscript", "template", "meta", "link"}
# Block-level tags that should introduce a line break in the extracted text.
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "div", "dl", "dd", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "tr", "ul",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # Self-closing equivalents (e.g. <br/>, <hr/>): treat as a block break.
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


class HTMLReader(BaseReader):
    """Extracts visible text from an HTML document."""

    extensions = (".html", ".htm")

    def read(self, data: bytes, filename: str = "") -> ReaderResult:
        parser = _TextExtractor()
        parser.feed(self._decode(data))
        text = parser.get_text()
        return ReaderResult(raw_text=text, pages=[text], num_pages=1, format="html")
