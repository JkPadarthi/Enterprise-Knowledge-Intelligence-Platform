"""Domain models and the shared pipeline state.

``AgentState`` is the single object threaded through every LangGraph node. Agents
read what they need and return a *partial* dictionary of fields they update; LangGraph
merges those updates back into the state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Entity(BaseModel):
    """A named entity extracted from a document."""

    id: str
    text: str
    label: str
    doc_id: str = ""
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A typed relationship between two entities (subject → relation → object)."""

    id: str
    subject: str
    relation: str
    object: str
    doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A contiguous text segment stored in the vector database."""

    id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMeta(BaseModel):
    """Lightweight metadata record for an ingested document."""

    id: str
    filename: str
    uploaded_at: datetime = Field(default_factory=_now)
    status: str = "pending"
    language: Optional[str] = None
    doc_type: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    num_pages: Optional[int] = None
    error: Optional[str] = None


class Citation(BaseModel):
    """A provenance pointer attached to a generated answer."""

    chunk_id: str
    source: str = "vector"  # "vector" | "graph"
    text_excerpt: str = ""
    score: Optional[float] = None
    node_ref: Optional[str] = None


class QARequest(BaseModel):
    """Incoming question to the QA orchestrator."""

    question: str
    doc_id: Optional[str] = None
    top_k: int = 5


class QAResponse(BaseModel):
    """Answer plus the citations that support it."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str = ""


class AgentState(BaseModel):
    """The shared, mutable state passed between pipeline agents.

    Every field has a safe default so a node can be unit-tested by constructing a
    minimal state and asserting only the fields it touches.
    """

    # Identity
    doc_id: str = ""
    filename: str = ""
    file_bytes: Optional[bytes] = None

    # Ingestion
    raw_text: str = ""
    pages: list[str] = Field(default_factory=list)
    num_pages: int = 0

    # Language / translation
    language: str = "en"
    language_confidence: float = 0.0
    translated_text: Optional[str] = None

    # Classification / sentiment
    doc_type: str = ""
    doc_type_scores: dict[str, float] = Field(default_factory=dict)
    sentiment_label: str = ""
    sentiment_score: float = 0.0

    # NER / graph
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relation] = Field(default_factory=list)
    graph_written: bool = False

    # Embeddings / vector
    chunk_ids: list[str] = Field(default_factory=list)

    # Summary
    summary: dict[str, str] = Field(default_factory=dict)

    # QA
    qa_question: str = ""
    qa_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)

    # Observability
    errors: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


__all__ = [
    "AgentState",
    "Citation",
    "Chunk",
    "DocumentMeta",
    "Entity",
    "QARequest",
    "QAResponse",
    "Relation",
]
