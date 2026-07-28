"""Application settings, loaded from environment variables / .env file.

Every configurable value used across the engine lives here so the rest of the
codebase reads configuration through one typed object (`get_settings()`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM provider (provider-agnostic; OpenRouter is the default backend) ──
    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    worker_model: str = "google/gemini-flash-1.5"
    qa_model: str = "google/gemini-pro-1.5"
    ollama_base_url: str = "http://localhost:11434"

    # ── Embeddings (local Sentence-Transformers) ─────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Knowledge graph (Neo4j) ───────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector store (ChromaDB) ───────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str | None = None
    chroma_port: int = 8000

    # ── Text chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Translation (conditional: only when source != target) ─────────────────
    translate_target_lang: str = "en"

    # ── NLP models (pretrained / zero-shot) ───────────────────────────────────
    classifier_model: str = "facebook/bart-large-mnli"
    doc_type_labels: str = "report,invoice,contract,email,article,legal,technical,other"
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    gliner_model: str = "urchade/gliner_large-v1.1"
    ner_labels: str = "PERSON,ORGANIZATION,LOCATION,DATE,MONEY,PRODUCT,EVENT"

    # ── Runtime ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    upload_dir: str = "./data/uploads"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── API auth (empty = open access) ────────────────────────────────────────
    api_key: str = ""

    @property
    def doc_type_label_list(self) -> list[str]:
        """Parsed document-type label set for the zero-shot classifier."""
        return [label.strip() for label in self.doc_type_labels.split(",") if label.strip()]

    @property
    def ner_label_list(self) -> list[str]:
        """Parsed entity label set for GLiNER."""
        return [label.strip() for label in self.ner_labels.split(",") if label.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
