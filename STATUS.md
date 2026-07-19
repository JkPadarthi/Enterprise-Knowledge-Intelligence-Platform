# Status

> Updated as work progresses. Last update: **Phase 3 complete (graph write + retrieval)**.

## Completed
- [x] Repository scaffold: `pyproject.toml`, `.env.example`, `.gitignore`, `docker-compose.yml`
- [x] Package structure (`agents/`, `api/`, `config/`, `models/`, `llm/`, `graph/`, `vector/`, `orchestration/`, `frontend/`, `tests/`)
- [x] Central configuration (`config/settings.py`, pydantic-settings) + structured logging
- [x] Domain models + shared `AgentState` (`models/schema.py`)
- [x] Provider-agnostic `LLMClient` (OpenRouter / Ollama / Mock backends + factory)
- [x] ChromaDB vector store wrapper (`vector/chroma_store.py`)
- [x] Neo4j store stub with connection plumbing (`graph/neo4j_store.py`, Phase 3)
- [x] `BaseAgent` (logging + retry)
- [x] **Phase 1 — ReaderAgent** (PyMuPDF PDF text extraction)
- [x] **Phase 1 — EmbeddingAgent** (chunker + Sentence-Transformers + ChromaDB)
- [x] **Phase 1 — QAOrchestrator** (vector retrieval + LLM answer + citations)
- [x] **Phase 1 — LangGraph ingestion skeleton** (`reader → embeddings`)
- [x] **Phase 1 — FastAPI service** (upload, list, QA endpoints + health)
- [x] Phase 1 test suite (reader, embeddings, qa, pipeline, mock LLM)
- [x] Documentation: PROJECT / ROADMAP / STATUS / DECISIONS / README
- [x] **Phase 2 — LanguageDetectionAgent** (langdetect: sets state.language + language_confidence)
- [x] **Phase 2 — TranslationAgent** (conditional: translates only when language != target "en" via LLMClient)
- [x] **Phase 2 — ClassificationAgent** (zero-shot transformers over doc_type_label_list)
- [x] **Phase 2 — SentimentAgent** (DistilBERT sentiment → state.sentiment_label/score)
- [x] **Phase 2 — LangGraph ingestion graph** extended: reader → language_detect → (conditional) translator → classifier → sentiment → embeddings
- [x] **Phase 2 — Ingest API** enriches DocumentMeta + response with doc_type / sentiment_label / sentiment_score
- [x] **Phase 2 — test suite** (language/translator/classifier/sentiment agents + pipeline + API tests)
- [x] **Phase 3 — NERAgent** (GLiNER entities + LLM relation extraction; injectable extractor/LLM)
- [x] **Phase 3 — KnowledgeGraphAgent** (idempotent upsert into injectable graph store)
- [x] **Phase 3 — ingestion graph extended** with `ner` node + parallel `embeddings`/`graph` branch; `run_ingest` injects `graph_store`
- [x] **Phase 3 — Neo4j store writes** (`replace_document_graph` doc-scoped delete-then-MERGE of `:Entity` nodes + typed `:RELATION` edges; `upsert_entities`/`upsert_relationships`/`query_graph` implemented, verified against live Neo4j)
- [x] **Phase 3 — Graph retrieval API** (`GET /documents/{doc_id}/graph` returns doc-scoped entities + relations via `query_graph`)
- [x] **Phase 3 — Ingest API** surfaces `num_entities` / `graph_written` in `DocumentMeta` + response
- [x] **Phase 3 — test suite** (NER agent + KnowledgeGraphAgent + pipeline graph-branch, no model/DB required)
- [x] Translator agent bugfix: uses `self.settings` instead of instantiating `Settings()` twice

## In Progress
- Phase 4: hybrid GraphRAG QA + summarization

## Verified
- `pip install -e ".[dev]"` succeeds (venv at `.venv/`); all heavy deps resolved (incl. `langgraph`).
- `pytest` passes — 32 tests (reader, chunker, embeddings, QA, pipeline, language detection,
  translation, classification, sentiment, NER, graph builder, graph-branch pipeline, API metadata).
- Live Neo4j smoke test passes: `replace_document_graph` writes a doc-scoped subgraph
  (2 `:Entity` nodes + 1 typed `:RELATION` edge) and idempotent re-write deletes the
  prior subgraph cleanly; deleting one doc leaves other docs' nodes untouched.
- `GET /documents/{doc_id}/graph` returns the doc-scoped subgraph from live Neo4j (200,
  empty list + `graph_written:false` when absent).
- End-to-end smoke test passes: `POST /documents/upload` indexes a PDF into ChromaDB
  (SentenceTransformer model loads), and `POST /qa` returns an answer with a `vector`
  citation (chunk id + score + node ref) using the mock LLM provider.

## Blockers / Risks
- Real QA answers require `OPENROUTER_API_KEY` (or `LLM_PROVIDER=ollama` with a local model);
  `LLM_PROVIDER=mock` returns canned text for dry-runs/offline tests.
- Hybrid graph retrieval in the QA path is Phase 4 (the graph store + retrieval endpoint
  already exist; QAOrchestrator does not yet consume the graph).

## Next Steps
1. Phase 4: full LangGraph wiring + hybrid GraphRAG QA (vector + graph) + summarization.
2. Phase 5: Streamlit dashboard, agent timeline, graph viewer.
