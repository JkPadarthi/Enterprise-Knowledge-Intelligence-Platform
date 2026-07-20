# Status

> Updated as work progresses. Last update: **Phase 5 complete (Streamlit dashboard + execution timeline + graph viewer)**.

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
- [x] **Phase 4 — `QAOrchestrator` hybrid retrieval**: vector top-k (ChromaDB) **and** Neo4j graph context merged into one LLM prompt; citations tagged `source="vector"` / `source="graph"`
- [x] **Phase 4 — `SummaryAgent`** (`agents/summarizer.py`): extractive (lead/centrality over chunks) + abstractive (LLMClient) → `state.summary = {abstractive, extractive}`
- [x] **Phase 4 — API**: `/qa` injects `graph_store` (dual-source citations); new `GET /documents/{doc_id}/summary` returns `state.summary` from the vector store
- [x] **Phase 4 — test suite** (hybrid QA + SummaryAgent, injected fakes)
- [x] **Phase 5 — Execution timeline telemetry**: `run_ingest` records per-agent order/start/end/duration/status into `AgentState.execution_log` (LangGraph reducer channel; parallel `embeddings`/`graph` branch supported)
- [x] **Phase 5 — `GET /documents/{doc_id}/timeline`** endpoint + `DocumentMeta.execution_log` persistence from `ingest`
- [x] **Phase 5 — Streamlit dashboard** (`frontend/app.py` + `frontend/client.py`): upload, document library, summary view, QA chat (dual-source citations), interactive knowledge-graph viewer (streamlit-agraph), execution timeline (table + duration bar chart)
- [x] **Phase 5 — test suite** (`tests/api/test_timeline.py`: upload echo + `/timeline` + 404)

## In Progress
- (none — all 5 phases delivered)

## Verified
- `pip install -e ".[dev]"` succeeds (venv at `.venv/`); all heavy deps resolved (incl. `langgraph`).
- `pytest` passes — 39 tests (reader, chunker, embeddings, QA [vector + hybrid], pipeline [phases 1–3],
  language detection, translation, classification, sentiment, NER, graph builder, summarizer,
  graph-branch pipeline, API metadata/ingest, timeline endpoint).
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
- `langgraph` and `neo4j` are only importable from the project venv — run `pytest` from
  inside `.venv` (e.g. `source .venv/bin/activate && pytest`), not the system Python, or
  collection fails on `orchestration/pipeline.py` / `graph/neo4j_store.py` imports.

## Next Steps
1. (Optional) Richer dashboard: graph filtering/search/layouts, timeline token/retry telemetry.
2. Future enhancements (see ROADMAP): non-PDF ingestion, new agents, production scaling (Celery/ARQ, auth, multi-tenancy), managed Neo4j/ChromaDB, Kubernetes manifests.

## Running the dashboard
```bash
source .venv/bin/activate
# Start the API (in one terminal)
uvicorn api.main:app --reload --port 8000
# Start the dashboard (in another terminal)
streamlit run frontend/app.py
# Point the dashboard at a different API with API_BASE_URL=http://host:port
```
