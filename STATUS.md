# Status

> Updated as work progresses. Last update: **Phase 2 complete**.

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

## In Progress
- (none)

## Verified (2026-07-15)
- `pip install -e ".[dev]"` succeeds (venv at `.venv/`); all heavy deps resolved.
- `pytest` passes — 24 tests (reader, chunker, embeddings, QA, pipeline, language detection, translation, classification, sentiment, API metadata).
- End-to-end smoke test passes: `POST /documents/upload` indexes a PDF into ChromaDB
  (SentenceTransformer model loads), and `POST /qa` returns an answer with a `vector`
  citation (chunk id + score + node ref) using the mock LLM provider.

## Blockers / Risks
- Phase 3+ agents (NER, graph build, summarizer) remain stubbed pending their phases.
- Neo4j writes and hybrid graph retrieval require Phase 3 before the QA path uses the graph.
- Real QA answers require `OPENROUTER_API_KEY` (or `LLM_PROVIDER=ollama` with a local model);
  `LLM_PROVIDER=mock` returns canned text for dry-runs/offline tests.

## Next Steps
1. Phase 3: NER + Relation extraction → Neo4j knowledge-graph build.
2. Phase 4: full LangGraph wiring + hybrid GraphRAG QA + summarization.
3. Phase 5: Streamlit dashboard, agent timeline, graph viewer.
