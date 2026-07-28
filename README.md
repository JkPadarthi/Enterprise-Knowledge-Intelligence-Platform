# Multi-Agent GraphRAG Enterprise Intelligence Engine

Ingest PDFs, build a knowledge graph + vector index, and answer questions with cited,
explainable answers.

> **Status:** All five phases are implemented — PDF ingestion, ChromaDB vector RAG, language
> detection / conditional translation / classification / sentiment, Neo4j knowledge graph
> (write + retrieval), hybrid GraphRAG QA (vector + graph) with a `SummaryAgent`, and a
> Streamlit dashboard (upload, document library, summary, QA chat, interactive graph viewer,
> execution timeline). See [ROADMAP.md](ROADMAP.md) / [STATUS.md](STATUS.md).

## Quickstart

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (or switch LLM_PROVIDER=mock for a dry run)
```

### 3. Start infrastructure (Neo4j for graph store)
```bash
docker compose up -d neo4j
```
ChromaDB runs embedded (no container needed) by default. Neo4j data persists to
`~/docker/neo4j/data` (bind-mount, see `docker-compose.yml`).

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Try it
```bash
# Upload a PDF
curl -F "file=@sample.pdf" http://localhost:8000/documents/upload

# Ask a question (vector RAG over indexed chunks)
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 3}'
```

Interactive docs: http://localhost:8000/docs

## API
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents/upload` | Ingest a PDF → returns `doc_id` + chunk ids + execution log |
| GET | `/documents` | List ingested documents |
| GET | `/documents/{doc_id}` | Document metadata/status |
| GET | `/documents/{doc_id}/graph` | Document entity-relation subgraph (Neo4j) |
| GET | `/documents/{doc_id}/timeline` | Per-agent execution timeline (Phase 5) |
| POST | `/qa` | Answer a question (hybrid vector + graph retrieval, dual-source citations) |
| GET | `/documents/{doc_id}/summary` | Extractive + abstractive summary (Phase 4) |
| GET | `/health` | Liveness probe |

Citations returned by `/qa` carry `source: "vector"` (chunk-based) or `source: "graph"`
(entity/relation-based), so every answer is traceable to its origin.

## Dashboard (Phase 5)

A Streamlit dashboard wraps the API:

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run frontend/app.py
# optional: API_BASE_URL=http://other-host:8000 streamlit run frontend/app.py
```

Pages: Upload, Document Library, Summary, QA Chat (with citations), Knowledge Graph
(interactive streamlit-agraph viewer), and Execution Timeline.

## Project Layout
```
agents/      # pipeline agents (reader, embeddings, qa, + Phase 2–4 stubs)
llm/         # provider-agnostic LLM client (openrouter / ollama / mock)
graph/       # Neo4j store (Phase 3)
vector/      # ChromaDB store
models/      # pydantic domain models + shared AgentState
config/      # settings + logging
orchestration/ # LangGraph pipeline
api/         # FastAPI service
tests/       # pytest suite (mock LLM, ephemeral Chroma, fake embedder)
```

## Tests
```bash
pytest
```
Agents are tested with injected mocks — no API keys, model downloads, or running
services required.

## Documentation
- [PROJECT.md](PROJECT.md) — vision, architecture, scope
- [ROADMAP.md](ROADMAP.md) — phased delivery plan
- [STATUS.md](STATUS.md) — current implementation status
- [DECISIONS.md](DECISIONS.md) — key architectural decisions and reasoning
