# GraphRAG Engine

Autonomous Multi-Agent GraphRAG Intelligence Engine — ingest PDFs, build a knowledge
graph + vector index, and answer questions with cited, explainable answers.

> **Status:** Phase 1 (PDF ingestion + ChromaDB + FastAPI + LangGraph skeleton + basic
> vector RAG) and Phase 2 (language detection, conditional translation, classification, sentiment) are implemented. Phases 3–5 are scaffolded and tracked in
> [ROADMAP.md](ROADMAP.md).

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

## API (Phase 1)
| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents/upload` | Ingest a PDF → returns `doc_id` + chunk ids |
| GET | `/documents` | List ingested documents |
| GET | `/documents/{doc_id}` | Document metadata/status |
| GET | `/documents/{doc_id}/graph` | Document entity-relation subgraph (Neo4j) |
| POST | `/qa` | Answer a question (vector retrieval + citations) |
| GET | `/health` | Liveness probe |

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
