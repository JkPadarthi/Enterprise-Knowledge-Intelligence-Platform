# GraphRAG Engine — Complete Guide

An autonomous multi-agent pipeline that ingests PDFs, builds **two parallel indexes** (a
vector store and a knowledge graph), and answers questions with **cited, traceable
answers** drawn from both. This guide covers the whole system end to end: how to run
it, what happens to a document from upload to Q/A, and every component involved.

---

## 1. What It Does

```
PDF ──► read ──► detect language ──► [translate if needed] ──► classify ──► sentiment
                                                                    │
                                                          extract entities (NER)
                                                                    │
                                              ┌─────────────────────┴─────────────────────┐
                                              ▼                                           ▼
                                    embed chunks ──► ChromaDB              build graph ──► Neo4j
                                              └─────────────────────┬─────────────────────┘
                                                                    ▼
                                          Question ──► search BOTH ──► LLM ──► answer + citations
```

Every answer cites its origin: `source: "vector"` (a text chunk) or `source: "graph"`
(an entity/relation fact), so you can always trace why the system said what it said.

---

## 2. Technology Stack — What Is Used and Why

| Layer | Component | Role |
|---|---|---|
| API | **FastAPI** + Uvicorn | HTTP endpoints, dependency injection, request validation |
| Orchestration | **LangGraph** (`StateGraph`) | Compiles the agent pipeline into a graph with a conditional branch (translate?) and a parallel fan-out (embeddings + graph) |
| PDF reading | **PyMuPDF** (`fitz`) | Extracts raw text page by page |
| Language detection | **langdetect** | Decides the document language; non-English docs get translated |
| Translation | **LLM (worker role)** | Conditional — only runs when `language != translate_target_lang` |
| Classification | **facebook/bart-large-mnli** (zero-shot, Hugging Face) | Assigns a `doc_type` from a configurable label set (invoice, contract, report, …) without any training |
| Sentiment | **distilbert-base-uncased-finetuned-sst-2-english** | POSITIVE/NEGATIVE label + confidence score |
| NER | **GLiNER** (`urchade/gliner_multi-v2.1`) | Extracts typed entities (PERSON, ORG, LOCATION, DATE, MONEY, …) and subject–relation–object triples |
| Embeddings | **sentence-transformers/all-MiniLM-L6-v2** | Local, free vector embeddings for chunks and queries |
| Vector store | **ChromaDB** (server mode, `HttpClient`) | Stores chunk text + embeddings; cosine-similarity retrieval filtered by `doc_id` |
| Graph store | **Neo4j 5.20** (async driver) | Stores `(:Entity)` nodes and `[:RELATION]` edges per document |
| LLM | **OpenRouter** (provider-agnostic client; also supports `ollama` and `mock`) | Abstractive answers, translations, summaries. Two model slots: `WORKER_MODEL` (fast) and `QA_MODEL` (strong) |
| Extractive summary | Custom TextRank-like scorer | Sentence embeddings → cosine-similarity matrix → top-3 centrality sentences, stdlib only |
| Dashboard | **Streamlit** + `streamlit-agraph` | Upload UI, document library, QA chat, interactive graph viewer, execution timeline |
| Config | **pydantic-settings** + `.env` | Every knob in one typed `Settings` object |
| Tests | **pytest** + fakes (no network/keys/models) | 60 passing, 2 skipped |

---

## 3. How to Run It

### 3.1 Infrastructure

Two backing services run in Docker:

```bash
docker start graphrag-neo4j graphrag-chroma
```

- **Neo4j** — `graphrag-neo4j` container, bolt on `localhost:7687` (managed by `docker-compose.yml`, data in `~/docker/neo4j/data`).
- **ChromaDB** — `graphrag-chroma` container on port **8001** (`chromadb/chroma` image).

> **Why ChromaDB runs as a server here:** this machine is aarch64, and the embedded
> ChromaDB Rust bindings crash with a bus error. `ChromaStore` already supports
> `HttpClient`, so the app talks to the container instead. `CHROMA_HOST` is therefore
> **mandatory** — without it the first upload kills the worker.

### 3.2 Start the API

```bash
cd "Enterprise Knowledge Intelligence Platform"
source venv/bin/activate
CHROMA_HOST=localhost CHROMA_PORT=8001 uvicorn api.main:app --port 8010
```

> Port **8010**, not 8000 — port 8000 on this machine belongs to an unrelated
> docker service (Honcho). Interactive docs: `http://localhost:8010/docs`.

### 3.3 Optional dashboard

```bash
source venv/bin/activate
API_BASE_URL=http://localhost:8010 streamlit run frontend/app.py
```

### 3.4 Optional auth

Set `API_KEY=some-secret` in `.env`. Every endpoint except `/health` then requires
`Authorization: Bearer some-secret` (enforced by an HTTP middleware in `api/main.py`).
Empty (default) means open access.

---

## 4. From Upload to Q/A — Step by Step

### Step 0 — Upload

```bash
curl -F "file=@sample.pdf" http://localhost:8010/documents/upload
```

`api/routes/ingest.py` rejects non-PDFs (400), saves the file to
`data/uploads/<doc_id>.pdf`, registers a `DocumentMeta` (status `processing`) in the
in-memory `DOCUMENT_REGISTRY`, and hands the bytes to the LangGraph pipeline.

### The ingestion pipeline (`orchestration/pipeline.py`)

Each stage is an **agent** (`agents/base.py` gives retry — 3 attempts — and logging).
Every stage is timed; the steps land in `state.execution_log` and are returned in the
upload response and via `/documents/{id}/timeline`.

| # | Agent | What it does | Real timing (first run, models warm) |
|---|---|---|---|
| 1 | **ReaderAgent** | PyMuPDF extracts text + page count | ~26 ms |
| 2 | **LanguageDetectionAgent** | `langdetect` on the text sample | ~3 ms |
| 3 | **TranslationAgent** | *conditional*: skipped for English docs; otherwise the worker LLM translates to `translate_target_lang` | skipped |
| 4 | **ClassificationAgent** | bart-large-mnli zero-shot over `doc_type_labels` | ~17 s |
| 5 | **SentimentAgent** | DistilBERT SST-2 label + score (token-aware truncation to the model's 512-token limit) | ~2.7 s |
| 6 | **NERAgent** | GLiNER extracts entities + relations with `doc_id` attached | ~29 s |
| 7 | **EmbeddingAgent** | `chunk_text` (size 1000 / overlap 200) → MiniLM embeddings → `ChromaStore.add_chunks` | ~7 s |
| 8 | **KnowledgeGraphAgent** | `replace_document_graph(doc_id, entities, relations)` → Neo4j MERGE of `:Entity` nodes and `:RELATION` edges | ~3 s |

Steps 7 and 8 run as a **parallel branch** off NER — vector index and graph index are
written independently, so a graph failure can never block vector search.

Verified upload response (real `sample.pdf` run):

```json
{
  "status": "indexed", "doc_type": "invoice",
  "sentiment_label": "POSITIVE", "sentiment_score": 0.87,
  "num_entities": 4, "graph_written": true,
  "chunk_ids": ["<doc_id>::c0"]
}
```

### What lives where after ingestion

- **ChromaDB** `documents` collection: chunk id, raw text, embedding, metadata `{doc_id, index}`.
- **Neo4j**: `(:Entity {id, text, label, doc_id})` nodes, `-[r:RELATION {relation, doc_id}]->` edges — e.g. `(Acme Corporation)-[:headquartered in]->(Paris)`.
- **Registry (RAM)**: `DocumentMeta` with status, doc_type, sentiment, counts, execution log. *Volatile — resets on API restart.*

### Step 5 — Ask a question

```bash
curl -X POST http://localhost:8010/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is Acme headquartered?", "doc_id": "<doc_id>", "top_k": 3}'
```

`agents/qa.py::QAOrchestrator.answer` does **hybrid retrieval**:

1. Embeds the question with the same MiniLM model.
2. **Vector leg:** ChromaDB cosine search, top-k, filtered by `doc_id` → chunks become `[chunk_id]` context + `source: "vector"` citations (score = 1 − distance).
3. **Graph leg:** one Cypher query over the doc's `-[r]->` edges → `subject relation object` fact lines + `source: "graph"` citations.
4. Concatenates both contexts into a grounded prompt ("use ONLY the retrieved context, cite with [chunk_id]") and calls the `qa`-role LLM.

Verified answer:

```json
{
  "answer": "Acme Corporation is headquartered in Paris [<doc_id>::c0].",
  "citations": [
    {"source": "vector", "chunk_id": "<doc_id>::c0", "score": 0.788, "text_excerpt": "Acme Corporation is headquartered in Paris…"},
    {"source": "graph", "node_ref": "Acme Corporation", "text_excerpt": "Acme Corporation headquartered in Paris"}
  ],
  "model": "google/gemma-4-31b-it"
}
```

### Step 6 — Inspect the by-products

```bash
curl http://localhost:8010/documents                      # library (registry)
curl http://localhost:8010/documents/<id>/graph           # entities + relations from Neo4j
curl http://localhost:8010/documents/<id>/summary         # extractive + abstractive
curl http://localhost:8010/documents/<id>/timeline        # per-agent durations/status
```

- **Summary**: extractive = sentence-embedding centrality (top 3 sentences, original
  order); abstractive = one LLM paragraph. Chunks are pulled via
  `ChromaStore.get_by_doc` — ChromaDB's `get()` returns a **flat** `documents` list
  (the `query()` shape is nested; the route normalizes both).
- **Timeline**: every agent's `ExecutionStep` (order, started/ended, duration_ms,
  status, detail), renumbered by start time after the parallel branch.

---

## 5. API Reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents/upload` | Ingest a PDF through the full pipeline |
| GET | `/documents` | List registered documents |
| GET | `/documents/{id}` | One document's metadata/status |
| GET | `/documents/{id}/graph` | Entity-relation subgraph from Neo4j |
| GET | `/documents/{id}/timeline` | Per-agent execution log |
| GET | `/documents/{id}/summary` | Extractive + abstractive summary |
| POST | `/qa` | Hybrid vector+graph answer with citations |
| GET | `/health` | Liveness (always open, even with `API_KEY` set) |

Error contract: non-PDF upload → 400; ingestion exception → 500 with the failed agent
in `detail` (and `status: "failed"` in the registry); unknown doc on `/timeline` → 404;
unknown doc on `/graph` or `/summary` → 200 with empty payload (by design).

---

## 6. Configuration (`.env` → `config/settings.py`)

```ini
LLM_PROVIDER=openrouter            # openrouter | ollama | mock
OPENROUTER_API_KEY=...
WORKER_MODEL=google/gemma-4-26b-a4b-it   # fast slot: translation, classification aids
QA_MODEL=google/gemma-4-31b-it           # strong slot: answers, summaries

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CLASSIFIER_MODEL=facebook/bart-large-mnli
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
GLINER_MODEL=urchade/gliner_multi-v2.1   # matches the local HF cache on this machine

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

CHROMA_HOST=localhost   # REQUIRED on this machine (see §3.1)
CHROMA_PORT=8001

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TRANSLATE_TARGET_LANG=en
DOC_TYPE_LABELS=report,invoice,contract,email,article,legal,technical,other
NER_LABELS=PERSON,ORGANIZATION,LOCATION,DATE,MONEY,PRODUCT,EVENT
UPLOAD_DIR=./data/uploads
API_KEY=                             # empty = open
```

`get_settings()` is `@lru_cache`d — one `Settings` object per process.

---

## 7. Testing

```bash
source venv/bin/activate
pytest            # 60 passed, 2 skipped
ruff check .
```

Tests run **fully offline**: `tests/conftest.py` provides a `FakeEmbedder`
(deterministic 8-dim vectors), an in-memory `FakeChromaStore` (dict + cosine scoring)
and `FakeNeo4jStore`, and a `MockBackend` LLM; API tests use FastAPI
`dependency_overrides`. No API keys, model downloads, ChromaDB, or Neo4j needed.

---

## 8. Maintenance

**Clear all data (what was done on 2026-07-26):**

```bash
# ChromaDB — delete every collection
python -c "import chromadb; c=chromadb.HttpClient(host='localhost',port=8001); \
[c.delete_collection(x.name) for x in c.list_collections()]"

# Neo4j — wipe the graph
docker exec graphrag-neo4j cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"

# Local artifacts + in-memory registry
rm -f data/uploads/*.pdf data/document_registry.json && rm -rf data/chroma
# then restart the API (the document registry lives in process memory)
```

**Project layout:** `agents/` (pipeline agents), `orchestration/` (LangGraph graph),
`api/` (FastAPI routes + deps), `vector/` (ChromaStore), `graph/` (Neo4jStore),
`llm/` (openrouter/ollama/mock backends), `models/` (pydantic schemas + AgentState),
`config/` (settings + logging), `frontend/` (Streamlit), `tests/`.
