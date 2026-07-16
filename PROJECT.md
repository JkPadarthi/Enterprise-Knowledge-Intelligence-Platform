# Project: Autonomous Multi-Agent GraphRAG Intelligence Engine

An enterprise-grade AI platform that ingests documents, builds a knowledge graph,
performs semantic indexing, and answers complex multi-document questions using a
multi-agent architecture. The system is modular, scalable, and designed for future
expansion (additional document types, new agents, production hardening).

## Vision

Turn unstructured documents into a queryable, explainable knowledge layer that
combines a **knowledge graph** (structured entity/relation facts) with a **vector
store** (semantic similarity). Questions are answered by *hybrid retrieval* over both
stores and returned with citations, so every answer is traceable to source chunks.

## Architecture

```
Upload PDF
    │
    ▼
Reader Agent ──► Language Detection ──► Translation (conditional) ──► Classification
    │                                                                      │
    │                                                                      ▼
    │                                                              Sentiment ──► NER & Relation
    │                                                                              │
    └──────────────────────────────────────────────┬───────────────────────────┘
                                                       ▼
                                            Knowledge Graph Agent ──► Neo4j
                                                       │
                                            Embedding Agent ──► ChromaDB
                                                       │
                                                  Summary Agent
                                                       │
                                                  QA Orchestrator (Hybrid GraphRAG)
                                                  ├─ Vector retrieval (ChromaDB)
                                                  ├─ Graph retrieval (Neo4j)
                                                  └─ Cited, explainable answer
```

Orchestration is handled by **LangGraph** (`orchestration/pipeline.py`). Each stage is
an independent, dependency-injected agent (`agents/`) so it can be unit-tested in
isolation.

## Tech Stack

| Concern | Choice |
| --- | --- |
| Language / API | Python 3.12+, FastAPI |
| Orchestration | LangGraph (`StateGraph`) |
| Knowledge graph | Neo4j (async driver v5) |
| Vector store | ChromaDB (embedded persistent / HTTP) |
| PDF parsing | PyMuPDF |
| Language detection | langdetect |
| Embeddings | Sentence-Transformers (local) |
| LLM | Provider-agnostic `LLMClient` — **OpenRouter** default, Ollama offline, Mock for tests |
| Frontend | Streamlit (Phase 5) |

## Scope & Constraints

- **In scope (Phase 1 complete):** PDF ingestion, ChromaDB indexing, FastAPI service,
  LangGraph ingestion skeleton, basic vector RAG QA, full test harness.
- **Planned:** language detection, conditional translation, classification, sentiment,
  NER + relations, Neo4j graph build, hybrid GraphRAG, Streamlit dashboard, agent
  execution timeline, knowledge-graph viewer.
- **Out of scope for now:** non-PDF document types (Word/PowerPoint/HTML/Markdown),
  authentication/multi-tenancy, production scaling (Celery/ARQ), Kubernetes manifests.
- Every agent is independently testable; configuration lives in environment variables;
  logging and retries are built into the agent base class.

See [ROADMAP.md](ROADMAP.md), [STATUS.md](STATUS.md), and [DECISIONS.md](DECISIONS.md)
for progression and the reasoning behind key choices.
