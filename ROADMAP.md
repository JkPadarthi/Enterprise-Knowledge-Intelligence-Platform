# Roadmap

Phased delivery of the Autonomous Multi-Agent GraphRAG Intelligence Engine. Each phase
builds on the previous one and keeps every agent independently testable.

## Phase 1 — Ingestion & Basic RAG ✅ (implemented)
- PDF ingestion (`ReaderAgent`, PyMuPDF)
- Text chunking + local Sentence-Transformers embeddings → ChromaDB (`EmbeddingAgent`)
- FastAPI service: `POST /documents/upload`, `GET /documents`, `POST /qa`
- LangGraph ingestion skeleton (`reader → embeddings`)
- Basic vector RAG QA with citations
- Test harness (pytest, mock LLM, ephemeral Chroma, fake embedder)

## Phase 2 — Language, Translation, Classification, Sentiment ✅ (implemented)
- `LanguageDetectionAgent` (langdetect)
- `TranslationAgent` (conditional: translate only if non-English) via `LLMClient`
- `ClassificationAgent` (zero-shot transformer)
- `SentimentAgent` (DistilBERT)
- Insert into the LangGraph pipeline after `reader`

## Phase 3 — Knowledge Graph
- `NERAgent` (GLiNER entities + LLM relation extraction)
- `KnowledgeGraphAgent` (Neo4j, idempotent MERGE)
- Graph retrieval endpoint for visualization
- Insert into the pipeline after NER; write to Neo4j in parallel with embeddings

## Phase 4 — Orchestration & Hybrid GraphRAG
- Full LangGraph `StateGraph` wiring all agents (with conditional translate edge)
- `QAOrchestrator` hybrid retrieval: vector (ChromaDB) + graph (Neo4j) → cited answer
- Explainable answers with chunk-id + node-id citations
- `SummaryAgent` (extractive + abstractive)

## Phase 5 — Dashboard & Experience
- Streamlit dashboard: upload, document list, summary view, QA chat
- Agent execution timeline
- Knowledge-graph viewer (pyvis / streamlit-agraph)
- Cross-document search

## Future Enhancements
- Additional document types (DOCX, PPTX, HTML, Markdown) via new `ReaderAgent` backends
- New agents (e.g., table extraction, claim verification) without touching the core
- Production scaling: background task queue (Celery/ARQ), auth, multi-tenancy
- Managed Neo4j / ChromaDB, Kubernetes deployment manifests
