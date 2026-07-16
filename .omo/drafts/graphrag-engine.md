---
slug: graphrag-engine
status: approved
intent: clear
pending-action: write .omo/plans/graphrag-engine.md
approach: Build the full Autonomous Multi-Agent GraphRAG Intelligence Engine from scratch as a modular Python package: a provider-agnostic LLMClient (OpenRouter default, Ollama fallback, Mock for tests), 9 independently-testable agents orchestrated by a LangGraph StateGraph, with Neo4j (graph) + ChromaDB (vector) stores, a FastAPI API, and a Streamlit dashboard. Covers all 5 roadmap phases in one decision-complete plan.
---

# Draft: graphrag-engine

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
| id | outcome (one line) | status | evidence path |
| --- | --- | --- | --- |
| foundation | Packaging, env config, domain models, LLM client abstraction, Neo4j + ChromaDB store wrappers, docker-compose for local infra | active | pyproject.toml, config/, models/, llm/, graph/, vector/, docker-compose.yml |
| agents-local | Reader, LanguageDetect, Classifier, Sentiment agents (local ML, no external LLM) | active | agents/reader.py, language_detect.py, classifier.py, sentiment.py |
| agents-llm | Translator (conditional), NER+Relation, Summarizer, Embeddings agents (LLM / local embeddings) | active | agents/translator.py, ner.py, summarizer.py, embeddings.py |
| graph-build | Knowledge Graph builder agent that writes entities/relations to Neo4j | active | agents/graph_builder.py |
| orchestration | LangGraph StateGraph pipeline (linear + conditional translator + parallel embeddings/graph) and QA orchestrator (hybrid retrieval) | active | orchestration/pipeline.py, agents/qa.py |
| api | FastAPI app: ingest PDF, query, documents/graph/search routes with DI | active | api/main.py, api/deps.py, api/routes/* |
| frontend | Streamlit dashboard: upload, agent timeline, summary, QA, KG viewer, multi-doc search | active | frontend/app.py |
| docs | PROJECT/ROADMAP/STATUS/DECISIONS/README kept updated | active | PROJECT.md, ROADMAP.md, STATUS.md, DECISIONS.md, README.md |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| LLM provider | Provider-agnostic `LLMClient`; default backend = OpenRouter; models via env (default model IDs point at Gemini Flash/Pro *through* OpenRouter, never hardcoded to the Gemini SDK) | User decision: OpenRouter default, never hardcode Gemini, mock for tests, Ollama offline fallback | yes (env/config) |
| Embeddings | Local Sentence Transformers, multilingual model, configurable | User decision; offline + free + privacy | yes |
| Translation | Conditional: detect language, translate ONLY if not English; routed through `LLMClient` (OpenRouter default); backend pluggable for future NLLB/MarianMT | User decision | yes |
| NER | GLiNER (multilingual-capable) for entities; relations via LLM prompt over extracted entities | Spec preferred GLiNER; relations need context | yes |
| Classification | Zero-shot classifier (bart-large-mnli) with configurable label set, no training | Avoids labeled-data requirement for enterprise doc types | yes |
| Sentiment | DistilBERT SST-2 (English); multilingual model configurable via env | Spec named DistilBERT | yes |
| Infra | docker-compose with Neo4j (APOC) + ChromaDB for local dev; clients also accept external URIs | Best practice for reproducible local env | yes |
| QA orchestration | Separate QA graph/function invoked by API; hybrid = vector top-k + graph entity context fed to LLM with citation instruction | Spec: Hybrid GraphRAG with citations | yes |
| Async | Neo4j driver async API + async FastAPI; agents are async node functions | Non-blocking ingestion/query | yes |

## Findings (cited - path:lines)
- Workspace is greenfield: only `.codegraph/` index dir exists; no source, no `pyproject.toml`, not a git repo yet. → Bootstrap from scratch.
- No existing `.omo/plans` or `.omo/drafts` (stale run-continuation artifact ignored).
- User resolved all 3 open forks (see Decisions): Local ST embeddings; OpenRouter-default provider-agnostic LLM; conditional LLM-based translation.
- Stack is stable/well-known: LangGraph `StateGraph` (current API, prebuilt supervisors optional), `neo4j` v5 async driver, `chromadb` persistent client, `pydantic-settings` `BaseSettings`, OpenRouter OpenAI-compatible `/chat/completions`.

## Decisions (with rationale)
1. **Provider-agnostic LLM, OpenRouter default.** `llm/client.py` defines `LLMClient` ABC with `complete(messages, **kw) -> str` and `complete_json(...) -> dict`. Backends: `OpenRouterBackend` (openai lib → `https://openrouter.ai/api/v1`), `OllamaBackend` (http to `OLLAMA_BASE_URL`), `MockBackend` (tests). Default model IDs `google/gemini-flash-1.5` (worker) / `google/gemini-pro-1.5` (QA) selected *via OpenRouter* — preserves spec's model intent without hardcoding the Gemini SDK. Rationale: user explicitly chose OpenRouter + testability.
2. **Local Sentence-Transformers embeddings.** `EMBEDDING_MODEL` env (default multilingual). Offline, free, privacy-preserving. Rationale: user choice.
3. **Conditional translation.** After `language_detect`, translator runs only when `language != TRANSLATE_TARGET_LANG` ("en"); otherwise a no-op that passes `translated_text=None`. LangGraph conditional edge implements the skip. Rationale: user choice; avoids needless LLM cost.
4. **GLiNER + LLM relations.** Entities via GLiNER (labels from `NER_LABELS` env). Relations via LLM prompt over (entities + text) → typed `Relation` triples. Rationale: spec prefers GLiNER; relations need context.
5. **Zero-shot classification, DistilBERT sentiment.** No training data required; labels configurable. Rationale: enterprise doc types vary; spec named DistilBERT.
6. **Two stores, one pipeline.** `graph_builder` → Neo4j (MERGE idempotent). `embeddings` → ChromaDB chunks. Both branch from NER output and run in parallel in the graph. Rationale: spec architecture.
7. **Hybrid QA with citations.** `qa.py`: vector top-k + graph entity context → LLM with explicit "cite chunk_id / node id" instruction → structured `QAResponse` with `citations`. Rationale: spec Phase 4.
8. **LangGraph for orchestration, not a monolith.** Each agent = node; typed `AgentState` (pydantic) carries fields between nodes. Rationale: spec + dev rules (modular, independently testable).
9. **DI everywhere; agents take injected deps (llm, stores, embedder).** Enables unit tests with Mock backend + in-memory Chroma + fake/fake-or-testcontainer Neo4j. Rationale: dev rules.
10. **docker-compose for infra.** Neo4j (with APOC) + ChromaDB; `.env.example` documents all vars. Rationale: reproducible local dev.

## Scope IN
- PDF ingestion → full 9-agent pipeline → Neo4j graph + ChromaDB vectors → summaries → hybrid QA with citations.
- All 5 roadmap phases (ingestion, language/translate/classify/sentiment, NER/graph, LangGraph orchestration + hybrid RAG, Streamlit dashboard).
- Provider-agnostic LLM with OpenRouter default + Ollama + Mock.
- Independent unit tests per agent; agent-executed QA per todo.
- Documentation set auto-maintained.

## Scope OUT (Must NOT have)
- Word/PowerPoint/HTML/Markdown ingestion (future; architecture must not preclude it, but not built now).
- Hardcoded Gemini SDK dependency.
- Training custom classifiers/sentiment models (zero-shot / pretrained only).
- Production auth, multi-tenant accounts, billing.
- Distributed task queue (Celery/ARQ) — use FastAPI BackgroundTasks; note as future scaling path only.
- Kubernetes / cloud deployment manifests (local docker-compose only).

## Open questions
- None blocking. All forks resolved by user. Model IDs and label sets are env-configurable post-build.

## Approval gate
status: approved
- User replied "write the md files" → authorization to generate the plan file granted.
- Approach recorded above. Proceed to write `.omo/plans/graphrag-engine.md` with full todos + TL;DR.
