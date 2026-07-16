# Decisions

Key architectural decisions and the reasoning behind them. These are the choices most
likely to be questioned later, recorded so the "why" is not lost.

## 1. Provider-agnostic LLM — OpenRouter default, never hardcoded Gemini
The original brief named Gemini Flash/Pro, but the owner chose a **provider-agnostic
`LLMClient`** with **OpenRouter as the default backend**, plus an **Ollama** backend for
offline dev and a **Mock** backend for tests. Default model IDs (`google/gemini-flash-1.5`
worker / `google/gemini-pro-1.5` QA) are resolved *through OpenRouter*, preserving the
brief's model intent without coupling to the Gemini SDK. Rationale: swappable providers,
offline testability, and a single injection point for every LLM call.

## 2. Local Sentence-Transformers embeddings
Embeddings use a local `sentence-transformers` model (no API cost, offline, private).
The `EmbeddingClient`/embedder is injectable so Gemini Embeddings can be swapped in later.
Rationale: the brief's "enterprise, future-expansion" goal favours a self-hosted,
cost-free vector path.

## 3. Conditional translation
Language is detected first; translation runs **only when the source language differs from
the target** (`TRANSLATE_TARGET_LANG`, default `en`). Rationale: avoids needless LLM calls
and preserves fidelity when content is already English.

## 4. Two stores, one pipeline
After NER, the pipeline branches: `KnowledgeGraphAgent → Neo4j` and `EmbeddingAgent →
ChromaDB` run in parallel. Rationale: the brief's architecture and the hybrid-retrieval
QA design both depend on having structured (graph) and semantic (vector) indices.

## 5. Zero-shot classification, DistilBERT sentiment
Classification uses a zero-shot transformer (no labeled training data). Sentiment uses a
pretrained DistilBERT model. Rationale: enterprise document types vary widely; training
data is not assumed.

## 6. GLiNER for NER, LLM for relations
Entities come from GLiNER (preferred in the brief). Relations are extracted via an LLM
prompt over the extracted entities (mockable, deterministic in tests). Rationale: GLiNER
covers entities well; relation extraction needs context that an LLM provides cleanly.

## 7. LangGraph for orchestration, not a monolith
Each agent is a pure `run(state) -> dict` node; a typed `AgentState` (pydantic) is
threaded through the graph. Rationale: matches the brief's "modular, independently
testable" rule and makes inserting future agents a one-line `add_node` change.

## 8. Dependency injection + TDD
Every external dependency (LLM, embedder, Neo4j, ChromaDB) is an interface injected into
agents. Tests inject mocks (Mock LLM, fake embedder, ephemeral Chroma). Rationale: the
brief requires every agent to be independently testable.

## 9. ChromaDB embedded persistent; Neo4j via docker-compose
ChromaDB runs embedded (file-backed) by default — no extra container. Neo4j is provided
via `docker-compose.yml` for reproducible local dev; both accept external URIs via env.
Rationale: minimal local infra, easy to point at managed services later.
