# graphrag-engine - Work Plan

## TL;DR (For humans)
<!-- Filled post-build: the base scaffold + Phases 1–4 are delivered. -->

**What you'll get:** A modular Python package that ingests PDFs, builds a Neo4j knowledge
graph + ChromaDB vector index, and answers questions with hybrid (vector + graph) cited
answers plus extractive/abstractive summaries. Phases 1–4 are implemented; Phase 5
(Streamlit dashboard) is scaffolded but not built.

**Why this approach:** Provider-agnostic LLM (OpenRouter default, Ollama offline, Mock for
tests), local Sentence-Transformers embeddings, conditional translation, and LangGraph
orchestration with dependency injection — so every agent is independently testable.

**What it will NOT do:** No Streamlit dashboard yet (Phase 5); no auth/multi-tenancy/Celery/
K8s; no non-PDF ingestion; no training of custom models.

**Effort:** XL (built across Phases 1–4)
**Risk:** Low–Medium — additive, DI-driven; graph→text rendering and summary prompt were the only novel surfaces.
**Decisions to sanity-check:** (resolved during build) local ST embeddings; OpenRouter-default
provider-agnostic LLM; conditional LLM translation; GLiNER+LLM relations; zero-shot
classification; DistilBERT sentiment.

---

> TL;DR (machine): XL effort, Low-Medium risk. Delivered: repo scaffold, config, models, LLM
> client (OpenRouter/Ollama/Mock), ChromaDB + Neo4j stores, 9 agents (reader, language_detect,
> translator, classifier, sentiment, ner, graph_builder, embeddings, summarizer) + QAOrchestrator,
> LangGraph pipeline, FastAPI (upload/documents/graph/qa/summary), Phase 1–4 docs. Phase 5 pending.

## Scope
### Must have
- Repo scaffold: `pyproject.toml`, `.env.example`, `.gitignore`, `docker-compose.yml` (Neo4j), package `__init__` files.
- `config/` (pydantic-settings + structured logging); `models/schema.py` (AgentState, Entity, Relation, Chunk, DocumentMeta, Citation, QARequest/Response).
- `llm/` provider-agnostic client: OpenRouter / Ollama / Mock backends + `get_llm_client` factory.
- `vector/chroma_store.py` (persistent + HTTP); `graph/neo4j_store.py` (async connect/close + query_graph, MERGE writes).
- `agents/`: BaseAgent (logging + retry), ReaderAgent (PyMuPDF), LanguageDetectionAgent, TranslationAgent (conditional), ClassificationAgent (zero-shot), SentimentAgent (DistilBERT), NERAgent (GLiNER + LLM relations), KnowledgeGraphAgent, EmbeddingAgent, SummaryAgent, QAOrchestrator (hybrid vector+graph).
- `orchestration/pipeline.py` LangGraph `StateGraph` (reader → language_detect → conditional translator → classifier → sentiment → ner → {embeddings, graph}); `run_ingest` with injected deps.
- `api/`: FastAPI app + deps; routes `ingest`, `documents`, `graph`, `query` (`/qa`), `summary`.
- Tests (pytest, injected fakes, no model downloads): reader, embeddings, qa (vector + hybrid), pipeline (phases 1–3), language/translator/classifier/sentiment, ner, graph_builder, summarizer, llm mock.
- Docs: PROJECT / ROADMAP / STATUS / DECISIONS / README kept updated through Phase 4.

### Must NOT have (guardrails, anti-slops, scope boundaries)
- Streamlit dashboard (Phase 5 — scaffolded only).
- Auth, multi-tenancy, Celery/ARQ task queue, Kubernetes manifests.
- Non-PDF ingestion (DOCX/PPTX/HTML/MD).
- Training custom classifiers/sentiment models (zero-shot / pretrained only).
- Hardcoded Gemini SDK dependency.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + pytest
- Evidence: <attemptDir>/task-<N>-graphrag-engine.<ext> (attemptDir = currentAttemptDir from 'omo ulw-loop status --json', .omo/evidence/ulw/<session>/<goalId>/a<attempt>; outside ulw-loop use .omo/evidence/)

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Scaffold repo: pyproject.toml, .env.example, .gitignore, docker-compose.yml, package __init__ files
  What to do / Must NOT do: Create project metadata file with correct Python version and dependencies; create environment template with all required variables; initialize git ignore for Python; create docker-compose for Neo4j (and optional ChromaDB); create empty __init__.py files for all packages to make them importable.
  Must NOT do: Include actual secrets in .env.example; forget to initialize any package directory; omit any package from docker-compose or pyproject.
  Parallelization: Wave 1 | Blocked by: None | Blocks: All other todos (foundation)
  References (executor has NO interview context - be exhaustive): pyproject.toml, .env.example, .gitignore, docker-compose.py, agents/__init__.py, api/__init__.py, config/__init__.py, models/__init__.py, llm/__init__.py, vector/__init__.py, graph/__init__.py, orchestration/__init__.py, frontend/__init__.py, tests/__init__.py
  Acceptance criteria (agent-executable): 
    - File pyproject.toml exists with [project] requires-python = ">=3.12" and dependencies including fastapi, uvicorn, pydantic, pydantic-settings, langgraph, chromadb, neo4j, sentence-transformers, transformers, ollama, openai, sentencepiece, sentence-transformers, etc.
    - File .env.example exists with all required environment variables (OPENROUTER_API_KEY, LLM_PROVIDER, WORKER_MODEL, QA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, CHROMA_PERSIST_DIR, etc.)
    - File .gitignore excludes __pycache__, *.pyc, .env, data/, etc.
    - File docker-compose.yml defines a neo4j service with appropriate ports and environment variables.
    - All package directories have an __init__.py file.
  QA scenarios (name the exact tool + invocation): 
    - happy: Check each file exists with correct content; verify docker-compose syntax with `docker compose config`; verify pyproject can be parsed by pip.
    - failure: Missing .env.example → should fail on missing env vars; incorrect docker-compose syntax → should fail on docker compose up.
  Evidence: <attemptDir>/task-1-graphrag-engine.<ext>
  Commit: Y | feat(project): scaffold repository with config, docker-compose, and package structure

- [ ] 2. Implement config/ (settings.py pydantic-settings, logging.py)
  What to do / Must NOT do: Create pydantic-settings BaseSettings class with all environment variables (LLM provider, API keys, model names, Neo4j credentials, ChromaDB settings, chunking parameters, NLP model names, log level, upload dir, app host/port). Create logging configuration that sets up structured logging to stdout with configurable log level.
  Must NOT do: Hardcode any values; forget to mark secrets as optional; omit any required environment variable; use basic logging instead of structured.
  Parallelization: Wave 1 | Blocked by: Todo 1 (package __init__ files) | Blocks: Todos 3-14 (all implementation depends on config)
  References (executor has no interview context - be exhaustive): config/settings.py, config/logging.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from config.settings import Settings; s = Settings(); assert s.llm_provider == 'openrouter'; assert s.worker_model == 'google/gemini-flash-1.5'; assert s.embedding_model == 'sentence-transformers/all-MiniLM-L6-v2'"
    - python3 -c "from config.logging import setup_logging; import logging; logger = setup_logging('test'); assert logger.level == logging.INFO; logger.info('test'); assert len(logger.handlers) == 1"
  QA scenarios (name the exact tool + invocation): 
    - happy: Settings loads correctly from environment; logging outputs structured JSON or formatted lines.
    - failure: Missing required env var → Settings validation error; invalid log level → ValueError.
  Evidence: <attemptDir>/task-2-graphrag-engine.<ext>
  Commit: Y | feat(config): add pydantic-settings configuration and structured logging

- [ ] 3. Implement models/schema.py (AgentState, Entity, Relation, QARequest/Response, etc.)
  What to do / Must NOT do: Define pydantic models for the shared pipeline state (AgentState) and domain objects (Entity, Relation, Chunk, DocumentMeta, QARequest, QAResponse, Citation). AgentState must contain all fields passed between pipeline agents (raw text, language, translation, classification, sentiment, entities, relationships, chunk ids, summary, qa answer, citations, errors, logs).
  Must NOT do: Use plain dictionaries instead of pydantic models; omit any required field; make fields mutable when they should be immutable; forget to include necessary imports.
  Parallelization: Wave 1 | Blocked by: Todo 2 (config) | Blocks: Todos 4-14 (all agents and stores depend on models)
  References (executor has NO interview context - be exhaustive): models/schema.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from models.schema import AgentState, Entity, Relation, Chunk, DocumentMeta, QARequest, QAResponse, Citation; st = AgentState(); assert hasattr(st, 'raw_text'); assert hasattr(st, 'entities'); assert len(st.entities) == 0"
    - python3 -c "from models.schema import Entity; e = Entity(id='e1', text='test', label='PER'); assert e.text == 'test'; assert e.label == 'PER'"
  QA scenarios (name the exact tool + invocation): 
    - happy: All models instantiate correctly with default values; fields can be set and retrieved.
    - failure: Missing required field → ValidationError; invalid field type → ValidationError.
  Evidence: <attemptDir>/task-3-graphrag-engine.<ext>
  Commit: Y | feat(models): add pydantic domain models and shared AgentState

- [ ] 4. Implement llm/ client abstraction (client + openrouter + ollama + mock backends + factory)
  What to do / Must NOT do: Create abstract base class LLMClient with acomplete method; implement OpenRouterBackend (using openai library), OllamaBackend (using ollama library), and MockBackend (returns canned responses); create factory get_llm_client(role, settings) that returns appropriate backend based on LLM_PROVIDER env var. Support async and sync completion methods.
  Must NOT do: Hardcode Gemini SDK; forget to make backends injectable for testing; omit JSON completion method; lack proper error handling for network issues.
  Parallelization: Wave 1 | Blocked by: Todo 3 (models) | Blocks: Todos 5-14 (LLM-dependent agents: translator, summarizer, qa)
  References (executor has NO interview context - be exhaustive): llm/client.py, llm/openrouter.py, llm/ollama.py, llm/mock.py, llm/__init__.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from llm import get_llm_client; from llm.mock import MockBackend; from config.settings import Settings; s = Settings(); s.llm_provider = 'mock'; client = get_llm_client('worker', s); assert isinstance(client, MockBackend); assert asyncio.run(client.acomplete([{'role':'user','content':'test'}])) == 'test'"
    - python3 -c "from llm import get_llm_client; from config.settings import Settings; s = Settings(); s.llm_provider = 'openrouter'; s.openrouter_api_key = 'fake'; client = get_llm_client('worker', s); assert client.__class__.__name__ == 'OpenRouterBackend'"
  QA scenarios (name the exact tool + invocation): 
    - happy: Factory returns correct backend based on LLM_PROVIDER; MockBackend returns canned text; OpenRouterBackend formats requests correctly.
    - failure: Invalid LLM_PROVIDER → ValueError; missing API key → OpenRouter authentication error (when actually calling).
  Evidence: <attemptDir>/task-4-graphrag-engine.<ext>
  Commit: Y | feat(llm): add provider-agnostic LLM client with OpenRouter, Ollama, and Mock backends

- [ ] 5. Implement vector/chroma_store.py (ChromaDB wrapper, persistent + http)
  What to do / Must NOT do: Create ChromaStore class wrapping chromadb.PersistentClient (embedded mode) or chromadb.HttpClient (server mode); implement add_chunks method to store text chunks with embeddings; implement query method for similarity search; support filtering by doc_id. Use settings for configuration.
  Must NOT do: Forget to handle both persistent and HTTP modes; omit metadata storage (doc_id, chunk index); use hardcoded collection name; fail to close client properly.
  Parallelization: Wave 1 | Blocked by: Todo 4 (llm client) | Blocks: Todos 8-10 (embeddings agent) and Todo 12 (api)
  References (executor has NO interview context - be exhaustive): vector/chroma_store.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from vector.chroma_store import ChromaStore; from config.settings import Settings; s = Settings(); s.chroma_persist_dir = '/tmp/test_chroma'; store = ChromaStore(s); assert store.count() == 0"
    - python3 -c "from vector.chroma_store import ChromaStore; from models.schema import Chunk; import numpy as np; s = Settings(); s.chroma_persist_dir = '/tmp/test_chroma2'; store = ChromaStore(s); chunks = [Chunk(id='t1', doc_id='d1', text='hello', index=0)]; embeds = np.array([[0.1,0.2,0.3]]); store.add_chunks(chunks, embeds); assert store.count() = 1"
  QA scenarios (name the exact tool + invocation): 
    - happy: Store correctly adds and retrieves chunks; query returns correct top-k results; filtering by doc_id works.
    - failure: Adding chunks with missing text → error; querying with invalid embedding dimension → error.
  Evidence: <attemptDir>/task-5-graphrag-engine.<ext>
  Commit: Y | feat(vector): add ChromaDB wrapper for persistent and HTTP modes

- [ ] 6. Implement graph/neo4j_store.py (Phase 3 stub with connection + interface)
  What to do / Must NOT do: Create Neo4jStore class with async connect and close methods; implement stub methods for upsert_entities, upsert_relationships, and query_graph that raise NotImplementedError with message indicating Phase 3 implementation. Include proper async context manager support.
  Must NOT do: Implement actual Neo4j logic now (save for Phase 3); forget to include async methods; omit connection parameters (uri, user, password); leave methods as empty pass.
  Parallelization: Wave 1 | Blocked by: Todo 5 (vector store) | Blocks: Todo 11 (orchestration) and Todo 12 (api)
  References (executor has NO interview context - be exhaustive): graph/neo4j_store.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from graph.neo4j_store import Neo4jStore; from config.settings import Settings; s = Settings(); store = Neo4jStore(s); assert store._driver is None"
    - python3 -c "import asyncio; from graph.neo4j_store import Neo4jStore; from config.settings import Settings; s = Settings(); async def test(): store = Neo4jStore(s); await store.connect(); assert store._driver is not None; await store.close(); assert store._driver is None; asyncio.run(test())"
  QA scenarios (name the exact tool + invocation): 
    - happy: Store instantiates without error; connect/open and close/work correctly; stub methods raise NotImplementedError.
    - failure: Missing connection parameters → error during connect; attempting to call stub methods → NotImplementedError.
  Evidence: <attemptDir>/task-6-graphrag-engine.<ext>
  Commit: Y | feat(graph): add Neo4j store stub with connection plumbing (Phase 3)

- [ ] 7. Implement agents/base.py (logging + retry mixin)
  What to do / Must NOT do: Create BaseAgent class with structured logging (prefixed with agent name) and retry wrapper with exponential backoff. Agents should inherit from this class and implement run(state, **deps) -> dict.
  Must NOT do: Forget to include logger name prefix; use fixed retry delay instead of exponential backoff; omit logging of attempts; make retry count non-configurable.
  Parallelization: Wave 1 | Blocked: None | Blocks: Todos 8-14 (all agents inherit from base)
  References (executor has NO interview context - be exhaustive): agents/base.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from agents.base import BaseAgent; class TestAgent(BaseAgent): name = 'test'; async def run(self, state, **deps): return {'test': True}; agent = TestAgent(); import asyncio; async def f(): return await agent.run_with_retry({}, retries=2); result = asyncio.run(f()); assert result['test'] == True"
    - python3 -c "from agents.base import BaseAgent; agent = BaseAgent(name='test'); agent._log('test message', logging.INFO); # should log without error"
  QA scenarios (name the exact tool + invocation): 
    - happy: BaseAgent provides logging and retry; custom agents inherit correctly; retry works on transient failures.
    - failure: Logger not configured → no output; retry exhausted after max attempts → raises last exception.
  Evidence: <attemptDir>/task-7-graphrag-engine.<ext>
  Commit: Y | feat(agents): add base agent with logging and retry

- [ ] 8. Implement Phase 1 ReaderAgent (PyMuPDF PDF ingestion)
  What to do / Must NOT do: Create ReaderAgent that inherits from BaseAgent; extracts text from PDF bytes using PyMuPDF (fitz); sets state.raw_text, state.pages, and state.num_pages. Handles multi-page PDFs; logs page count and character count.
  Must NOT do: Forget to close the PDF document; assume PDF is text-based (should handle binary PDFs); omit error handling for corrupted or encrypted PDFs; return empty text on failure.
  Parallelization: Wave 2 | Blocked by: Todo 7 (agents base) | Blocks: Todos 9-10 (embeddings and qa agents depend on raw text)
  References (executor has NO interview context - be exhaustive): agents/reader.py
  Acceptance criteria (agent-executable): 
    - python3 -c "import fitz; from agents.reader import ReaderAgent; import asyncio; doc = fitz.open(); page = doc.new_page(); page.insert_text((50, 100), 'Hello World'); data = doc.tobytes(); doc.close(); agent = ReaderAgent(); state = await agent.run(AgentState(file_bytes=data)); assert state.num_pages == 1; assert 'Hello World' in state.raw_text"
    - python3 -c "import fitz; from agents.reader import ReaderAgent; agent = ReaderAgent(); import asyncio; async def f(): return await agent.run(AgentState(file_bytes=b'not a pdf')); result = await f(); assert isinstance(result, Exception) or 'Failed to open PDF' in str(result)"
  QA scenarios (name the exact tool + invocation): 
    - happy: Extracts text correctly from single and multi-page PDFs; returns page count and raw text; handles empty PDFs gracefully.
    - failure: Corrupted PDF → ValueError; encrypted PDF → PermissionError; missing file_bytes → ValueError.
  Evidence: <attemptDir>/task-8-graphrag-engine.<ext>
  Commit: Y | feat(agents-reader): add PDF text extraction agent

- [ ] 9. Implement Phase 1 EmbeddingAgent (Sentence-Transformers + ChromaDB indexing + chunker)
  What to do / Must NOT do: Create EmbeddingAgent that inherits from BaseAgent; chunks text (translated or raw) using character-based overlap; embeds chunks using Sentence-Transformer model; stores chunks and embeddings in ChromaDB via ChromaStore. Includes chunker helper function. Embedder and vector store are injectable for testing.
  Must NOT do: Forget to chunk text before embedding; embed raw text instead of translated text when available; use fixed chunk size without overlap; omit injectability for embedder and store; load SentenceTransformer at module level (should be lazy).
  Parallelization: Wave 2 | Blocked by: Todo 8 (reader agent) | Blocks: Todo 10 (qa agent depends on embeddings for context) and Todo 11 (orchestration)
  References (executor has NO interview context - be exhaustive): agents/embeddings.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from agents.embeddings import EmbeddingAgent, chunk_text; from models.schema import AgentState, Chunk; from config.settings import Settings; from tests.conftest import FakeEmbedder; from chromadb import EphemeralClient; from vector.chroma_store import ChromaStore; s = Settings(chroma_persist_dir='/tmp/test_chroma'); store = ChromaStore(s); store._client = EphemeralClient(); store._collection = store._client.get_or_create_collection('test', metadata={'hnsw:space':'cosine'}); embedder = FakeEmbedder(); agent = EmbeddingAgent(s, embedder=embedder, vector_store=store); state = AgentState(doc_id='d1', raw_text='Hello world. This is a test.' * 3); result = await agent.run(state); assert len(result['chunk_ids']) > 0; assert store.count() = len(result['chunk_ids'])"
    - python3 -c "from agents.embeddings import EmbeddingAgent; from config.settings import Settings; s = Settings(); agent = EmbeddingAgent(s); import asyncio; async def f(): return await agent.run(AgentState(raw_text='')); result = await f(); assert result['chunk_ids'] == []"
  QA scenarios (name the exact tool + invocation): 
    - happy: Correctly chunks text; embeds and stores chunks; returns list of chunk ids; store count matches.
    - failure: Empty input text → returns empty chunk ids; invalid chunk size/overlap → ValueError.
  Evidence: <attemptDir>/task-9-graphrag-engine.<ext>
  Commit: Y | feat(agents-embeddings): add text chunking, embedding, and ChromaDB storage agent

- [ ] 10. Implement Phase 1 basic RAG QA agent (vector retrieval + LLM answer + citations)
  What to do / Must NOT do: Create QAOrchestrator that inherits from BaseAgent; given a question, retrieves top-k similar chunks from ChromaDB, builds context with [chunk_id] citations, and asks LLM to answer with inline citations. Uses injectable LLM, embedder, and vector store. Includes happy and failure paths for QA.
  Must NOT do: Forget to include citations in context; use hardcoded prompt instead of configurable; omit embedder injection; fail to handle empty retrieval results; return citations without text excerpts.
  Parallelization: Wave 2 | Blocked by: Todo 9 (embeddings agent) | Blocks: Todo 11 (orchestration) and Todo 12 (api)
  References (executor has NO interview context - be exhaustive): agents/qa.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from agents.qa import QAOrchestrator; from models.schema import AgentState, Chunk, QAResponse; from config.settings import Settings; from tests.conftest import FakeEmbedder; from chromadb import EphemeralClient; from vector.chroma_store import ChromaStore; s = Settings(chroma_persist_dir='/tmp/test_chroma'); store = ChromaStore(s); store._client = EphemeralClient(); store._collection = store._client.get_or_create_collection('test', metadata={'hnsw:space':'cosine'}); embedder = FakeEmbedder(); llm = MockBackend(text_response='The answer is 42.'); orch = QAOrchestrator(s, llm=llm, embedder=embedder, vector_store=store); state = AgentState(doc_id='d1'); result = await orch.answer('What is the answer?', doc_id='d1', top_k=1); assert result['qa_answer'] == 'The answer is 42.'; assert len(result['citations']) == 1; assert result['citations'][0].source == 'vector'"
    - python3 -c "from agents.qa import QAOrchestrator; from config.settings import Settings; s = Settings(); orch = QAOrchestrator(s); import asyncio; async def f(): return await orch.answer('test', doc_id='d1'); result = await f(); assert result['qa_answer'] == '' and len(result['citations']) == 0"
  QA scenarios (name the exact tool + invocation): 
    - happy: Returns answer with correct citations when context is available; returns empty answer when no relevant chunks.
    - failure: Invalid top_k (negative) → ValueError; missing doc_id when required → error.
  Evidence: <attemptDir>/task-10-graphrag-engine.<ext>
  Commit: Y | feat(agents-qa): add basic vector-RAG QA agent with citations

- [ ] 11. Implement orchestration/pipeline.py (LangGraph StateGraph skeleton: reader → embeddings → qa)
  What to do / Must NOT do: Create LangGraph StateGraph that wires ReaderAgent → EmbeddingAgent → QAOrchestrator in sequence; provide run_ingest helper function to run the graph end-to-end; include async node functions that call agent.run. The graph should represent the Phase 1 ingestion pipeline (reader → embeddings → qa for QA).
  Must NOT do: Forget to include conditional edges for future phases (e.g., translation); omit the run_ingest helper; use synchronous calls instead of async; fail to pass dependencies (embedder, vector store) to agents correctly.
  Parallelization: Wave 3 | Blocked by: Todo 10 (qa agent) | Blocks: Todo 12 (api)
  References (executor has NO interview context - be exhaustive): orchestration/pipeline.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from orchestration.pipeline import run_ingest; from config.settings import Settings; from tests.conftest import FakeEmbedder; from chromadb import EphemeralClient; from vector.chroma_store import ChromaStore; import fitz; s = Settings(chroma_persist_dir='/tmp/test_chroma'); store = ChromaStore(s); store._client = EphemeralClient(); store._collection = store._client.get_or_create_collection('test', metadata={'hnsw:space':'cosine'}); def make_pdf(text): d=fitz.open(); p=d.new_page(); p.insert_text((50,100),text); b=d.tobytes(); d.close(); return b; pdf=make_pdf('Test. ' * 3); st=await run_ingest(pdf, 'test.pdf', 'doc1', s, embedder=FakeEmbedder(), vector_store=store); assert isinstance(st, AgentState); assert st.num_pages == 1; assert len(st.chunk_ids) > 0"
    - python3 -c "from orchestration.pipeline import run_ingest; from config.settings import Settings; s = Settings(); import asyncio; async def f(): return await run_ingest(b'not a pdf', 'test.pdf', 'doc1', s, embedder=FakeEmbedder(), vector_store=ChromaStore(s)); result = await f(); assert isinstance(result, Exception) or 'Failed to open PDF' in str(result)"
  QA scenarios (name the exact tool + invocation): 
    - happy: Graph runs successfully; output state contains expected fields (raw_text, pages, num_pages, chunk_ids); run_ingest returns AgentState.
    - failure: Invalid PDF bytes → ValueError; missing dependencies → error during graph execution.
  Evidence: <attemptDir>/task-11-graphrag-engine.<ext>
  Commit: Y | feat(orchestration): add LangGraph ingestion skeleton and run_ingest helper

- [ ] 12. Implement api/ (FastAPI app, deps, routes: upload, query, documents)
  What to do / Must NOT do: Create FastAPI application with dependency injection for settings, LLM client, vector store, and ChromaStore; define routes for POST /documents/upload (PDF ingestion), GET /documents (list), GET /documents/{doc_id} (get), POST /qa (question answering); include health check endpoint. Use BackgroundTasks for upload to avoid blocking.
  Must NOT do: Hardcode secrets; forget to include Pydantic models for request/response bodies; omit error handling for invalid file types; lack logging for requests; fail to inject dependencies correctly.
  Parallelization: Wave 4 | Blocked by: Todo 11 (orchestration) | Blocks: Todo 13 (tests) and Todo 14 (docs)
  References (executor has NO interview context - be exhaustive): api/main.py, api/deps.py, api/routes/ingest.py, api/routes/documents.py, api/routes/qa.py
  Acceptance criteria (agent-executable): 
    - python3 -c "from fastapi.testclient import TestClient; from api.main import app; c = TestClient(app); r = c.get('/health'); assert r.status_code = 200; assert r.json() == {'status': 'ok'}"
    - python3 -c "from fastapi.testclient import TestClient; from api.main import app; c = TestClient(app); r = c.post('/documents/upload', files={'file':('test.pdf', b'%PDF-1.4\n1 0 obj<</Type/Catalog>>\ntrailer<<>>%%EOF', 'application/pdf')}); assert r.status_code == 200; assert 'doc_id' in r.json(); assert r.json()['status'] == 'indexed'"
  QA scenarios (name the exact tool + invocation): 
    - happy: Health endpoint returns ok; upload endpoint accepts PDF and returns doc_id with status indexed; qa endpoint returns answer with citations; list endpoint returns list of documents.
    - failure: Upload endpoint rejects non-PDF files (e.g., .txt) with 400 error; qa endpoint with invalid JSON returns 422 error.
  Evidence: <attemptDir>/task-12-graphrag-engine.<ext>
  Commit: Y | feat(api): add FastAPI app with dependency injection, upload, list, and get endpoints; QA endpoint for question answering

- [ ] 13. Write tests/ for Phase 1 (reader, embeddings, qa, pipeline, llm mock)
  What to do / Must NOT do: Write pytest test suite for Phase 1 components: ReaderAgent (text extraction), EmbeddingAgent (chunking + storage), QAOrchestrator (answer + citations), orchestration pipeline (end-to-end run), and LLM mock (text and json responses). Use fixtures for settings, fake embedder, ephemeral chroma store, and mock LLM. Include happy and failure scenarios for each component.
  Must NOT do: Forget to test error conditions; omit tests for any Phase 1 component; use real LLM or model downloads in tests; make tests dependent on external services.
  Parallelization: Wave 5 | Blocked by: Todo 12 (api) | Blocks: Todo 14 (docs)
  References (executor has NO interview context - be exhaustive): tests/conftest.py, tests/llm/test_mock.py, tests/agents/__init__.py, tests/agents/test_reader.py, tests/agents/test_embeddings.py, tests/agents/test_qa.py, tests/test_pipeline.py
  Acceptance criteria (agent-executable): 
    - python3 -c "import pytest; result = pytest.main(['-q', 'tests/']); assert result == 0"
    - python3 -c "from tests.conftest import Settings, FakeEmbedder; from chromadb import EphemeralClient; from vector.chroma_store import ChromaStore; s = Settings(); e = FakeEmbedder(); store = ChromaStore(s); store._client = EphemeralClient(); store._collection = store._client.get_or_create_collection('test', metadata={'hnsw:space': 'cosine'}); assert isinstance(store, ChromaStore)"
  QA scenarios (name the exact tool + invocation): 
    - happy: All 9 tests pass; mock LLM returns correct text/json; embeddings agent stores and retrieves chunks correctly; qa agent answers with citations; pipeline runs end-to-end.
    - failure: Any test fails → non-zero exit code; missing fixture → ImportError.
  Evidence: <attemptDir>/task-13-graphrag-engine.<ext>
  Commit: Y | feat(tests): add pytest test suite for Phase 1 components

- [ ] 14. Write doc files (PROJECT, ROADMAP, STATUS, DECISIONS, README) and update STATUS as work progresses
  What to do / Must NOT do: Create or update PROJECT.md (vision, architecture, scope), ROADMAP.md (phased delivery plan), STATUS.md (current status, completed tasks, blockers, next steps), DECISIONS.md (key architectural decisions and reasoning), README.md (setup, usage, examples, badges). Update STATUS.md after each major milestone to reflect progress.
  Must NOT do: Forget to update STATUS.md after completing a major milestone; omit any required document; include outdated or incorrect information; make documents overly technical without summary for non-engineers.
  Parallelization: Wave 6 | Blocked by: Todo 13 (tests) | Blocks: None (final wave)
  References (executor has NO interview context - be exhaustive): PROJECT.md, ROADMAP.md, STATUS.md, DECISIONS.md, README.md
  Acceptance criteria (agent-executable): 
    - python3 -c "import os; assert os.path.exists('PROJECT.md'); assert os.path.exists('ROADMAP.md'); assert os.path.exists('STATUS.md'); assert os.path.exists('DECISIONS.md'); assert os.path.exists('README.md')"
    - python3 -c "with open('STATUS.md', 'r') as f: content = f.read(); assert 'Phase 1' in content and 'completed' in content"
  QA scenarios (name the exact tool + invocation): 
    - happy: All documents exist and contain correct, up-to-date information; STATUS.md reflects current progress.
    - failure: Missing document → FileNotFoundError; outdated information → incorrect status or instructions.
  Evidence: <attemptDir>/task-14-graphrag-engine.<ext>
  Commit: Y | feat(docs): add and update project documentation set

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity

## Commit strategy
- One commit per todo (see each todo's Commit line), conventional-commit style.
- No squashing; each commit independently buildable and `pytest` green at the agent/test/docs commits.
- Do NOT commit `.omo/` plan artifacts unless the user asks.

## Success criteria
- Repo scaffolds and `pip install -e ".[dev]"` resolves all deps (incl. `langgraph`, `neo4j`).
- `pytest` green with injected fakes (no model downloads / network) across reader, embeddings,
  qa (vector + hybrid), pipeline (phases 1–3), language/translator/classifier/sentiment, ner,
  graph_builder, summarizer, llm mock.
- `POST /documents/upload` ingests a PDF → ChromaDB chunks + (Phase 3) Neo4j subgraph; response
  carries `language`, `doc_type`, `sentiment_label`, `sentiment_score`, `num_entities`, `graph_written`.
- `GET /documents/{doc_id}/graph` returns doc-scoped entities + relations; `POST /qa` returns
  dual-source (`vector`/`graph`) citations; `GET /documents/{doc_id}/summary` returns extractive +
  abstractive summary.
- STATUS/ROADMAP/README reflect Phases 1–4 implemented; Phase 5 noted as next.