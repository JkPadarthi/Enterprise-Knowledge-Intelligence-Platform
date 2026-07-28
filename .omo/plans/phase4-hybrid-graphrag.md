# phase4-hybrid-graphrag - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->

**What you'll get:** The QA endpoint stops being vector-only. It becomes *hybrid GraphRAG* — for each question it retrieves the top-k vector chunks **and** pulls the relevant entity/relation subgraph from Neo4j, feeds both into the LLM with a citation scheme that tags each fact as `[chunk_id]` (vector) or `[entity:text]` (graph), and returns citations carrying `source: "vector"` or `source: "graph"`. Plus a `SummaryAgent` that produces extractive + abstractive summaries stored on `state.summary`, surfaced via a new `GET /documents/{doc_id}/summary` endpoint. Every change is additive and reuses the existing DI/injected-store patterns.

**Why this approach:** Phase 3 already built and verified the graph write path + `query_graph` + the `/graph` endpoint. Phase 4's only missing piece is *consuming* that graph at query time and adding summarization. The `Citation` model already has `source: "vector" | "graph"` and `node_ref` — the schema was designed for exactly this. So this plan is mostly orchestration + two new retrieval methods, not new abstractions.

**What it will NOT do:** It does not build the Streamlit dashboard (that's Phase 5). It does not add auth, multi-tenancy, or a task queue. It does not change the ingestion pipeline or the NER/graph-write agents. It does not change `AgentState`/`Citation`/`QAResponse` field shapes (they already fit).

**Effort:** Medium
**Risk:** Low–Medium — additive, follows established DI patterns; the only new surface is graph→text rendering and the summary LLM prompt.
**Decisions to sanity-check:** (1) graph retrieval strategy — keyword/entity-overlap expansion from the question vs. a fixed doc-scoped subgraph; (2) how graph context is rendered into the prompt; (3) summary storage key shape.

Your next move: approve, or run a high-accuracy review. Full execution detail follows below.

---

> TL;DR (machine): Medium effort, Low-Medium risk. Implement Roadmap Phase 4: hybrid GraphRAG QA (vector + Neo4j graph context, dual-source citations) + SummaryAgent (extractive + abstractive) with a summary endpoint. All additive, reuses DI/injected stores, no schema changes.

## Scope
### Must have
- `QAOrchestrator.answer()` extended to **hybrid retrieval**: after the existing vector `store.query(...)`, also call an injected `graph_store.query_graph(...)` to fetch entity/relation context for the question's doc (or globally when `doc_id` is None), render it as readable text, and prepend it to the LLM context.
- Dual-source `Citation` objects: vector chunks keep `source="vector"` (already done); graph facts get `source="graph"`, `chunk_id=""`, `node_ref=<entity text or relation subject>`, `text_excerpt=<rendered fact>`.
- A `_get_graph_store()` helper on `QAOrchestrator` mirroring `_get_store`/`_get_llm` (injectable `graph_store` kwarg, else `Neo4jStore(self.settings)` with connect/close lifecycle like `KnowledgeGraphAgent`).
- `SummaryAgent.run()` implemented: extractive summary (lexrank/lead sentences over chunks via injected embedder) + abstractive summary (LLMClient over concatenated chunks), sets `state.summary = {"abstractive": str, "extractive": str}`. Injectable LLM + embedder.
- API: new `GET /documents/{doc_id}/summary` route returning `state.summary` (read from an injected/fake store or recomputed). Plus the `/qa` route injects `graph_store` into the orchestrator.
- Unit tests (injected fakes, no real models/Neo4j): hybrid QA returns graph citations; summary agent sets both keys; QA endpoint wiring injects graph store.
- STATUS/ROADMAP/README updated to mark Phase 4 done.

### Must NOT have (guardrails, anti-slops, scope boundaries)
- Streamlit dashboard (Phase 5).
- Auth, multi-tenancy, Celery/ARQ, K8s (future work).
- Changes to `AgentState` / `Citation` / `QAResponse` / `DocumentMeta` field shapes.
- New pip dependencies unless strictly required (lexrank needs `sumy` or `networkx` — see open question 1; default to a dependency-free lead-sentence extractor to avoid new deps).
- Any model download or network call during `pytest` (inject fakes).
- Breaking the existing vector-only `/qa` contract for callers that don't pass a graph store (graph retrieval must be a graceful no-op when no graph store is available).

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + pytest (`asyncio_mode = "auto"`, `pythonpath = ["."]` per `pyproject.toml`).
- Hybrid QA tested by injecting a `FakeGraphStore` whose `query_graph` returns canned entity/relation rows; assert citations include `source="graph"` entries and the prompt sent to the mock LLM contains the graph fact text.
- Summary agent tested with a `MockBackend` (abstractive) + `FakeEmbedder` (extractive); assert `state.summary` has both keys.
- Evidence: `.omo/evidence/phase4-hybrid-graphrag/` (or attempt dir from `omo ulw-loop status --json`); capture `pytest -q` output and per-agent invocation transcripts.
- Live smoke (manual, post-tests): with Neo4j + Chroma up, ask a question whose answer lives in the graph (e.g. a relation), confirm a `source="graph"` citation appears.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 QAOrchestrator hybrid retrieval | none | 2 (api), 4 (tests) | 3 |
| 2 QAOrchestrator graph-store lifecycle helper | none | 1 | 1, 3 |
| 3 SummaryAgent implementation | none | 5 (api summary), 4 (tests) | 1, 2 |
| 4 Hybrid QA + summary unit tests | 1, 3 | 5, 6 | — |
| 5 API: /qa injects graph_store + new /summary route | 1, 3 | 6 | 4 |
| 6 Docs update + commit | 4, 5 | none | — |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Extend `QAOrchestrator.answer()` with hybrid graph retrieval (`agents/qa.py`)
  What to do / Must NOT do: After the vector `results = store.query(...)` block, add a graph branch: get `graph_store = self._get_graph_store(deps.get("graph_store"))`. If a graph store is available, build a Cypher query that fetches entities + relations for the question's `doc_id` (or all docs when `doc_id is None`) — e.g. `MATCH (s:Entity)-[r]->(o:Entity) WHERE s.doc_id=$doc_id RETURN s.text AS subject, r.relation AS relation, o.text AS object`. Call `await graph_store.query_graph(cypher, params)`. Render each row as a fact line `subject relation object` and collect into `graph_context`. For each fact, append a `Citation(source="graph", chunk_id="", node_ref=subject, text_excerpt=fact[:200])`. Prepend `graph_context` to the LLM `context` string under a clear `## Knowledge graph` heading. If no graph store is injected/available, skip silently (vector-only behavior preserved). Must NOT change the vector retrieval path or break existing citations; must NOT require a graph store to be present; must NOT catch broad exceptions silently.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 4, 5
  References (executor has NO interview context - be exhaustive): `agents/qa.py:74-119` (`answer`, vector path, `Citation` construction, `_get_store`/`_get_llm` patterns), `graph/neo4j_store.py:133` (`query_graph` signature + parameterized Cypher), `models/schema.py:70-77` (`Citation` with `source`/`node_ref`), `agents/graph_builder.py:33-58` (the connect/close-when-opened lifecycle pattern to mirror for the graph store here).
  Acceptance criteria (agent-executable):
    - With a `FakeGraphStore.query_graph` returning `[{"subject":"Acme Corp","relation":"HEADQUARTERED_IN","object":"Paris"}]`, assert the `MockBackend.acomplete` received a prompt containing "Acme Corp HEADQUARTERED_IN Paris" and the returned `citations` include one with `source=="graph"` and `node_ref=="Acme Corp"`.
    - With `graph_store=None` (not injected), assert `answer()` still returns vector citations only and no graph citation, and the prompt contains no "Knowledge graph" heading.
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/agents/test_qa.py -q` extended; hybrid path yields graph citations; vector path unchanged.
    - failure: graph store raises → exception propagates (not swallowed); missing `doc_id` + global query still returns rows.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-1-hybrid-qa.txt`
  Commit: Y | feat(agents): hybrid GraphRAG QA — merge Neo4j graph context into answer with dual-source citations

- [ ] 2. Add `QAOrchestrator._get_graph_store()` lifecycle helper (`agents/qa.py`)
  What to do / Must NOT do: Mirror `_get_store`: accept `graph_store` kwarg, fall back to `self._graph_store`, then default-construct `Neo4jStore(self.settings)`. In `answer()`, if the resolved store's driver is `None` and it has `connect`, `await store.connect()` and set `opened_here=True`; `close()` in a `finally` only if `opened_here`. This exactly mirrors the fix already applied in `agents/graph_builder.py:47-58`. Must NOT close an injected store; must NOT import neo4j at module top.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 1
  References: `agents/qa.py:65-72` (`_get_store`), `agents/graph_builder.py:33-58` (the connect-when-opened pattern), `graph/neo4j_store.py:25-40` (`connect`/`close`).
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.qa import QAOrchestrator; from config.settings import Settings; from tests.conftest import FakeEmbedder; a=QAOrchestrator(Settings(), embedder=FakeEmbedder()); s=a._get_graph_store(None); assert s is not None"` (default constructs a store).
    - Injected store is returned as-is and never closed by the orchestrator.
  QA scenarios:
    - happy: default store constructed; injected store passed through.
    - failure: connecting a default store that has no driver opens it; closing only happens for the opened one.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-2-graph-store-helper.txt`
  Commit: Y | feat(agents): add graph-store lifecycle helper to QAOrchestrator

- [ ] 3. Implement `SummaryAgent.run()` (`agents/summarizer.py`)
  What to do / Must NOT do: Replace the `NotImplementedError` body. Extractive: rank chunks by centrality/lead-position using the injected embedder (cosine over chunk texts; pick top-N sentences) — keep it dependency-free (no `sumy`/`networkx` unless present). Abstractive: build a prompt with the concatenated chunk texts and call `self._get_llm().acomplete(messages)`; `self._get_llm` mirrors `QAOrchestrator._get_llm`. Set `state.summary = {"abstractive": <llm text>, "extractive": <joined lead sentences>}`. Read text from `state.raw_text` (or `state.translated_text or state.raw_text`); if empty, return `{}`. Must NOT import transformers/llm SDK at module top; must NOT add new pip deps by default (dependency-free extractive).
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 4, 5
  References: `agents/summarizer.py:11-19` (stub), `agents/qa.py:51-63` (`_get_llm`/`_get_embedder` patterns), `models/schema.py:132-133` (`summary: dict[str,str]`), `agents/base.py` (`run` returns partial dict).
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.summarizer import SummaryAgent; from models.schema import AgentState; from llm.mock import MockBackend; a=SummaryAgent(Settings(), llm=MockBackend(text_response='Summary.')); st=AgentState(doc_id='d', raw_text='Sentence one. Sentence two. Sentence three.'); r=asyncio.run(a.run(st)); assert set(r['summary'])=={'abstractive','extractive'} and r['summary']['abstractive']=='Summary.'"`
    - empty text → `{}`.
  QA scenarios:
    - happy: `pytest tests/agents/test_summarizer.py -q` passes; both summary keys populated via MockBackend.
    - failure: empty input returns `{}`; abstractive LLM failure propagates or is caught per base retry.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-3-summarizer.txt`
  Commit: Y | feat(agents): implement SummaryAgent (extractive + abstractive)

- [ ] 4. Add Phase 4 test suite (`tests/agents/test_qa.py` hybrid + `tests/agents/test_summarizer.py`)
  What to do / Must NOT do: Extend `test_qa.py` with hybrid-retrieval tests (FakeGraphStore + MockBackend asserting graph citations + prompt contents). Add `test_summarizer.py` with MockBackend + FakeEmbedder asserting both summary keys. Add a pipeline/API test asserting `/qa` injects graph store and `/summary` returns. Must NOT add network/model downloads; must NOT break the existing Phase 1 QA tests.
  Parallelization: Wave 2 | Blocked by: Todo 1, 3 | Blocks: Todo 6
  References: `tests/agents/test_qa.py` (existing patterns), `tests/conftest.py` (`FakeEmbedder`, `MockBackend`, `settings` fixtures), `api/routes/query.py` (QA route to mirror for summary route).
  Acceptance criteria (agent-executable):
    - `pytest -q` exits 0; hybrid QA + summarizer tests green with injected fakes; existing vector QA tests still pass.
  QA scenarios:
    - happy: graph citations present when FakeGraphStore returns rows; summary has both keys.
    - failure: no graph store → vector-only citations; empty summary input → `{}`.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-4-tests.txt`
  Commit: Y | test(phase4): hybrid QA + SummaryAgent unit tests with injected fakes

- [ ] 5. API: inject graph_store into `/qa` + add `GET /documents/{doc_id}/summary` (`api/routes/query.py`, new `api/routes/summary.py`, `api/main.py`, `api/deps.py`)
  What to do / Must NOT do: In `api/routes/query.py`, inject `graph_store: Neo4jStore = Depends(get_graph_store_connected)` (reuse the existing connected-dependency pattern from `api/routes/graph.py`) and pass it to `orchestrator.answer(..., graph_store=store)`. Create `api/routes/summary.py` with `GET /documents/{doc_id}/summary` that builds a `SummaryAgent`, calls `run` over the doc's text (fetch via injected vector store `get_by_doc` or a document registry), and returns `state.summary`. Reuse `get_settings_dep`/`get_vector_store`/`get_graph_store_connected` from `api/deps.py` (do NOT add new deps unless missing). Register the new router in `api/main.py`. Must NOT change the `/qa` response shape; must NOT require Neo4j for the summary endpoint if the doc text is available from the vector store.
  Parallelization: Wave 3 | Blocked by: Todo 1, 3 | Blocks: Todo 6
  References: `api/routes/query.py:17-31` (QA route), `api/routes/graph.py` (`get_graph_store_connected` usage, prefix pattern), `api/deps.py:23-40` (graph store dependency), `vector/chroma_store.py:62` (`get_by_doc`), `api/main.py:27-32` (router registration).
  Acceptance criteria (agent-executable):
    - `python3 -c "from fastapi.testclient import TestClient; from api.main import app; from unittest.mock import patch; from graph.neo4j_store import Neo4jStore; c=TestClient(app); fake=Neo4jStore(Settings()); fake._driver='FAKE'; async def q(cypher, params): return [{'subject':'Acme Corp','relation':'HEADQUARTERED_IN','object':'Paris'}]; fake.query_graph=q; with patch('api.routes.query.get_graph_store_connected', return_value=fake): r=c.post('/qa', json={'question':'Where is Acme headquartered?','doc_id':'d1'}); assert r.status_code==200; import json; j=r.json(); assert any(c['source']=='graph' for c in j['citations'])"` (graph citation present when graph store wired).
    - `GET /documents/d1/summary` returns 200 with `abstractive`/`extractive` keys (mocked).
  QA scenarios:
    - happy: `/qa` returns dual-source citations when graph store present; `/summary` returns both keys.
    - failure: `/qa` without graph store → vector-only (200); `/summary` for unknown doc → 404 or empty gracefully.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-5-api.txt`
  Commit: Y | feat(api): wire graph store into /qa + add /summary endpoint

- [ ] 6. Update STATUS.md, ROADMAP.md, README.md to mark Phase 4 complete
  What to do / Must NOT do: ROADMAP Phase 4 header → `✅ (implemented)`. STATUS: move Phase 4 from "In Progress" to Completed; update "Last update" line; note hybrid QA + summary done; keep Phase 5 as next. README: mention hybrid GraphRAG QA and the summary endpoint in the API table. Must NOT alter Phase 1-3 content or the architecture diagram.
  Parallelization: Wave 4 | Blocked by: Todo 4, 5 | Blocks: none
  References: `STATUS.md` (structure), `ROADMAP.md:30-35` (Phase 4 section), `README.md:48-55` (API table).
  Acceptance criteria (agent-executable):
    - `python3 -c "t=open('ROADMAP.md').read(); assert 'Phase 4' in t and '✅' in t.split('Phase 4')[1].split('\n')[0]"`
    - `STATUS.md` lists hybrid QA + SummaryAgent under Completed.
  QA scenarios:
    - happy: docs render consistently; ROADMAP Phase 4 marked done.
    - failure: no stale "Phase 4 in progress" wording remains.
  Evidence: `.omo/evidence/phase4-hybrid-graphrag/task-6-docs.txt`
  Commit: Y | docs(phase4): mark hybrid GraphRAG QA + summarization complete

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
- `QAOrchestrator.answer()` performs hybrid retrieval: vector top-k + Neo4j graph context, merged into one LLM prompt with `source="vector"` and `source="graph"` citations.
- `/qa` returns dual-source citations; vector-only behavior preserved when no graph store is injected.
- `SummaryAgent` produces `state.summary = {abstractive, extractive}`; `GET /documents/{doc_id}/summary` surfaces it.
- `pytest -q` green with NO model downloads / network calls (fakes injected); existing Phase 1-3 tests still pass.
- STATUS/ROADMAP/README reflect Phase 4 as implemented; no Phase 5 work leaked in.

## Open questions (for user sanity-check before approval)
1. **Extractive summary dependency:** default to a dependency-free lead-sentence/centrality extractor (no new pip dep). Alternative: add `sumy` (LexRank) — cleaner summaries but a new dependency. **Recommendation: dependency-free default.**
2. **Graph retrieval scope:** when `doc_id` is given, query only that doc's subgraph (precise). When `doc_id is None` (cross-document question), query globally and let the LLM pick relevant facts. **Recommendation: doc-scoped when doc_id present, global otherwise** (matches Phase 3's "doc-scoped write, global search" decision).
3. **Summary endpoint data source:** read chunk texts from the vector store (`get_by_doc`) rather than re-running NER/raw text, so it works for already-ingested docs. **Recommendation: vector store `get_by_doc`.**
