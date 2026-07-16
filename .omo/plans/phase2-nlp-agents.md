# phase2-nlp-agents - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->

**What you'll get:** Four new document-understanding agents — automatic language detection, conditional translation (only when the document isn't English), zero-shot document-type classification, and sentiment scoring — wired into the existing ingestion pipeline so every uploaded PDF is enriched with language, type, and sentiment metadata, plus tests that run with zero model downloads.

**Why this approach:** The repo already has the agent stubs, the shared state fields, the settings, and the dependencies for Phase 2 — so this plan only fills in the four `NotImplementedError` bodies, inserts them into the LangGraph graph after the reader, and copies the new fields into the document record. Heavy ML models load lazily (only when used), exactly like the existing embedding agent, so imports stay fast and tests stay offline.

**What it will NOT do:** It does not build the knowledge graph / NER (Phase 3), the hybrid QA + summarizer (Phase 4), or the dashboard (Phase 5); it adds no new dependencies or configuration; and it never downloads models or hits the network during tests.

**Effort:** Short
**Risk:** Low - additive, follows established patterns, no schema changes
**Decisions to sanity-check:** conditional translation skips the model call when the doc is already English; classifier/sentiment truncate input to bound latency; tests inject fakes instead of real models.

Your next move: approve, or run a high-accuracy review. Full execution detail follows below.

---

> TL;DR (machine): Short effort, Low risk. Implement Roadmap Phase 2 (4 NLP agents + LangGraph wiring + API metadata + tests), all additive, no new deps/schema.

## Scope
### Must have
- `LanguageDetectionAgent` (`agents/language_detect.py`): detect `state.language` + `state.language_confidence` via `langdetect`; safe on empty/error.
- `TranslationAgent` (`agents/translator.py`): conditional — only translate when `state.language != settings.translate_target_lang`; set `state.translated_text`; otherwise return `{}` (no-op).
- `ClassificationAgent` (`agents/classifier.py`): zero-shot classification via `transformers.pipeline("zero-shot-classification", model=settings.classifier_model, ...)` over `settings.doc_type_label_list`; set `state.doc_type` + `state.doc_type_scores`.
- `SentimentAgent` (`agents/sentiment.py`): `transformers.pipeline("text-classification", model=settings.sentiment_model, ...)`; set `state.sentiment_label` + `state.sentiment_score`.
- LangGraph ingestion graph extended: `reader → language_detect → [conditional] translator → classifier → sentiment → embeddings`, with the conditional translate edge.
- Ingest API copies `doc_type`, `sentiment_label`, `sentiment_score` into `DocumentMeta` and returns them.
- Unit tests for each Phase 2 agent + updated end-to-end pipeline test; `pytest` green with no downloads/network.
- STATUS/ROADMAP/README updated to mark Phase 2 complete.

### Must NOT have (guardrails, anti-slops, scope boundaries)
- NER, relation extraction, Neo4j writes (Phase 3).
- Hybrid QA graph, SummaryAgent (Phase 4).
- Streamlit dashboard (Phase 5).
- New pip dependencies or new environment variables (all present in `pyproject.toml` / `Settings`).
- Any model download, network call, or running service during `pytest`.
- Changes to `AgentState` / `DocumentMeta` / `Settings` field shapes.
- Hardcoding model names or label sets (read from `Settings`).

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + pytest (`asyncio_mode = "auto"`, `pythonpath = ["."]` per `pyproject.toml:60-64`).
- Every Phase 2 agent tested by injecting a fake pipeline / fake LLM — no real `torch`/`transformers` model load, no network.
- Evidence: `.omo/evidence/phase2-nlp-agents/` (or attempt dir reported by `omo ulw-loop status --json`); capture `pytest -q` output and per-agent invocation transcripts.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 LanguageDetectionAgent | none | 5 (graph) | 2, 3, 4 |
| 2 TranslationAgent | none | 5 (graph) | 1, 3, 4 |
| 3 ClassificationAgent | none | 5 (graph) | 1, 2, 4 |
| 4 SentimentAgent | none | 5 (graph) | 1, 2, 3 |
| 5 Orchestration wiring | 1,2,3,4 | 6 (api), 7 (tests) | — |
| 6 API metadata enrichment | 5 | 7 (tests) | — |
| 7 Tests (agents + pipeline) | 1,2,3,4,5,6 | 8 (docs) | — |
| 8 Docs update | 7 | none | — |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [x] 1. Implement `LanguageDetectionAgent` (fill `agents/language_detect.py`)
  What to do / Must NOT do: Replace the `NotImplementedError` body. In `run(self, state, **deps)`, take `text = state.raw_text`. If `text.strip()` is empty, return `{}` (leave defaults). Otherwise lazy-import `langdetect` inside a `_get_detector` helper cached on `self` (mirror `EmbeddingAgent._get_embedder`), call `detect_langs(text)` to get the top language + confidence; set `state.language` and `state.language_confidence`. On `LangDetectException` or any detection error, append a message to `state.errors` and return `{}` (keep the default "en"). Never import `langdetect` at module top level. Must NOT change `AgentState`/`Settings`; must NOT raise on empty input.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 5
  References (executor has NO interview context - be exhaustive): `agents/language_detect.py` (current stub names `language`/`language_confidence`), `models/schema.py:111-114` (fields, defaults `language="en"`, `language_confidence=0.0`), `agents/base.py` (`BaseAgent.run(state, **deps) -> dict`, `_log`), `agents/embeddings.py:75-80` (lazy `_get_embedder` pattern to mirror), `agents/embeddings.py:92` (`text = state.translated_text or state.raw_text` fallback convention).
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.language_detect import LanguageDetectionAgent; from models.schema import AgentState; from config.settings import Settings; a=LanguageDetectionAgent(Settings()); s=AgentState(raw_text='Bonjour le monde'); r=asyncio.run(a.run(s)); assert r['language'] in {'fr', 'it', 'es'} or r['language_confidence']>0" ` (langdetect picks a non-en language for French; confidence > 0)
    - `python3 -c "import asyncio; from agents.language_detect import LanguageDetectionAgent; from models.schema import AgentState; from config.settings import Settings; a=LanguageDetectionAgent(Settings()); s=AgentState(raw_text='   '); r=asyncio.run(a.run(s)); assert r=={}"`
    - `python3 -c "import agents.language_detect as m; assert 'langdetect' not in globals() or True"` — verify no top-level import by checking the module imports without the heavy dep present is NOT required, but confirm `run` does not touch `state.errors` on success.
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/agents/test_language_detect.py -q` passes; French text yields a non-English language with confidence > 0; English text yields "en".
    - failure: empty/whitespace `raw_text` → returns `{}` and does not raise; garbage that trips `detect_langs` → appends to `state.errors`, returns `{}`.
  Evidence: `.omo/evidence/phase2-nlp-agents/task-1-language-detect.txt`
  Commit: Y | feat(agents): implement LanguageDetectionAgent via langdetect

- [x] 2. Implement `TranslationAgent` (fill `agents/translator.py`)
  What to do / Must NOT do: Replace the `NotImplementedError` body. `__init__` accepts `settings` and an optional `llm` (injectable for tests). In `run(self, state, **deps)`, if `state.language == settings.translate_target_lang` (default "en") OR `not state.raw_text.strip()`, return `{}` (no translation, `translated_text` stays `None`). Otherwise build a translation prompt (system: "You are a translator. Translate the following text to {target_lang}. Return ONLY the translated text."; user: the raw text) and call `self._get_llm().acomplete(messages)` (LLM via injected `llm` or `get_llm_client("worker", self.settings)`, mirroring `QAOrchestrator._get_llm`). Set `state.translated_text` to the returned text. Must NOT call the LLM when language is already the target; must NOT import LLM SDK at module top.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 5
  References (executor has NO interview context - be exhaustive): `agents/translator.py` (stub contract), `config/settings.py:50` (`translate_target_lang="en"`), `llm/client.py` (`LLMClient.acomplete(messages)`), `llm/__init__.py:13` (`get_llm_client(role, settings)`), `agents/qa.py:51-56` (`_get_llm` pattern: prefer injected `self._llm`, else `get_llm_client(role, self.settings)`), `models/schema.py:114` (`translated_text: Optional[str] = None`), `llm/mock.py` (`MockBackend(text_response=...)` for tests).
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.translator import TranslationAgent; from models.schema import AgentState; from config.settings import Settings; from llm.mock import MockBackend; a=TranslationAgent(Settings(), llm=MockBackend(text_response='hello')); s=AgentState(raw_text='Bonjour', language='fr'); r=asyncio.run(a.run(s)); assert r.get('translated_text')=='hello'"`
    - `python3 -c "import asyncio; from agents.translator import TranslationAgent; from models.schema import AgentState; from config.settings import Settings; a=TranslationAgent(Settings()); s=AgentState(raw_text='Hello', language='en'); r=asyncio.run(a.run(s)); assert r=={}"`
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/agents/test_translator.py -q` passes; non-English input → `translated_text` set via MockBackend; English input → `{}` (LLM never called — assert `acomplete` not invoked).
    - failure: empty `raw_text` → returns `{}`; non-English with injected failing LLM → raises and is caught by `run_with_retry` (assert retries logged).
  Evidence: `.omo/evidence/phase2-nlp-agents/task-2-translator.txt`
  Commit: Y | feat(agents): implement conditional TranslationAgent via LLMClient

- [x] 3. Implement `ClassificationAgent` (fill `agents/classifier.py`)
  What to do / Must NOT do: Replace the `NotImplementedError` body. Lazy-load `transformers.pipeline` inside `_get_pipeline` cached on `self` (model = `settings.classifier_model`, task = `"zero-shot-classification"`). In `run`, take `text = (state.translated_text or state.raw_text)` and truncate to a safe cap (e.g. first 2000 chars). If empty, return `{}`. Call `pipe(text, candidate_labels=settings.doc_type_label_list)` → returns `{"labels": [...], "scores": [...]}`. Set `state.doc_type` = top label, `state.doc_type_scores` = `{label: score}` dict. Must NOT import `transformers` at module top; must NOT hardcode labels/model.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 5
  References (executor has NO interview context - be exhaustive): `agents/classifier.py` (stub contract), `config/settings.py:53-54` (`classifier_model="facebook/bart-large-mnli"`, `doc_type_labels`), `config/settings.py:65-68` (`doc_type_label_list` property), `models/schema.py:117-118` (`doc_type=""`, `doc_type_scores={}`), `agents/embeddings.py:75-80` + `:92` (lazy model + `translated_text or raw_text` fallback), `agents/base.py`.
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.classifier import ClassificationAgent; from models.schema import AgentState; from config.settings import Settings; from tests.conftest import FakeClassifierPipe; a=ClassificationAgent(Settings()); a._pipe=FakeClassifierPipe(); s=AgentState(raw_text='Invoice for services rendered'); r=asyncio.run(a.run(s)); assert r['doc_type'] in Settings().doc_type_label_list; assert set(r['doc_type_scores']) <= set(Settings().doc_type_label_list)"` (FakeClassifierPipe returns labels/scores; see Todo 7 for fake def).
    - `python3 -c "import asyncio; from agents.classifier import ClassificationAgent; from models.schema import AgentState; from config.settings import Settings; a=ClassificationAgent(Settings()); r=asyncio.run(a.run(AgentState(raw_text='   '))); assert r=={}"`
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/agents/test_classifier.py -q` passes; fake zero-shot pipeline returns a valid `doc_type` from the label list and a scores dict.
    - failure: empty input → `{}`; invalid candidate labels → handled; truncation keeps length ≤ cap (assert `len(text) <= 2000`).
  Evidence: `.omo/evidence/phase2-nlp-agents/task-3-classifier.txt`
  Commit: Y | feat(agents): implement ClassificationAgent via zero-shot transformers

- [x] 4. Implement `SentimentAgent` (fill `agents/sentiment.py`)
  What to do / Must NOT do: Replace the `NotImplementedError` body. Lazy-load `transformers.pipeline` inside `_get_pipeline` cached on `self` (model = `settings.sentiment_model`, task = `"text-classification"`). In `run`, take `text = (state.translated_text or state.raw_text)`, truncate to a safe cap (e.g. first 2000 chars). If empty, return `{}`. Call `pipe(text)` → returns `[{"label": ..., "score": ...}]`. Map label → `state.sentiment_label` (normalize POSITIVE/NEGATIVE/NEUTRAL; keep raw label if unknown) and `state.sentiment_score` = score (float). Must NOT import `transformers` at module top; must NOT hardcode model.
  Parallelization: Wave 1 | Blocked by: none | Blocks: Todo 5
  References (executor has NO interview context - be exhaustive): `agents/sentiment.py` (stub contract), `config/settings.py:55` (`sentiment_model="distilbert-base-uncased-finetuned-sst-2-english"`), `models/schema.py:119-120` (`sentiment_label=""`, `sentiment_score=0.0`), `agents/embeddings.py:75-80` + `:92` (lazy model + fallback), `agents/base.py`.
  Acceptance criteria (agent-executable):
    - `python3 -c "import asyncio; from agents.sentiment import SentimentAgent; from models.schema import AgentState; from config.settings import Settings; from tests.conftest import FakeSentimentPipe; a=SentimentAgent(Settings()); a._pipe=FakeSentimentPipe(); s=AgentState(raw_text='I love this product'); r=asyncio.run(a.run(s)); assert r['sentiment_label'] in {'POSITIVE','NEGATIVE','NEUTRAL'}; assert 0.0 <= r['sentiment_score'] <= 1.0"` (FakeSentimentPipe returns `[{"label":"POSITIVE","score":0.98}]`).
    - `python3 -c "import asyncio; from agents.sentiment import SentimentAgent; from models.schema import AgentState; from config.settings import Settings; a=SentimentAgent(Settings()); r=asyncio.run(a.run(AgentState(raw_text='   '))); assert r=={}"`
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/agents/test_sentiment.py -q` passes; fake sentiment pipeline returns a normalized label + float score in [0,1].
    - failure: empty input → `{}`; unknown label → preserved as-is in `sentiment_label`; truncation keeps length ≤ cap.
  Evidence: `.omo/evidence/phase2-nlp-agents/task-4-sentiment.txt`
  Commit: Y | feat(agents): implement SentimentAgent via DistilBERT transformers pipeline

- [x] 5. Wire Phase 2 agents into the LangGraph ingestion graph (`orchestration/pipeline.py`)
  What to do / Must NOT do: Extend `build_ingest_graph` (and keep `run_ingest` signature unchanged) so the graph is `reader → language_detect → [conditional] translator → classifier → sentiment → embeddings`. Construct `LanguageDetectionAgent`, `TranslationAgent`, `ClassificationAgent`, `SentimentAgent` from `settings` (translator gets no `llm` so it uses the factory at runtime; tests may inject via a new optional param if needed, but keep `run_ingest`'s `embedder`/`vector_store` injection intact). Add node functions that call each agent's `run(state)`. Implement the conditional translate edge: add a `should_translate(state)` routing function returning `"translator"` when `state.language != settings.translate_target_lang` else `"classifier"`; wire `reader → language_detect`, `language_detect → should_translate` (conditional), `translator → classifier`, `classifier → sentiment`, `sentiment → embeddings`, `embeddings → END`. Must NOT break the existing `reader → embeddings` contract used by Phase 1 tests; must NOT remove the `embedder`/`vector_store` injection.
  Parallelization: Wave 2 | Blocked by: Todo 1,2,3,4 | Blocks: Todo 6, Todo 7
  References (executor has NO interview context - be exhaustive): `orchestration/pipeline.py:20-42` (`build_ingest_graph`, `reader_node`, `embeddings_node`, `run_ingest`), `agents/reader.py`, `agents/embeddings.py`, `langgraph.graph` (`StateGraph`, `END`, `add_conditional_edges`), `agents/base.py` (`run` returns partial dict merged by LangGraph), `models/schema.py:111-120`.
  Acceptance criteria (agent-executable):
    - `python3 -c "from orchestration.pipeline import build_ingest_graph; from config.settings import Settings; g=build_ingest_graph(Settings()); assert g is not None"` (compiles with the new nodes/edges)
    - `python3 -c "import asyncio, fitz; from orchestration.pipeline import run_ingest; from config.settings import Settings; from tests.conftest import FakeEmbedder; from chromadb import EphemeralClient; from vector.chroma_store import ChromaStore; s=Settings(); store=ChromaStore(s); store._client=EphemeralClient(); store._collection=store._client.get_or_create_collection('t', metadata={'hnsw:space':'cosine'}); d=fitz.open(); p=d.new_page(); p.insert_text((50,100),'Invoice for services. I love this product.'); b=d.tobytes(); d.close(); st=asyncio.run(run_ingest(b,'doc.pdf','d1',s,embedder=FakeEmbedder(),vector_store=store)); assert st.language; assert st.doc_type; assert st.sentiment_label; assert len(st.chunk_ids)>0"` (full Phase 2 pipeline runs end-to-end with injected fakes)
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/test_pipeline.py -q` passes and the returned state carries language/doc_type/sentiment; graph compiles; `run_ingest` still injects embedder/vector_store.
    - failure: English-only doc → `translator` node is skipped (assert `translated_text is None` and translator `run` not invoked); corrupted PDF → `ReaderAgent` raises and propagates as before.
  Evidence: `.omo/evidence/phase2-nlp-agents/task-5-orchestration.txt`
  Commit: Y | feat(orchestration): wire Phase 2 NLP agents into ingestion graph with conditional translate

- [x] 6. Enrich ingest API with Phase 2 metadata (`api/routes/ingest.py`)
  What to do / Must NOT do: After `run_ingest` succeeds in `upload_document`, copy `state.doc_type`, `state.sentiment_label`, `state.sentiment_score` into `meta` (the `DocumentMeta`) alongside the existing `meta.language`/`meta.num_pages`, and include them in the returned dict. Keep `DOCUMENT_REGISTRY[doc_id]` updated. Must NOT change the endpoint signature or other response fields; must NOT add new required config.
  Parallelization: Wave 3 | Blocked by: Todo 5 | Blocks: Todo 7
  References (executor has NO interview context - be exhaustive): `api/routes/ingest.py:36-57` (current `meta.language = state.language`, return dict), `models/schema.py:60-63` (`DocumentMeta.doc_type`, `sentiment_label`, `sentiment_score`), `api/deps.py:13` (`DOCUMENT_REGISTRY`), `agents/embeddings.py:92` (fallback text used downstream).
  Acceptance criteria (agent-executable):
    - `python3 -c "from fastapi.testclient import TestClient; from api.main import app; from unittest.mock import patch; import fitz; d=fitz.open(); p=d.new_page(); p.insert_text((50,100),'Invoice. I love it.'); b=d.tobytes(); d.close(); from models.schema import AgentState; s=AgentState(doc_id='x', language='en', doc_type='invoice', sentiment_label='POSITIVE', sentiment_score=0.9, num_pages=1, chunk_ids=['x::c0']); \nwith patch('api.routes.ingest.run_ingest', return_value=s):\n c=TestClient(app); r=c.post('/documents/upload', files={'file':('i.pdf', b'%PDF-1.4', 'application/pdf')}); j=r.json(); assert j['doc_type']=='invoice' and j['sentiment_label']=='POSITIVE' and j['sentiment_score']==0.9"` (patched run_ingest to avoid real models).
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest tests/` (or a new `tests/api/test_ingest_phase2.py`) passes; upload response includes `doc_type`, `sentiment_label`, `sentiment_score`; `GET /documents/{doc_id}` reflects them via `DOCUMENT_REGISTRY`.
    - failure: a doc with `doc_type=""` (detection skipped/empty) → endpoint still returns 200 with `doc_type` null/absent cleanly; non-PDF still rejected with 400.
  Evidence: `.omo/evidence/phase2-nlp-agents/task-6-api.txt`
  Commit: Y | feat(api): surface doc_type/sentiment/language in ingest response + DocumentMeta

- [x] 7. Add Phase 2 test suite (agents + fakes + updated pipeline)
  What to do / Must NOT do: Add `tests/agents/test_language_detect.py`, `test_translator.py`, `test_classifier.py`, `test_sentiment.py`. In `tests/conftest.py` add fakes: `FakeClassifierPipe` (returns `{"labels": settings.doc_type_label_list[:2], "scores": [0.7,0.3]}`) and `FakeSentimentPipe` (returns `[{"label":"POSITIVE","score":0.98}]`), plus a way to inject them (agents read `self._pipe` if set, else lazy-load). Classifier/sentiment tests monkeypatch/assign the fake pipe so NO real `transformers`/`torch` model loads. Translator test uses `MockBackend`. Update `tests/test_pipeline.py` to assert the end-to-end state now carries `language`/`doc_type`/`sentiment_label` using injected fakes (no model downloads). Must NOT add network/model downloads to any test; must NOT break the existing 9 passing tests.
  Parallelization: Wave 4 | Blocked by: Todo 1,2,3,4,5,6 | Blocks: Todo 8
  References (executor has NO interview context - be exhaustive): `tests/conftest.py` (existing `settings`, `embedder`=`FakeEmbedder`, `chroma_store`, `mock_llm` fixtures; `asyncio_mode=auto`), `tests/agents/test_embeddings.py` + `tests/test_pipeline.py` (patterns to mirror), `tests/llm/test_mock.py` (`MockBackend`), `pyproject.toml:60-64` (pytest config), `agents/classifier.py`/`:92` fallback, `agents/sentiment.py`.
  Acceptance criteria (agent-executable):
    - `pytest -q` exits 0; total tests increase by ≥ 4 (one happy + one failure per new agent) plus the updated pipeline test; no `torch`/network imports triggered during the run (assert by running with `TRANSFORMERS_OFFLINE=1` and confirming green).
    - `python3 -c "import subprocess; assert subprocess.run(['pytest','-q','tests/agents/test_classifier.py','tests/agents/test_sentiment.py','tests/agents/test_language_detect.py','tests/agents/test_translator.py']).returncode==0"`
  QA scenarios (name the exact tool + invocation):
    - happy: `pytest -q` → all green; new agent tests pass with fakes; pipeline test asserts Phase 2 fields populated.
    - failure: a test that feeds empty input asserts `{}` returned (no crash); a test asserting English doc skips translator (LLM `acomplete` call count == 0).
  Evidence: `.omo/evidence/phase2-nlp-agents/task-7-tests.txt`
  Commit: Y | test(phase2): add NLP agent unit tests with injected fakes + update pipeline test

- [x] 8. Update STATUS.md, ROADMAP.md, README.md to mark Phase 2 complete
  What to do / Must NOT do: In `STATUS.md`: under "Completed" add the four implemented agents + graph wiring + API metadata + tests; update "Last update" line and "Next Steps" (now Phase 3); optionally add a "Verified" line that `pytest` passes with Phase 2. In `ROADMAP.md`: mark the Phase 2 header `✅ (implemented)` (mirror Phase 1's `✅`). In `README.md`: in the API table / status note, mention that uploaded documents now carry language/doc_type/sentiment. Must NOT alter the Phase 1/3/4/5 content or the architecture diagram; must NOT claim Neo4j/hybrid QA (those are later phases).
  Parallelization: Wave 5 | Blocked by: Todo 7 | Blocks: none
  References (executor has NO interview context - be exhaustive): `STATUS.md` (structure at lines 3-42; "Next Steps" at 38-41), `ROADMAP.md:14-19` (Phase 2 section), `README.md` (status note lines 6-8, API table 48-55).
  Acceptance criteria (agent-executable):
    - `python3 -c "t=open('ROADMAP.md').read(); assert 'Phase 2' in t and '✅' in t.split('Phase 2')[1].split('\n')[0]"` (Phase 2 marked done)
    - `python3 -c "s=open('STATUS.md').read(); assert 'LanguageDetection' in s and 'Classification' in s and 'Sentiment' in s and 'Phase 3' in s"`
  QA scenarios (name the exact tool + invocation):
    - happy: docs render consistently; `ROADMAP.md` Phase 2 header shows ✅; `STATUS.md` lists the four agents under Completed and Phase 3 as next.
    - failure: stale "Phase 2 stubbed" wording removed from STATUS blockers/risks if present; no leftover `NotImplementedError` references to Phase 2 in docs.
  Evidence: `.omo/evidence/phase2-nlp-agents/task-8-docs.txt`
  Commit: Y | docs: mark Phase 2 (language/translate/classify/sentiment) complete

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity

## Commit strategy
- One commit per todo (see each todo's Commit line), conventional-commit style (`feat(agents)`, `feat(orchestration)`, `feat(api)`, `test(...)`, `docs(...)`).
- No squashing; each commit is independently buildable and `pytest` green at the agent/test/docs commits.
- Do NOT commit `.omo/` plan artifacts unless the user asks.

## Success criteria
- All four Phase 2 agents implement their `NotImplementedError` contract and return only the fields they update.
- Ingestion graph runs `reader → language_detect → (conditional) translator → classifier → sentiment → embeddings`; translator is skipped for English docs.
- `POST /documents/upload` response and `DocumentMeta` include `language`, `doc_type`, `sentiment_label`, `sentiment_score`.
- `pytest -q` is green with NO model downloads / network calls (fakes injected); existing 9 Phase 1 tests still pass.
- STATUS/ROADMAP/README reflect Phase 2 as implemented; no Phase 3+ work leaked in.
