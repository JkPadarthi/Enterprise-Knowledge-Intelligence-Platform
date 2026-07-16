---
slug: phase2-nlp-agents
status: awaiting-approval
intent: clear
review_required: false
pending-action: write .omo/plans/phase2-nlp-agents.md
approach: Implement Roadmap Phase 2: four NLP agents (LanguageDetection, conditional Translation, Classification, Sentiment) as pure BaseAgent subclasses already stubbed in the repo, wire them into the LangGraph ingestion graph after reader (with conditional translate edge), surface doc_type/sentiment/language on DocumentMeta in the ingest API, add unit tests using injected fakes + the existing conftest fixtures, and update STATUS/ROADMAP/README. Reuse established patterns: lazy heavy imports, DI (settings/llm/embedder injected), AgentState fields + Settings already present, deps already in pyproject.toml.
---

# Draft: phase2-nlp-agents

## Context
A prior approved plan (`graphrag-engine`, status: approved) built the full scaffold and Phase 1
(reader → embeddings → vector QA) end-to-end. Phase 1 is verified complete: `agents/reader.py`,
`agents/embeddings.py`, `agents/qa.py`, `orchestration/pipeline.py` (reader → embeddings), the
FastAPI service, and the pytest suite all exist and pass. Phase 2–5 agents remain `NotImplementedError`
stubs. This plan covers ONLY Roadmap Phase 2.

## Components (topology ledger)
| id | outcome (one line) | status | evidence path |
| --- | --- | --- | --- |
| langdetect-agent | LanguageDetectionAgent detects `state.language` + `state.language_confidence` via langdetect, runs in-process | active | agents/language_detect.py |
| translator-agent | TranslationAgent conditionally sets `state.translated_text` via LLMClient only when `language != translate_target_lang` | active | agents/translator.py, llm/ |
| classifier-agent | ClassificationAgent sets `state.doc_type` + `state.doc_type_scores` via zero-shot transformers pipeline | active | agents/classifier.py, config/settings.py |
| sentiment-agent | SentimentAgent sets `state.sentiment_label` + `state.sentiment_score` via DistilBERT pipeline | active | agents/sentiment.py |
| orchestration | Extended LangGraph ingestion graph: reader → language_detect → conditional translate → classifier → sentiment → embeddings | active | orchestration/pipeline.py |
| api | Ingest endpoint copies doc_type/sentiment/language into DocumentMeta + returns them | active | api/routes/ingest.py, api/deps.py |
| tests | Unit tests per Phase 2 agent (fakes/injected LLM, no model downloads) + pipeline test updated | active | tests/agents/test_*.py, tests/test_pipeline.py |
| docs | STATUS/ROADMAP/README updated to mark Phase 2 done | active | STATUS.md, ROADMAP.md, README.md |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Heavy-model loading | Lazy import inside `_get_*` helpers, cached on the instance (mirror EmbeddingAgent) | Avoids import-time torch/transformers cost; matches repo convention | yes |
| Fake for unit tests | Inject a `pipeline`-like callable / monkeypatch `transformers.pipeline`; no real model download in tests | Repo rule: "no model downloads in tests"; conftest provides no transformers fake yet — plan adds them | yes |
| Conditional translate edge | LangGraph conditional edge on `state.language != translate_target_lang`; else skip translator node, fall through to classifier | ROADMAP "Insert into the LangGraph pipeline after reader" + conditional translate is a known Phase 2/4 concern; Phase 1 stub message already says conditional | yes |
| Text the later agents read | language/translate/read raw_text; classifier + sentiment read `state.translated_text or state.raw_text` (same fallback EmbeddingAgent uses) | Consistency with embeddings agent | yes |
| classifier/sentiment run on short input | Truncate input to a cap (e.g. first ~2000 chars) before the pipeline to bound latency | transformer pipelines are slow on long text; safe default | yes |

## Findings (cited - path:lines)
- Phase 2 stubs already exist with `NotImplementedError` carrying the exact intended contract:
  `agents/language_detect.py:16` (set `state.language`/`language_confidence`),
  `agents/translator.py:22` (conditional, `state.translated_text`),
  `agents/classifier.py:16` (set `state.doc_type`/`doc_type_scores`),
  `agents/sentiment.py:16` (set `state.sentiment_label`/`sentiment_score`).
- `AgentState` already has every Phase 2 field with safe defaults:
  `models/schema.py:111-120` (`language`, `language_confidence`, `translated_text`, `doc_type`,
  `doc_type_scores`, `sentiment_label`, `sentiment_score`).
- `DocumentMeta` already has `language`, `doc_type`, `sentiment_label`, `sentiment_score`
  (`models/schema.py:60-63`).
- `Settings` already has `translate_target_lang`, `classifier_model`, `doc_type_labels`
  (→ `doc_type_label_list` property), `sentiment_model` (`config/settings.py:50-55, 65-68`).
- `pyproject.toml:22-24` already lists `langdetect`, `transformers`, `torch`. No new dependency.
- `BaseAgent` pattern (`agents/base.py`) gives logging + retry; `run(state, **deps) -> dict`.
- `EmbeddingAgent._get_embedder` (`agents/embeddings.py:75-80`) is the reference for lazy,
  instance-cached model loading. `run` reads `state.translated_text or state.raw_text`
  (`agents/embeddings.py:92`) — the fallback convention to mirror.
- `LLMClient` abstract + `get_llm_client(role, settings)` factory (`llm/client.py`, `llm/__init__.py`)
  — translator uses `get_llm_client("worker", settings)` (or an injected `llm` for tests),
  `acomplete(messages)`, mirroring `QAOrchestrator._get_llm`.
- `orchestration/pipeline.py:20-42` builds `reader → embeddings`; `run_ingest` injects
  `embedder`/`vector_store`. Phase 2 must insert nodes between reader and embeddings and add the
  conditional translate edge, while keeping `embedder`/`vector_store` injection working.
- `api/routes/ingest.py:36-57` already copies `state.language` and `state.num_pages` into
  `DocumentMeta` and returns `chunk_ids`; needs `doc_type`/`sentiment_*` added. `DOCUMENT_REGISTRY`
  in `api/deps.py:13` is the in-memory store (Phase 3 → Neo4j).
- `tests/conftest.py` provides `settings`, `embedder` (FakeEmbedder), `chroma_store`, `mock_llm`
  fixtures. Phase 2 tests add transformers fakes. Existing tests:
  `tests/agents/test_reader.py`, `test_embeddings.py`, `test_qa.py`, `tests/test_pipeline.py`,
  `tests/llm/test_mock.py`. `pytest` config: `asyncio_mode = "auto"` (`pyproject.toml:62`).

## Decisions
1. Each Phase 2 agent is a pure `BaseAgent` subclass: lazy-loads its model in a `_get_*` helper,
   cached on the instance, reads the relevant text from `state`, returns only the fields it updates.
2. `TranslationAgent` is conditional: if `state.language == settings.translate_target_lang` it
   returns `{}` and leaves `translated_text=None`; otherwise it calls the LLM with a translation
   prompt and sets `translated_text`. LLM injectable (`llm=`) or via factory.
3. `ClassificationAgent` / `SentimentAgent` use `transformers.pipeline(...)` (zero-shot for
   classifier using `settings.doc_type_label_list`; text-classification for sentiment using
   `settings.sentiment_model`). Input truncated to a cap before the pipeline.
4. `LanguageDetectionAgent` uses `langdetect.detect_langs(text)` for confidence; falls back to
   `state.language` default ("en") on empty text / detection error (append to `state.errors`).
5. Graph wiring: insert `language_detect` after `reader`; add a conditional edge to `translator`
   (only when non-target language) else straight to `classifier`; `classifier → sentiment →
   embeddings`. `run_ingest` signature unchanged (still injects embedder/vector_store); agents
   pull `settings` they were constructed with.
6. `api/routes/ingest.py` copies `state.doc_type`, `state.sentiment_label`, `state.sentiment_score`
   into `DocumentMeta` and returns them in the upload response.
7. Tests inject fakes (a callable standing in for `transformers.pipeline`; a stub LLM / MockBackend
   for translator) so no model downloads or network occur. `pytest` green required.

## Scope IN
- Four Phase 2 agents implemented per ROADMAP, conditional translation, LangGraph wiring with
  conditional translate edge, DocumentMeta enrichment in ingest API, unit tests, STATUS/ROADMAP/README update.

## Scope OUT (Must NOT have)
- NER / relation extraction / Neo4j graph build (Phase 3).
- Full LangGraph StateGraph hybrid QA + SummaryAgent (Phase 4).
- Streamlit dashboard (Phase 5).
- New dependencies / new env vars (all already present).
- Real model downloads or network calls in tests.
- Changing AgentState / DocumentMeta / Settings field shapes (already sufficient).

## Open questions
- None blocking. All reversible design choices defaulted per repo conventions above.

## Approval gate
status: awaiting-approval
- Approach recorded above. Pending user approval to write `.omo/plans/phase2-nlp-agents.md`.
