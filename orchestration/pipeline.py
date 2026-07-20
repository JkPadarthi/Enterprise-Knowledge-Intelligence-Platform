"""LangGraph orchestration of the agent pipeline.

Phase 2 ships the *ingestion* graph through sentiment. Phase 3 extends it with NER
and a parallel branch that writes the knowledge graph (Neo4j) alongside the vector
embeddings (ChromaDB). The graph is built from injected dependencies so it is fully
testable without real models or stores.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from agents.classifier import ClassificationAgent
from agents.embeddings import EmbeddingAgent
from agents.graph_builder import KnowledgeGraphAgent
from agents.language_detect import LanguageDetectionAgent
from agents.ner import NERAgent
from agents.reader import ReaderAgent
from agents.sentiment import SentimentAgent
from agents.translator import TranslationAgent
from models.schema import AgentState, ExecutionStep


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _timed(state: AgentState, agent_name: str, coro):
    """Time an agent coroutine and return (result, [ExecutionStep]).

    The step list is returned (not mutated onto state) so each node can emit it via the
    LangGraph `execution_log` reducer channel, which concatenates contributions from the
    parallel embeddings/graph branch. Exceptions propagate unchanged so the pipeline keeps
    its existing fail-loud behavior (a failed ingest still 500s).
    """
    started = time.monotonic()
    sa = _now_iso()
    result = await coro
    step = ExecutionStep(
        order=0,  # renumbered after collection; see run_ingest
        agent=agent_name,
        started_at=sa,
        ended_at=_now_iso(),
        duration_ms=round((time.monotonic() - started) * 1000, 2),
        status="success",
        detail="",
    )
    return result, [step]


def build_ingest_graph(
    settings: Any,
    *,
    embedder: Any | None = None,
    vector_store: Any | None = None,
    graph_store: Any | None = None,
):
    """Compile the Phase-3 ingestion graph:

    ``reader -> language_detect -> [conditional] translator -> classifier ->
    sentiment -> ner -> {embeddings (ChromaDB), graph (Neo4j)} -> END``
    """
    builder = StateGraph(AgentState)

    # Instantiate agents
    reader = ReaderAgent(settings)
    language_detect = LanguageDetectionAgent(settings)
    translator = TranslationAgent(settings)
    classifier = ClassificationAgent(settings)
    sentiment = SentimentAgent(settings)
    ner = NERAgent(settings)
    embeddings = EmbeddingAgent(settings, embedder=embedder, vector_store=vector_store)
    graph_builder = KnowledgeGraphAgent(settings, graph_store=graph_store)

    # Define node functions. Each returns its agent result merged with a one-element
    # execution_log list; the LangGraph reducer channel concatenates all of them.
    async def reader_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "reader", reader.run(state))
        return {**res, "execution_log": step}

    async def language_detect_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "language_detect", language_detect.run(state))
        return {**res, "execution_log": step}

    async def translator_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "translator", translator.run(state))
        return {**res, "execution_log": step}

    async def classifier_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "classifier", classifier.run(state))
        return {**res, "execution_log": step}

    async def sentiment_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "sentiment", sentiment.run(state))
        return {**res, "execution_log": step}

    async def ner_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "ner", ner.run(state))
        return {**res, "execution_log": step}

    async def embeddings_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "embeddings", embeddings.run(state, vector_store=vector_store))
        return {**res, "execution_log": step}

    async def graph_node(state: AgentState) -> dict[str, Any]:
        res, step = await _timed(state, "graph", graph_builder.run(state, graph_store=graph_store))
        return {**res, "execution_log": step}

    # Add nodes
    builder.add_node("reader", reader_node)
    builder.add_node("language_detect", language_detect_node)
    builder.add_node("translator", translator_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("ner", ner_node)
    builder.add_node("embeddings", embeddings_node)
    builder.add_node("graph", graph_node)

    # Define edges
    builder.set_entry_point("reader")
    builder.add_edge("reader", "language_detect")

    # Conditional edge: translate if language != target language
    def should_translate(state: AgentState) -> str:
        # If language is not the target language (default "en"), translate
        target_lang = getattr(settings, "translate_target_lang", "en")
        if state.language != target_lang:
            return "translate"
        return "skip"

    builder.add_conditional_edges(
        "language_detect",
        should_translate,
        {
            "translate": "translator",
            "skip": "classifier",
        },
    )

    builder.add_edge("translator", "classifier")
    builder.add_edge("classifier", "sentiment")
    builder.add_edge("sentiment", "ner")
    # After NER, branch into the two parallel index writes.
    builder.add_edge("ner", "embeddings")
    builder.add_edge("ner", "graph")
    builder.add_edge("embeddings", END)
    builder.add_edge("graph", END)

    return builder.compile()


async def run_ingest(
    pdf_bytes: bytes,
    filename: str,
    doc_id: str,
    settings: Any,
    *,
    embedder: Any | None = None,
    vector_store: Any | None = None,
    graph_store: Any | None = None,
) -> AgentState:
    """Run the ingestion graph end-to-end and return the final state."""
    graph = build_ingest_graph(settings, embedder=embedder, vector_store=vector_store, graph_store=graph_store)
    initial = AgentState(doc_id=doc_id, filename=filename, file_bytes=pdf_bytes)
    result = await graph.ainvoke(initial)
    if not isinstance(result, AgentState):
        result = AgentState(**result)
    # Renumber steps sequentially (nodes emit order=0; the reducer concatenates them
    # in execution order, but parallel branches may interleave — sort by start time).
    result.execution_log.sort(key=lambda s: s.started_at)
    for i, step in enumerate(result.execution_log):
        step.order = i + 1
    return result