"""LangGraph orchestration of the agent pipeline.

Phase 2 ships the *ingestion* graph through sentiment. Phase 3 extends it with NER
and a parallel branch that writes the knowledge graph (Neo4j) alongside the vector
embeddings (ChromaDB). The graph is built from injected dependencies so it is fully
testable without real models or stores.
"""

from __future__ import annotations

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
from models.schema import AgentState


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

    # Define node functions
    async def reader_node(state: AgentState) -> dict[str, Any]:
        return await reader.run(state)

    async def language_detect_node(state: AgentState) -> dict[str, Any]:
        return await language_detect.run(state)

    async def translator_node(state: AgentState) -> dict[str, Any]:
        return await translator.run(state)

    async def classifier_node(state: AgentState) -> dict[str, Any]:
        return await classifier.run(state)

    async def sentiment_node(state: AgentState) -> dict[str, Any]:
        return await sentiment.run(state)

    async def ner_node(state: AgentState) -> dict[str, Any]:
        return await ner.run(state)

    async def embeddings_node(state: AgentState) -> dict[str, Any]:
        return await embeddings.run(state, vector_store=vector_store)

    async def graph_node(state: AgentState) -> dict[str, Any]:
        return await graph_builder.run(state, graph_store=graph_store)

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
    if isinstance(result, AgentState):
        return result
    return AgentState(**result)