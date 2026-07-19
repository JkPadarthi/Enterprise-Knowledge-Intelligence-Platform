"""Tests for the KnowledgeGraphAgent (Phase 3)."""

from __future__ import annotations

import asyncio

from agents.graph_builder import KnowledgeGraphAgent
from config.settings import Settings
from models.schema import AgentState, Entity, Relation


class FakeGraphStore:
    """Records idempotent writes so we can assert on graph-build behaviour."""

    def __init__(self) -> None:
        self.written: dict[str, tuple[list, list]] = {}

    async def replace_document_graph(self, doc_id: str, entities: list, relationships: list) -> None:
        self.written[doc_id] = (entities, relationships)


def test_graph_writes_idempotent_subgraph():
    """Agent forwards entities/relations to an idempotent store write."""
    store = FakeGraphStore()
    agent = KnowledgeGraphAgent(Settings(), graph_store=store)
    state = AgentState(
        doc_id="d1",
        entities=[Entity(id="e0", text="Acme", label="ORGANIZATION", doc_id="d1")],
        relationships=[Relation(id="r0", subject="Acme", relation="IN", object="Paris", doc_id="d1")],
    )
    result = asyncio.run(agent.run(state, graph_store=store))
    assert result["graph_written"] is True
    assert "d1" in store.written
    ents, rels = store.written["d1"]
    assert len(ents) == 1
    assert len(rels) == 1


def test_graph_skips_when_nothing_to_write():
    """With no entities/relations the agent reports graph_written=False."""
    store = FakeGraphStore()
    agent = KnowledgeGraphAgent(Settings(), graph_store=store)
    state = AgentState(doc_id="d2")
    result = asyncio.run(agent.run(state, graph_store=store))
    assert result["graph_written"] is False
    assert store.written == {}
