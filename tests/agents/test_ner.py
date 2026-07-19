"""Tests for the NER agent (Phase 3)."""

from __future__ import annotations

import asyncio

from agents.ner import NERAgent, _build_relation_prompt
from config.settings import Settings
from models.schema import AgentState
from llm.mock import MockBackend


class FakeGlinerExtractor:
    """Deterministic fake GLiNER extractor returning canned spans."""

    def __init__(self, spans: list[dict]) -> None:
        self._spans = spans
        self.calls: list[tuple[str, list[str]]] = []

    def predict_entities(self, text: str, labels: list[str]) -> list[dict]:
        self.calls.append((text, labels))
        return self._spans


def test_ner_happy_path_with_relations():
    """Entities from GLiNER + relations from LLM populate state."""
    spans = [
        {"text": "Acme Corp", "label": "ORGANIZATION", "start": 0, "end": 9},
        {"text": "Paris", "label": "LOCATION", "start": 20, "end": 25},
    ]
    extractor = FakeGlinerExtractor(spans)
    relations = [{"subject": "Acme Corp", "relation": "HEADQUARTERED_IN", "object": "Paris"}]
    llm = MockBackend(json_response=relations)
    agent = NERAgent(Settings(), extractor=extractor, llm=llm)

    state = AgentState(doc_id="d1", raw_text="Acme Corp is based in Paris.")
    result = asyncio.run(agent.run(state))

    assert len(result["entities"]) == 2
    assert result["entities"][0].text == "Acme Corp"
    assert all(e.doc_id == "d1" for e in result["entities"])
    assert len(result["relationships"]) == 1
    assert result["relationships"][0].relation == "HEADQUARTERED_IN"


def test_ner_empty_text_skips():
    """Whitespace-only text yields no entities and no relations."""
    extractor = FakeGlinerExtractor([])
    agent = NERAgent(Settings(), extractor=extractor, llm=MockBackend())
    state = AgentState(raw_text="   ")
    result = asyncio.run(agent.run(state))
    assert result == {}


def test_ner_relation_json_malformed_is_skipped():
    """Malformed LLM JSON is logged and skipped, not raised."""
    extractor = FakeGlinerExtractor([{"text": "X", "label": "PERSON"}])
    llm = MockBackend(text_response="not json at all")
    agent = NERAgent(Settings(), extractor=extractor, llm=llm)
    state = AgentState(doc_id="d2", raw_text="X did something.")
    result = asyncio.run(agent.run(state))
    assert len(result["entities"]) == 1
    assert result["relationships"] == []


def test_relation_prompt_shape():
    """Prompt carries entities and asks for a JSON array."""
    from models.schema import Entity

    entities = [Entity(id="e0", text="Acme", label="ORGANIZATION", doc_id="d")]
    prompt = _build_relation_prompt(entities, "text")
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "Acme" in prompt[1]["content"]
