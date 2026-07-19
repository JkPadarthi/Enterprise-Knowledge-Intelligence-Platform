"""NER + relation extraction agent (Phase 3).

Extracts entities with GLiNER (labels from ``settings.ner_label_list``) and
relations with the provider-agnostic ``LLMClient``. Both the GLiNER extractor and
the LLM are injectable so the agent is unit-testable without downloading models or
hitting the network. Relation extraction is also guarded: if the LLM returns
malformed JSON it is logged and skipped rather than crashing ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from config.settings import Settings
from llm import get_llm_client
from models.schema import AgentState, Entity, Relation


def _build_relation_prompt(entities: list[Entity], text: str) -> list[dict[str, str]]:
    """Prompt the LLM to find relations among the extracted entity texts."""
    entity_lines = "\n".join(f"- {e.text} ({e.label})" for e in entities)
    return [
        {
            "role": "system",
            "content": (
                "You are a relation-extraction engine. Given the entities and the "
                "source text, return a JSON array of relations. Each item must have "
                "'subject', 'relation', and 'object' keys, where subject/object are "
                "exact entity texts from the list. Only output the JSON array, no prose."
            ),
        },
        {
            "role": "user",
            "content": f"Entities:\n{entity_lines}\n\nText:\n{text}\n\nRelations (JSON array):",
        },
    ]


class NERAgent(BaseAgent):
    """Extracts entities (GLiNER) and relations (LLM) into ``state.entities``/``state.relationships``."""

    name = "ner"

    def __init__(
        self,
        settings: Settings | None = None,
        logger: Any | None = None,
        extractor: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        super().__init__(settings, logger)
        self._extractor = extractor
        self._llm = llm

    def _get_extractor(self) -> Any:
        if self._extractor is not None:
            return self._extractor
        from gliner import GLiNER

        self._extractor = GLiNER.from_pretrained(self.settings.gliner_model)
        return self._extractor

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        return get_llm_client("worker", self.settings)

    def _extract_entities(self, text: str, labels: list[str]) -> list[Entity]:
        extractor = self._get_extractor()
        spans = extractor.predict_entities(text, labels)
        entities: list[Entity] = []
        for i, span in enumerate(spans):
            label = str(span.get("label", "MISC"))
            ent_text = str(span.get("text", "")).strip()
            if not ent_text:
                continue
            entities.append(
                Entity(
                    id=f"e{i}",
                    text=ent_text,
                    label=label,
                    start=span.get("start"),
                    end=span.get("end"),
                )
            )
        return entities

    async def _extract_relations(self, entities: list[Entity], text: str) -> list[Relation]:
        if not entities:
            return []
        llm = self._get_llm()
        try:
            raw = await llm.acomplete_json(_build_relation_prompt(entities, text))
        except Exception as exc:  # noqa: BLE001
            self._log("relation extraction failed: %s", logging.ERROR, exc)
            return []
        if not isinstance(raw, list):
            return []
        relations: list[Relation] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            relation = item.get("relation")
            obj = item.get("object")
            if not (subject and relation and obj):
                continue
            relations.append(
                Relation(id=f"r{i}", subject=str(subject), relation=str(relation), object=str(obj))
            )
        return relations

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        text = (state.translated_text or state.raw_text).strip()
        if not text:
            self._log("no text to extract entities; skipping", level=logging.INFO)
            return {}

        labels = self.settings.ner_label_list
        entities = self._extract_entities(text, labels)
        # Attach doc_id to entities for downstream graph writes.
        for ent in entities:
            ent.doc_id = state.doc_id

        relations = await self._extract_relations(entities, text)
        for rel in relations:
            rel.doc_id = state.doc_id

        self._log(
            "extracted %d entities, %d relations", logging.INFO, len(entities), len(relations)
        )
        return {"entities": entities, "relationships": relations}
