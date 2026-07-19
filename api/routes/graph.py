"""Graph retrieval endpoint (Phase 3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path

from api.deps import get_settings_dep, get_graph_store_connected
from config.settings import Settings
from graph.neo4j_store import Neo4jStore

router = APIRouter(prefix="/documents", tags=["graph"])


@router.get("/{doc_id}/graph")
async def get_document_graph(
    doc_id: str = Path(..., description="Document ID"),
    settings: Settings = Depends(get_settings_dep),
    store: Neo4jStore = Depends(get_graph_store_connected),
) -> dict:
    """Return the entity-relation subgraph for a given document ID."""
    entity_query = """
        MATCH (n:Entity {doc_id:$doc_id})
        RETURN n.text AS text, n.label AS label
    """
    relation_query = """
        MATCH (s:Entity {doc_id:$doc_id})-[r {doc_id:$doc_id}]->(o:Entity {doc_id:$doc_id})
        RETURN s.text AS subject, r.relation AS relation, o.text AS object
    """
    entity_result = await store.query_graph(entity_query, {"doc_id": doc_id})
    relation_result = await store.query_graph(relation_query, {"doc_id": doc_id})

    entities = [
        {"text": record["text"], "label": record["label"]}
        for record in entity_result
    ]
    relations = [
        {"subject": record["subject"], "relation": record["relation"], "object": record["object"]}
        for record in relation_result
    ]

    # Absence of data is a valid "not yet built" state: 200 with empty lists.
    graph_written = bool(entities) or bool(relations)

    return {
        "doc_id": doc_id,
        "entities": entities,
        "relations": relations,
        "graph_written": graph_written,
    }