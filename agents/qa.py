"""QA orchestrator — basic vector RAG (Phase 1).

Retrieves the top-k nearest chunks from ChromaDB for a question, builds a
grounded prompt, and asks the LLM to answer with inline ``[chunk_id]`` citations.
Phase 4 extends this into *hybrid* retrieval by also querying Neo4j and merging
graph context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from agents.base import BaseAgent
from config.settings import Settings
from models.schema import AgentState, Citation

_SYSTEM_PROMPT = (
    "You are a precise document-question answering assistant. "
    "Use ONLY the retrieved context to answer. Cite the supporting chunk using "
    "the format [chunk_id] next to each fact. If the answer is not in the context, "
    "say you don't know. Be concise."
)


class Embedder(Protocol):
    """Minimal embedder protocol: map texts -> float matrix."""

    def encode(self, texts: list[str]) -> Any:  # noqa: D102
        ...


class QAOrchestrator(BaseAgent):
    """Answers questions over indexed documents using vector retrieval + LLM."""

    name = "qa"

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
        embedder: Embedder | None = None,
        vector_store: Any | None = None,
        graph_store: Any | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._graph_store = graph_store

    def _get_llm(self, role: str = "qa") -> Any:
        if self._llm is not None:
            return self._llm
        from llm import get_llm_client

        return get_llm_client(role, self.settings)

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _get_store(self, vector_store: Any | None) -> Any:
        if vector_store is not None:
            return vector_store
        if self._vector_store is not None:
            return self._vector_store
        from vector.chroma_store import ChromaStore

        return ChromaStore(self.settings)

    def _get_graph_store(self, graph_store: Any | None) -> Any:
        if graph_store is not None:
            return graph_store
        if self._graph_store is not None:
            return self._graph_store
        from graph.neo4j_store import Neo4jStore

        return Neo4jStore(self.settings)

    async def answer(
        self,
        question: str,
        *,
        doc_id: str | None = None,
        top_k: int = 5,
        **deps: Any,
    ) -> dict[str, Any]:
        store = self._get_store(deps.get("vector_store"))
        embedder = self._get_embedder()
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None, lambda: embedder.encode([question])[0].tolist()
        )

        where = {"doc_id": doc_id} if doc_id else None
        results = await loop.run_in_executor(None, store.query, query_embedding, top_k, where)

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        context_parts: list[str] = []
        citations: list[Citation] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=True):
            context_parts.append(f"[{cid}]\n{doc}")
            citations.append(
                Citation(
                    chunk_id=cid,
                    source="vector",
                    text_excerpt=(doc or "")[:200],
                    score=(1.0 - dist) if dist is not None else None,
                    node_ref=(meta or {}).get("doc_id"),
                )
            )

        context = "\n\n".join(context_parts)
        
        # Hybrid graph retrieval
        graph_store = self._get_graph_store(deps.get("graph_store"))
        opened_here = False
        try:
            if graph_store is not None:
                # Connect if needed
                if getattr(graph_store, "_driver", None) is None and hasattr(graph_store, "connect"):
                    await graph_store.connect()
                    opened_here = True

                # Build Cypher query
                if doc_id is not None:
                    cypher = (
                        "MATCH (s:Entity)-[r]->(o:Entity) WHERE s.doc_id=$doc_id "
                        "RETURN s.text AS subject, r.relation AS relation, o.text AS object "
                        "LIMIT 1000"
                    )
                    params = {"doc_id": doc_id}
                else:
                    cypher = (
                        "MATCH (s:Entity)-[r]->(o:Entity) "
                        "RETURN s.text AS subject, r.relation AS relation, o.text AS object "
                        "LIMIT 1000"
                    )
                    params = {}

                rows = await graph_store.query_graph(cypher, params)

                fact_lines = []
                for row in rows:
                    subject = row.get("subject", "")
                    relation = row.get("relation", "")
                    obj = row.get("object", "")
                    fact = f"{subject} {relation} {obj}".strip()
                    if fact:  # only add non-empty facts
                        fact_lines.append(fact)
                        # Create a citation for this graph fact
                        citations.append(
                            Citation(
                                chunk_id="",  # graph citations don't have chunk_id
                                source="graph",
                                text_excerpt=fact[:200],
                                score=None,
                                node_ref=subject,  # using subject as node_ref
                            )
                        )

                if fact_lines:
                    context += "\n\n## Knowledge graph\n" + "\n".join(fact_lines)

        finally:
            if opened_here and hasattr(graph_store, "close"):
                await graph_store.close()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        answer = await self._get_llm("qa").acomplete(messages)

        self._log("answered question (top_k=%d, citations=%d)", logging.INFO, top_k, len(citations))
        return {"qa_answer": answer, "citations": citations}

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        """LangGraph entry point: answers ``state.qa_question`` if present."""
        if not state.qa_question:
            return {}
        return await self.answer(state.qa_question, doc_id=state.doc_id or None, **deps)