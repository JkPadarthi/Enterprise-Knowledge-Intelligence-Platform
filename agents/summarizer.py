"""Summarization agent (Phase 4).

Produces an extractive summary (centrality/lead over chunk sentences, computed
with the stdlib only) and an abstractive summary (LLM over concatenated text),
returning both on ``state.summary``.
"""

from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent
from models.schema import AgentState


class SummaryAgent(BaseAgent):
    """Produces extractive (TextRank) + abstractive (LLM) summaries."""

    name = "summarizer"

    def __init__(
        self,
        settings=None,
        logger=None,
        llm=None,
        embedder=None,
    ) -> None:
        super().__init__(settings, logger)
        self._llm = llm
        self._embedder = embedder

    def _get_llm(self, role: str = "summary") -> Any:
        if self._llm is not None:
            return self._llm
        from llm import get_llm_client

        return get_llm_client(role, self.settings)

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        # Get text to summarize
        text = (state.translated_text or state.raw_text or "").strip()
        if not text:
            return {}

        # Extractive summary (TextRank-like)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            extractive = ""
        else:
            # Get embeddings for each sentence
            embedder = self._get_embedder()
            embeddings = embedder.encode(sentences).tolist()
            n = len(embeddings)
            # Compute similarity matrix (cosine similarity) using pure Python
            similarity = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                    norm_i = sum(a * a for a in embeddings[i]) ** 0.5
                    norm_j = sum(a * a for a in embeddings[j]) ** 0.5
                    if norm_i == 0.0 or norm_j == 0.0:
                        similarity[i][j] = 0.0
                    else:
                        similarity[i][j] = dot / (norm_i * norm_j)
            # Score each sentence by sum of similarities with all others
            scores = [sum(similarity[i]) for i in range(n)]
            # Get top k indices (k = min(3, len(sentences)))
            k = min(3, n)
            # Get indices of top k scores, in descending order of score
            top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            # Sort indices to preserve original order in the text
            top_indices_sorted = sorted(top_indices)
            # Build extractive summary
            extractive = " ".join([sentences[i] for i in top_indices_sorted])

        # Abstractive summary
        messages = [
            {"role": "system", "content": "You are a concise document summarizer. Write one short paragraph."},
            {"role": "user", "content": f"Summarize:\n{text}"},
        ]
        abstractive = await self._get_llm("summary").acomplete(messages)

        return {"summary": {"abstractive": abstractive, "extractive": extractive}}