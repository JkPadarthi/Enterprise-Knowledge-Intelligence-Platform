"""Document classification agent (Phase 2 — not yet implemented)."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from config.settings import Settings
from models.schema import AgentState


class ClassificationAgent(BaseAgent):
    """Classifies document type with a zero-shot transformer (e.g. bart-large-mnli)."""

    name = "classifier"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._pipe: Any = None  # Injected for tests; lazy-loaded otherwise

    def _get_pipeline(self):
        """Lazy-load zero-shot classification pipeline or return injected mock."""
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline(
            "zero-shot-classification",
            model=self.settings.classifier_model,
        )
        return self._pipe

    @staticmethod
    def _truncate(text: str, pipe: Any) -> str:
        """Truncate to the model's token limit (with a 2-token margin for special tokens).

        bart-large caps at 512 tokens; a raw char cap can still overflow the tokenizer
        and raise at inference. Token-based truncation avoids that.
        """
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return text[:1000]
        max_tokens = getattr(tokenizer, "model_max_length", 512) or 512
        limit = max(1, max_tokens - 2)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            return text
        return tokenizer.decode(ids[:limit], skip_special_tokens=True)

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        text = (state.translated_text or state.raw_text).strip()
        if not text:
            self._log("no text to classify; skipping", level=logging.INFO)
            return {}

        pipe = self._get_pipeline()
        text = self._truncate(text, pipe)
        if not text:
            return {}

        labels = self.settings.doc_type_label_list
        result = pipe(text, candidate_labels=labels)

        # result: {"sequence": str, "labels": [str], "scores": [float]}
        # labels are sorted descending by score
        state.doc_type = result["labels"][0]
        state.doc_type_scores = {
            label: score for label, score in zip(result["labels"], result["scores"])
        }
        self._log(
            "classified as %s (scores: %s)",
            logging.INFO,
            state.doc_type,
            state.doc_type_scores,
        )
        return {
            "doc_type": state.doc_type,
            "doc_type_scores": state.doc_type_scores,
        }