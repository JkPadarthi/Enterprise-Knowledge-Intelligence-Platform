"""Sentiment analysis agent (Phase 2)."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from config.settings import Settings
from models.schema import AgentState


class SentimentAgent(BaseAgent):
    """Computes sentiment with a DistilBERT pipeline and sets ``state.sentiment_*``."""

    name = "sentiment"

    def __init__(self, settings: Settings | None = None, logger: Any | None = None) -> None:
        super().__init__(settings, logger)
        self._pipe: Any = None  # Injected for tests; lazy-loaded otherwise

    def _get_pipeline(self):
        """Lazy-load sentiment pipeline or return injected mock."""
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline("text-classification", model=self.settings.sentiment_model)
        return self._pipe

    @staticmethod
    def _truncate(text: str, pipe: Any) -> str:
        """Truncate to the model's token limit (with a 2-token margin for special tokens).

        DistilBERT caps at 512 tokens; a raw char cap (e.g. 2000) can still overflow the
        tokenizer and raise at inference. Token-based truncation avoids that.
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
            self._log("no text to analyze sentiment; skipping", level=logging.INFO)
            return {}

        pipe = self._get_pipeline()
        text = self._truncate(text, pipe)
        if not text:
            return {}
        result = pipe(text)
        # result: [{"label": str, "score": float}]
        state.sentiment_label = result[0]["label"]
        state.sentiment_score = float(result[0]["score"])
        self._log("sentiment %s (%.3f)", logging.INFO, state.sentiment_label, state.sentiment_score)
        return {"sentiment_label": state.sentiment_label, "sentiment_score": state.sentiment_score}
