"""Language detection agent (Phase 2 — not yet implemented)."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from models.schema import AgentState


class LanguageDetectionAgent(BaseAgent):
    """Detects the document language with ``langdetect`` and sets ``state.language``."""

    name = "language_detect"

    def _get_detector(self):
        """Lazy-load langdetect detector and exception class."""
        if not hasattr(self, "_detect_langs"):
            from langdetect import detect_langs, LangDetectException

            self._detect_langs = detect_langs
            self._LangDetectException = LangDetectException

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        text = state.raw_text
        if not text.strip():
            # Empty or whitespace-only text: no language detection
            return {}

        self._get_detector()
        try:
            # detect_langs returns a list of LangDetect instances sorted by probability
            langs = self._detect_langs(text)
            if not langs:
                # Should not happen for non-empty text, but guard anyway
                return {}
            top = langs[0]
            lang_code = str(top.lang)
            confidence = float(top.prob)
            # Update state as instructed
            state.language = lang_code
            state.language_confidence = confidence
            return {"language": lang_code, "language_confidence": confidence}
        except self._LangDetectException as e:
            # On detection error, record the error and return empty dict
            state.errors.append(f"Language detection failed: {e}")
            return {}