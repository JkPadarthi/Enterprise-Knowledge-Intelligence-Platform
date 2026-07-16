"""Translation agent (Phase 2 — not yet implemented).

When built, it will run *conditionally*: only translate when ``state.language``
differs from ``TRANSLATE_TARGET_LANG`` ("en"). Translation goes through the
provider-agnostic ``LLMClient``, so a local NLLB/MarianMT backend can be added
later without touching this agent.
"""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from config.settings import Settings
from llm import get_llm_client
from models.schema import AgentState


class TranslationAgent(BaseAgent):
    """Translates non-English documents into the target language via the LLMClient."""

    name = "translator"

    def __init__(
        self,
        settings: Settings | None = None,
        logger: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        super().__init__(settings, logger)
        self._llm = llm

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        from llm import get_llm_client

        return get_llm_client("worker", self.settings)

    async def run(self, state: AgentState, **deps: Any) -> dict[str, Any]:
        # If the text is empty or only whitespace, or the language is already the target, skip.
        if not state.raw_text.strip() or state.language == Settings().translate_target_lang:
            return {}

        # Build the translation prompt.
        target_lang = Settings().translate_target_lang
        system_msg = f"You are a precise translator. Translate the following document text into {target_lang}. Return ONLY the translated text, with no commentary."
        user_msg = state.raw_text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Call the LLM.
        llm = self._get_llm()
        translated_text = await llm.acomplete(messages)

        # Update the state and return the result.
        state.translated_text = translated_text
        return {"translated_text": translated_text}