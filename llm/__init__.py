"""LLM client factory.

Selects a backend from ``LLM_PROVIDER`` and picks the model for the requested role
(``worker`` vs ``qa``). Backend imports are deferred into :func:`get_llm_client` so
importing the ``llm`` package never pulls in heavy SDKs unnecessarily.
"""

from __future__ import annotations

from llm.client import LLMClient


def get_llm_client(role: str = "worker", settings=None) -> LLMClient:
    """Build an :class:`LLMClient` for ``role`` ("worker" | "qa").

    Args:
        role: Which model slot to use - ``worker`` (fast) or ``qa`` (strong).
        settings: Optional pre-built Settings; fetched via ``get_settings`` if omitted.

    Returns:
        A configured LLM backend instance.

    Raises:
        ValueError: If ``LLM_PROVIDER`` names an unknown backend.
    """
    from config.settings import get_settings

    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    model = settings.qa_model if role == "qa" else settings.worker_model

    if provider == "openrouter":
        from llm.openrouter import OpenRouterBackend

        return OpenRouterBackend(model=model, api_key=settings.openrouter_api_key)
    if provider == "ollama":
        from llm.ollama import OllamaBackend

        return OllamaBackend(model=model, base_url=settings.ollama_base_url)
    if provider == "mock":
        from llm.mock import MockBackend

        return MockBackend(model=model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected openrouter|ollama|mock)")


__all__ = ["LLMClient", "get_llm_client"]

