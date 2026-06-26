"""LLM factory — OpenAI / Ollama with optional LangSmith tracing."""

import logging
import os

from langchain_core.language_models import BaseChatModel

from backend.config import settings

logger = logging.getLogger(__name__)

# ── LangSmith tracing setup ───────────────────────────────────────────────────
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"]     = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"]    = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"]     = settings.LANGSMITH_PROJECT
    logger.info("LangSmith tracing ENABLED  project=%s", settings.LANGSMITH_PROJECT)

try:
    from langsmith import traceable
except ImportError:
    traceable = lambda *a, **k: lambda f: f  # noqa: E731

if not settings.LANGCHAIN_TRACING_V2:
    traceable = lambda *a, **k: lambda f: f  # noqa: E731


class LLMFactory:
    """Return a LangChain ChatModel for the requested provider."""

    @staticmethod
    def get_llm(provider: str | None = None) -> BaseChatModel:
        provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
        logger.info("LLMFactory: provider=%s", provider)

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in .env")
            return ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY)

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.2", base_url=settings.OLLAMA_BASE_URL)

        raise ValueError(f"Unknown LLM provider: {provider!r} — use openai | ollama")
