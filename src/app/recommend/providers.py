"""
providers.py — ILLMProvider adapters.

LangChain lives HERE and nowhere else in the domain. Each real adapter wraps a
LangChain chat model and exposes the plain `complete()` port. Heavy SDKs are
imported lazily inside the constructor so that importing this module (e.g. for
the FakeProvider in tests) never requires langchain or a running model.

    OllamaProvider     — local, offline, reproducible default (LangChain ChatOllama)
    AnthropicProvider  — hosted Claude via LangChain ChatAnthropic
    FakeProvider       — pure-Python canned responses for offline unit tests
"""
from __future__ import annotations

import logging

from . import config as C

log = logging.getLogger(__name__)


class OllamaProvider:
    """Local LLM via LangChain's ChatOllama. Requires a running Ollama daemon
    and a pulled model (`ollama pull llama3.1`)."""

    def __init__(self, model: str = C.OLLAMA_MODEL, temperature: float = C.TEMPERATURE) -> None:
        import os
        self.name = f"ollama:{model}"
        try:
            from langchain_ollama import ChatOllama  # preferred package
        except ImportError:  # pragma: no cover - fallback for older installs
            from langchain_community.chat_models import ChatOllama  # type: ignore
        # OLLAMA_BASE_URL lets the container reach an Ollama host (e.g. in Docker,
        # http://host.docker.internal:11434); defaults to the local daemon.
        base = os.getenv("OLLAMA_BASE_URL")
        extra = {"base_url": base} if base else {}
        self._chat = ChatOllama(model=model, temperature=temperature, **extra)

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return msg.content if hasattr(msg, "content") else str(msg)


class AnthropicProvider:
    """Hosted Claude via LangChain's ChatAnthropic. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str = C.ANTHROPIC_MODEL, temperature: float = C.TEMPERATURE) -> None:
        self.name = f"anthropic:{model}"
        from langchain_anthropic import ChatAnthropic
        self._chat = ChatAnthropic(
            model=model, temperature=temperature, max_tokens=C.MAX_TOKENS
        )

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return msg.content if hasattr(msg, "content") else str(msg)


class OpenRouterProvider:
    """OpenRouter via LangChain's ChatOpenAI (OpenRouter is OpenAI-compatible).
    Needs OPENROUTER_API_KEY. Use any OpenRouter model slug (free-tier slugs end
    in ':free', e.g. 'meta-llama/llama-3.3-70b-instruct:free')."""

    def __init__(self, model: str = C.OPENROUTER_MODEL, temperature: float = C.TEMPERATURE) -> None:
        import os
        self.name = f"openrouter:{model}"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from langchain_openai import ChatOpenAI
        self._chat = ChatOpenAI(
            model=model, api_key=api_key, base_url=C.OPENROUTER_BASE_URL,
            temperature=temperature, max_tokens=C.MAX_TOKENS,
            extra_body={"reasoning": C.OPENROUTER_REASONING,},
        )

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return msg.content if hasattr(msg, "content") else str(msg)


class FakeProvider:
    """Deterministic canned provider for offline tests / demos (no LangChain, no
    network). Returns whatever JSON string it is constructed with, so a test can
    exercise the full parse -> guard -> render pipeline without a real model."""

    def __init__(self, response: str, name: str = "fake") -> None:
        self.name = name
        self._response = response

    def complete(self, system: str, user: str) -> str:  # noqa: D401
        return self._response


def build_provider(kind: str = C.DEFAULT_PROVIDER, **kwargs):
    """Factory: return a configured provider by name ('ollama'|'anthropic'|'openrouter')."""
    kind = (kind or C.DEFAULT_PROVIDER).lower()
    if kind == "ollama":
        return OllamaProvider(**kwargs)
    if kind == "anthropic":
        return AnthropicProvider(**kwargs)
    if kind == "openrouter":
        return OpenRouterProvider(**kwargs)
    raise ValueError(f"unknown provider {kind!r}; expected 'ollama', 'anthropic', or 'openrouter'")
