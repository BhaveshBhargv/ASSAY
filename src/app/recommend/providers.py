"""LLM providers for Ollama, Anthropic and OpenRouter, plus a fake one for tests.

The real providers wrap LangChain chat models. LangChain is imported inside each
constructor, so this module can be imported without it installed.
"""
from __future__ import annotations

import logging

from . import config as C

log = logging.getLogger(__name__)


class OllamaProvider:
    """A local model through Ollama. Needs the Ollama server running and the model pulled."""

    def __init__(self, model: str = C.OLLAMA_MODEL, temperature: float = C.TEMPERATURE) -> None:
        import os
        self.name = f"ollama:{model}"
        try:
            from langchain_ollama import ChatOllama
        except ImportError:  # pragma: no cover
            from langchain_community.chat_models import ChatOllama  # type: ignore
        # Set OLLAMA_BASE_URL to use an Ollama server other than the local one.
        base = os.getenv("OLLAMA_BASE_URL")
        extra = {"base_url": base} if base else {}
        self._chat = ChatOllama(model=model, temperature=temperature, **extra)

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return msg.content if hasattr(msg, "content") else str(msg)


class AnthropicProvider:
    """Claude through the Anthropic API. Needs ANTHROPIC_API_KEY."""

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
    """Any OpenRouter model, through its OpenAI-compatible API. Needs OPENROUTER_API_KEY."""

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
            extra_body={"reasoning": C.OPENROUTER_REASONING},
        )

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return msg.content if hasattr(msg, "content") else str(msg)


class FakeProvider:
    """Always returns the response it was created with. Used in tests."""

    def __init__(self, response: str, name: str = "fake") -> None:
        self.name = name
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


def build_provider(kind: str = C.DEFAULT_PROVIDER, **kwargs):
    """Create a provider by name: "ollama", "anthropic" or "openrouter"."""
    kind = (kind or C.DEFAULT_PROVIDER).lower()
    if kind == "ollama":
        return OllamaProvider(**kwargs)
    if kind == "anthropic":
        return AnthropicProvider(**kwargs)
    if kind == "openrouter":
        return OpenRouterProvider(**kwargs)
    raise ValueError(f"unknown provider {kind!r}; expected 'ollama', 'anthropic', or 'openrouter'")
