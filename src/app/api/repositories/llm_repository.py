"""llm_repository.py — builds LLM providers (behind the ILLMProvider port)."""
from __future__ import annotations

import logging

from ..core.errors import LLMUnavailableError

log = logging.getLogger("assay.api")


class LLMRepository:
    """Factory for LLM providers. Construction is cheap (no network), so a fresh
    provider is built per request; connection failures surface at generation time."""

    def provider(self, kind: str, model: str | None = None):
        from app.recommend.providers import build_provider
        try:
            kwargs = {"model": model} if model else {}
            return build_provider(kind, **kwargs)
        except Exception as exc:  # noqa: BLE001 - e.g. missing SDK / bad config
            raise LLMUnavailableError(
                f"Could not initialise provider '{kind}': {exc}",
                detail={"provider": kind, "model": model}) from exc
