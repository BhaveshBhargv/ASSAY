"""Creates LLM providers for the API."""
from __future__ import annotations

import logging

from ..core.errors import LLMUnavailableError

log = logging.getLogger("assay.api")


class LLMRepository:
    """Creates providers on demand.

    Creating a provider doesn't connect to anything, so a new one is made for
    each request and connection problems show up when generating.
    """

    def provider(self, kind: str, model: str | None = None):
        from app.recommend.providers import build_provider
        try:
            kwargs = {"model": model} if model else {}
            return build_provider(kind, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailableError(
                f"Could not initialise provider '{kind}': {exc}",
                detail={"provider": kind, "model": model}) from exc
