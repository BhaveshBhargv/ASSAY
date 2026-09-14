"""Loads the guideline retriever."""
from __future__ import annotations

import logging

log = logging.getLogger("assay.api")


class RetrievalRepository:
    def __init__(self) -> None:
        from app.rag.embedder import DEFAULT_MODEL
        from app.rag.retriever import GuidelineRetriever
        self._retriever = GuidelineRetriever.load()
        self._embedder_name = DEFAULT_MODEL
        log.info("RAG retriever loaded (%d passages, embedder %s)",
                 self._retriever.size, self._embedder_name)

    @property
    def retriever(self):
        return self._retriever

    def info(self) -> dict:
        dim = None
        try:
            dim = self._retriever._embedder.dim  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        return {"embedder": self._embedder_name, "dim": dim, "index_size": self._retriever.size}
