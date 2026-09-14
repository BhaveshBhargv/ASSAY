"""Semantic search over the guideline index.

Passages tagged with one of the patient's flagged biomarkers get a small score
boost, so they rank above more general matches.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import config as C
from .embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from .ingest import load_chunks
from .models import GuidelineChunk, RetrievalResult
from .ports import IEmbedder
from .vector_store import FaissVectorStore

log = logging.getLogger(__name__)


class GuidelineRetriever:
    def __init__(self, embedder: IEmbedder, store: FaissVectorStore,
                 chunks: list[GuidelineChunk]) -> None:
        self._embedder = embedder
        self._store = store
        self._chunks = chunks

    @classmethod
    def load(cls, index_dir: Path = C.INDEX_DIR,
             embedder: IEmbedder | None = None) -> "GuidelineRetriever":
        embedder = embedder or SentenceTransformerEmbedder(DEFAULT_MODEL)
        chunks = load_chunks(C.CHUNKS_PATH)
        store = FaissVectorStore(dim=embedder.dim)
        store.load(C.FAISS_PATH)
        return cls(embedder, store, chunks)

    def retrieve(
        self, query: str, k: int = 5, biomarker_boost: tuple[str, ...] = (),
        boost: float = 0.05, pool: int = 20,
    ) -> list[RetrievalResult]:
        """Top k passages for the query, boosting ones tagged with biomarker_boost."""
        qvec = self._embedder.embed([query])[0]
        hits = self._store.search(qvec, k=max(pool, k))
        results = []
        for idx, score in hits:
            chunk = self._chunks[idx]
            if biomarker_boost and set(chunk.biomarkers) & set(biomarker_boost):
                score += boost
            results.append(RetrievalResult(chunk=chunk, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def retrieve_for_assessment(
        self, severity: str, flagged: list[dict], k: int = 5,
    ) -> list[RetrievalResult]:
        """Search using the assessment. flagged is a list of {code, name, status} dicts."""
        query = build_query(severity, flagged)
        codes = tuple(f.get("code", "") for f in flagged)
        return self.retrieve(query, k=k, biomarker_boost=codes)

    @property
    def size(self) -> int:
        return len(self._chunks)


def build_query(severity: str, flagged: list[dict]) -> str:
    if flagged:
        parts = []
        for f in flagged:
            name = f.get("name") or f.get("code", "")
            status = f.get("status", "")
            parts.append(f"{status} {name}".strip())
        markers = "; ".join(parts)
        # Start with the flagged markers rather than naming a condition, otherwise
        # a liver or anaemia query gets pulled towards cardiometabolic passages.
        return (f"Lifestyle guidance for {markers}. "
                f"Evidence-based advice for {severity} health risk.")
    return f"Evidence-based lifestyle recommendations for {severity} health risk."
