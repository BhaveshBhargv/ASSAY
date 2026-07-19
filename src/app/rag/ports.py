"""
ports.py — abstract interfaces (Dependency Inversion).

The ingestion & retrieval logic depends only on these ports; concrete adapters
(SentenceTransformer, FAISS) are injected. Swap the embedder or the vector store
without touching the RAG logic (Liskov / Open-Closed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IEmbedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings."""


class IVectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[int]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(id, score), ...] for the top-k nearest passages."""

    @abstractmethod
    def save(self, path) -> None: ...

    @abstractmethod
    def load(self, path) -> None: ...
