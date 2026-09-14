"""Base classes for the embedder and vector store, so either can be swapped out."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IEmbedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of normalised embeddings."""


class IVectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[int]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(id, score), ...] for the k nearest passages."""

    @abstractmethod
    def save(self, path) -> None: ...

    @abstractmethod
    def load(self, path) -> None: ...
