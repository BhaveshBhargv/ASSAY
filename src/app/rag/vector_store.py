"""
vector_store.py — FAISS vector store adapter (implements IVectorStore).

Uses IndexFlatIP over L2-normalised vectors => exact cosine similarity. Ideal for
a small, fixed guideline corpus (exact search, zero infra, fully reproducible).
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .ports import IVectorStore


class FaissVectorStore(IVectorStore):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        vector = np.ascontiguousarray(vector.reshape(1, -1), dtype="float32")
        k = min(k, self._index.ntotal)
        if k == 0:
            return []
        scores, ids = self._index.search(vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def save(self, path) -> None:
        faiss.write_index(self._index, str(Path(path)))

    def load(self, path) -> None:
        self._index = faiss.read_index(str(Path(path)))
        self.dim = self._index.d
