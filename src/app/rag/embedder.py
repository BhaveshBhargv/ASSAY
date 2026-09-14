"""Sentence-transformer embeddings for the guideline index.

Vectors are L2-normalised, so inner-product search in FAISS gives cosine
similarity.
"""
from __future__ import annotations

import logging

import numpy as np

from .ports import IEmbedder

log = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(IEmbedder):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # slow import, so only when needed
        log.info("loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")
