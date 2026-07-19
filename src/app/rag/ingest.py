"""
ingest.py — build the FAISS index from the guideline corpus (offline, build-time).

Pipeline:  load (curated YAML + PDFs) -> embed -> FAISS index -> persist.
Run via scripts/build_rag_index.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config as C
from .embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from .loaders import load_all
from .models import GuidelineChunk
from .ports import IEmbedder
from .vector_store import FaissVectorStore

log = logging.getLogger(__name__)


def build_index(
    curated_path: Path = C.CURATED_PATH,
    pdf_dir: Path = C.PDF_DIR,
    index_dir: Path = C.INDEX_DIR,
    embedder: IEmbedder | None = None,
) -> dict:
    embedder = embedder or SentenceTransformerEmbedder(DEFAULT_MODEL)
    chunks = load_all(curated_path, pdf_dir)
    if not chunks:
        raise RuntimeError("no guideline chunks found to index")

    vectors = embedder.embed([c.text for c in chunks])
    store = FaissVectorStore(dim=embedder.dim)
    store.add(vectors, ids=list(range(len(chunks))))

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(C.FAISS_PATH)
    _save_chunks(chunks, C.CHUNKS_PATH, model_name=getattr(embedder, "model_name", "?"))

    log.info("indexed %d chunks (dim=%d) -> %s", len(chunks), embedder.dim, index_dir)
    return {
        "n_chunks": len(chunks),
        "dim": embedder.dim,
        "model": getattr(embedder, "model_name", "?"),
        "sources": sorted({c.source for c in chunks if c.source}),
    }


def _save_chunks(chunks: list[GuidelineChunk], path: Path, model_name: str) -> None:
    payload = {"model": model_name, "chunks": [c.to_dict() for c in chunks]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_chunks(path: Path = C.CHUNKS_PATH) -> list[GuidelineChunk]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for d in payload["chunks"]:
        out.append(GuidelineChunk(
            chunk_id=d["chunk_id"], text=d["text"], source=d["source"],
            code=d.get("code", ""), title=d.get("title", ""),
            biomarkers=tuple(d.get("biomarkers", ())),
            categories=tuple(d.get("categories", ())),
            severities=tuple(d.get("severities", ())),
            url=d.get("url", ""),
        ))
    return out
