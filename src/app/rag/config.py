"""Shared paths for the RAG subsystem."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
CURATED_PATH = GUIDELINES_DIR / "curated_corpus.yaml"
PDF_DIR = GUIDELINES_DIR / "pdfs"
INDEX_DIR = Path(__file__).resolve().parent / "index"
FAISS_PATH = INDEX_DIR / "guidelines.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
