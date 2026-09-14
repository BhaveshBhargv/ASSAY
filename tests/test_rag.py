"""RAG tests.

The retrieval test needs the built index and the embedding model, and is skipped
if the index isn't there.

Run:  python tests/test_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.rag import config as C  # noqa: E402
from app.rag.loaders import load_curated_yaml  # noqa: E402
from app.rag.retriever import build_query  # noqa: E402
from app.rag.splitter import RecursiveTextSplitter  # noqa: E402


def test_splitter_short_text_single_chunk():
    s = RecursiveTextSplitter(chunk_size=600, chunk_overlap=100)
    assert s.split("A short passage.") == ["A short passage."]


def test_splitter_long_text_chunks_with_bound():
    s = RecursiveTextSplitter(chunk_size=120, chunk_overlap=20)
    text = ("Sentence one is here. " * 40)
    chunks = s.split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + 20 for c in chunks)  # allow for the overlap


def test_curated_corpus_loads_with_provenance():
    chunks = load_curated_yaml(C.CURATED_PATH)
    assert len(chunks) >= 25
    assert all(c.source in {"NICE", "NHS", "WHO"} for c in chunks)
    assert all(c.text for c in chunks)
    assert all(c.citation() for c in chunks)


def test_build_query_mentions_flags():
    q = build_query("serious", [
        {"code": "hba1c_pct", "name": "HbA1c", "status": "high"},
        {"code": "hdl_mgdl", "name": "HDL cholesterol", "status": "low"},
    ])
    assert "serious" in q and "HbA1c" in q and "HDL" in q


def test_retrieval_if_index_exists():
    if not C.FAISS_PATH.exists():
        print("SKIP retrieval (index not built)")
        return
    from app.rag.retriever import GuidelineRetriever
    r = GuidelineRetriever.load()
    res = r.retrieve("high cholesterol diet advice", k=3)
    assert len(res) == 3
    assert res[0].score >= res[-1].score
    # a cholesterol query should bring up a lipid passage first
    top_bm = set(res[0].chunk.biomarkers)
    assert top_bm & {"total_chol_mgdl", "ldl_mgdl", "hdl_mgdl"}
    print("retrieval OK — top:", res[0].chunk.citation(), "-", res[0].chunk.title)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
