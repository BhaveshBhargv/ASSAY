"""
build_rag_index.py — Phase 5 entrypoint: build the FAISS guideline index.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --query "high cholesterol and low HDL"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.rag.ingest import build_index          # noqa: E402
from app.rag.retriever import GuidelineRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the RAG guideline index")
    ap.add_argument("--query", help="run a sample search after building")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    stats = build_index()
    print("\n=== RAG index built ===")
    print(f"chunks : {stats['n_chunks']}")
    print(f"model  : {stats['model']} (dim={stats['dim']})")
    print(f"sources: {', '.join(stats['sources'])}")

    if args.query:
        r = GuidelineRetriever.load()
        print(f"\n=== top {args.k} for: {args.query!r} ===")
        for res in r.retrieve(args.query, k=args.k):
            c = res.chunk
            print(f"[{res.score:.3f}] {c.citation()} — {c.title}")
            print(f"        {c.text[:120].strip()}...")


if __name__ == "__main__":
    main()
