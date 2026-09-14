"""Retrieval quality for the evaluation report.

For each test query this computes Precision@K, Recall@K, the mean similarity
score of the top K and the share of the top K tagged with a queried biomarker.
Relevance comes from the corpus's own biomarker tags rather than human labels.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.rag.retriever import GuidelineRetriever, build_query

from . import config as C

log = logging.getLogger("assay.eval")


def _relevant_total(retriever: GuidelineRetriever, codes: set[str]) -> int:
    return sum(1 for c in retriever._chunks if set(c.biomarkers) & codes)  # noqa: SLF001


def evaluate_rag(retriever: GuidelineRetriever | None = None,
                 ks: tuple[int, ...] = C.RAG_KS) -> dict:
    retriever = retriever or GuidelineRetriever.load()
    kmax = max(ks)
    per_query = []

    for q in C.RAG_QUERIES:
        codes = set(q["relevant_codes"])
        total_rel = _relevant_total(retriever, codes)
        query = build_query(q["severity"], q["flagged"])
        results = retriever.retrieve(query, k=kmax, biomarker_boost=tuple(codes))

        rel_flags = [bool(set(r.chunk.biomarkers) & codes) for r in results]
        scores = [float(r.score) for r in results]

        row = {"name": q["name"], "total_relevant": total_rel}
        for k in ks:
            hits = sum(rel_flags[:k])
            row[f"P@{k}"] = hits / k
            row[f"R@{k}"] = (hits / total_rel) if total_rel else 0.0
        row["context_score"] = sum(scores[:kmax]) / len(scores[:kmax]) if scores else 0.0
        row["tag_overlap"] = sum(rel_flags[:kmax]) / len(rel_flags[:kmax]) if rel_flags else 0.0
        per_query.append(row)

    agg = {"n_queries": len(per_query)}
    for k in ks:
        agg[f"precision@{k}"] = _mean(per_query, f"P@{k}")
        agg[f"recall@{k}"] = _mean(per_query, f"R@{k}")
    agg["context_relevance_score"] = _mean(per_query, "context_score")
    agg["context_tag_overlap"] = _mean(per_query, "tag_overlap")

    _plot_pr_at_k(agg, ks)
    log.info("RAG: P@1=%.2f P@3=%.2f R@3=%.2f ctx=%.2f",
             agg.get("precision@1", 0), agg.get("precision@3", 0),
             agg.get("recall@3", 0), agg["context_relevance_score"])
    return {"aggregate": agg, "per_query": per_query}


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def _plot_pr_at_k(agg: dict, ks: tuple[int, ...]) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prec = [agg[f"precision@{k}"] for k in ks]
    rec = [agg[f"recall@{k}"] for k in ks]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ks, prec, "-o", color="#0E7C86", label="Precision@K")
    ax.plot(ks, rec, "-o", color="#C8871B", label="Recall@K")
    for k, p, r in zip(ks, prec, rec):
        ax.text(k, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(k, r - 0.05, f"{r:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(ks))
    ax.set_xlabel("K")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Retrieval Precision@K / Recall@K "
                 f"(context relevance {agg['context_relevance_score']:.2f})", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "rag_precision_recall_at_k.png", dpi=140)
    plt.close(fig)
