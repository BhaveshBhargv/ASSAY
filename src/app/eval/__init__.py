"""
eval — Phase 9 end-to-end system evaluation.

Four evaluators, one report:
  ml_eval   — accuracy / precision / recall / F1 / ROC-AUC (+ SHAP, importance)
  rag_eval  — Retrieval Precision@K / Recall@K / context relevance
  llm_eval  — faithfulness / groundedness / hallucination rate

Run: `python scripts/evaluate_system.py`
"""
from __future__ import annotations

__all__ = ["ml_eval", "rag_eval", "llm_eval", "config"]
