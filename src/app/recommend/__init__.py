"""
recommend — Phase 6 LLM recommendation engine.

Turns a fused clinical assessment (rule engine + Random Forest) plus retrieved
guideline evidence into a grounded, non-diagnostic, patient-friendly lifestyle
recommendation report.

Public entrypoint:
    from app.recommend import RecommendationEngine
    engine = RecommendationEngine.build()          # local Ollama by default
    bundle = engine.recommend(demographics, biomarkers)
    print(bundle.render())
"""
from __future__ import annotations

from .engine import RecommendationEngine
from .fusion import FusedAssessment, fuse
from .report import AdviceItem, RecommendationReport
from .bundle import RecommendationBundle

__all__ = [
    "RecommendationEngine",
    "FusedAssessment",
    "fuse",
    "AdviceItem",
    "RecommendationReport",
    "RecommendationBundle",
]
