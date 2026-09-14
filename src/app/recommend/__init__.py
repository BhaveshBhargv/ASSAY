"""Generates grounded lifestyle recommendations from an assessment.

    from app.recommend import RecommendationEngine
    engine = RecommendationEngine.build()
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
