"""
enums.py — controlled vocabularies for the clinical rule engine.

Kept in the domain layer with no external dependencies (Clean Architecture).
"""
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    """Per-biomarker clinical status (the five requested categories)."""
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    BORDERLINE = "borderline"
    SEVERE = "severe"


class Severity(str, Enum):
    """Aggregate severity taxonomy feeding Fusion (Stage 3)."""
    NORMAL = "normal"
    BORDERLINE = "borderline"
    SERIOUS = "serious"


class Direction(str, Enum):
    """Which side of the reference range a value falls on."""
    LOW = "low"          # below range
    IN_RANGE = "in_range"
    HIGH = "high"        # above range


# Ordinal ranks for max/priority aggregation.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.BORDERLINE: 1,
    Severity.SERIOUS: 2,
}
RANK_TO_SEVERITY: dict[int, Severity] = {v: k for k, v in SEVERITY_RANK.items()}


def max_severity(items: list[Severity]) -> Severity:
    """Safety-dominant merge: the highest severity wins."""
    if not items:
        return Severity.NORMAL
    return RANK_TO_SEVERITY[max(SEVERITY_RANK[s] for s in items)]
