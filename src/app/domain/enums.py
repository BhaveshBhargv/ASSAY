"""Enums used by the clinical rule engine."""
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    """Clinical status of a single biomarker."""
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    BORDERLINE = "borderline"
    SEVERE = "severe"


class Severity(str, Enum):
    """Overall severity used when results are combined."""
    NORMAL = "normal"
    BORDERLINE = "borderline"
    SERIOUS = "serious"


class Direction(str, Enum):
    """Which side of the reference range a value falls on."""
    LOW = "low"
    IN_RANGE = "in_range"
    HIGH = "high"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.BORDERLINE: 1,
    Severity.SERIOUS: 2,
}
RANK_TO_SEVERITY: dict[int, Severity] = {v: k for k, v in SEVERITY_RANK.items()}


def max_severity(items: list[Severity]) -> Severity:
    """Highest severity in the list, or normal if it's empty."""
    if not items:
        return Severity.NORMAL
    return RANK_TO_SEVERITY[max(SEVERITY_RANK[s] for s in items)]
