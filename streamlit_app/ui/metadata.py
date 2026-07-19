"""
metadata.py — UI-facing biomarker metadata.

Thin adapter over the shared engine catalog (`app.ingestion.catalog`) so the
dashboard and the API share one source of truth. `display` keeps the field names
the form expects.
"""
from __future__ import annotations

from app.ingestion.catalog import (  # re-exported for the UI
    ALL_CODES,
    PANEL_OF,
    PANELS,
    SYNONYMS,
    biomarker_meta,
    reference_range,
    rule_for,
)

__all__ = ["ALL_CODES", "PANEL_OF", "PANELS", "SYNONYMS", "rule_for",
           "reference_range", "display"]


def display(code: str) -> dict:
    """Name/unit/tier for one biomarker (form label helper)."""
    return biomarker_meta(code)
