"""Biomarker metadata for the UI, re-exported from app.ingestion.catalog."""
from __future__ import annotations

from app.ingestion.catalog import (
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
    """Name, unit and tier of a biomarker, for the form labels."""
    return biomarker_meta(code)
