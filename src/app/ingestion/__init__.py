"""
ingestion — shared blood-report catalog and file parsing.

Lifted out of the Streamlit UI so the API and the dashboard use ONE source of
truth for biomarker panels, lab-report synonyms, and report parsing.
"""
from __future__ import annotations

from .catalog import (
    ALL_CODES,
    PANELS,
    PANEL_OF,
    SYNONYMS,
    biomarker_meta,
    reference_range,
    rule_for,
)
from .report_parser import ParseResult, parse_upload

__all__ = [
    "ALL_CODES", "PANELS", "PANEL_OF", "SYNONYMS",
    "biomarker_meta", "reference_range", "rule_for",
    "ParseResult", "parse_upload",
]
