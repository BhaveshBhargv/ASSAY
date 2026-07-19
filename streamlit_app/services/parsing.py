"""
parsing.py — re-export the shared report parser for the dashboard.

The implementation now lives in the engine (`app.ingestion.report_parser`) so the
API and the UI parse blood reports identically.
"""
from __future__ import annotations

from app.ingestion.report_parser import ParseResult, parse_upload

__all__ = ["ParseResult", "parse_upload"]
