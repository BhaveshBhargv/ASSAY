"""Re-exports the report parser so the dashboard and the API use the same code."""
from __future__ import annotations

from app.ingestion.report_parser import ParseResult, parse_upload

__all__ = ["ParseResult", "parse_upload"]
