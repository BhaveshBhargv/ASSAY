"""Parses an uploaded report for /upload."""
from __future__ import annotations

import logging

from app.ingestion.report_parser import parse_upload

from ..core.errors import ParsingError
from ..schemas.upload import UploadResponse

log = logging.getLogger("assay.api")

_FAIL_HINTS = ("Unsupported file type", "Could not read")


class UploadService:
    def parse(self, filename: str, data: bytes) -> UploadResponse:
        res = parse_upload(filename, data)
        if not res.biomarkers and any(h in n for n in res.notes for h in _FAIL_HINTS):
            raise ParsingError(res.notes[0] if res.notes else "Could not parse the file.",
                               detail={"filename": filename})
        log.info("parsed %s: %d biomarker(s)", filename, len(res.biomarkers))
        return UploadResponse(
            biomarkers=res.biomarkers, demographics=res.demographics,
            recognised=len(res.biomarkers), notes=res.notes, filename=filename,
        )
