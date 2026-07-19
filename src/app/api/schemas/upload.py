"""upload.py — /upload response (parsed blood report for review)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    biomarkers: dict[str, float] = Field(..., description="Recognised biomarker code → value.")
    demographics: dict = Field(default_factory=dict, description="Any demographics found in the file.")
    recognised: int = Field(..., description="Number of biomarker values recognised.")
    notes: list[str] = Field(default_factory=list,
                             description="Human-readable parse notes; review before calling /predict or /recommend.")
    filename: str
