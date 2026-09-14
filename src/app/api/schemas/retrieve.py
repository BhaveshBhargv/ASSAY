"""Request and response models for /retrieve."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class FlaggedIn(BaseModel):
    code: str = ""
    name: str = ""
    status: str = ""


class RetrieveRequest(BaseModel):
    query: Optional[str] = Field(None, description="Free-text query. If omitted, one is built from severity + flagged.")
    severity: Optional[Literal["normal", "borderline", "serious"]] = None
    flagged: list[FlaggedIn] = Field(default_factory=list,
                                     description="Flagged markers to bias retrieval toward.")
    k: int = Field(6, ge=1, le=20, description="Number of passages to return.")

    @model_validator(mode="after")
    def _need_a_signal(self):
        if not self.query and not self.severity and not self.flagged:
            raise ValueError("provide a query, or a severity and/or flagged markers")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "severity": "serious",
                "flagged": [{"code": "hba1c_pct", "name": "HbA1c", "status": "borderline"},
                            {"code": "hdl_mgdl", "name": "HDL cholesterol", "status": "low"}],
                "k": 5,
            }
        }
    }


class PassageOut(BaseModel):
    id: str = Field(..., description="Prompt-local evidence id (E1, E2, …).")
    score: float
    source: str
    code: str
    title: str
    citation: str
    text: str
    biomarkers: list[str] = Field(default_factory=list)
    url: str = ""


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The query actually used (echoed for transparency).")
    count: int
    passages: list[PassageOut]
