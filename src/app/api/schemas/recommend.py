"""recommend.py — /recommend request/response (full grounded pipeline)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .common import EXAMPLE_BIOMARKERS, AssessmentOut, Demographics, validate_biomarkers


class RecommendRequest(BaseModel):
    demographics: Demographics
    biomarkers: dict[str, float]
    provider: Literal["ollama", "anthropic", "openrouter"] = Field(
        "ollama", description="LLM provider (behind the port). 'openrouter' needs OPENROUTER_API_KEY.")
    model: Optional[str] = Field(None, description="Override the model name (else the provider default).")
    k: int = Field(6, ge=1, le=20, description="Guideline passages to ground on.")
    llm_recheck: bool = Field(False, description="Run the extra LLM groundedness recheck (slower).")

    _v = field_validator("biomarkers")(validate_biomarkers)

    model_config = {
        "json_schema_extra": {
            "example": {
                "demographics": {"age": 54, "sex": "male"},
                "biomarkers": EXAMPLE_BIOMARKERS,
                "provider": "ollama", "k": 6,
            }
        }
    }


class AdviceItemOut(BaseModel):
    advice: str
    rationale: str
    evidence: list[str]


class ReportOut(BaseModel):
    explanation: str
    lifestyle: list[AdviceItemOut] = Field(default_factory=list)
    diet: list[AdviceItemOut] = Field(default_factory=list)
    exercise: list[AdviceItemOut] = Field(default_factory=list)
    sleep: list[AdviceItemOut] = Field(default_factory=list)
    weight_management: list[AdviceItemOut] = Field(default_factory=list)
    hydration: list[AdviceItemOut] = Field(default_factory=list)
    smoking: list[AdviceItemOut] = Field(default_factory=list)
    alcohol: list[AdviceItemOut] = Field(default_factory=list)
    follow_up: list[AdviceItemOut] = Field(default_factory=list)


class EvidenceRefOut(BaseModel):
    id: str
    citation: str
    title: str
    text: str


class GroundednessOut(BaseModel):
    groundedness: float
    total_items: int
    grounded_items: int
    dropped_items: list[dict] = Field(default_factory=list)
    diagnostic_violations: list[str] = Field(default_factory=list)
    passed: bool


class RecommendResponse(BaseModel):
    provider: str
    assessment: AssessmentOut
    report: ReportOut
    evidence: list[EvidenceRefOut]
    groundedness: GroundednessOut
    urgent_note: str = ""
    clinician_note: str = ""
    disclaimer: str
