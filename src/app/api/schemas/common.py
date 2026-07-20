"""
common.py — shared request/response models and validators.

`Demographics` and the biomarker-map validator are reused across endpoints so
validation rules live in exactly one place.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.ingestion.catalog import ALL_CODES

_CODES = set(ALL_CODES)

# Example panel reused in OpenAPI docs.
EXAMPLE_BIOMARKERS = {
    "hba1c_pct": 6.1, "total_chol_mgdl": 232, "hdl_mgdl": 34,
    "triglycerides_mgdl": 205, "alt": 46,
}


def validate_biomarkers(v: dict) -> dict[str, float]:
    if not v:
        raise ValueError("provide at least one biomarker value")
    unknown = sorted(set(v) - _CODES)
    if unknown:
        raise ValueError(f"unknown biomarker code(s): {unknown}; see GET /model-info for valid codes")
    bad = sorted(k for k, val in v.items() if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0)
    if bad:
        raise ValueError(f"biomarker values must be positive numbers: {bad}")
    return {k: float(val) for k, val in v.items()}


class Demographics(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Adult age in years.")
    sex: Literal["male", "female"] = Field(..., description="Biological sex (drives sex-specific ranges).")
    eth_code: Optional[int] = Field(None, ge=1, le=7, description="NHANES RIDRETH3 ethnicity code (model input).")
    pir: Optional[float] = Field(None, ge=0, le=5, description="Income-to-poverty ratio (model input).")
    educ_code: Optional[int] = Field(None, ge=1, le=5, description="Education code (model input).")

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class BiomarkerResultOut(BaseModel):
    code: str
    name: str
    unit: str
    value: float
    status: str = Field(..., description="normal | low | high | borderline | severe")
    direction: str
    severity: str = Field(..., description="normal | borderline | serious")
    urgent: bool
    tier: str = Field(..., description="actionable | flag_only")
    reference_range: dict
    interpretation: str


class MarkerOut(BaseModel):
    code: str
    name: str
    status: str
    severity: str
    interpretation: str


class RiskDriverOut(BaseModel):
    feature: str = Field(..., description="Model feature column that drove the read.")
    label: str = Field(..., description="Patient-friendly label for the feature.")
    value: Optional[float] = Field(None, description="The patient's value for this feature.")
    contribution: float = Field(..., description="Positive SHAP contribution toward the predicted class.")


class AssessmentOut(BaseModel):
    severity: str = Field(..., description="Fused overall severity (safety-dominant max of rules and model).")
    rule_severity: str
    rf_severity: Optional[str] = Field(None, description="Random Forest predicted class (null if unavailable).")
    rf_probabilities: dict = Field(default_factory=dict)
    escalated_by_rf: bool = Field(..., description="True when the model raised risk above the rules (hidden-pattern signal).")
    rf_drivers: list[RiskDriverOut] = Field(
        default_factory=list,
        description="Per-patient SHAP drivers of the model's read (only for a raised-risk prediction).")
    urgent_referral: bool
    flagged: list[MarkerOut] = Field(..., description="Actionable abnormal markers → lifestyle advice.")
    signpost: list[MarkerOut] = Field(..., description="Flag-only abnormal markers → discuss with clinician.")
    biomarkers: list[BiomarkerResultOut] = Field(..., description="Every evaluated biomarker.")


class ErrorBody(BaseModel):
    type: str
    message: str
    detail: Optional[object] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str


def _bm_field():
    return Field(..., description="Map of biomarker code → positive value.",
                 json_schema_extra={"example": EXAMPLE_BIOMARKERS})
