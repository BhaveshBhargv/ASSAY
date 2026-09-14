"""Response model for GET /model-info."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BiomarkerInfo(BaseModel):
    code: str
    name: str
    unit: str
    tier: str
    panel: str


class RuleEngineInfo(BaseModel):
    ruleset_version: str
    biomarker_count: int
    actionable_count: int
    flag_only_count: int


class RandomForestInfo(BaseModel):
    available: bool
    algorithm: Optional[str] = None
    classes: list[str] = Field(default_factory=list)
    n_features: Optional[int] = None
    feature_names: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class RagInfo(BaseModel):
    embedder: str
    dim: Optional[int] = None
    index_size: int


class ModelInfoResponse(BaseModel):
    app: str
    version: str
    rule_engine: RuleEngineInfo
    random_forest: RandomForestInfo
    rag: RagInfo
    default_provider: str
    biomarkers: list[BiomarkerInfo]
    disclaimer: str
