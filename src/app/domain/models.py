"""Dataclasses for the rule set and the rule engine's results.

A threshold is either a single number or a {"male": ..., "female": ...} map,
resolved against the patient's sex when a value is classified.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from .enums import Direction, Severity, Status

Threshold = Union[float, dict]


def resolve_threshold(value: Optional[Threshold], sex: Optional[str], default: float) -> float:
    if value is None:
        return default
    if isinstance(value, dict):
        if sex not in value:
            raise ValueError(f"sex-specific threshold requires sex in {list(value)}, got {sex!r}")
        return float(value[sex])
    return float(value)


@dataclass(frozen=True)
class Guideline:
    source: str
    code: str


@dataclass(frozen=True)
class Band:
    status: Status
    severity: Severity
    min: Optional[Threshold] = None  # inclusive, None means no lower bound
    max: Optional[Threshold] = None  # exclusive, None means no upper bound
    direction: Direction = Direction.IN_RANGE
    urgent: Optional[bool] = None  # defaults to status == SEVERE
    guideline: Optional[Guideline] = None
    interpretation: str = ""

    def is_urgent(self) -> bool:
        return self.urgent if self.urgent is not None else self.status == Status.SEVERE

    def contains(self, value: float, sex: Optional[str]) -> bool:
        lo = resolve_threshold(self.min, sex, -math.inf)
        hi = resolve_threshold(self.max, sex, math.inf)
        return lo <= value < hi


@dataclass(frozen=True)
class BiomarkerRule:
    code: str
    name: str
    unit: str
    bands: tuple[Band, ...]
    category: str = "blood"
    tier: str = "actionable"  # or "flag_only"
    label_role: str = "core"  # "core", "secondary" or "none"
    reference_range_low: Optional[Threshold] = None
    reference_range_high: Optional[Threshold] = None

    def reference_range(self, sex: Optional[str]) -> dict:
        return {
            "low": resolve_threshold(self.reference_range_low, sex, -math.inf),
            "high": resolve_threshold(self.reference_range_high, sex, math.inf),
        }


@dataclass(frozen=True)
class RuleSet:
    version: str
    rules: dict[str, BiomarkerRule]
    secondary_borderline_min: int = 2

    def get(self, code: str) -> Optional[BiomarkerRule]:
        return self.rules.get(code)


@dataclass(frozen=True)
class PatientContext:
    sex: Optional[str] = None  # "male" or "female"
    age: Optional[int] = None


@dataclass
class BiomarkerResult:
    code: str
    name: str
    category: str
    tier: str
    label_role: str
    value: float
    unit: str
    status: Status
    direction: Direction
    severity: Severity
    urgent: bool
    reference_range: dict
    interpretation: str
    guideline: Optional[Guideline]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["direction"] = self.direction.value
        d["severity"] = self.severity.value
        return d


@dataclass
class RuleEngineResult:
    ruleset_version: str
    context: PatientContext
    biomarkers: list[BiomarkerResult] = field(default_factory=list)
    unknown_codes: list[str] = field(default_factory=list)
    secondary_borderline_min: int = 2

    @property
    def overall_severity(self) -> Severity:
        # Only actionable markers count towards the grade. Flag-only markers can
        # still trigger an urgent referral, but leaving them out here keeps this
        # the same as the training label definition.
        actionable = [b for b in self.biomarkers if b.tier == "actionable"]
        if any(b.severity == Severity.SERIOUS for b in actionable):
            return Severity.SERIOUS
        core_borderline = any(
            b.severity == Severity.BORDERLINE and b.label_role == "core"
            for b in actionable
        )
        secondary_borderline = sum(
            1 for b in actionable
            if b.severity == Severity.BORDERLINE and b.label_role == "secondary"
        )
        if core_borderline or secondary_borderline >= self.secondary_borderline_min:
            return Severity.BORDERLINE
        return Severity.NORMAL

    @property
    def urgent_referral(self) -> bool:
        return any(b.urgent for b in self.biomarkers)

    @property
    def flagged(self) -> list[str]:
        return [b.code for b in self.biomarkers if b.severity != Severity.NORMAL]

    @property
    def recommendation_targets(self) -> list[str]:
        """Abnormal actionable markers, which get lifestyle advice."""
        return [b.code for b in self.biomarkers
                if b.severity != Severity.NORMAL and b.tier == "actionable"]

    @property
    def clinician_signpost(self) -> list[str]:
        """Abnormal flag-only markers, which are referred to a clinician instead."""
        return [b.code for b in self.biomarkers
                if b.severity != Severity.NORMAL and b.tier == "flag_only"]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.biomarkers:
            counts[b.status.value] = counts.get(b.status.value, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "ruleset_version": self.ruleset_version,
            "context": {"age": self.context.age, "sex": self.context.sex},
            "biomarkers": [b.to_dict() for b in self.biomarkers],
            "summary": {
                "overall_severity": self.overall_severity.value,
                "urgent_referral": self.urgent_referral,
                "flagged": self.flagged,
                "recommendation_targets": self.recommendation_targets,
                "clinician_signpost": self.clinician_signpost,
                "status_counts": self.status_counts(),
                "unknown_codes": self.unknown_codes,
            },
        }
