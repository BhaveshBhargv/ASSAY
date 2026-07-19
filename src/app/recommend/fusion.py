"""
fusion.py — Stage 3: combine the rule engine and Random Forest into ONE label.

Design (safety-dominant): the training label IS the weighted-core rule label, and
the RF is trained to predict it. The RF earns its place on the *novelty slice* —
cases where the individual biomarkers look normal/borderline but their pattern
signals higher risk. So fusion escalates, never de-escalates:

    final_severity = max(rule_severity, rf_severity)          # by ordinal rank

When the RF is strictly higher than the rules, `escalated_by_rf` is True — this is
the hidden-risk signal that drives the "why we still recommend caution" narrative.

The fused object also splits abnormal markers into:
    • recommendation_targets  — actionable markers → get lifestyle advice
    • clinician_signpost      — flag-only markers  → "discuss with clinician"
so the LLM is only ever asked to advise on markers that carry lifestyle actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..domain.enums import SEVERITY_RANK, Severity
from ..domain.models import RuleEngineResult


@dataclass
class FusedAssessment:
    severity: str                              # final fused label
    rule_severity: str
    rf_severity: Optional[str]
    rf_probabilities: dict = field(default_factory=dict)
    escalated_by_rf: bool = False
    urgent_referral: bool = False
    # markers the LLM will advise on: [{code, name, status, severity, interpretation}]
    flagged: list[dict] = field(default_factory=list)
    # flag-only abnormal markers → clinician signpost text, never lifestyle advice
    signpost: list[dict] = field(default_factory=list)

    @property
    def flagged_names(self) -> list[str]:
        return [f["name"] for f in self.flagged]

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "rule_severity": self.rule_severity,
            "rf_severity": self.rf_severity,
            "rf_probabilities": self.rf_probabilities,
            "escalated_by_rf": self.escalated_by_rf,
            "urgent_referral": self.urgent_referral,
            "flagged": self.flagged,
            "signpost": self.signpost,
        }


def _max_severity(a: str, b: str) -> str:
    """Ordinal max over the 3-class taxonomy (unknown labels treated as normal)."""
    ra = SEVERITY_RANK.get(Severity(a), 0) if a in Severity._value2member_map_ else 0
    rb = SEVERITY_RANK.get(Severity(b), 0) if b in Severity._value2member_map_ else 0
    return a if ra >= rb else b


def fuse(rule_result: RuleEngineResult, rf_output: Optional[dict]) -> FusedAssessment:
    """Merge rule-engine output with the RF prediction into one assessment.

    rf_output: dict from RiskModel.predict, or None to run rules-only.
    """
    rule_sev = rule_result.overall_severity.value
    rf_sev = rf_output.get("predicted_severity") if rf_output else None
    rf_probs = rf_output.get("probabilities", {}) if rf_output else {}

    final = _max_severity(rule_sev, rf_sev) if rf_sev else rule_sev
    escalated = bool(
        rf_sev
        and SEVERITY_RANK.get(Severity(rf_sev), 0) > SEVERITY_RANK.get(Severity(rule_sev), 0)
    )

    flagged: list[dict] = []
    signpost: list[dict] = []
    for b in rule_result.biomarkers:
        if b.severity == Severity.NORMAL:
            continue
        entry = {
            "code": b.code,
            "name": b.name,
            "status": b.status.value,
            "severity": b.severity.value,
            "interpretation": b.interpretation,
        }
        if b.tier == "actionable":
            flagged.append(entry)
        else:  # flag_only → clinician signpost, never lifestyle advice
            signpost.append(entry)

    return FusedAssessment(
        severity=final,
        rule_severity=rule_sev,
        rf_severity=rf_sev,
        rf_probabilities=rf_probs,
        escalated_by_rf=escalated,
        urgent_referral=rule_result.urgent_referral,
        flagged=flagged,
        signpost=signpost,
    )
