"""
mappers.py — translate engine domain objects into API response schemas.

Kept separate so services stay focused on orchestration and the wire format lives
in exactly one place.
"""
from __future__ import annotations

from ..schemas.common import AssessmentOut, BiomarkerResultOut, MarkerOut
from ..schemas.recommend import EvidenceRefOut, GroundednessOut, ReportOut
from ..schemas.retrieve import PassageOut


def _biomarker_out(b) -> BiomarkerResultOut:
    d = b.to_dict()
    return BiomarkerResultOut(
        code=d["code"], name=d["name"], unit=d["unit"], value=d["value"],
        status=d["status"], direction=d["direction"], severity=d["severity"],
        urgent=d["urgent"], tier=d["tier"], reference_range=d["reference_range"],
        interpretation=d["interpretation"],
    )


def assessment_out(rule_result, fused) -> AssessmentOut:
    return AssessmentOut(
        severity=fused.severity,
        rule_severity=fused.rule_severity,
        rf_severity=fused.rf_severity,
        rf_probabilities=fused.rf_probabilities,
        escalated_by_rf=fused.escalated_by_rf,
        urgent_referral=fused.urgent_referral,
        flagged=[MarkerOut(**f) for f in fused.flagged],
        signpost=[MarkerOut(**s) for s in fused.signpost],
        biomarkers=[_biomarker_out(b) for b in rule_result.biomarkers],
    )


def report_out(report) -> ReportOut:
    return ReportOut.model_validate(report.model_dump())


def evidence_refs(evidence) -> list[EvidenceRefOut]:
    return [EvidenceRefOut(id=e.id, citation=e.citation, title=e.title, text=e.text)
            for e in evidence]


def groundedness_out(audit) -> GroundednessOut:
    return GroundednessOut(**audit.to_dict())


def passages_out(results) -> list[PassageOut]:
    out = []
    for i, res in enumerate(results, start=1):
        c = res.chunk
        out.append(PassageOut(
            id=f"E{i}", score=round(float(res.score), 4), source=c.source, code=c.code,
            title=c.title, citation=c.citation(), text=c.text,
            biomarkers=list(c.biomarkers), url=c.url,
        ))
    return out
