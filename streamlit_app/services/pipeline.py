"""The dashboard's wrapper around the engine.

assess() runs the rules, Random Forest, fusion and retrieval, and always works.
generate() runs the LLM step, which fails if no model can be reached.

The rule engine, retriever and model are loaded once with st.cache_resource.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.domain.models import PatientContext
from app.recommend.bundle import RecommendationBundle
from app.recommend.config import DISCLAIMER
from app.recommend.fusion import FusedAssessment, fuse
from app.recommend.prompt import EvidenceItem, build_evidence_pack


@dataclass
class Assessment:
    rule_result: object
    fused: FusedAssessment
    evidence: list[EvidenceItem]


@dataclass
class Recommendations:
    report: object  # RecommendationReport
    audit: object  # GuardReport
    error: str = ""


@st.cache_resource(show_spinner="Loading models and guideline index…")
def load_core():
    """Load the rule engine, retriever and RF model."""
    from app.domain.services.rule_engine import RuleEngine
    from app.rag.retriever import GuidelineRetriever
    from app.recommend.engine import RiskAdapter
    from app.rules.loader import load_ruleset

    rule_engine = RuleEngine(load_ruleset())
    retriever = GuidelineRetriever.load()
    try:
        risk = RiskAdapter()
    except Exception:  # noqa: BLE001 - no model, so rules only
        risk = None
    return rule_engine, retriever, risk


def assess(demographics: dict, biomarkers: dict, k: int = 6) -> Assessment:
    rule_engine, retriever, risk = load_core()
    ctx = PatientContext(sex=demographics.get("sex"), age=demographics.get("age"))
    rule_result = rule_engine.evaluate(biomarkers, ctx)
    rf_output = risk.predict(demographics, biomarkers) if risk else None
    fused = fuse(rule_result, rf_output)
    results = retriever.retrieve_for_assessment(fused.severity, fused.flagged, k=k)
    return Assessment(rule_result, fused, build_evidence_pack(results))


def rf_available() -> bool:
    return load_core()[2] is not None


def generate(provider_kind: str, model: str | None, demographics: dict,
             assessment: Assessment, llm_recheck: bool = False) -> Recommendations:
    """Run the LLM step. Errors are returned in `error` instead of being raised."""
    from app.recommend.generator import generate_report, GenerationError
    from app.recommend.guards import verify
    from app.recommend.prompt import build_messages
    from app.recommend.providers import build_provider

    try:
        kwargs = {"model": model} if model else {}
        provider = build_provider(provider_kind, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return Recommendations(None, None, error=f"Could not initialise provider: {exc}")

    system, user = build_messages(demographics, assessment.fused, assessment.evidence)
    try:
        report = generate_report(provider, system, user)
    except GenerationError as exc:
        return Recommendations(None, None, error=f"The model returned an unusable response. {exc}")
    except Exception as exc:  # noqa: BLE001 - usually Ollama isn't running
        return Recommendations(None, None,
                               error=f"Could not reach the language model ({provider_kind}). "
                                     f"Is it running? Details: {exc}")

    cleaned, audit = verify(report, assessment.evidence, provider=provider, llm_recheck=llm_recheck)
    return Recommendations(cleaned, audit)


def build_bundle(assessment: Assessment, recs: Recommendations,
                 provider_name: str = "") -> RecommendationBundle | None:
    """Bundle for the PDF export, or None when there are no recommendations."""
    if recs is None or recs.report is None:
        return None
    return RecommendationBundle(
        assessment=assessment.fused, evidence=assessment.evidence,
        report=recs.report, audit=recs.audit, disclaimer=DISCLAIMER,
        provider=provider_name,
    )
