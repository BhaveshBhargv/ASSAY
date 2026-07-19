"""
recommendation_service.py — the /recommend use case: full grounded pipeline.

assess (rules+RF+fusion) → retrieve evidence → generate (LLM) → verify (guards).
LLM/connection failures are converted to a typed LLMUnavailableError so the
controller returns a clean 502 rather than a stack trace.
"""
from __future__ import annotations

import logging

from app.recommend.bundle import RecommendationBundle
from app.recommend.config import DISCLAIMER
from app.recommend.generator import GenerationError, generate_report
from app.recommend.guards import verify
from app.recommend.prompt import build_evidence_pack, build_messages

from ..core.errors import LLMUnavailableError
from ..repositories.llm_repository import LLMRepository
from ..repositories.retrieval_repository import RetrievalRepository
from ..schemas.recommend import RecommendRequest, RecommendResponse
from . import mappers
from .assessment_service import AssessmentService

log = logging.getLogger("assay.api")


class RecommendationService:
    def __init__(self, assessment: AssessmentService,
                 retrieval: RetrievalRepository, llm: LLMRepository) -> None:
        self._assessment = assessment
        self._retrieval = retrieval
        self._llm = llm

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        demographics = req.demographics.to_dict()
        bundle = self._assessment.assess(demographics, req.biomarkers)
        fused = bundle.fused

        results = self._retrieval.retriever.retrieve_for_assessment(
            fused.severity, fused.flagged, k=req.k)
        evidence = build_evidence_pack(results)

        provider = self._llm.provider(req.provider, req.model)
        system, user = build_messages(demographics, fused, evidence)
        try:
            report = generate_report(provider, system, user)
        except GenerationError as exc:
            raise LLMUnavailableError(f"The model returned an unusable response: {exc}",
                                      detail={"provider": provider.name}) from exc
        except Exception as exc:  # noqa: BLE001 - typically no daemon / network
            raise LLMUnavailableError(
                f"Could not reach the language model ({req.provider}). Is it running?",
                detail={"provider": req.provider, "error": str(exc)}) from exc

        cleaned, audit = verify(report, evidence, provider=provider, llm_recheck=req.llm_recheck)

        rec_bundle = RecommendationBundle(
            assessment=fused, evidence=evidence, report=cleaned, audit=audit,
            disclaimer=DISCLAIMER, provider=provider.name)
        log.info("recommendation ready: provider=%s groundedness=%.2f",
                 provider.name, audit.groundedness)

        return RecommendResponse(
            provider=provider.name,
            assessment=mappers.assessment_out(bundle.rule_result, fused),
            report=mappers.report_out(cleaned),
            evidence=mappers.evidence_refs(evidence),
            groundedness=mappers.groundedness_out(audit),
            urgent_note=rec_bundle.urgent_note,
            clinician_note=rec_bundle.clinician_note,
            disclaimer=DISCLAIMER,
        )
