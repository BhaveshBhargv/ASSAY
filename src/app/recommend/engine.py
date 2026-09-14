"""Runs the whole pipeline from raw inputs to a recommendation bundle.

rules -> Random Forest -> fusion -> retrieval -> LLM -> guards
"""
from __future__ import annotations

import logging
from typing import Optional

from ..domain.models import PatientContext
from ..domain.services.rule_engine import RuleEngine
from . import config as C
from .bundle import RecommendationBundle
from .fusion import fuse
from .generator import generate_report
from .guards import verify
from .ports import ILLMProvider
from .prompt import build_evidence_pack, build_messages

log = logging.getLogger(__name__)

# NHANES sex codes (RIAGENDR): 1 = male, 2 = female.
_SEX_CODE = {"male": 1, "female": 2, "m": 1, "f": 2, 1: 1, 2: 2}


class RiskAdapter:
    """Builds the model's features from raw biomarkers and runs the prediction.

    Features are built with the same functions used in data prep, so training
    and inference match. If prediction fails, predict returns None and fusion
    falls back to the rules alone.
    """

    def __init__(self) -> None:
        from ..ml.predict import RiskModel
        self._model = RiskModel()

    def predict(self, demographics: dict, biomarkers: dict) -> Optional[dict]:
        try:
            feats = self._build_features(demographics, biomarkers)
            return self._model.predict(feats)
        except Exception as exc:  # noqa: BLE001
            log.warning("RF prediction unavailable (%s); using rules-only fusion", exc)
            return None

    @staticmethod
    def _build_features(demographics: dict, biomarkers: dict) -> dict:
        import pandas as pd

        from data_prep.encode_scale import build_feature_matrix
        from data_prep.features import add_features

        # The model only uses the biomarkers, age and sex.
        row: dict = dict(biomarkers)
        row["age"] = demographics.get("age")
        row["sex_code"] = _SEX_CODE.get(demographics.get("sex"), demographics.get("sex_code"))

        df = add_features(pd.DataFrame([row]))
        X = build_feature_matrix(df)
        return X.iloc[0].to_dict()


class RecommendationEngine:
    def __init__(
        self,
        rule_engine: RuleEngine,
        retriever,
        provider: ILLMProvider,
        risk_adapter: Optional[RiskAdapter] = None,
        k: int = C.TOP_K_EVIDENCE,
        llm_recheck: bool = C.LLM_ENTAILMENT_RECHECK,
    ) -> None:
        self.rule_engine = rule_engine
        self.retriever = retriever
        self.provider = provider
        self.risk_adapter = risk_adapter
        self.k = k
        self.llm_recheck = llm_recheck

    @classmethod
    def build(
        cls,
        provider="ollama",
        use_rf: bool = True,
        k: int = C.TOP_K_EVIDENCE,
        llm_recheck: bool = C.LLM_ENTAILMENT_RECHECK,
        **provider_kwargs,
    ) -> "RecommendationEngine":
        """Create an engine with the real components. provider can be a name or a provider."""
        from ..rag.retriever import GuidelineRetriever
        from ..rules.loader import load_ruleset

        rule_engine = RuleEngine(load_ruleset())
        retriever = GuidelineRetriever.load()

        if isinstance(provider, str):
            from .providers import build_provider
            provider = build_provider(provider, **provider_kwargs)

        risk = None
        if use_rf:
            try:
                risk = RiskAdapter()
            except Exception as exc:  # noqa: BLE001
                log.warning("RF model unavailable (%s); rules-only fusion", exc)

        return cls(rule_engine, retriever, provider, risk, k=k, llm_recheck=llm_recheck)

    def recommend(self, demographics: dict, biomarkers: dict) -> RecommendationBundle:
        context = PatientContext(
            sex=demographics.get("sex"), age=demographics.get("age")
        )
        rule_result = self.rule_engine.evaluate(biomarkers, context)

        rf_output = self.risk_adapter.predict(demographics, biomarkers) if self.risk_adapter else None
        assessment = fuse(rule_result, rf_output)

        results = self.retriever.retrieve_for_assessment(
            assessment.severity, assessment.flagged, k=self.k
        )
        evidence = build_evidence_pack(results)

        system, user = build_messages(demographics, assessment, evidence)
        report = generate_report(self.provider, system, user)

        cleaned, audit = verify(
            report, evidence, provider=self.provider, llm_recheck=self.llm_recheck
        )

        provider_name = getattr(self.provider, "name", "unknown")
        return RecommendationBundle(
            assessment=assessment,
            evidence=evidence,
            report=cleaned,
            audit=audit,
            disclaimer=C.DISCLAIMER,
            provider=provider_name,
        )
