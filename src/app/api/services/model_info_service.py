"""model_info_service.py — the /model-info use case: capabilities & provenance."""
from __future__ import annotations

from app.ingestion.catalog import ALL_CODES, biomarker_meta
from app.recommend.config import DISCLAIMER

from ..core.config import settings
from ..repositories.model_repository import ModelRepository
from ..repositories.retrieval_repository import RetrievalRepository
from ..repositories.rule_repository import RuleRepository
from ..schemas.model_info import (
    BiomarkerInfo,
    ModelInfoResponse,
    RagInfo,
    RandomForestInfo,
    RuleEngineInfo,
)


class ModelInfoService:
    def __init__(self, rules: RuleRepository, models: ModelRepository,
                 retrieval: RetrievalRepository) -> None:
        self._rules = rules
        self._models = models
        self._retrieval = retrieval

    def info(self) -> ModelInfoResponse:
        biomarkers = [BiomarkerInfo(**biomarker_meta(c)) for c in ALL_CODES]
        return ModelInfoResponse(
            app=settings.app_name,
            version=settings.version,
            rule_engine=RuleEngineInfo(ruleset_version=self._rules.version, **self._rules.counts()),
            random_forest=RandomForestInfo(**self._models.info()),
            rag=RagInfo(**self._retrieval.info()),
            default_provider=settings.default_provider,
            biomarkers=biomarkers,
            disclaimer=DISCLAIMER,
        )
