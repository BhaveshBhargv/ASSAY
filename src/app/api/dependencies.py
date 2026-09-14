"""FastAPI dependencies.

Repositories load models and indexes, so each one is created once and cached.
Services are cheap and are built for every request.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from .repositories.llm_repository import LLMRepository
from .repositories.model_repository import ModelRepository
from .repositories.retrieval_repository import RetrievalRepository
from .repositories.rule_repository import RuleRepository
from .services.assessment_service import AssessmentService
from .services.model_info_service import ModelInfoService
from .services.recommendation_service import RecommendationService
from .services.retrieval_service import RetrievalService
from .services.upload_service import UploadService


# Repositories, created once.
@lru_cache(maxsize=1)
def get_rule_repository() -> RuleRepository:
    return RuleRepository()


@lru_cache(maxsize=1)
def get_model_repository() -> ModelRepository:
    return ModelRepository()


@lru_cache(maxsize=1)
def get_retrieval_repository() -> RetrievalRepository:
    return RetrievalRepository()


@lru_cache(maxsize=1)
def get_llm_repository() -> LLMRepository:
    return LLMRepository()


# Services, created per request.
def get_assessment_service(
    rules: RuleRepository = Depends(get_rule_repository),
    models: ModelRepository = Depends(get_model_repository),
) -> AssessmentService:
    return AssessmentService(rules, models)


def get_recommendation_service(
    assessment: AssessmentService = Depends(get_assessment_service),
    retrieval: RetrievalRepository = Depends(get_retrieval_repository),
    llm: LLMRepository = Depends(get_llm_repository),
) -> RecommendationService:
    return RecommendationService(assessment, retrieval, llm)


def get_retrieval_service(
    retrieval: RetrievalRepository = Depends(get_retrieval_repository),
) -> RetrievalService:
    return RetrievalService(retrieval)


def get_upload_service() -> UploadService:
    return UploadService()


def get_model_info_service(
    rules: RuleRepository = Depends(get_rule_repository),
    models: ModelRepository = Depends(get_model_repository),
    retrieval: RetrievalRepository = Depends(get_retrieval_repository),
) -> ModelInfoService:
    return ModelInfoService(rules, models, retrieval)


def warm_up() -> None:
    """Load the repositories at startup so the first request isn't slow."""
    get_rule_repository()
    get_model_repository()
    get_retrieval_repository()
    get_llm_repository()
