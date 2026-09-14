"""Guideline search for /retrieve."""
from __future__ import annotations

import logging

from app.rag.retriever import build_query

from ..repositories.retrieval_repository import RetrievalRepository
from ..schemas.retrieve import RetrieveRequest, RetrieveResponse
from . import mappers

log = logging.getLogger("assay.api")


class RetrievalService:
    def __init__(self, retrieval: RetrievalRepository) -> None:
        self._retrieval = retrieval

    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        boost = tuple(f.code for f in req.flagged if f.code)
        if req.query:
            query = req.query
        else:
            flagged = [f.model_dump() for f in req.flagged]
            query = build_query(req.severity or "borderline", flagged)

        results = self._retrieval.retriever.retrieve(query, k=req.k, biomarker_boost=boost)
        passages = mappers.passages_out(results)
        log.info("retrieved %d passage(s) for query %r", len(passages), query[:80])
        return RetrieveResponse(query=query, count=len(passages), passages=passages)
