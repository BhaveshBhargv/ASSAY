"""rule_repository.py — owns the clinical rule engine and ruleset."""
from __future__ import annotations

import logging

log = logging.getLogger("assay.api")


class RuleRepository:
    def __init__(self) -> None:
        from app.domain.services.rule_engine import RuleEngine
        from app.rules.loader import load_ruleset
        self._ruleset = load_ruleset()
        self._engine = RuleEngine(self._ruleset)
        log.info("rule engine loaded (ruleset %s, %d biomarkers)",
                 self._engine.ruleset_version, len(self._engine.codes))

    @property
    def engine(self):
        return self._engine

    @property
    def version(self) -> str:
        return self._engine.ruleset_version

    def counts(self) -> dict:
        codes = self._engine.codes
        actionable = set(self._engine.actionable_codes)
        return {
            "biomarker_count": len(codes),
            "actionable_count": len(actionable),
            "flag_only_count": len(codes) - len(actionable),
        }
