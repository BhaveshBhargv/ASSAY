"""Loads config/clinical_rules.yaml into a validated RuleSet.

Bands are checked when the file is loaded, so a broken rule fails straight
away rather than during a request. Sex-specific bands can't be checked for gaps
here and are resolved when a value is classified.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..domain.enums import Direction, Severity, Status
from ..domain.models import Band, BiomarkerRule, Guideline, RuleSet

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "clinical_rules.yaml"


def _band(raw: dict) -> Band:
    g = raw.get("guideline")
    return Band(
        status=Status(raw["status"]),
        severity=Severity(raw["severity"]),
        min=raw.get("min"),
        max=raw.get("max"),
        direction=Direction(raw.get("direction", "in_range")),
        urgent=raw.get("urgent"),
        guideline=Guideline(source=g["source"], code=g["code"]) if g else None,
        interpretation=raw.get("interpretation", ""),
    )


def _validate_contiguity(code: str, bands: tuple[Band, ...]) -> None:
    """Check scalar bands are in order and cover every value with no gaps."""
    scalar = all(
        not isinstance(b.min, dict) and not isinstance(b.max, dict) for b in bands
    )
    if not scalar:
        return  # sex-specific bands are checked at classification time
    prev_hi = -math.inf
    for i, b in enumerate(bands):
        lo = b.min if b.min is not None else -math.inf
        hi = b.max if b.max is not None else math.inf
        if lo > hi:
            raise ValueError(f"{code}: band {i} has min > max")
        if i == 0 and lo != -math.inf:
            raise ValueError(f"{code}: first band must be open-ended below (no min)")
        if lo != prev_hi:
            raise ValueError(f"{code}: gap/overlap at band {i} (expected min={prev_hi}, got {lo})")
        prev_hi = hi
    if prev_hi != math.inf:
        raise ValueError(f"{code}: last band must be open-ended above (no max)")


def load_ruleset(path: Path = DEFAULT_RULES_PATH) -> RuleSet:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    meta = cfg.get("meta", {})
    version = meta.get("ruleset_version", "unknown")
    secondary_min = int(meta.get("secondary_borderline_min", 2))
    secondary_markers = set(meta.get("label_secondary_markers", []))
    rules: dict[str, BiomarkerRule] = {}

    for code, spec in cfg["biomarkers"].items():
        bands = tuple(_band(b) for b in spec["bands"])
        _validate_contiguity(code, bands)
        ref = spec.get("reference_range", {})
        tier = spec.get("tier", "actionable")
        if code in secondary_markers:
            label_role = "secondary"
        elif tier == "actionable":
            label_role = "core"
        else:
            label_role = "none"
        rules[code] = BiomarkerRule(
            code=code,
            name=spec["name"],
            unit=spec.get("unit", ""),
            bands=bands,
            category=spec.get("category", "blood"),
            tier=tier,
            label_role=label_role,
            reference_range_low=ref.get("low"),
            reference_range_high=ref.get("high"),
        )

    return RuleSet(version=version, rules=rules, secondary_borderline_min=secondary_min)
