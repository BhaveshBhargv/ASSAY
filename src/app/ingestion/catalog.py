"""
catalog.py — the biomarker catalog: panels, synonyms, and metadata.

Panels and lab-report synonyms are declared here; names, units, tiers, and
reference ranges come from the rule engine's ruleset (single source of truth), so
the API, the dashboard form, the CSV template, and the PDF extractor can never
drift from the clinical config.
"""
from __future__ import annotations

import math
from functools import lru_cache

# Ordered clinical panels (mirror src/data_prep/config.py groupings).
PANELS: list[tuple[str, list[str]]] = [
    ("Cardiometabolic", ["hba1c_pct", "fasting_glucose_mgdl", "total_chol_mgdl",
                          "ldl_mgdl", "hdl_mgdl", "triglycerides_mgdl"]),
    ("Complete blood count", ["hemoglobin", "hematocrit", "rbc", "mcv", "mch",
                              "mchc", "rdw", "wbc", "platelets"]),
    ("Liver", ["alt", "ast", "alp", "albumin", "total_bilirubin"]),
    ("Kidney", ["creatinine", "bun"]),
    ("Electrolytes", ["sodium", "potassium", "chloride", "calcium"]),
    ("Vitamins", ["vitamin_d"]),
]

PANEL_OF: dict[str, str] = {code: panel for panel, codes in PANELS for code in codes}
ALL_CODES: list[str] = [code for _p, codes in PANELS for code in codes]

# Lab-report name variants for best-effort PDF extraction. Longest/most specific
# first; single-letter electrolyte symbols are intentionally omitted (too noisy).
SYNONYMS: dict[str, list[str]] = {
    # NB: the "…glycosylated/glycated haemoglobin" spellings MUST be listed (and are
    # longer than plain "haemoglobin"), otherwise a label like
    # "Glycosylated Hemoglobin (HbA1c)" resolves to `hemoglobin`. LABEL_PRIORITY
    # below is the belt-and-braces guard for the same collision.
    "hba1c_pct": ["glycosylated haemoglobin", "glycosylated hemoglobin",
                  "glycated haemoglobin", "glycated hemoglobin",
                  "glycohaemoglobin", "glycohemoglobin",
                  "haemoglobin a1c", "hemoglobin a1c", "hba1c", "a1c"],
    "fasting_glucose_mgdl": ["fasting blood glucose", "fasting glucose", "glucose fasting",
                             "blood glucose", "glucose", "fbg", "fbs"],
    "total_chol_mgdl": ["total cholesterol", "cholesterol total", "serum cholesterol", "cholesterol"],
    "ldl_mgdl": ["ldl cholesterol", "ldl-c", "ldl"],
    "hdl_mgdl": ["hdl cholesterol", "hdl-c", "hdl"],
    "triglycerides_mgdl": ["triglycerides", "triglyceride", "trig"],
    "hemoglobin": ["haemoglobin", "hemoglobin", "hgb", "hb"],
    "hematocrit": ["packed cell volume", "haematocrit", "hematocrit", "hct", "pcv"],
    "rbc": ["red blood cell count", "red blood cells", "rbc count", "erythrocyte", "rbc"],
    "mcv": ["mean corpuscular volume", "mcv"],
    "mchc": ["mean corpuscular haemoglobin concentration", "mean corpuscular hemoglobin concentration", "mchc"],
    "mch": ["mean corpuscular haemoglobin", "mean corpuscular hemoglobin", "mch"],
    "rdw": ["red cell distribution width", "rdw-cv", "rdw"],
    "wbc": ["total leucocyte count", "total leukocyte count", "white blood cell count",
            "white blood cells", "wbc count", "leukocyte", "leucocyte", "wbc", "tlc"],
    "platelets": ["platelet count", "platelets", "platelet", "plt"],
    "alt": ["alanine aminotransferase", "alt/sgpt", "sgpt", "alt"],
    "ast": ["aspartate aminotransferase", "ast/sgot", "sgot", "ast"],
    "alp": ["alkaline phosphatase", "alk phos", "alp"],
    "albumin": ["serum albumin", "albumin"],
    "total_bilirubin": ["total bilirubin", "bilirubin total", "bilirubin"],
    "creatinine": ["serum creatinine", "creatinine"],
    # NB: bare "urea" (Blood Urea) is a different quantity (~2.14x BUN); only map
    # BUN-specific labels so we never feed a urea value into the BUN field.
    "bun": ["blood urea nitrogen", "urea nitrogen", "bun"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "chloride": ["chloride"],
    "calcium": ["calcium"],
    "vitamin_d": ["25-hydroxyvitamin d", "25-oh vitamin d", "25(oh)d", "vitamin d3", "vitamin d",
                  "cholecalciferol"],
}


# Codes that must win when a label also contains a broader analyte's name.
# "Glycosylated Hemoglobin (HbA1c)" is HbA1c, never haemoglobin — resolved by
# priority first, then by longest matching term.
LABEL_PRIORITY: dict[str, int] = {"hba1c_pct": 2}


@lru_cache(maxsize=1)
def _ruleset():
    from app.rules.loader import load_ruleset
    return load_ruleset()


def rule_for(code: str):
    return _ruleset().get(code)


def biomarker_meta(code: str) -> dict:
    """Name, unit, tier, and label role for one biomarker code."""
    r = rule_for(code)
    if r is None:
        return {"code": code, "name": code, "unit": "", "tier": "actionable",
                "label_role": "core", "panel": PANEL_OF.get(code, "")}
    return {"code": code, "name": r.name, "unit": r.unit.strip(), "tier": r.tier,
            "label_role": r.label_role, "panel": PANEL_OF.get(code, "")}


def reference_range(code: str, sex: str | None) -> dict:
    """Sex-resolved reference range; open-ended bounds become None."""
    r = rule_for(code)
    if r is None:
        return {"low": None, "high": None}
    rng = r.reference_range(sex if sex in ("male", "female") else "male")
    lo = None if rng["low"] == -math.inf else rng["low"]
    hi = None if rng["high"] == math.inf else rng["high"]
    return {"low": lo, "high": hi}


@lru_cache(maxsize=1)
def label_index() -> dict[str, str]:
    """label (lowercased) -> canonical code, from codes + names + synonyms."""
    idx: dict[str, str] = {}
    for code in ALL_CODES:
        idx[code.lower()] = code
        r = rule_for(code)
        if r is not None:
            idx[r.name.lower()] = code
    for code, alts in SYNONYMS.items():
        for a in alts:
            idx.setdefault(a.lower(), code)
    return idx
