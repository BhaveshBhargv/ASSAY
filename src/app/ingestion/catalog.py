"""Biomarker panels, lab-report synonyms and per-code metadata.

Names, units, tiers and reference ranges are read from the rule set, so they
always match the clinical config.
"""
from __future__ import annotations

import math
from functools import lru_cache

# Same grouping as src/data_prep/config.py.
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

# Names labs use for each test, most specific first. Single-letter electrolyte
# symbols (Na, K, ...) are left out because they match far too much text.
SYNONYMS: dict[str, list[str]] = {
    # The glycated/glycosylated spellings have to be here, otherwise a label like
    # "Glycosylated Hemoglobin (HbA1c)" matches plain haemoglobin instead.
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
    # Plain "urea" is a different measurement (about 2.14x BUN), so only
    # BUN-specific names are mapped.
    "bun": ["blood urea nitrogen", "urea nitrogen", "bun"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "chloride": ["chloride"],
    "calcium": ["calcium"],
    "vitamin_d": ["25-hydroxyvitamin d", "25-oh vitamin d", "25(oh)d", "vitamin d3", "vitamin d",
                  "cholecalciferol"],
}


# Codes that win when a label also contains another test's name, e.g.
# "Glycosylated Hemoglobin (HbA1c)" is HbA1c, not haemoglobin.
LABEL_PRIORITY: dict[str, int] = {"hba1c_pct": 2}


@lru_cache(maxsize=1)
def _ruleset():
    from app.rules.loader import load_ruleset
    return load_ruleset()


def rule_for(code: str):
    return _ruleset().get(code)


def biomarker_meta(code: str) -> dict:
    """Name, unit, tier, label role and panel for a code."""
    r = rule_for(code)
    if r is None:
        return {"code": code, "name": code, "unit": "", "tier": "actionable",
                "label_role": "core", "panel": PANEL_OF.get(code, "")}
    return {"code": code, "name": r.name, "unit": r.unit.strip(), "tier": r.tier,
            "label_role": r.label_role, "panel": PANEL_OF.get(code, "")}


def reference_range(code: str, sex: str | None) -> dict:
    """Reference range for the given sex, with open-ended bounds as None."""
    r = rule_for(code)
    if r is None:
        return {"low": None, "high": None}
    rng = r.reference_range(sex if sex in ("male", "female") else "male")
    lo = None if rng["low"] == -math.inf else rng["low"]
    hi = None if rng["high"] == math.inf else rng["high"]
    return {"low": lo, "high": hi}


@lru_cache(maxsize=1)
def label_index() -> dict[str, str]:
    """Maps every lowercased code, test name and synonym to its code."""
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
