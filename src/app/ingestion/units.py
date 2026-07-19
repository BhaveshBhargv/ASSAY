"""
units.py — unit normalisation, conversion, and plausibility for biomarkers.

Two jobs used by the report parser:
  1. Decide whether a printed value's unit is acceptable for a given biomarker
     (unit-anchoring stops "199 pg/ml" from being read as fasting glucose).
  2. Convert a printed value in a report's unit to the app's canonical unit
     (e.g. Vitamin D ng/mL → nmol/L ×2.5), and reject physiologically
     impossible values.

Canonical units match `config/clinical_rules.yaml`. Conversion factors multiply
the printed value to reach the canonical unit.
"""
from __future__ import annotations

# Canonical unit per biomarker (normalised token form).
CANON_UNIT: dict[str, str] = {
    "hba1c_pct": "%", "fasting_glucose_mgdl": "mg/dl", "total_chol_mgdl": "mg/dl",
    "ldl_mgdl": "mg/dl", "hdl_mgdl": "mg/dl", "triglycerides_mgdl": "mg/dl",
    "hemoglobin": "g/dl", "hematocrit": "%", "rbc": "10^6/ul", "mcv": "fl",
    "mch": "pg", "mchc": "g/dl", "rdw": "%", "wbc": "10^3/ul", "platelets": "10^3/ul",
    "alt": "u/l", "ast": "u/l", "alp": "u/l", "albumin": "g/dl",
    "total_bilirubin": "mg/dl", "creatinine": "mg/dl", "bun": "mg/dl",
    "sodium": "mmol/l", "potassium": "mmol/l", "chloride": "mmol/l",
    "calcium": "mg/dl", "vitamin_d": "nmol/l",
}

# Accepted printed units → factor to the canonical unit (1.0 = already canonical).
CONVERSIONS: dict[str, dict[str, float]] = {
    "hba1c_pct": {"%": 1.0},
    "fasting_glucose_mgdl": {"mg/dl": 1.0, "mmol/l": 18.016},
    "total_chol_mgdl": {"mg/dl": 1.0, "mmol/l": 38.67},
    "ldl_mgdl": {"mg/dl": 1.0, "mmol/l": 38.67},
    "hdl_mgdl": {"mg/dl": 1.0, "mmol/l": 38.67},
    "triglycerides_mgdl": {"mg/dl": 1.0, "mmol/l": 88.57},
    "hemoglobin": {"g/dl": 1.0, "g/l": 0.1},
    "hematocrit": {"%": 1.0, "l/l": 100.0},
    "rbc": {"10^6/ul": 1.0},
    "mcv": {"fl": 1.0},
    "mch": {"pg": 1.0},
    "mchc": {"g/dl": 1.0, "g/l": 0.1},
    "rdw": {"%": 1.0},
    "wbc": {"10^3/ul": 1.0},
    "platelets": {"10^3/ul": 1.0},
    "alt": {"u/l": 1.0}, "ast": {"u/l": 1.0}, "alp": {"u/l": 1.0},
    "albumin": {"g/dl": 1.0, "g/l": 0.1},
    "total_bilirubin": {"mg/dl": 1.0, "umol/l": 0.0585},
    "creatinine": {"mg/dl": 1.0, "umol/l": 0.011312},
    "bun": {"mg/dl": 1.0, "mmol/l": 2.801},
    "sodium": {"mmol/l": 1.0, "meq/l": 1.0},
    "potassium": {"mmol/l": 1.0, "meq/l": 1.0},
    "chloride": {"mmol/l": 1.0, "meq/l": 1.0},
    "calcium": {"mg/dl": 1.0, "mmol/l": 4.008},
    "vitamin_d": {"nmol/l": 1.0, "ng/ml": 2.496},
}

# Physiological plausibility bounds in CANONICAL units (reject garbage extractions).
PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "hba1c_pct": (3, 20), "fasting_glucose_mgdl": (30, 600), "total_chol_mgdl": (50, 500),
    "ldl_mgdl": (10, 400), "hdl_mgdl": (5, 150), "triglycerides_mgdl": (20, 2000),
    "hemoglobin": (3, 25), "hematocrit": (10, 65), "rbc": (2, 8), "mcv": (50, 130),
    "mch": (15, 45), "mchc": (25, 40), "rdw": (8, 30), "wbc": (0.5, 50),
    "platelets": (10, 1000), "alt": (2, 500), "ast": (2, 500), "alp": (20, 400),
    "albumin": (1, 6), "total_bilirubin": (0.05, 20), "creatinine": (0.2, 15),
    "bun": (2, 100), "sodium": (110, 170), "potassium": (2, 8), "chloride": (70, 130),
    "calcium": (5, 15), "vitamin_d": (3, 250),
}

# Printed-unit spellings → normalised token. Longest keys checked first so, e.g.,
# "10^3/ul" wins before "3", and "pg/ml" before "pg".
_UNIT_ALIASES: list[tuple[str, str]] = sorted([
    ("10^3/ul", "10^3/ul"), ("10³/ul", "10^3/ul"), ("x10^3/ul", "10^3/ul"),
    ("10^9/l", "10^3/ul"), ("k/ul", "10^3/ul"), ("thou/ul", "10^3/ul"), ("thous/ul", "10^3/ul"),
    ("10^6/ul", "10^6/ul"), ("10^12/l", "10^6/ul"), ("x10^6/ul", "10^6/ul"),
    ("million/ul", "10^6/ul"), ("mill/ul", "10^6/ul"), ("mil/ul", "10^6/ul"),
    ("mg/dl", "mg/dl"), ("mgs/dl", "mg/dl"),
    ("ng/ml", "ng/ml"), ("ng/dl", "ng/dl"),
    ("nmol/l", "nmol/l"), ("umol/l", "umol/l"),
    ("mmol/l", "mmol/l"), ("meq/l", "meq/l"),
    ("g/dl", "g/dl"), ("gm/dl", "g/dl"), ("g/l", "g/l"),
    ("iu/l", "u/l"), ("u/l", "u/l"), ("units/l", "u/l"),
    ("pg/ml", "pg/ml"), ("pg", "pg"),
    ("fl", "fl"), ("l/l", "l/l"), ("%", "%"),
], key=lambda t: -len(t[0]))


def detect_unit(rest: str) -> str:
    """Normalised unit token that the text immediately after a value starts with."""
    s = rest.lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    for spelling, norm in _UNIT_ALIASES:
        if s.startswith(spelling):
            return norm
    return ""


def to_canonical(code: str, value: float, unit: str):
    """Return (canonical_value, converted_from) or (None, None) if the unit isn't
    acceptable for this biomarker. `converted_from` is the source unit when a
    non-trivial conversion was applied, else None."""
    table = CONVERSIONS.get(code)
    if table is None or unit not in table:
        return None, None
    factor = table[unit]
    converted_from = unit if unit != CANON_UNIT.get(code) else None
    return value * factor, converted_from


def is_plausible(code: str, canonical_value: float) -> bool:
    lo, hi = PLAUSIBILITY.get(code, (float("-inf"), float("inf")))
    return lo <= canonical_value <= hi
