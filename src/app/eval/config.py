"""Evaluation settings, test patients and the retrieval test queries."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase9"

# An advice item counts as faithful if its embedding has at least this cosine
# similarity with one of the passages it cites.
FAITHFULNESS_TAU = 0.35

# K values for Precision@K and Recall@K.
RAG_KS = (1, 3, 5)

# Made-up patients covering the patterns the panel flags.
EVAL_PATIENTS: list[dict] = [
    {"name": "Metabolic cluster", "demographics": {"age": 54, "sex": "male"},
     "biomarkers": {"hba1c_pct": 6.1, "fasting_glucose_mgdl": 108, "total_chol_mgdl": 232,
                    "ldl_mgdl": 150, "hdl_mgdl": 34, "triglycerides_mgdl": 205, "alt": 46}},
    {"name": "Fatty-liver pattern", "demographics": {"age": 48, "sex": "male"},
     "biomarkers": {"alt": 92, "ast": 74, "hba1c_pct": 5.9, "triglycerides_mgdl": 190,
                    "total_chol_mgdl": 215, "hdl_mgdl": 38}},
    {"name": "Anaemia pattern", "demographics": {"age": 41, "sex": "female"},
     "biomarkers": {"hemoglobin": 10.8, "hematocrit": 33, "rbc": 3.8, "mcv": 74,
                    "mch": 24, "rdw": 16.2, "hba1c_pct": 5.3}},
    {"name": "Vitamin D low", "demographics": {"age": 36, "sex": "female"},
     "biomarkers": {"vitamin_d": 28, "hba1c_pct": 5.4, "total_chol_mgdl": 188, "hdl_mgdl": 60}},
    {"name": "Largely healthy", "demographics": {"age": 34, "sex": "female"},
     "biomarkers": {"hba1c_pct": 5.1, "total_chol_mgdl": 178, "hdl_mgdl": 62,
                    "triglycerides_mgdl": 96, "ldl_mgdl": 96}},
]

# Retrieval test queries. A passage counts as relevant if it's tagged with any of
# relevant_codes, so relevance comes from the corpus's own tags, not manual labels.
RAG_QUERIES: list[dict] = [
    {"name": "Diabetes / raised HbA1c", "severity": "serious",
     "flagged": [{"code": "hba1c_pct", "name": "HbA1c", "status": "high"},
                 {"code": "fasting_glucose_mgdl", "name": "Fasting glucose", "status": "high"}],
     "relevant_codes": {"hba1c_pct", "fasting_glucose_mgdl"}},
    {"name": "Dyslipidaemia", "severity": "borderline",
     "flagged": [{"code": "total_chol_mgdl", "name": "Total cholesterol", "status": "high"},
                 {"code": "ldl_mgdl", "name": "LDL cholesterol", "status": "high"},
                 {"code": "hdl_mgdl", "name": "HDL cholesterol", "status": "low"}],
     "relevant_codes": {"total_chol_mgdl", "ldl_mgdl", "hdl_mgdl"}},
    {"name": "Fatty liver / raised ALT", "severity": "borderline",
     "flagged": [{"code": "alt", "name": "ALT", "status": "high"},
                 {"code": "ast", "name": "AST", "status": "high"}],
     "relevant_codes": {"alt", "ast"}},
    {"name": "Reduced kidney function", "severity": "serious",
     "flagged": [{"code": "creatinine", "name": "Creatinine", "status": "high"},
                 {"code": "bun", "name": "Blood urea nitrogen", "status": "high"}],
     "relevant_codes": {"creatinine", "bun"}},
    {"name": "Anaemia", "severity": "borderline",
     "flagged": [{"code": "hemoglobin", "name": "Haemoglobin", "status": "low"},
                 {"code": "mcv", "name": "MCV", "status": "low"}],
     "relevant_codes": {"hemoglobin", "mcv", "hematocrit", "rbc"}},
    {"name": "Vitamin D deficiency", "severity": "borderline",
     "flagged": [{"code": "vitamin_d", "name": "Vitamin D", "status": "low"}],
     "relevant_codes": {"vitamin_d"}},
    {"name": "Metabolic cluster", "severity": "borderline",
     "flagged": [{"code": "hba1c_pct", "name": "HbA1c", "status": "borderline"},
                 {"code": "hdl_mgdl", "name": "HDL cholesterol", "status": "low"},
                 {"code": "triglycerides_mgdl", "name": "Triglycerides", "status": "high"}],
     "relevant_codes": {"hba1c_pct", "hdl_mgdl", "triglycerides_mgdl"}},
]
