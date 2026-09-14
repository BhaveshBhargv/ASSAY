"""Sample patients for the sidebar. The values are made up, not taken from real people."""
from __future__ import annotations

SAMPLES: dict[str, dict] = {
    "Metabolic risk — male, 54": {
        "demographics": {"age": 54, "sex": "male", "eth_code": 3, "pir": 2.5, "educ_code": 4},
        "biomarkers": {
            "hba1c_pct": 6.1, "fasting_glucose_mgdl": 108, "total_chol_mgdl": 232,
            "ldl_mgdl": 150, "hdl_mgdl": 34, "triglycerides_mgdl": 205,
            "alt": 46, "ast": 38, "creatinine": 0.9, "bun": 16,
            "hemoglobin": 14.5, "hematocrit": 43, "wbc": 7.2, "platelets": 250,
            "vitamin_d": 38,
        },
    },
    "Largely healthy — female, 34": {
        "demographics": {"age": 34, "sex": "female", "eth_code": 3, "pir": 3.5, "educ_code": 5},
        "biomarkers": {
            "hba1c_pct": 5.1, "fasting_glucose_mgdl": 88, "total_chol_mgdl": 178,
            "ldl_mgdl": 96, "hdl_mgdl": 62, "triglycerides_mgdl": 96,
            "alt": 22, "ast": 20, "creatinine": 0.8, "bun": 12,
            "hemoglobin": 13.6, "hematocrit": 40, "wbc": 6.1, "platelets": 265,
            "vitamin_d": 72,
        },
    },
    "Anaemia pattern — female, 41": {
        "demographics": {"age": 41, "sex": "female", "eth_code": 4, "pir": 2.0, "educ_code": 3},
        "biomarkers": {
            "hba1c_pct": 5.3, "fasting_glucose_mgdl": 92, "total_chol_mgdl": 190,
            "ldl_mgdl": 110, "hdl_mgdl": 58, "triglycerides_mgdl": 110,
            "hemoglobin": 10.8, "hematocrit": 33, "rbc": 3.8, "mcv": 74,
            "mch": 24, "mchc": 31, "rdw": 16.2, "wbc": 6.4, "platelets": 290,
            "creatinine": 0.8, "vitamin_d": 44,
        },
    },
}
