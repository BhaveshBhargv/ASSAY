"""Paths, NHANES variable names and column roles for data preparation.

The column lists decide which columns are model features and which are only
used to build the label, which is what keeps the questionnaire answers out of
the features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "nhanes"
PROCESSED_DIR = DATA_DIR / "processed"

NHANES_FILES_YAML = CONFIG_DIR / "nhanes_files.yaml"
THRESHOLDS_YAML = CONFIG_DIR / "clinical_thresholds.yaml"


@dataclass(frozen=True)
class Cycle:
    name: str  # "2013-2014"
    dir: str  # directory name on the CDC site
    kind: Literal["suffix", "prefix"]
    token: str  # "_H", "_I" or "P_"

    def filename(self, base: str) -> str:
        """File name for a component in this cycle, e.g. DEMO_H.XPT."""
        if self.kind == "suffix":
            return f"{base}{self.token}.XPT"
        return f"{self.token}{base}.XPT"


# NHANES variable codes and the names used in this project. The survey weight is
# handled separately because its name changes between cycles.
CANONICAL_VARS: dict[str, str] = {
    # demographics
    "RIDAGEYR": "age",
    "RIAGENDR": "sex_code",       # 1 = male, 2 = female
    "RIDRETH3": "eth_code",       # 1-7 race/ethnicity
    "DMDEDUC2": "educ_code",      # education (adults 20+)
    "INDFMPIR": "pir",            # poverty-income ratio
    "SDMVPSU": "psu",
    "SDMVSTRA": "strata",
    # labs
    "LBXGH":  "hba1c_pct",
    "LBXGLU": "fasting_glucose_mgdl",
    "LBXTC":  "total_chol_mgdl",
    "LBDHDD": "hdl_mgdl",
    "LBDLDL": "ldl_mgdl",
    "LBXTR":  "triglycerides_mgdl",
    # complete blood count (CBC)
    "LBXHGB":   "hemoglobin",
    "LBXHCT":   "hematocrit",
    "LBXRBCSI": "rbc",
    "LBXWBCSI": "wbc",
    "LBXPLTSI": "platelets",
    "LBXMCVSI": "mcv",
    "LBXMCHSI": "mch",
    "LBXMC":    "mchc",
    "LBXRDW":   "rdw",
    # biochemistry profile (BIOPRO): liver, kidney, electrolytes
    "LBXSATSI": "alt",
    "LBXSASSI": "ast",
    "LBXSAPSI": "alp",
    "LBXSAL":   "albumin",
    "LBXSTB":   "total_bilirubin",
    "LBXSCR":   "creatinine",
    "LBXSBU":   "bun",
    "LBXSNASI": "sodium",
    "LBXSKSI":  "potassium",
    "LBXSCLSI": "chloride",
    "LBXSCA":   "calcium",
    # vitamin D (VID), not in the 2017-2020 release
    "LBXVIDMS": "vitamin_d",
    # diabetes questionnaire (label only)
    "DIQ010": "diq_diabetes",
    "DIQ050": "diq_insulin",
    "DIQ070": "diq_pills",
    # blood pressure / cholesterol questionnaire (label only)
    "BPQ080":  "bpq_high_chol",
    "BPQ090D": "bpq_chol_med",
    "BPQ020":  "bpq_high_bp",
    "BPQ040A": "bpq_bp_med",
    # medical conditions (label only)
    "MCQ160C": "mcq_chd",
    "MCQ160E": "mcq_mi",
    "MCQ160F": "mcq_stroke",
    "MCQ160B": "mcq_chf",
}

# Survey weight: WTMEC2YR in the 2-year cycles, WTMECPRP in the P_ release.
WEIGHT_VARS = ("WTMEC2YR", "WTMECPRP")

# Biomarkers by panel.
CARDIOMETABOLIC = [
    "hba1c_pct", "fasting_glucose_mgdl",
    "total_chol_mgdl", "ldl_mgdl", "hdl_mgdl", "triglycerides_mgdl",
]
CBC_ANAEMIA = ["hemoglobin", "hematocrit", "rbc", "mcv", "mch", "mchc", "rdw"]
LIVER = ["alt", "ast", "alp", "albumin", "total_bilirubin"]
KIDNEY = ["creatinine", "bun"]
VITAMINS = ["vitamin_d"]  # missing for the 2017-2020 cycle

# Reported and escalated by the rule engine, but never used for lifestyle advice
# and left out of the model, since they aren't really lifestyle-driven.
FLAG_ONLY_BIOMARKERS = ["wbc", "platelets", "sodium", "potassium", "chloride", "calcium"]

# Blood pressure, BMI and waist aren't blood tests, so they aren't used at all.

# These drive the severity label and the recommendations.
ACTIONABLE_BLOOD = CARDIOMETABOLIC + CBC_ANAEMIA + LIVER + KIDNEY + VITAMINS

# Rows missing any of these are dropped, since the label depends on them.
MANDATORY_BIOMARKERS = ["hba1c_pct", "total_chol_mgdl", "hdl_mgdl"]

# Model features: the actionable markers minus vitamin D, which is missing for a
# whole cycle.
FEATURE_BIOMARKERS = [b for b in ACTIONABLE_BLOOD if b != "vitamin_d"]

# Feature columns that get imputed.
SUPPLEMENTARY_BIOMARKERS = [b for b in FEATURE_BIOMARKERS if b not in MANDATORY_BIOMARKERS]

# Everything that goes through cleaning, plausibility checks and winsorising.
ALL_BIOMARKERS = ACTIONABLE_BLOOD + FLAG_ONLY_BIOMARKERS

# Demographic columns kept through data preparation.
DEMOGRAPHIC_FEATURES = ["age", "sex_code", "eth_code", "educ_code", "pir"]

# Only used to build the label, never as model features.
LABEL_ONLY_COLUMNS = [
    "diq_diabetes", "diq_insulin", "diq_pills",
    "bpq_high_chol", "bpq_chol_med", "bpq_high_bp", "bpq_bp_med",
    "mcq_chd", "mcq_mi", "mcq_stroke", "mcq_chf",
    "statin_or_metformin",  # derived from RXQ_RX
]

# Survey design columns, kept for reference but not used as features.
DESIGN_COLUMNS = ["psu", "strata", "wtmec", "cycle"]

# Adults only. Children's results use age-based reference ranges, and children
# don't have the adult diagnosis and medication questionnaires.
ADULT_MIN_AGE = 18


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_file_registry() -> dict:
    return load_yaml(NHANES_FILES_YAML)


def load_thresholds() -> dict:
    return load_yaml(THRESHOLDS_YAML)


def get_cycles() -> list[Cycle]:
    reg = load_file_registry()
    return [Cycle(**c) for c in reg["cycles"]]
