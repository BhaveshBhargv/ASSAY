"""
config.py — Phase 2 configuration loading and canonical schema.

Single source of truth for:
  * which NHANES files to pull (from config/nhanes_files.yaml)
  * clinical thresholds & label rules (from config/clinical_thresholds.yaml)
  * the canonical variable-name map (raw NHANES codes -> readable names)
  * which columns are features vs label-only (leakage control)

Keeping all of this here means every downstream module depends on one
stable contract (Single Responsibility + Dependency Inversion).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "nhanes"
PROCESSED_DIR = DATA_DIR / "processed"

NHANES_FILES_YAML = CONFIG_DIR / "nhanes_files.yaml"
THRESHOLDS_YAML = CONFIG_DIR / "clinical_thresholds.yaml"


# --------------------------------------------------------------------------- #
# Cycle description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cycle:
    name: str          # "2013-2014"
    dir: str           # CDC directory segment
    kind: Literal["suffix", "prefix"]
    token: str         # "_H" | "_I" | "P_"

    def filename(self, base: str) -> str:
        """Build the .XPT filename for a component base name in this cycle."""
        if self.kind == "suffix":
            return f"{base}{self.token}.XPT"
        return f"{self.token}{base}.XPT"


# --------------------------------------------------------------------------- #
# Canonical variable map: raw NHANES code -> readable canonical name.
# Blood-pressure and survey-weight variables are handled separately because
# their raw names differ across cycles (see loaders).
# --------------------------------------------------------------------------- #
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
    # complete blood count (CBC file)
    "LBXHGB":   "hemoglobin",
    "LBXHCT":   "hematocrit",
    "LBXRBCSI": "rbc",
    "LBXWBCSI": "wbc",
    "LBXPLTSI": "platelets",
    "LBXMCVSI": "mcv",
    "LBXMCHSI": "mch",
    "LBXMC":    "mchc",
    "LBXRDW":   "rdw",
    # standard biochemistry profile (BIOPRO file) — liver / kidney / electrolytes
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
    # vitamin D (VID file) — absent in the 2017-2020 pre-pandemic release
    "LBXVIDMS": "vitamin_d",
    # diabetes questionnaire (label-only)
    "DIQ010": "diq_diabetes",
    "DIQ050": "diq_insulin",
    "DIQ070": "diq_pills",
    # bp / cholesterol questionnaire (label-only)
    "BPQ080":  "bpq_high_chol",
    "BPQ090D": "bpq_chol_med",
    "BPQ020":  "bpq_high_bp",
    "BPQ040A": "bpq_bp_med",
    # medical conditions (label-only)
    "MCQ160C": "mcq_chd",
    "MCQ160E": "mcq_mi",
    "MCQ160F": "mcq_stroke",
    "MCQ160B": "mcq_chf",
}

# Survey weight variable differs by cycle: WTMEC2YR (H/I) vs WTMECPRP (P_).
WEIGHT_VARS = ("WTMEC2YR", "WTMECPRP")

# --------------------------------------------------------------------------- #
# Column role definitions — the leakage-control contract.
# --------------------------------------------------------------------------- #
# =========================================================================== #
# Biomarker panels (blood-report analytes), grouped by clinical panel.
# =========================================================================== #
CARDIOMETABOLIC = [
    "hba1c_pct", "fasting_glucose_mgdl",
    "total_chol_mgdl", "ldl_mgdl", "hdl_mgdl", "triglycerides_mgdl",
]
CBC_ANAEMIA = ["hemoglobin", "hematocrit", "rbc", "mcv", "mch", "mchc", "rdw"]
LIVER = ["alt", "ast", "alp", "albumin", "total_bilirubin"]
KIDNEY = ["creatinine", "bun"]
VITAMINS = ["vitamin_d"]   # optional: absent for the 2017-2020 cycle

# Flag-only blood analytes: reported & escalated by the rule engine, but NEVER
# used to generate lifestyle advice, and EXCLUDED from the model (medical, not
# lifestyle-driven). They do not feed the RF label or features.
FLAG_ONLY_BIOMARKERS = ["wbc", "platelets", "sodium", "potassium", "chloride", "calcium"]

# NOTE: BP / BMI / waist are intentionally OUT OF SCOPE — this is a blood-test-
# report system. Non-blood measurements are not ingested, labelled, or modelled;
# weight & blood pressure are signposted to the GP instead.

# --- Roles ---------------------------------------------------------------- #
# Actionable BLOOD analytes: drive the multi-system severity label & recommendations.
ACTIONABLE_BLOOD = CARDIOMETABOLIC + CBC_ANAEMIA + LIVER + KIDNEY + VITAMINS

# Mandatory (drop-if-missing): the full-sample analytes that anchor the label.
MANDATORY_BIOMARKERS = ["hba1c_pct", "total_chol_mgdl", "hdl_mgdl"]

# RF FEATURES = actionable blood analytes, minus vitamin_d (structurally missing
# for a whole cycle). Flag-only markers are excluded.
FEATURE_BIOMARKERS = [b for b in ACTIONABLE_BLOOD if b != "vitamin_d"]

# Imputed (non-mandatory feature columns). vitamin_d & flag-only are NOT imputed.
SUPPLEMENTARY_BIOMARKERS = [b for b in FEATURE_BIOMARKERS if b not in MANDATORY_BIOMARKERS]

# Everything carried through cleaning/winsorising/plausibility (for reporting too).
ALL_BIOMARKERS = ACTIONABLE_BLOOD + FLAG_ONLY_BIOMARKERS

# Demographic features fed to the model.
DEMOGRAPHIC_FEATURES = ["age", "sex_code", "eth_code", "educ_code", "pir"]

# LABEL-ONLY columns: used to build the target, NEVER used as RF features.
LABEL_ONLY_COLUMNS = [
    "diq_diabetes", "diq_insulin", "diq_pills",
    "bpq_high_chol", "bpq_chol_med", "bpq_high_bp", "bpq_bp_med",
    "mcq_chd", "mcq_mi", "mcq_stroke", "mcq_chf",
    "statin_or_metformin",   # engineered flag from RXQ_RX
]

# Survey-design columns kept for provenance but excluded from features.
DESIGN_COLUMNS = ["psu", "strata", "wtmec", "cycle"]

# Adult-only cohort: NHANES surveys all ages, but this system is an adult
# cardiometabolic recommender. Paediatric records use percentile-based reference
# ranges (not the adult thresholds here) and lack the adult diagnosis/medication
# questionnaires, so they are excluded before labelling.
ADULT_MIN_AGE = 18


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
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
