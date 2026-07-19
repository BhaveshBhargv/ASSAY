"""
missing.py — hybrid missing-value strategy (approved).

Rule 1 (integrity): drop any row missing a MANDATORY (label-defining) biomarker.
        We never fabricate the ground truth the model learns.
Rule 2 (retention): median-impute SUPPLEMENTARY biomarkers and soft demographic
        predictors (education, income ratio, height). Imputers are FIT ON TRAIN
        ONLY and applied to test — no leakage.

Note: labels are built (labeling.py) BEFORE supplementary imputation, so Layer-1
severity is only ever driven by genuinely observed values.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from .config import MANDATORY_BIOMARKERS, SUPPLEMENTARY_BIOMARKERS

log = logging.getLogger(__name__)

# Soft predictors that may be missing and are safe to impute.
_SOFT_PREDICTORS = ["pir", "educ_code"]


def drop_missing_mandatory(df: pd.DataFrame) -> pd.DataFrame:
    """Complete-case on the mandatory biomarkers only."""
    present = [c for c in MANDATORY_BIOMARKERS if c in df.columns]
    before = len(df)
    out = df.dropna(subset=present).reset_index(drop=True)
    log.info("dropped %d rows missing mandatory biomarkers (%d -> %d)",
             before - len(out), before, len(out))
    return out


def fit_imputer(train: pd.DataFrame, strategy: str = "median", knn_neighbors: int = 5):
    """
    Fit an imputer on the supplementary + soft-predictor columns using TRAIN only.
    Returns (imputer, columns).
    """
    cols = [c for c in (SUPPLEMENTARY_BIOMARKERS + _SOFT_PREDICTORS) if c in train.columns]
    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        imputer = SimpleImputer(strategy=strategy)
    imputer.fit(train[cols])
    return imputer, cols


def apply_imputer(df: pd.DataFrame, imputer, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = imputer.transform(out[cols])
    return out
