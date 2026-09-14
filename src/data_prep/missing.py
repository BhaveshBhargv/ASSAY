"""Missing values: drop rows without the label-defining markers, impute the rest.

Rows missing a mandatory biomarker are dropped, since the label can't be built
without it. Supplementary biomarkers and a couple of demographic columns are
imputed, with the imputer fitted on the training set only. Labels are built
before this step, so they only ever use real measurements.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from .config import MANDATORY_BIOMARKERS, SUPPLEMENTARY_BIOMARKERS

log = logging.getLogger(__name__)

# Demographic columns that are fine to impute.
_SOFT_PREDICTORS = ["pir", "educ_code"]


def drop_missing_mandatory(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any mandatory biomarker."""
    present = [c for c in MANDATORY_BIOMARKERS if c in df.columns]
    before = len(df)
    out = df.dropna(subset=present).reset_index(drop=True)
    log.info("dropped %d rows missing mandatory biomarkers (%d -> %d)",
             before - len(out), before, len(out))
    return out


def fit_imputer(train: pd.DataFrame, strategy: str = "median", knn_neighbors: int = 5):
    """Fit an imputer on the training set. Returns (imputer, columns)."""
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
