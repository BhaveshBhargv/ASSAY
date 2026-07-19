"""
encode_scale.py — categorical encoding + optional scaling, with train/test alignment.

The model is intentionally CLINICAL-ONLY: biomarkers + age + sex. Socio-demographic
inputs (ethnicity, income-to-poverty ratio, education) were dropped — a blood
report doesn't carry them, and they aren't clinically actionable, so the model
must not depend on them.

Encoding:
  sex_code                      -> one-hot (nominal; fixed categories so train and
                                   test always share identical columns)
  age_band                      -> kept as an ordinal integer
Scaling:
  The Random Forest is scale-invariant, so we return an UNSCALED matrix for it.
  We ALSO return a StandardScaler-transformed copy (fit on train only) for any
  distance-based / RAG-side use. Demonstrating that trees need no scaling is a
  deliberate design point.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_BIOMARKERS
from .features import ENGINEERED_NUMERIC, ENGINEERED_ORDINAL

# Fixed nominal categories.
_SEX_CATS = [1, 2]

# Numeric (continuous) features carried through (flag-only & vitamin_d excluded).
_NUMERIC_BASE = ["age"] + FEATURE_BIOMARKERS + ENGINEERED_NUMERIC
# Ordinal integer features.
_ORDINAL = list(ENGINEERED_ORDINAL)


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    sex = pd.Categorical(df["sex_code"], categories=_SEX_CATS)
    sex_d = pd.get_dummies(sex, prefix="sex")
    sex_d.index = df.index
    return sex_d


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model-ready UNSCALED feature matrix (leakage-safe columns only)."""
    numeric = [c for c in _NUMERIC_BASE if c in df.columns]
    ordinal = [c for c in _ORDINAL if c in df.columns]
    base = df[numeric + ordinal].apply(pd.to_numeric, errors="coerce")
    dummies = _one_hot(df).astype("int8")
    X = pd.concat([base, dummies], axis=1)
    return X


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, list[str]]:
    """Fit a StandardScaler on the continuous columns of TRAIN only."""
    cont = [c for c in _NUMERIC_BASE if c in X_train.columns]
    scaler = StandardScaler().fit(X_train[cont])
    return scaler, cont


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler, cont: list[str]) -> pd.DataFrame:
    out = X.copy()
    out[cont] = scaler.transform(out[cont])
    return out
