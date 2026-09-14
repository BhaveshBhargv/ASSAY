"""One-hot encoding and scaling for the feature matrix.

The model only uses biomarkers, age and sex. Ethnicity, income and education
were dropped because a blood report doesn't include them. Sex is one-hot encoded
with fixed categories so train and test always end up with the same columns.
The Random Forest uses the unscaled matrix; a scaled copy fitted on the training
set is saved as well.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_BIOMARKERS
from .features import ENGINEERED_NUMERIC, ENGINEERED_ORDINAL

_SEX_CATS = [1, 2]

# Continuous features (no flag-only markers or vitamin D).
_NUMERIC_BASE = ["age"] + FEATURE_BIOMARKERS + ENGINEERED_NUMERIC
_ORDINAL = list(ENGINEERED_ORDINAL)


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    sex = pd.Categorical(df["sex_code"], categories=_SEX_CATS)
    sex_d = pd.get_dummies(sex, prefix="sex")
    sex_d.index = df.index
    return sex_d


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The unscaled feature matrix the model is trained on."""
    numeric = [c for c in _NUMERIC_BASE if c in df.columns]
    ordinal = [c for c in _ORDINAL if c in df.columns]
    base = df[numeric + ordinal].apply(pd.to_numeric, errors="coerce")
    dummies = _one_hot(df).astype("int8")
    X = pd.concat([base, dummies], axis=1)
    return X


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, list[str]]:
    """Fit a StandardScaler on the continuous columns of the training set."""
    cont = [c for c in _NUMERIC_BASE if c in X_train.columns]
    scaler = StandardScaler().fit(X_train[cont])
    return scaler, cont


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler, cont: list[str]) -> pd.DataFrame:
    out = X.copy()
    out[cont] = scaler.transform(out[cont])
    return out
