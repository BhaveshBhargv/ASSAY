"""Outlier handling, in two steps.

Before the train/test split, impossible values (outside fixed plausibility
bounds) are set to NaN so they get imputed later. After the split, values are
winsorised to the 1st and 99th percentiles of the training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_BIOMARKERS


def apply_plausibility(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Set physiologically impossible values to NaN."""
    out = df.copy()
    for name, bounds in thresholds.get("plausibility", {}).items():
        if name not in out.columns:
            continue
        v = pd.to_numeric(out[name], errors="coerce")
        lo, hi = bounds.get("min"), bounds.get("max")
        v = v.mask((v < lo) | (v > hi))
        out[name] = v
    return out


def fit_winsor_bounds(
    train: pd.DataFrame, cols: list[str] | None = None, lower: float = 0.01, upper: float = 0.99
) -> dict[str, tuple[float, float]]:
    """Winsorisation bounds from the training set."""
    cols = cols or [c for c in ALL_BIOMARKERS if c in train.columns]
    bounds: dict[str, tuple[float, float]] = {}
    for c in cols:
        v = pd.to_numeric(train[c], errors="coerce")
        bounds[c] = (float(v.quantile(lower)), float(v.quantile(upper)))
    return bounds


def apply_winsor(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip values to the given bounds."""
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
    return out
