"""
features.py — clinically-motivated feature engineering (stateless, no leakage).

Derived features give the Random Forest interaction signal in explicit form
(all blood-derived — no anthropometric inputs):
  tc_hdl_ratio      total cholesterol / HDL     (atherogenic ratio)
  tg_hdl_ratio      triglycerides / HDL          (insulin-resistance surrogate)
  tyg_index         ln(TG * fasting glucose / 2) (validated IR marker)
  age_band          0..4 ordinal decade-ish bands

Applied AFTER imputation so ratios are not spuriously NaN. All row-wise
transforms — identical on train and test, so nothing to fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"total_chol_mgdl", "hdl_mgdl"} <= set(out.columns):
        out["tc_hdl_ratio"] = _safe_div(out["total_chol_mgdl"], out["hdl_mgdl"])
    if {"triglycerides_mgdl", "hdl_mgdl"} <= set(out.columns):
        out["tg_hdl_ratio"] = _safe_div(out["triglycerides_mgdl"], out["hdl_mgdl"])
    if {"triglycerides_mgdl", "fasting_glucose_mgdl"} <= set(out.columns):
        prod = out["triglycerides_mgdl"] * out["fasting_glucose_mgdl"] / 2.0
        out["tyg_index"] = np.log(prod.clip(lower=1e-6))

    if "age" in out.columns:
        out["age_band"] = pd.cut(
            out["age"], bins=[-np.inf, 30, 45, 60, 75, np.inf], labels=[0, 1, 2, 3, 4]
        ).astype("float").astype("Int64")

    return out


ENGINEERED_NUMERIC = ["tc_hdl_ratio", "tg_hdl_ratio", "tyg_index"]
ENGINEERED_ORDINAL = ["age_band"]
