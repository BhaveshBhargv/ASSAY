"""Engineered features. Each row is handled on its own, so there's nothing to fit.

tc_hdl_ratio   total cholesterol / HDL
tg_hdl_ratio   triglycerides / HDL, linked to insulin resistance
tyg_index      ln(TG * fasting glucose / 2), another insulin resistance marker
age_band       age group from 0 to 4

These are added after imputation so the ratios don't come out as NaN.
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
