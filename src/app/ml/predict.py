"""Run the saved Random Forest on a single patient's features.

For a raised-risk prediction it also returns the features whose SHAP values
pushed the prediction up the most.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import INT_TO_LABEL, LABEL_ORDER, REGISTRY_DIR
from .train import load_model

# Readable names for the feature columns, shown in the UI and the PDF.
_FEATURE_LABELS = {
    "hba1c_pct": "HbA1c",
    "fasting_glucose_mgdl": "Fasting glucose",
    "total_chol_mgdl": "Total cholesterol",
    "ldl_mgdl": "LDL cholesterol",
    "hdl_mgdl": "HDL cholesterol",
    "triglycerides_mgdl": "Triglycerides",
    "hemoglobin": "Haemoglobin",
    "hematocrit": "Haematocrit",
    "rbc": "Red blood cell count",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "albumin": "Albumin",
    "total_bilirubin": "Total bilirubin",
    "creatinine": "Creatinine",
    "bun": "Urea (BUN)",
    "tc_hdl_ratio": "Total-cholesterol-to-HDL ratio",
    "tg_hdl_ratio": "Triglyceride-to-HDL ratio",
    "tyg_index": "TyG index (insulin-resistance marker)",
    "age": "Age",
    "age_band": "Age group",
}


def feature_label(feature: str) -> str:
    if feature.startswith("sex_"):
        return "Sex"
    return _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


class RiskModel:
    def __init__(self, registry_dir: Path = REGISTRY_DIR) -> None:
        self.model = load_model(registry_dir)
        self.feature_names = list(self.model.feature_names_in_)
        self._explainer = None  # SHAP explainer, created on first use

    def predict(self, features: dict[str, float]) -> dict:
        row = pd.DataFrame([{f: features.get(f, np.nan) for f in self.feature_names}])
        proba = self.model.predict_proba(row)[0]
        idx = int(np.argmax(proba))
        out = {
            "predicted_severity": INT_TO_LABEL[idx],
            "probability": float(proba[idx]),
            "probabilities": {LABEL_ORDER[i]: float(proba[i]) for i in range(len(LABEL_ORDER))},
        }
        # Drivers only make sense for a raised-risk prediction.
        if idx > 0:
            out["drivers"] = self._drivers(row, idx)
        return out

    def _drivers(self, row: pd.DataFrame, class_idx: int, top: int = 4) -> list[dict]:
        """Features that pushed the prediction towards class_idx, strongest first.

        Returns an empty list if SHAP fails, so prediction still works without it.
        """
        try:
            import shap
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            contribs = _class_shap(self._explainer.shap_values(row), class_idx)
        except Exception:  # noqa: BLE001
            return []

        drivers = []
        for i, feat in enumerate(self.feature_names):
            contrib = float(contribs[i])
            if contrib <= 0:
                continue  # only features that increased the risk
            if feat.startswith("sex_") and _num(row.iloc[0][feat]) == 0:
                continue  # the sex column that doesn't apply to this patient
            drivers.append({
                "feature": feat,
                "label": feature_label(feat),
                "value": _num(row.iloc[0][feat]),
                "contribution": contrib,
            })
        drivers.sort(key=lambda d: d["contribution"], reverse=True)

        # Keep only the strongest entry per label (both sex columns show as "Sex").
        seen, deduped = set(), []
        for d in drivers:
            if d["label"] in seen:
                continue
            seen.add(d["label"])
            deduped.append(d)
        return deduped[:top]


def _class_shap(sv, class_idx: int) -> np.ndarray:
    """SHAP values for the first row and one class, for either shap output format."""
    if isinstance(sv, list):  # older versions: a list per class
        return np.asarray(sv[class_idx])[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        return sv[0, :, class_idx]
    if sv.ndim == 2:  # binary case
        return sv[0]
    raise ValueError(f"unexpected SHAP shape {sv.shape}")


def _num(v):
    """Round a value for display, or None if it isn't a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, 2)
