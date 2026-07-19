"""
predict.py — thin inference wrapper around the saved Random Forest.

Used later by the API/fusion layer. Aligns an incoming feature dict to the
model's training columns (missing -> NaN handled by imputation upstream) and
returns the predicted severity class + class probabilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import INT_TO_LABEL, LABEL_ORDER, REGISTRY_DIR
from .train import load_model


class RiskModel:
    def __init__(self, registry_dir: Path = REGISTRY_DIR) -> None:
        self.model = load_model(registry_dir)
        self.feature_names = list(self.model.feature_names_in_)

    def predict(self, features: dict[str, float]) -> dict:
        row = pd.DataFrame([{f: features.get(f, np.nan) for f in self.feature_names}])
        proba = self.model.predict_proba(row)[0]
        idx = int(np.argmax(proba))
        return {
            "predicted_severity": INT_TO_LABEL[idx],
            "probability": float(proba[idx]),
            "probabilities": {LABEL_ORDER[i]: float(proba[i]) for i in range(len(LABEL_ORDER))},
        }
