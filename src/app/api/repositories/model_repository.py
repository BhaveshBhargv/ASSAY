"""Loads the Random Forest and its metadata."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("assay.api")

_REGISTRY = Path(__file__).resolve().parents[2] / "ml" / "registry"


class ModelRepository:
    def __init__(self) -> None:
        self._adapter = None
        self._metadata: dict = {}
        try:
            from app.recommend.engine import RiskAdapter
            self._adapter = RiskAdapter()
            log.info("random forest loaded")
        except Exception as exc:  # noqa: BLE001
            log.warning("random forest unavailable (%s); predictions will be rules-only", exc)

        meta_path = _REGISTRY / "rf_model_metadata.json"
        if meta_path.exists():
            try:
                self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read RF metadata: %s", exc)

    @property
    def available(self) -> bool:
        return self._adapter is not None

    @property
    def adapter(self):
        return self._adapter

    def predict(self, demographics: dict, biomarkers: dict):
        """The model's output, or None if the model isn't available."""
        if self._adapter is None:
            return None
        return self._adapter.predict(demographics, biomarkers)

    def info(self) -> dict:
        m = self._metadata
        return {
            "available": self.available,
            "algorithm": m.get("model_type", "RandomForestClassifier" if self.available else None),
            "classes": m.get("label_order", []),
            "n_features": m.get("n_features"),
            "feature_names": m.get("feature_names", []),
            "metrics": {k: m[k] for k in ("cv_best_macro_f1", "best_params", "scoring", "n_train")
                        if k in m},
        }
