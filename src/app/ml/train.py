"""
train.py — train the Random Forest with GridSearchCV + stratified CV.

Design (approved):
  * RandomForestClassifier(class_weight="balanced")  — handle class imbalance
  * GridSearchCV over a focused grid, scoring = macro-F1
  * StratifiedKFold(5) — preserves class proportions per fold
  * unscaled features (trees are scale-invariant)

The model is saved with joblib together with a metadata sidecar (best params,
CV score, feature names, label order, data fingerprint) for reproducibility.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import LABEL_ORDER, REGISTRY_DIR, Dataset

log = logging.getLogger(__name__)

# Focused, defensible grid (keeps GridSearch tractable: 2*3*3*2 = 36 configs).
PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 12, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.5],
}


@dataclass
class TrainResult:
    model: RandomForestClassifier
    best_params: dict
    cv_best_macro_f1: float
    model_path: Path
    metadata_path: Path


def train(
    data: Dataset,
    param_grid: dict | None = None,
    cv_splits: int = 5,
    seed: int = 42,
    registry_dir: Path = REGISTRY_DIR,
) -> TrainResult:
    param_grid = param_grid or PARAM_GRID
    registry_dir.mkdir(parents=True, exist_ok=True)

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    log.info("fitting GridSearchCV over %d configs x %d folds ...",
             _grid_size(param_grid), cv_splits)
    search.fit(data.X_train, data.y_train)

    model: RandomForestClassifier = search.best_estimator_
    log.info("best macro-F1 (CV): %.4f | params: %s",
             search.best_score_, search.best_params_)

    model_path = registry_dir / "rf_model.joblib"
    joblib.dump(model, model_path, compress=3)  # keep the committed artefact small

    metadata = {
        "model_type": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_weight": "balanced",
        "scoring": "f1_macro",
        "cv_splits": cv_splits,
        "seed": seed,
        "best_params": search.best_params_,
        "cv_best_macro_f1": float(search.best_score_),
        "label_order": LABEL_ORDER,
        "n_features": data.X_train.shape[1],
        "feature_names": data.feature_names,
        "n_train": int(len(data.X_train)),
    }
    metadata_path = registry_dir / "rf_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log.info("saved model -> %s", model_path)

    return TrainResult(
        model=model,
        best_params=search.best_params_,
        cv_best_macro_f1=float(search.best_score_),
        model_path=model_path,
        metadata_path=metadata_path,
    )


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def load_model(registry_dir: Path = REGISTRY_DIR) -> RandomForestClassifier:
    return joblib.load(registry_dir / "rf_model.joblib")
