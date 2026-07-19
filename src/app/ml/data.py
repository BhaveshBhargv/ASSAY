"""
data.py — load the Phase-2 processed dataset for the Random Forest.

The feature matrices (unscaled, model-ready) and labels were produced by the
Phase-2 pipeline. This module just loads them, encodes the 3-class label into a
fixed ordinal (normal < borderline < serious), and exposes the Layer-2 novelty
mask used to evaluate the model's core purpose — detecting hidden risk where
individual biomarkers look normal/borderline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"

# Fixed ordinal class order (kept stable so encodings/plots are reproducible).
LABEL_ORDER = ["normal", "borderline", "serious"]
LABEL_TO_INT = {c: i for i, c in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {i: c for c, i in LABEL_TO_INT.items()}


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray            # encoded ints
    y_test: np.ndarray
    feature_names: list[str]
    # Layer-2 novelty slice mask over the TEST set (bloods normal/borderline but
    # diagnosis/medication evidence upgraded the label — the RF must recover this
    # from biomarker patterns alone).
    test_novelty_mask: np.ndarray


def _load_labels(path: Path) -> np.ndarray:
    s = pd.read_csv(path)
    col = s.columns[0]
    return s[col].map(LABEL_TO_INT).to_numpy()


def load_dataset(processed_dir: Path = PROCESSED_DIR) -> Dataset:
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = _load_labels(processed_dir / "y_train.csv")
    y_test = _load_labels(processed_dir / "y_test.csv")

    # Novelty mask aligns row-for-row with X_test (same split order).
    test_meta = pd.read_csv(processed_dir / "test.csv")
    if "label_upgraded_by_l2" in test_meta.columns:
        mask = test_meta["label_upgraded_by_l2"].astype(bool).to_numpy()
    else:
        mask = np.zeros(len(X_test), dtype=bool)

    if len(mask) != len(X_test):
        raise ValueError("test.csv and X_test.csv are misaligned")

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_train.columns),
        test_novelty_mask=mask,
    )
