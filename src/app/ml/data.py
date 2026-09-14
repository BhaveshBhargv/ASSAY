"""Load the processed dataset used to train and evaluate the Random Forest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_DIR = Path(__file__).resolve().parent / "registry"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"

# Lowest to highest severity.
LABEL_ORDER = ["normal", "borderline", "serious"]

# Layer-2 columns meaning the patient is on medication. The analyte ones are
# drugs that lower something the panel measures (lipids, glucose); BP drugs
# don't, so BP-only patients work as a control when checking whether treatment
# hides the signal the model is supposed to find.
ANALYTE_TREATMENT_COLUMNS = ["diq_insulin", "diq_pills", "bpq_chol_med",
                             "statin_or_metformin"]
OTHER_TREATMENT_COLUMNS = ["bpq_bp_med"]
TREATMENT_COLUMNS = ANALYTE_TREATMENT_COLUMNS + OTHER_TREATMENT_COLUMNS
LABEL_TO_INT = {c: i for i, c in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {i: c for c, i in LABEL_TO_INT.items()}


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray  # label ints
    y_test: np.ndarray
    feature_names: list[str]
    # Test rows where Layer 2 raised the label above what the bloods alone gave.
    test_novelty_mask: np.ndarray
    # The rest are only set for the real dataset (None for synthetic test data).
    test_treated_mask: np.ndarray | None = None
    test_l1: np.ndarray | None = None  # rule-engine label from the bloods alone
    test_analyte_treated_mask: np.ndarray | None = None


def _load_labels(path: Path) -> np.ndarray:
    s = pd.read_csv(path)
    col = s.columns[0]
    return s[col].map(LABEL_TO_INT).to_numpy()


def load_dataset(processed_dir: Path = PROCESSED_DIR) -> Dataset:
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = _load_labels(processed_dir / "y_train.csv")
    y_test = _load_labels(processed_dir / "y_test.csv")

    # test.csv is in the same row order as X_test.csv.
    test_meta = pd.read_csv(processed_dir / "test.csv")
    if "label_upgraded_by_l2" in test_meta.columns:
        mask = test_meta["label_upgraded_by_l2"].astype(bool).to_numpy()
    else:
        mask = np.zeros(len(X_test), dtype=bool)

    if len(mask) != len(X_test):
        raise ValueError("test.csv and X_test.csv are misaligned")

    def _any_yes(columns: list[str]) -> np.ndarray:
        flag = np.zeros(len(X_test), dtype=bool)
        for col in columns:
            if col in test_meta.columns:
                flag |= pd.to_numeric(test_meta[col], errors="coerce").eq(1).to_numpy()
        return flag

    treated = _any_yes(TREATMENT_COLUMNS)
    analyte_treated = _any_yes(ANALYTE_TREATMENT_COLUMNS)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_train.columns),
        test_novelty_mask=mask,
        test_treated_mask=treated,
        test_analyte_treated_mask=analyte_treated,
        test_l1=(test_meta["label_l1"].map(LABEL_TO_INT).to_numpy()
                 if "label_l1" in test_meta.columns else None),
    )
