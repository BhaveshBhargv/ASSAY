"""Runs the full data preparation.

The order of the steps matters:
  1. merge the cycles
  2. set implausible values to NaN
  3. build the labels, before any imputation
  4. drop rows missing a mandatory marker
  5. stratified train/test split (nothing is fitted before this point)
  6. winsorise, fitted on train
  7. impute, fitted on train
  8. add the engineered features
  9. build the feature matrices, unscaled and scaled
 10. save everything
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import encode_scale, features, missing, outliers
from .config import (
    ADULT_MIN_AGE,
    ALL_BIOMARKERS,
    DEMOGRAPHIC_FEATURES,
    DESIGN_COLUMNS,
    LABEL_ONLY_COLUMNS,
    PROCESSED_DIR,
    load_thresholds,
)
from .labeling import build_labels, label_report
from .merge import build_merged

log = logging.getLogger(__name__)


def _select_working_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns used later on."""
    wanted = (
        ["SEQN"]
        + ALL_BIOMARKERS
        + DEMOGRAPHIC_FEATURES
        + LABEL_ONLY_COLUMNS
        + DESIGN_COLUMNS
    )
    keep = [c for c in wanted if c in df.columns]
    return df[keep].copy()


def run_pipeline(
    test_size: float = 0.20,
    seed: int = 42,
    impute_strategy: str = "median",
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    thresholds = load_thresholds()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(exist_ok=True)

    # 1-2. merge and plausibility
    merged = build_merged()
    merged = _select_working_columns(merged)
    merged = outliers.apply_plausibility(merged, thresholds)

    # adults only
    before = len(merged)
    merged = merged[merged["age"] >= ADULT_MIN_AGE].reset_index(drop=True)
    log.info("adult filter (age >= %d): %d -> %d rows", ADULT_MIN_AGE, before, len(merged))

    # 3. labels
    labelled = build_labels(merged, thresholds)
    rpt = label_report(labelled)
    log.info("label report: %s", rpt)

    # 4. drop rows missing mandatory markers
    clean = missing.drop_missing_mandatory(labelled)

    # 5. split
    train, test = train_test_split(
        clean, test_size=test_size, random_state=seed, stratify=clean["label"]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # 6. winsorise
    wbounds = outliers.fit_winsor_bounds(train)
    train = outliers.apply_winsor(train, wbounds)
    test = outliers.apply_winsor(test, wbounds)

    # 7. impute
    imputer, imp_cols = missing.fit_imputer(train, strategy=impute_strategy)
    train = missing.apply_imputer(train, imputer, imp_cols)
    test = missing.apply_imputer(test, imputer, imp_cols)

    # 8. engineered features
    train = features.add_features(train)
    test = features.add_features(test)

    # 9. feature matrices
    X_train = encode_scale.build_feature_matrix(train)
    X_test = encode_scale.build_feature_matrix(test)
    scaler, cont = encode_scale.fit_scaler(X_train)
    X_train_scaled = encode_scale.apply_scaler(X_train, scaler, cont)
    X_test_scaled = encode_scale.apply_scaler(X_test, scaler, cont)
    y_train, y_test = train["label"], test["label"]

    # 10. save
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    X_train_scaled.to_csv(out_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(out_dir / "X_test_scaled.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    joblib.dump(imputer, art_dir / "imputer.joblib")
    joblib.dump(scaler, art_dir / "scaler.joblib")
    (art_dir / "winsor_bounds.json").write_text(json.dumps(wbounds, indent=2))
    (art_dir / "feature_names.json").write_text(json.dumps(list(X_train.columns), indent=2))

    summary = {
        "rows_total": int(len(clean)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "imputed_columns": imp_cols,
        "label_report": rpt,
        "train_label_distribution": y_train.value_counts().to_dict(),
        "test_label_distribution": y_test.value_counts().to_dict(),
        "config": {
            "test_size": test_size, "seed": seed,
            "impute_strategy": impute_strategy, "adult_min_age": ADULT_MIN_AGE,
        },
    }
    (art_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Phase 2 complete: %d train / %d test rows, %d features",
             len(train), len(test), X_train.shape[1])
    return summary
