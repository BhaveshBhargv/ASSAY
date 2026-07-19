"""
explain.py — model interpretability: Gini importance + SHAP.

Produces:
  * feature_importance.csv  — RF Gini importance and mean|SHAP| side by side
  * feature_importance.png  — top-N bar chart
  * shap_summary_<class>.png — SHAP beeswarm for each class (esp. 'serious')

SHAP turns the forest from a black box into a per-prediction, auditable
explanation — essential for a clinical-adjacent system.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .data import LABEL_ORDER, REPORTS_DIR, Dataset

log = logging.getLogger(__name__)


def _shap_values_3d(model, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP array shaped (n_samples, n_features, n_classes), robust to
    the two shap output conventions (list-per-class vs stacked array)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                      # older API: list per class
        return np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # (n, features, classes)
        return sv
    if sv.ndim == 2:                              # binary edge case
        return np.stack([-sv, sv], axis=-1)
    raise ValueError(f"unexpected SHAP shape {sv.shape}")


def explain(
    model, data: Dataset, reports_dir: Path = REPORTS_DIR, sample: int = 1000, seed: int = 42
) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    X = data.X_test
    if len(X) > sample:
        X = X.sample(sample, random_state=seed)

    # --- SHAP ------------------------------------------------------------- #
    sv = _shap_values_3d(model, X)                      # (n, feat, classes)
    mean_abs_shap = np.abs(sv).mean(axis=0)             # (feat, classes)
    mean_abs_overall = mean_abs_shap.mean(axis=1)       # avg across classes

    # --- Gini importance -------------------------------------------------- #
    gini = model.feature_importances_

    imp = pd.DataFrame({
        "feature": data.feature_names,
        "gini_importance": gini,
        "mean_abs_shap": mean_abs_overall,
    })
    for i, cls in enumerate(LABEL_ORDER):
        imp[f"shap_{cls}"] = mean_abs_shap[:, i]
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(reports_dir / "feature_importance.csv", index=False)

    _plot_top_importance(imp, reports_dir / "feature_importance.png")

    # --- per-class SHAP beeswarm ----------------------------------------- #
    for i, cls in enumerate(LABEL_ORDER):
        try:
            plt.figure()
            shap.summary_plot(sv[:, :, i], X, show=False, max_display=15)
            plt.title(f"SHAP — class '{cls}'")
            plt.tight_layout()
            plt.savefig(reports_dir / f"shap_summary_{cls}.png", dpi=130)
            plt.close()
        except Exception as exc:  # plotting is best-effort
            log.warning("SHAP beeswarm for %s failed: %s", cls, exc)

    log.info("top features: %s", ", ".join(imp["feature"].head(8)))
    return imp


def _plot_top_importance(imp: pd.DataFrame, path: Path, top: int = 15) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#4C78A8")
    ax.set_xlabel("mean |SHAP| (avg across classes)")
    ax.set_title(f"Top {top} features")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
