"""
evaluate.py — full evaluation of the trained Random Forest.

Reports (all on the held-out test set):
  accuracy, precision/recall/F1 (macro, weighted, per-class), ROC-AUC
  (macro-OVR + per-class), confusion matrix (+ plot), ROC curves (+ plot).

Plus the study's core question — the **Layer-2 novelty slice**: records whose
individual biomarkers looked normal/borderline but whose diagnosis/medication
evidence indicated higher risk. The RF never sees that evidence, so its ability
to flag these from biomarker patterns alone measures whether it has learned the
clinically significant *interaction* signal.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .data import INT_TO_LABEL, LABEL_ORDER, REPORTS_DIR, Dataset

log = logging.getLogger(__name__)
_CLASSES = list(range(len(LABEL_ORDER)))


def evaluate(model, data: Dataset, reports_dir: Path = REPORTS_DIR) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    X_test, y_test = data.X_test, data.y_test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- headline + per-class metrics ------------------------------------- #
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, labels=_CLASSES, average=None, zero_division=0)

    # --- ROC-AUC (one-vs-rest) -------------------------------------------- #
    y_test_bin = label_binarize(y_test, classes=_CLASSES)
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    auc_per_class = roc_auc_score(y_test_bin, y_proba, average=None)

    # --- confusion matrix -------------------------------------------------- #
    cm = confusion_matrix(y_test, y_pred, labels=_CLASSES)
    _plot_confusion(cm, reports_dir / "confusion_matrix.png")
    _plot_roc(y_test_bin, y_proba, reports_dir / "roc_curves.png")

    report_txt = classification_report(
        y_test, y_pred, labels=_CLASSES, target_names=LABEL_ORDER, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(report_txt)

    metrics = {
        "accuracy": float(acc),
        "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f_macro)},
        "weighted": {"precision": float(p_wt), "recall": float(r_wt), "f1": float(f_wt)},
        "roc_auc_macro_ovr": float(auc_macro),
        "per_class": {
            LABEL_ORDER[i]: {
                "precision": float(p_c[i]), "recall": float(r_c[i]),
                "f1": float(f_c[i]), "support": int(sup_c[i]),
                "roc_auc_ovr": float(auc_per_class[i]),
            } for i in _CLASSES
        },
        "confusion_matrix": {"labels": LABEL_ORDER, "counts": cm.tolist()},
        "novelty_slice": _evaluate_novelty(y_test, y_pred, data.test_novelty_mask),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("test macro-F1=%.4f accuracy=%.4f roc_auc=%.4f",
             f_macro, acc, auc_macro)
    return metrics


def _evaluate_novelty(y_test, y_pred, mask: np.ndarray) -> dict:
    """The RF's core purpose: recover hidden risk on the Layer-2 slice."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "note": "no novelty-slice records in test set"}
    yt, yp = y_test[mask], y_pred[mask]
    # These records are truly elevated (borderline/serious). "Risk detected" =
    # model predicts anything other than 'normal' from biomarkers alone.
    risk_detected = float(np.mean(yp != 0))
    exact = float(accuracy_score(yt, yp))
    return {
        "n": n,
        "description": "records upgraded by Layer-2 (bloods normal/borderline, "
                       "diagnosis/medication indicated higher risk)",
        "risk_detected_rate": risk_detected,   # predicts non-normal (any risk)
        "exact_label_accuracy": exact,
        "predicted_class_distribution": {
            INT_TO_LABEL[c]: int(np.sum(yp == c)) for c in _CLASSES
        },
    }


# --------------------------------------------------------------------------- #
def _plot_confusion(cm: np.ndarray, path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(_CLASSES); ax.set_yticks(_CLASSES)
    ax.set_xticklabels(LABEL_ORDER); ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in _CLASSES:
        for j in _CLASSES:
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_roc(y_test_bin: np.ndarray, y_proba: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i in _CLASSES:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{LABEL_ORDER[i]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
