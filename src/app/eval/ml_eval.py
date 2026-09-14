"""ML metrics and feature importance for the evaluation report.

Reuses evaluate() and explain() from app.ml, writes their output to
reports/phase9 and adds a chart of the headline metrics.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.ml.data import load_dataset
from app.ml.evaluate import evaluate
from app.ml.explain import explain
from app.ml.train import load_model

from . import config as C

log = logging.getLogger("assay.eval")

_TEAL, _GREEN, _AMBER, _RED = "#0E7C86", "#1F9D74", "#C8871B", "#C24A57"


def evaluate_ml() -> dict:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    model = load_model()

    metrics = evaluate(model, data, reports_dir=C.REPORTS_DIR)
    imp = explain(model, data, reports_dir=C.REPORTS_DIR)

    headline = {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["macro"]["precision"],
        "recall_macro": metrics["macro"]["recall"],
        "f1_macro": metrics["macro"]["f1"],
        "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
    }
    _plot_headline(headline, metrics["per_class"])

    summary = {
        "n_test": int(len(data.y_test)),
        "n_features": len(data.feature_names),
        "headline": headline,
        "per_class": {k: {m: v[m] for m in ("precision", "recall", "f1", "support")}
                      for k, v in metrics["per_class"].items()},
        "confusion_matrix": metrics["confusion_matrix"],
        "novelty_slice": metrics["novelty_slice"],
        "top_features": list(imp["feature"].head(10)),
    }
    log.info("ML: acc=%.3f macroF1=%.3f rocauc=%.3f",
             headline["accuracy"], headline["f1_macro"], headline["roc_auc_macro_ovr"])
    return summary


def _plot_headline(headline: dict, per_class: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    names = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "ROC-AUC\n(macro)"]
    vals = [headline["accuracy"], headline["precision_macro"], headline["recall_macro"],
            headline["f1_macro"], headline["roc_auc_macro_ovr"]]
    bars = ax1.bar(names, vals, color=_TEAL)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Random Forest — headline metrics", fontsize=11)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    classes = list(per_class.keys())
    colors = {"normal": _GREEN, "borderline": _AMBER, "serious": _RED}
    x = range(len(classes))
    width = 0.26
    for i, m in enumerate(("precision", "recall", "f1")):
        offs = (i - 1) * width
        ax2.bar([xi + offs for xi in x], [per_class[c][m] for c in classes], width,
                label=m, color=[colors[c] for c in classes], alpha=0.55 + 0.2 * i)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-class precision / recall / F1", fontsize=11)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "ml_headline_metrics.png", dpi=140)
    plt.close(fig)
