"""
train_model.py — Phase 4 entrypoint: train -> evaluate -> explain.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --quick     # tiny grid for a fast smoke run

Requires the Phase-2 processed dataset in data/processed/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ml.data import load_dataset            # noqa: E402
from app.ml.evaluate import evaluate            # noqa: E402
from app.ml.explain import explain              # noqa: E402
from app.ml.train import train                  # noqa: E402

_QUICK_GRID = {"n_estimators": [200], "max_depth": [None], "min_samples_leaf": [5]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 — train the Random Forest")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data = load_dataset()
    print(f"loaded: {len(data.X_train)} train / {len(data.X_test)} test, "
          f"{len(data.feature_names)} features")

    result = train(data, param_grid=_QUICK_GRID if args.quick else None)
    metrics = evaluate(result.model, data)
    imp = explain(result.model, data)

    print("\n=== Phase 4 results ===")
    print(f"CV macro-F1 (best) : {result.cv_best_macro_f1:.4f}")
    print(f"best params        : {result.best_params}")
    print(f"TEST accuracy      : {metrics['accuracy']:.4f}")
    print(f"TEST macro-F1      : {metrics['macro']['f1']:.4f}")
    print(f"TEST ROC-AUC (macro): {metrics['roc_auc_macro_ovr']:.4f}")
    print("per-class F1:", {k: round(v["f1"], 3) for k, v in metrics["per_class"].items()})
    ns = metrics["novelty_slice"]
    if ns.get("n"):
        print(f"\nNOVELTY SLICE (n={ns['n']}): hidden-risk detection = "
              f"{ns['risk_detected_rate']*100:.1f}% | exact acc = {ns['exact_label_accuracy']*100:.1f}%")
    print("\ntop features:", ", ".join(imp["feature"].head(8)))
    print(f"\nmodel -> {result.model_path}")
    print(f"reports -> {ROOT / 'reports' / 'phase4'}")


if __name__ == "__main__":
    main()
