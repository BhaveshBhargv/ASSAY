"""Train, evaluate and explain the Random Forest.

Needs the processed dataset in data/processed/ (see run_data_prep.py).

    python scripts/train_model.py
    python scripts/train_model.py --quick     # small grid for a quick run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ml.data import load_dataset  # noqa: E402
from app.ml.evaluate import evaluate  # noqa: E402
from app.ml.explain import explain  # noqa: E402
from app.ml.train import train  # noqa: E402

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
        m, v = ns["model"], ns["vs_best_constant"]
        ci = m["exact_accuracy_ci95"]
        print(f"\nNOVELTY SLICE (n={ns['n']}) — true labels {ns['true_class_distribution']}")
        print(f"  model exact accuracy : {m['exact_accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for name, b in ns["baselines"].items():
            bci = b["exact_accuracy_ci95"]
            print(f"  {name:<22}: {b['exact_accuracy']:.3f} [{bci[0]:.3f}, {bci[1]:.3f}]")
        print(f"  vs {v['baseline']}: McNemar p={v['mcnemar_exact_p']:.3g} "
              f"(model wins {v['model_correct_baseline_wrong']}, "
              f"loses {v['model_wrong_baseline_correct']}) -> "
              f"{'model better' if v['model_beats_baseline'] else 'MODEL DOES NOT BEAT BASELINE'}")
    print("\ntop features:", ", ".join(imp["feature"].head(8)))
    print(f"\nmodel -> {result.model_path}")
    print(f"reports -> {ROOT / 'reports' / 'phase4'}")


if __name__ == "__main__":
    main()
