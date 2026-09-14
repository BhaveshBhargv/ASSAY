"""Smoke test of training, evaluation, SHAP and prediction on synthetic data.

Run:  python tests/test_ml.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ml.data import Dataset  # noqa: E402
from app.ml.evaluate import evaluate  # noqa: E402
from app.ml.explain import explain  # noqa: E402
from app.ml.train import train  # noqa: E402
from app.ml.predict import RiskModel  # noqa: E402


def _synthetic(n=600, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    feats = ["hba1c_pct", "ldl_mgdl", "alt", "hemoglobin", "tyg_index"]
    X = pd.DataFrame(rng.normal(size=(n, len(feats))), columns=feats)
    # the label depends on two of the features so there's something to learn
    score = X["hba1c_pct"] + 0.5 * X["ldl_mgdl"] + rng.normal(0, 0.3, n)
    y = np.digitize(score, np.quantile(score, [0.55, 0.85]))  # 3 unbalanced classes
    cut = int(n * 0.8)
    mask = np.zeros(n - cut, dtype=bool)
    mask[:20] = True
    return Dataset(
        X_train=X.iloc[:cut].reset_index(drop=True),
        X_test=X.iloc[cut:].reset_index(drop=True),
        y_train=y[:cut], y_test=y[cut:],
        feature_names=feats, test_novelty_mask=mask,
    )


def test_train_evaluate_explain_predict():
    data = _synthetic()
    with tempfile.TemporaryDirectory() as td:
        reg = Path(td) / "registry"
        rep = Path(td) / "reports"
        res = train(data, param_grid={"n_estimators": [50], "max_depth": [4]},
                    cv_splits=3, registry_dir=reg)
        assert res.model_path.exists()
        assert 0.0 <= res.cv_best_macro_f1 <= 1.0

        metrics = evaluate(res.model, data, reports_dir=rep)
        assert set(metrics) >= {"accuracy", "macro", "weighted", "roc_auc_macro_ovr",
                                "per_class", "confusion_matrix", "novelty_slice"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert (rep / "confusion_matrix.png").exists()
        assert (rep / "metrics.json").exists()
        assert metrics["novelty_slice"]["n"] == 20

        imp = explain(res.model, data, reports_dir=rep, sample=100)
        assert list(imp["feature"])
        assert (rep / "feature_importance.csv").exists()

        rm = RiskModel(registry_dir=reg)
        out = rm.predict({"hba1c_pct": 2.0, "ldl_mgdl": 2.0})
        assert out["predicted_severity"] in {"normal", "borderline", "serious"}
        assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
    print("PASS test_train_evaluate_explain_predict")


if __name__ == "__main__":
    test_train_evaluate_explain_predict()
    print("\nALL ML SMOKE TESTS PASSED")
