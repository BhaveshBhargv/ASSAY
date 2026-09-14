"""Evaluate the trained Random Forest on the test set.

As well as the usual metrics, this looks at the novelty slice: test records
whose bloods looked normal or borderline but whose diagnosis or medication data
raised the label. The model never sees that data, so its results on these
records show whether it learned anything beyond the rule thresholds.
"""
from __future__ import annotations

import json
import logging
import math
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

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, labels=_CLASSES, average=None, zero_division=0)

    # one-vs-rest ROC-AUC
    y_test_bin = label_binarize(y_test, classes=_CLASSES)
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    auc_per_class = roc_auc_score(y_test_bin, y_proba, average=None)

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
        "novelty_slice": _evaluate_novelty(
            y_test, y_pred, data.test_novelty_mask,
            y_train=data.y_train, reports_dir=reports_dir,
            treated_mask=data.test_treated_mask, l1_labels=data.test_l1,
            analyte_treated_mask=data.test_analyte_treated_mask),
        "rule_agreement": _rule_agreement(
            y_test, y_pred, data.test_l1, data.test_novelty_mask),
        "fusion_analysis": _fusion_analysis(
            y_test, y_pred, data.test_l1, data.test_novelty_mask),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("test macro-F1=%.4f accuracy=%.4f roc_auc=%.4f",
             f_macro, acc, auc_macro)
    return metrics


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n.

    Better behaved than the normal approximation near 0 or 1 and for small n.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for two predictors on the same records.

    b is the number of records only the first got right, c the number only the
    second got right.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _score_predictor(pred: np.ndarray, y_true: np.ndarray) -> dict:
    n = len(y_true)
    k = int(np.sum(pred == y_true))
    lo, hi = _wilson(k, n)
    return {
        "exact_accuracy": round(k / n, 4),
        "exact_accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "risk_detected_rate": round(float(np.mean(pred != 0)), 4),
        "predicted_distribution": {INT_TO_LABEL[c]: int(np.sum(pred == c)) for c in _CLASSES},
    }


def _fusion_analysis(y_true, y_pred, l1: np.ndarray | None, mask: np.ndarray) -> dict:
    """Compare the rules alone, the model alone and the fused max(rule, model).

    Fusion can only raise the rule result, so the only harm it can do is a false
    escalation. Outside the novelty slice L2 <= L1, so the true label equals the
    rule label there and any escalation on those rows is wrong by definition.
    """
    if l1 is None:
        return {"note": "Layer-1 labels unavailable (synthetic dataset)"}

    fused = np.maximum(l1, y_pred)
    n_total = len(y_true)
    safe = ~mask  # the rule label is already correct on these rows

    def scored(pred) -> dict:
        k = int(np.sum(pred == y_true))
        lo, hi = _wilson(k, n_total)
        return {"accuracy": round(k / n_total, 4),
                "accuracy_ci95": [round(lo, 4), round(hi, 4)]}

    escalated = fused > l1
    false_esc = escalated & safe
    warranted = escalated & mask
    n_safe = int(safe.sum())
    fe_k = int(false_esc.sum())
    fe_lo, fe_hi = _wilson(fe_k, n_safe) if n_safe else (0.0, 0.0)

    rule_ok, fused_ok = l1 == y_true, fused == y_true
    fixed = int(np.sum(~rule_ok & fused_ok))
    broken = int(np.sum(rule_ok & ~fused_ok))

    return {
        "configurations": {
            "rule_engine_only": scored(l1),
            "model_only": scored(y_pred),
            "fused_max_rule_model": scored(fused),
        },
        "escalations": {
            "total": int(escalated.sum()),
            "false_on_non_novelty": fe_k,
            "warranted_on_novelty": int(warranted.sum()),
            "warranted_reached_true_label": int(np.sum(warranted & (fused == y_true))),
            "warranted_still_too_low": int(np.sum(warranted & (fused < y_true))),
            "warranted_overshot": int(np.sum(warranted & (fused > y_true))),
        },
        "false_escalation_rate": {
            "rate": round(fe_k / n_safe, 4) if n_safe else 0.0,
            "ci95": [round(fe_lo, 4), round(fe_hi, 4)],
            "n_at_risk": n_safe,
            "by_severity": {
                f"{INT_TO_LABEL[a]}_to_{INT_TO_LABEL[b]}":
                    int(np.sum(false_esc & (l1 == a) & (fused == b)))
                for a in _CLASSES for b in _CLASSES if b > a
                and int(np.sum(false_esc & (l1 == a) & (fused == b)))
            },
        },
        "net_effect_vs_rules_alone": {
            "records_fixed": fixed,
            "records_broken": broken,
            "net": fixed - broken,
            "mcnemar_exact_p": float(f"{_mcnemar_exact(fixed, broken):.4g}"),
            "model_improves_system": bool(
                fixed > broken and _mcnemar_exact(fixed, broken) < 0.05),
        },
        "interpretation": (
            "Fusion can only escalate, so the false-escalation rate on the "
            "non-novelty population is the only harm this architecture can cause. "
            "Compare `records_fixed` against `records_broken` to judge whether the "
            "model earns its place in the pipeline at all."
        ),
    }


def _rule_agreement(y_true, y_pred, l1: np.ndarray | None, mask: np.ndarray) -> dict:
    """How often the model just agrees with the rule engine.

    Layer 1 is a deterministic function of the features, so high agreement means
    the forest has mostly re-learned the rules. Layer 1 is never "serious" on the
    novelty slice, so a model that copies it can't predict "serious" there either.
    """
    if l1 is None:
        return {"note": "Layer-1 labels unavailable (synthetic dataset)"}

    serious = _CLASSES[-1]
    rf_serious = y_pred == serious
    out = {
        "agreement_with_rule_engine": round(float(np.mean(y_pred == l1)), 4),
        "agreement_on_novelty_slice": round(float(np.mean(y_pred[mask] == l1[mask])), 4),
        "accuracy_vs_true_label": round(float(np.mean(y_pred == y_true)), 4),
        "counts": {
            "rule_serious": int(np.sum(l1 == serious)),
            "model_serious": int(rf_serious.sum()),
            "true_serious": int(np.sum(y_true == serious)),
        },
        "novelty_slice_counts": {
            "rule_serious": int(np.sum(l1[mask] == serious)),
            "model_serious": int(np.sum(y_pred[mask] == serious)),
            "true_serious": int(np.sum(y_true[mask] == serious)),
        },
    }
    if rf_serious.any():
        out["model_serious_already_flagged_by_rules"] = round(
            float(np.mean(l1[rf_serious] == serious)), 4)
        out["model_serious_beyond_rules"] = int(np.sum(l1[rf_serious] != serious))
    out["interpretation"] = (
        "Agreement close to 1.0 means the forest is largely a re-implementation of "
        "the rule engine, so the headline accuracy/ROC-AUC measure re-learned "
        "determinism rather than clinical inference. Layer-1 is never 'serious' on "
        "the novelty slice by construction, which bounds what that experiment could "
        "have shown regardless of model quality."
    )
    return out


def _stratify_by_treatment(yt, yp, treated: np.ndarray, l1=None,
                           analyte_treated: np.ndarray | None = None) -> dict:
    """Break down novelty-slice results by treatment status.

    Tests the idea that medication makes the bloods look normal and so hides the
    signal. If that were true, untreated patients should be predicted more
    accurately. Each group is compared with its own best constant predictor,
    because the groups have different class balances. When we know which drugs
    act on a measured analyte, BP-only patients are split out as a control, since
    BP drugs don't change anything in the panel.
    """
    out: dict = {}
    if analyte_treated is not None:
        strata = (("analyte_lowering_rx", analyte_treated),
                  ("bp_rx_only_control", treated & ~analyte_treated),
                  ("no_medication", ~treated))
    else:
        strata = (("treated", treated), ("diagnosed_unmedicated", ~treated))

    for name, sel in strata:
        n = int(sel.sum())
        if n == 0:
            continue
        yy, pp = yt[sel], yp[sel]
        k = int(np.sum(yy == pp))
        lo, hi = _wilson(k, n)
        best_k, best_c = max((int(np.sum(yy == c)), c) for c in _CLASSES)
        b_lo, b_hi = _wilson(best_k, n)
        entry = {
            "n": n,
            "model_exact_accuracy": round(k / n, 4),
            "model_exact_accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "best_constant": INT_TO_LABEL[best_c],
            "best_constant_exact_accuracy": round(best_k / n, 4),
            "best_constant_ci95": [round(b_lo, 4), round(b_hi, 4)],
            "gap_vs_best_constant": round(k / n - best_k / n, 4),
            "model_called_normal": int(np.sum(pp == 0)),
            "true_distribution": {INT_TO_LABEL[c]: int(np.sum(yy == c)) for c in _CLASSES},
        }
        if l1 is not None:
            # How normal the bloods look on their own, without the model.
            entry["bloods_read_normal_rate"] = round(float(np.mean(l1[sel] == 0)), 4)
        out[name] = entry

    ok = yp == yt
    drug = analyte_treated if analyte_treated is not None else treated
    table = [[int(np.sum(ok & drug)), int(np.sum(~ok & drug))],
             [int(np.sum(ok & ~drug)), int(np.sum(~ok & ~drug))]]
    out["independence_test"] = {"contingency_correct_by_stratum": table}
    try:
        from scipy.stats import fisher_exact
        p = float(fisher_exact(np.array(table))[1])
        out["independence_test"]["fisher_exact_p"] = float(f"{p:.4g}")
        out["independence_test"]["accuracy_differs_by_stratum"] = bool(p < 0.05)
    except Exception:  # noqa: BLE001
        pass

    # Spell out the verdict so a null result can't be read as support.
    medicated = out.get("analyte_lowering_rx") or out.get("treated")
    unmedicated = out.get("no_medication") or out.get("diagnosed_unmedicated")
    if medicated and unmedicated:
        diff = unmedicated["model_exact_accuracy"] - medicated["model_exact_accuracy"]
        p = out["independence_test"].get("fisher_exact_p")
        supported = diff > 0 and p is not None and p < 0.05
        out["verdict"] = {
            "hypothesis": "medication normalises the measured analytes, erasing the "
                          "signal the model must recover",
            "unmedicated_minus_medicated": round(diff, 4),
            "supported": bool(supported),
            "reading": (
                "Unmedicated records are recovered more accurately — consistent with "
                "treatment erasing the signal." if supported else
                "Unmedicated records are NOT recovered more accurately, so treatment-"
                "induced normalisation does not explain the failure. See "
                "`rule_agreement` for the structural explanation that does."),
        }
    return out


def _evaluate_novelty(y_test, y_pred, mask: np.ndarray, y_train=None,
                      reports_dir: Path | None = None, seed: int = 42,
                      treated_mask: np.ndarray | None = None,
                      l1_labels: np.ndarray | None = None,
                      analyte_treated_mask: np.ndarray | None = None) -> dict:
    """Score the model on the novelty slice against constant predictors.

    The raw numbers are misleading on this slice. It has no true "normal" rows,
    so any predictor that never says normal gets a risk_detected_rate of 1.0,
    and the rule engine scores 0 because these are exactly the rows it gets
    wrong. So the model is compared on exact accuracy with the best constant
    predictor, using Wilson intervals and an exact McNemar test.
    """
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "note": "no novelty-slice records in test set"}

    yt, yp = y_test[mask], y_pred[mask]
    model = _score_predictor(yp, yt)

    baselines = {f"constant_{INT_TO_LABEL[c]}": _score_predictor(np.full(n, c), yt)
                 for c in _CLASSES}
    if y_train is not None and len(y_train):
        rng = np.random.default_rng(seed)
        prior = np.bincount(y_train, minlength=len(_CLASSES)) / len(y_train)
        draws = [float(np.mean(rng.choice(len(_CLASSES), size=n, p=prior) == yt))
                 for _ in range(2000)]
        baselines["stratified_random"] = {
            "exact_accuracy": round(float(np.mean(draws)), 4),
            "exact_accuracy_ci95": [round(float(np.percentile(draws, 2.5)), 4),
                                    round(float(np.percentile(draws, 97.5)), 4)],
            "note": "2000 draws from the training class prior",
        }

    # Compare with the strongest constant predictor.
    best = max((k for k in baselines if k.startswith("constant_")),
               key=lambda k: baselines[k]["exact_accuracy"])
    best_pred = np.full(n, [c for c in _CLASSES if INT_TO_LABEL[c] == best.split("_", 1)[1]][0])
    model_ok, base_ok = yp == yt, best_pred == yt
    wins = int(np.sum(model_ok & ~base_ok))
    losses = int(np.sum(~model_ok & base_ok))
    p_value = _mcnemar_exact(wins, losses)

    if reports_dir is not None:
        _plot_novelty_baselines(model, baselines, n, reports_dir / "novelty_baselines.png")

    log.info("novelty slice n=%d: model exact=%.3f vs best constant (%s) %.3f, McNemar p=%.3g",
             n, model["exact_accuracy"], best, baselines[best]["exact_accuracy"], p_value)

    return {
        "n": n,
        "description": "records upgraded by Layer-2 (bloods normal/borderline, "
                       "diagnosis/medication indicated higher risk)",
        "true_class_distribution": {INT_TO_LABEL[c]: int(np.sum(yt == c)) for c in _CLASSES},
        # kept so older reports still line up
        "risk_detected_rate": model["risk_detected_rate"],
        "exact_label_accuracy": model["exact_accuracy"],
        "predicted_class_distribution": model["predicted_distribution"],
        "model": model,
        "baselines": baselines,
        "vs_best_constant": {
            "baseline": best,
            "baseline_exact_accuracy": baselines[best]["exact_accuracy"],
            "model_correct_baseline_wrong": wins,
            "model_wrong_baseline_correct": losses,
            "mcnemar_exact_p": float(f"{p_value:.4g}"),
            "model_beats_baseline": bool(
                model["exact_accuracy"] > baselines[best]["exact_accuracy"] and p_value < 0.05),
        },
        "metric_caveat": (
            "risk_detected_rate is degenerate on this slice: it contains no true "
            "'normal' records by construction, so any constant non-normal "
            "predictor scores 1.0. Compare exact_accuracy against `baselines`."
        ),
        **({"strata": _stratify_by_treatment(
            yt, yp,
            treated_mask[mask] if treated_mask is not None else np.zeros(n, bool),
            l1_labels[mask] if l1_labels is not None else None,
            analyte_treated=(analyte_treated_mask[mask]
                             if analyte_treated_mask is not None else None))}
           if treated_mask is not None else {}),
    }


def _plot_novelty_baselines(model: dict, baselines: dict, n: int, path: Path) -> None:
    names = ["Random Forest"] + [k.replace("constant_", "constant\n").replace("_", " ")
                                 for k in baselines]
    scores = [model] + list(baselines.values())
    vals = [s["exact_accuracy"] for s in scores]
    cis = [s.get("exact_accuracy_ci95", [s["exact_accuracy"]] * 2) for s in scores]
    err = np.array([[v - ci[0] for v, ci in zip(vals, cis)],
                    [ci[1] - v for v, ci in zip(vals, cis)]])
    colours = ["#C24A57"] + ["#0E7C86"] * len(baselines)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(names, vals, color=colours, yerr=err, capsize=4,
                  error_kw={"ecolor": "#3C4E5A", "lw": 1.2})
    # Put the value above the error bar so the two don't overlap.
    for bar, v, ci in zip(bars, vals, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, ci[1] + 0.025, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("exact accuracy")
    ax.set_title(f"Layer-2 novelty slice (n={n}): model vs constant predictors\n"
                 "error bars = 95% Wilson interval", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


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
