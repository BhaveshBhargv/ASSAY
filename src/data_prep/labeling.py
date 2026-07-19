"""
labeling.py — three-layer target construction (the project's methodological core).

Layer 1  Rule-based single-biomarker labelling — delegated to the SHARED clinical
         rule engine (`app.domain.services.rule_engine`). Training labels and the
         inference-time rule engine therefore use ONE clinical definition and can
         never drift (the Phase-3 unification decision).
Layer 2  Diagnosis & medication proxy labelling for interaction cases
         (records no single biomarker flags) using questionnaire data.
Layer 3  Conflict resolution: final = max(Layer 1, Layer 2). Clinical
         diagnosis/medication can only ever RAISE severity, never lower it.

Leakage guard: the Layer-2 inputs (DIQ/BPQ/MCQ/RXQ) are consumed HERE only and
are dropped from the feature matrix downstream.

Outputs added to the frame:
  label_l1  label_l2  label   (strings: normal/borderline/serious)
  label_upgraded_by_l2        (bool: the novelty slice — L2 > L1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.enums import SEVERITY_RANK
from app.domain.services.rule_engine import RuleEngine
from app.rules.loader import load_ruleset

_ORDER = {"normal": 0, "borderline": 1, "serious": 2}
_INV = {v: k for k, v in _ORDER.items()}


def default_engine() -> RuleEngine:
    return RuleEngine(load_ruleset())


def _sex_series(df: pd.DataFrame) -> pd.Series:
    """Map NHANES sex_code (1=male, 2=female) to 'male'/'female'."""
    return df["sex_code"].map({1: "male", 2: "female"})


# --------------------------------------------------------------------------- #
# Layer 1 — via the shared rule engine
# --------------------------------------------------------------------------- #
def layer1_severity(df: pd.DataFrame, engine: RuleEngine) -> pd.Series:
    """Weighted-core Layer-1 severity over the ACTIONABLE panel (flag-only
    markers never drive the label):
        * any actionable serious marker            -> serious
        * a CORE marker at borderline              -> borderline
        * SECONDARY markers                        -> borderline only when
          >= secondary_borderline_min are mildly abnormal
    Uses the engine's core/secondary split so the training label and the
    inference-time overall_severity share ONE policy.
    """
    sex = _sex_series(df).to_numpy()
    n = len(df)
    core = set(engine.core_actionable_codes)
    secondary = set(engine.secondary_actionable_codes)
    smin = engine.secondary_borderline_min

    has_serious = np.zeros(n, dtype=bool)
    core_borderline = np.zeros(n, dtype=bool)
    secondary_bl_count = np.zeros(n, dtype="int16")

    for code in engine.actionable_codes:
        if code not in df.columns:
            continue
        col = pd.to_numeric(df[code], errors="coerce").to_numpy()
        rank = np.fromiter(
            (SEVERITY_RANK[engine.layer1_severity(code, v, s)] for v, s in zip(col, sex)),
            dtype="int8", count=n,
        )
        has_serious |= rank == 2
        if code in core:
            core_borderline |= rank == 1
        elif code in secondary:
            secondary_bl_count += (rank == 1)

    borderline = core_borderline | (secondary_bl_count >= smin)
    out = np.where(has_serious, 2, np.where(borderline, 1, 0)).astype("int8")
    return pd.Series(out, index=df.index, name="label_l1_int")


# --------------------------------------------------------------------------- #
# Layer 2 — diagnosis & medication proxies
# --------------------------------------------------------------------------- #
def _yes(series: pd.Series, code: int = 1) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") == code


def layer2_severity(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Minimum severity enforced by clinical diagnosis / medication evidence."""
    n = len(df)
    sev = np.zeros(n, dtype="int8")

    def bump(mask: pd.Series, level: str) -> None:
        nonlocal sev
        lv = _ORDER[level]
        sev = np.where(mask.to_numpy(), np.maximum(sev, lv), sev)

    col = df.columns
    if "diq_diabetes" in col:
        bump(_yes(df["diq_diabetes"], 1), "borderline")
        bump(_yes(df["diq_diabetes"], 3), "borderline")
    if "diq_insulin" in col:
        bump(_yes(df["diq_insulin"], 1), "borderline")
    if "diq_pills" in col:
        bump(_yes(df["diq_pills"], 1), "borderline")
    if "bpq_high_chol" in col:
        bump(_yes(df["bpq_high_chol"], 1), "borderline")
    if "bpq_chol_med" in col:
        bump(_yes(df["bpq_chol_med"], 1), "borderline")
    if "bpq_high_bp" in col and "bpq_bp_med" in col:
        bump(_yes(df["bpq_high_bp"], 1) & _yes(df["bpq_bp_med"], 1), "borderline")
    for c in ("mcq_chd", "mcq_mi", "mcq_stroke"):
        if c in col:
            bump(_yes(df[c], 1), "serious")
    if "statin_or_metformin" in col:
        bump(_yes(df["statin_or_metformin"], 1), "borderline")

    return pd.Series(sev, index=df.index, name="label_l2_int")


# --------------------------------------------------------------------------- #
# Layer 3 — fuse
# --------------------------------------------------------------------------- #
def build_labels(df: pd.DataFrame, thresholds: dict, engine: RuleEngine | None = None) -> pd.DataFrame:
    """Attach label_l1, label_l2, label, and the novelty flag to a copy of df."""
    engine = engine or default_engine()
    out = df.copy()
    l1 = layer1_severity(out, engine)
    l2 = layer2_severity(out, thresholds)
    final = np.maximum(l1.to_numpy(), l2.to_numpy())

    out["label_l1"] = l1.map(_INV)
    out["label_l2"] = l2.map(_INV)
    out["label"] = pd.Series(final, index=out.index).map(_INV)
    out["label_upgraded_by_l2"] = l2.to_numpy() > l1.to_numpy()
    return out


def label_report(df: pd.DataFrame) -> dict:
    """Summary counts for the dissertation methods chapter."""
    return {
        "label_distribution": df["label"].value_counts().to_dict(),
        "layer1_distribution": df["label_l1"].value_counts().to_dict(),
        "upgraded_by_layer2": int(df["label_upgraded_by_l2"].sum()),
        "upgraded_share": round(float(df["label_upgraded_by_l2"].mean()), 4),
    }
