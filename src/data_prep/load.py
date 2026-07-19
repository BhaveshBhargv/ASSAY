"""
load.py — read NHANES .XPT files into pandas and apply canonical renaming.

Each .XPT is a SAS transport file; pandas reads it natively (no extra driver).
String columns come back as bytes and are decoded. Survey-weight columns are
harmonised to a single 'wtmec' name across cycles.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import CANONICAL_VARS, WEIGHT_VARS

log = logging.getLogger(__name__)


def read_xpt(path: Path) -> pd.DataFrame:
    """Read a single .XPT file, decoding byte strings to str."""
    df = pd.read_sas(path, format="xport")
    # Decode object (bytes) columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: v.decode("utf-8", "ignore").strip() if isinstance(v, bytes) else v
        )
    return df


def load_component(path: Path) -> pd.DataFrame | None:
    """
    Load a component file, keep SEQN + any recognised canonical variables,
    and harmonise the survey-weight column. Returns None if the file is absent.
    """
    if not path.exists():
        return None

    df = read_xpt(path)
    if "SEQN" not in df.columns:
        log.warning("no SEQN in %s — skipping", path.name)
        return None

    keep = ["SEQN"]
    rename: dict[str, str] = {}

    for raw, canon in CANONICAL_VARS.items():
        if raw in df.columns:
            keep.append(raw)
            rename[raw] = canon

    # Harmonise survey weight (name differs by cycle).
    for wvar in WEIGHT_VARS:
        if wvar in df.columns:
            keep.append(wvar)
            rename[wvar] = "wtmec"
            break

    out = df[keep].rename(columns=rename)
    out["SEQN"] = out["SEQN"].astype("int64")
    return out
