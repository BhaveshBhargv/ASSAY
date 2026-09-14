"""Read NHANES .XPT files into pandas and rename the columns we use."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import CANONICAL_VARS, WEIGHT_VARS

log = logging.getLogger(__name__)


def read_xpt(path: Path) -> pd.DataFrame:
    """Read one .XPT file, decoding any byte strings."""
    df = pd.read_sas(path, format="xport")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: v.decode("utf-8", "ignore").strip() if isinstance(v, bytes) else v
        )
    return df


def load_component(path: Path) -> pd.DataFrame | None:
    """Load a component file, keeping SEQN and the columns we know about.

    The survey weight column is renamed to wtmec. Returns None if the file
    doesn't exist or has no SEQN.
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

    # the weight column has a different name in each cycle
    for wvar in WEIGHT_VARS:
        if wvar in df.columns:
            keep.append(wvar)
            rename[wvar] = "wtmec"
            break

    out = df[keep].rename(columns=rename)
    out["SEQN"] = out["SEQN"].astype("int64")
    return out
