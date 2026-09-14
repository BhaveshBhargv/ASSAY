"""Merge all components and cycles into one table.

Within a cycle, every component is left-joined onto DEMO by SEQN, so a missing
lab result never drops a person. RXQ_RX is reduced to a single statin/metformin
flag. The cycles are then stacked with a `cycle` column.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, get_cycles, load_file_registry
from .load import load_component, read_xpt

log = logging.getLogger(__name__)

# Lowercase drug name fragments for statins and metformin.
_STATIN_METFORMIN = (
    "metformin",
    "statin",  # atorvastatin, simvastatin, rosuvastatin, ...
)


def _statin_metformin_flag(rxq_path: Path) -> pd.DataFrame | None:
    """SEQN with a 0/1 statin_or_metformin flag, from an RXQ_RX file."""
    if not rxq_path.exists():
        return None
    df = read_xpt(rxq_path)
    if "SEQN" not in df.columns or "RXDDRUG" not in df.columns:
        return None
    df = df[["SEQN", "RXDDRUG"]].copy()
    name = df["RXDDRUG"].astype(str).str.lower()
    df["hit"] = name.apply(lambda s: int(any(k in s for k in _STATIN_METFORMIN)))
    flag = df.groupby("SEQN")["hit"].max().reset_index()
    flag.columns = ["SEQN", "statin_or_metformin"]
    flag["SEQN"] = flag["SEQN"].astype("int64")
    return flag


def _load_cycle(cycle, raw_dir: Path, reg: dict) -> pd.DataFrame:
    cycle_dir = raw_dir / cycle.name
    components: list[str] = list(reg["components"])

    demo = load_component(cycle_dir / cycle.filename("DEMO"))
    if demo is None:
        raise FileNotFoundError(f"DEMO missing for cycle {cycle.name}")
    merged = demo

    # RXQ_RX is handled separately below.
    for base in components:
        if base in ("DEMO", "RXQ_RX"):
            continue
        comp = load_component(cycle_dir / cycle.filename(base))
        if comp is None:
            log.warning("cycle %s: component %s absent", cycle.name, base)
            continue
        merged = merged.merge(comp, on="SEQN", how="left")

    rxq = _statin_metformin_flag(cycle_dir / cycle.filename("RXQ_RX"))
    if rxq is not None:
        merged = merged.merge(rxq, on="SEQN", how="left")
    if "statin_or_metformin" not in merged.columns:
        merged["statin_or_metformin"] = 0
    merged["statin_or_metformin"] = merged["statin_or_metformin"].fillna(0).astype(int)

    merged["cycle"] = cycle.name
    log.info("cycle %s merged: %d rows, %d cols", cycle.name, len(merged), merged.shape[1])
    return merged


def build_merged(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Merge every component from every cycle into one DataFrame."""
    reg = load_file_registry()
    frames = [_load_cycle(cycle, raw_dir, reg) for cycle in get_cycles()]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    log.info("combined: %d rows, %d cols", len(combined), combined.shape[1])
    return combined
