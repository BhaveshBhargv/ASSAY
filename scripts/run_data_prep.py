"""
run_data_prep.py — Phase 2 entrypoint.

Usage:
    python -m scripts.run_data_prep --download        # fetch XPT then build
    python -m scripts.run_data_prep                   # build from cached XPT
    python -m scripts.run_data_prep --impute knn      # KNN instead of median

Run from the project root so `src` is importable, or `pip install -e .`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.download import download_all          # noqa: E402
from data_prep.pipeline import run_pipeline           # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="NHANES Phase 2 data preparation")
    parser.add_argument("--download", action="store_true", help="download XPT files first")
    parser.add_argument("--impute", default="median", choices=["median", "knn"])
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.download:
        download_all()

    summary = run_pipeline(
        test_size=args.test_size, seed=args.seed, impute_strategy=args.impute
    )
    print("\n=== Phase 2 summary ===")
    print(f"train rows : {summary['rows_train']}")
    print(f"test rows  : {summary['rows_test']}")
    print(f"features   : {summary['n_features']}")
    print(f"labels     : {summary['label_report']['label_distribution']}")
    print(f"L2 upgrades: {summary['label_report']['upgraded_by_layer2']} "
          f"({summary['label_report']['upgraded_share']*100:.1f}% — the novelty slice)")


if __name__ == "__main__":
    main()
