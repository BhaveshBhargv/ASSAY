"""
run_rule_engine.py — Phase 3 demo / CLI.

Classifies a sample blood panel and prints the standardized JSON output.

Usage:
    python scripts/run_rule_engine.py
    python scripts/run_rule_engine.py --sex female --age 61 \
        --set hba1c_pct=6.8 --set sbp_mmhg=182 --set hdl_mgdl=38
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.domain.models import PatientContext            # noqa: E402
from app.domain.services.rule_engine import RuleEngine  # noqa: E402
from app.rules.loader import load_ruleset               # noqa: E402

_SAMPLE = {
    "hba1c_pct": 6.8,
    "fasting_glucose_mgdl": 118,
    "total_chol_mgdl": 236,
    "hdl_mgdl": 38,
    "ldl_mgdl": 165,
    "triglycerides_mgdl": 180,
    "alt": 62,
    "hemoglobin": 11.5,
    "creatinine": 1.1,
    "vitamin_d": 38,
    "potassium": 5.4,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Clinical rule engine demo")
    p.add_argument("--sex", choices=["male", "female"], default="male")
    p.add_argument("--age", type=int, default=54)
    p.add_argument("--set", action="append", default=[], metavar="code=value",
                   help="override/add a reading, e.g. --set hba1c_pct=7.2")
    args = p.parse_args()

    readings = dict(_SAMPLE)
    for item in args.set:
        code, _, val = item.partition("=")
        readings[code.strip()] = float(val)

    engine = RuleEngine(load_ruleset())
    result = engine.evaluate(readings, PatientContext(sex=args.sex, age=args.age))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
