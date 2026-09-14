"""Run the whole pipeline on a sample patient and print the report.

Needs Ollama running with the model pulled (`ollama pull llama3.1`), unless you
pick another provider.

    python scripts/generate_recommendation.py
    python scripts/generate_recommendation.py --provider anthropic
    python scripts/generate_recommendation.py --no-rf --k 5
    python scripts/generate_recommendation.py --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.recommend import RecommendationEngine  # noqa: E402

# Made-up patient with raised HbA1c, low HDL and high triglycerides. No single
# result is severe, but together they're the usual metabolic risk pattern.
SAMPLE_DEMOGRAPHICS = {"age": 54, "sex": "male", "eth_code": 3, "pir": 2.5, "educ_code": 4}
SAMPLE_BIOMARKERS = {
    "hba1c_pct": 6.1,
    "fasting_glucose_mgdl": 108,
    "total_chol_mgdl": 232,
    "ldl_mgdl": 150,
    "hdl_mgdl": 34,
    "triglycerides_mgdl": 205,
    "alt": 46,
    "creatinine": 0.9,
    "hemoglobin": 14.5,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a grounded lifestyle recommendation")
    ap.add_argument("--provider", default="ollama", help="ollama | anthropic")
    ap.add_argument("--model", default=None, help="override the model name")
    ap.add_argument("--k", type=int, default=6, help="guideline passages to retrieve")
    ap.add_argument("--no-rf", action="store_true", help="skip the Random Forest (rules-only)")
    ap.add_argument("--recheck", action="store_true", help="enable LLM entailment recheck")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    kwargs = {"model": args.model} if args.model else {}
    engine = RecommendationEngine.build(
        provider=args.provider, use_rf=not args.no_rf, k=args.k,
        llm_recheck=args.recheck, **kwargs,
    )
    bundle = engine.recommend(SAMPLE_DEMOGRAPHICS, SAMPLE_BIOMARKERS)

    if args.json:
        print(json.dumps(bundle.to_dict(), indent=2))
    else:
        print("\n" + bundle.render())
        print("\n--- audit -------------------------------------------------")
        print(json.dumps(bundle.audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
