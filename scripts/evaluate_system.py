"""Run the ML, RAG and LLM evaluations.

The summary JSON and the plots are written to reports/phase9/.

    python scripts/evaluate_system.py                 # all three (the LLM step needs Ollama)
    python scripts/evaluate_system.py --no-llm
    python scripts/evaluate_system.py --provider anthropic
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.eval import config as C  # noqa: E402
from app.eval.ml_eval import evaluate_ml  # noqa: E402
from app.eval.rag_eval import evaluate_rag  # noqa: E402
from app.eval.llm_eval import run_llm_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9 — evaluate the complete system")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    if not args.no_ml:
        print("\n=== MACHINE LEARNING ===")
        ml = evaluate_ml()
        summary["ml"] = ml
        h = ml["headline"]
        print(f"accuracy {h['accuracy']:.3f} | precision {h['precision_macro']:.3f} | "
              f"recall {h['recall_macro']:.3f} | F1 {h['f1_macro']:.3f} | "
              f"ROC-AUC {h['roc_auc_macro_ovr']:.3f}")
        print("top features:", ", ".join(ml["top_features"][:8]))

    if not args.no_rag:
        print("\n=== RAG RETRIEVAL ===")
        rag = evaluate_rag()
        summary["rag"] = rag
        a = rag["aggregate"]
        for k in C.RAG_KS:
            print(f"P@{k} {a[f'precision@{k}']:.3f} | R@{k} {a[f'recall@{k}']:.3f}")
        print(f"context relevance: score {a['context_relevance_score']:.3f} | "
              f"tag-overlap {a['context_tag_overlap']:.3f}")

    if not args.no_llm:
        print("\n=== LLM RECOMMENDATIONS ===")
        llm = run_llm_eval(provider_kind=args.provider, model=args.model)
        summary["llm"] = llm
        if llm["status"] == "ok":
            a = llm["aggregate"]
            print(f"groundedness {a['groundedness']:.3f} | faithfulness {a['faithfulness_rate']:.3f} "
                  f"(cos {a['faithfulness_mean_cosine']:.3f}) | "
                  f"hallucination {a['hallucination_rate']:.3f}")
        else:
            print(f"skipped — {llm['status']}: {llm.get('error', llm.get('failures'))}")
            print("(start Ollama and re-run, or use --provider anthropic, for LLM metrics)")

    out = C.REPORTS_DIR / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary  -> {out}")
    print(f"plots    -> {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
