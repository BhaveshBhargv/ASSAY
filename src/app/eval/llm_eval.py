"""
llm_eval.py — recommendation quality (Phase 9).

Metrics over the generated recommendations:
  Groundedness    — share of advice items that cite a real retrieved passage
  Faithfulness    — share whose text is entailed by their cited evidence, judged
                    by embedding cosine >= tau (deterministic; no LLM judge needed)
  Hallucination   — share that are ungrounded OR unfaithful (1 - faithful share)

`score_recommendation` / `aggregate` are pure functions (embedding-based) so the
metric logic is unit-testable offline. `run_llm_eval` drives the full pipeline to
generate the recommendations first (needs a reachable LLM provider).
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

log = logging.getLogger("assay.eval")


def score_recommendation(report, evidence, embedder, tau: float = C.FAITHFULNESS_TAU) -> dict:
    """Groundedness / faithfulness / hallucination for one generated report."""
    ev_by_id = {e.id: e for e in evidence}
    items = report.all_items()
    if not items:
        return {"total_items": 0, "grounded_items": 0, "faithful_items": 0,
                "groundedness": 1.0, "faithfulness_rate": 1.0,
                "faithfulness_mean_cosine": 0.0, "hallucination_rate": 0.0, "items": []}

    adv_vecs = np.asarray(embedder.embed([f"{it.advice} {it.rationale}" for _s, it in items]))
    ev_ids = list(ev_by_id)
    ev_vecs = {i: v for i, v in zip(ev_ids, np.asarray(embedder.embed(
        [ev_by_id[i].text for i in ev_ids])))} if ev_ids else {}

    grounded = faithful = hallucinated = 0
    cos_sum = 0.0
    per = []
    for (sec, it), av in zip(items, adv_vecs):
        cited = [e for e in it.evidence if e in ev_by_id]
        is_grounded = bool(cited)
        best_cos = max((float(np.dot(av, ev_vecs[c])) for c in cited), default=0.0)
        is_faithful = is_grounded and best_cos >= tau
        grounded += int(is_grounded)
        faithful += int(is_faithful)
        cos_sum += best_cos if is_grounded else 0.0
        hallucinated += int(not is_faithful)
        per.append({"section": sec, "advice": it.advice, "grounded": is_grounded,
                    "cosine": round(best_cos, 3), "faithful": is_faithful})

    n = len(items)
    return {
        "total_items": n, "grounded_items": grounded, "faithful_items": faithful,
        "groundedness": grounded / n,
        "faithfulness_rate": faithful / n,
        "faithfulness_mean_cosine": (cos_sum / grounded) if grounded else 0.0,
        "hallucination_rate": hallucinated / n,
        "items": per,
    }


def aggregate(scores: list[dict]) -> dict:
    tot = sum(s["total_items"] for s in scores)
    if tot == 0:
        return {"n_reports": len(scores), "total_items": 0}
    grd = sum(s["grounded_items"] for s in scores)
    fth = sum(s["faithful_items"] for s in scores)
    cos = sum(s["faithfulness_mean_cosine"] * s["grounded_items"] for s in scores)
    return {
        "n_reports": len(scores),
        "total_items": tot,
        "groundedness": grd / tot,
        "faithfulness_rate": fth / tot,
        "faithfulness_mean_cosine": (cos / grd) if grd else 0.0,
        "hallucination_rate": (tot - fth) / tot,
    }


def run_llm_eval(provider_kind: str = "ollama", model: str | None = None,
                 patients: list[dict] | None = None, k: int = 6) -> dict:
    """Generate recommendations for the test patients and score them. Returns a
    status dict; on an unreachable LLM it reports the failure instead of raising."""
    from app.domain.models import PatientContext
    from app.domain.services.rule_engine import RuleEngine
    from app.rag.embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
    from app.rag.retriever import GuidelineRetriever
    from app.recommend.engine import RiskAdapter
    from app.recommend.fusion import fuse
    from app.recommend.generator import generate_report
    from app.recommend.guards import verify
    from app.recommend.prompt import build_evidence_pack, build_messages
    from app.recommend.providers import build_provider
    from app.rules.loader import load_ruleset

    patients = patients or C.EVAL_PATIENTS
    engine = RuleEngine(load_ruleset())
    retriever = GuidelineRetriever.load()
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    try:
        risk = RiskAdapter()
    except Exception:  # noqa: BLE001
        risk = None
    try:
        provider = build_provider(provider_kind, **({"model": model} if model else {}))
    except Exception as exc:  # noqa: BLE001
        return {"status": "provider_unavailable", "error": str(exc)}

    scores, per_patient, failures = [], [], []
    for p in patients:
        demo, bio = p["demographics"], p["biomarkers"]
        ctx = PatientContext(sex=demo.get("sex"), age=demo.get("age"))
        rule_result = engine.evaluate(bio, ctx)
        fused = fuse(rule_result, risk.predict(demo, bio) if risk else None)
        evidence = build_evidence_pack(
            retriever.retrieve_for_assessment(fused.severity, fused.flagged, k=k))
        system, user = build_messages(demo, fused, evidence)
        try:
            report = generate_report(provider, system, user)
        except Exception as exc:  # noqa: BLE001 - no daemon / bad output
            failures.append({"patient": p["name"], "error": str(exc)[:160]})
            continue
        cleaned, _audit = verify(report, evidence, provider=provider)
        s = score_recommendation(cleaned, evidence, embedder)
        scores.append(s)
        per_patient.append({"patient": p["name"], **{k2: s[k2] for k2 in
                            ("groundedness", "faithfulness_rate", "hallucination_rate",
                             "faithfulness_mean_cosine", "total_items")}})

    if not scores:
        return {"status": "no_generations", "provider": provider.name, "failures": failures}

    agg = aggregate(scores)
    _plot_llm(agg)
    return {"status": "ok", "provider": provider.name, "aggregate": agg,
            "per_patient": per_patient, "failures": failures}


def _plot_llm(agg: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    names = ["Groundedness", "Faithfulness", "Hallucination"]
    vals = [agg["groundedness"], agg["faithfulness_rate"], agg["hallucination_rate"]]
    colors = ["#1F9D74", "#0E7C86", "#C24A57"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("LLM recommendation quality", fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "llm_quality.png", dpi=140)
    plt.close(fig)
