"""
test_eval.py — Phase 9 evaluation logic (offline).

RAG evaluation runs against the built index; LLM metric logic is tested with a
fake embedder so it needs no model. Retrieval tests skip if the index is absent.

Run:  python tests/test_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from app.eval.llm_eval import aggregate, score_recommendation  # noqa: E402
from app.rag.models import GuidelineChunk  # noqa: E402
from app.recommend.prompt import EvidenceItem  # noqa: E402
from app.recommend.report import AdviceItem, RecommendationReport  # noqa: E402


class _FakeEmbedder:
    """Deterministic embedder: identical texts → identical unit vectors, so a
    grounded item whose evidence text matches scores cosine 1, a mismatch ~0."""
    _vocab = {"walk": 0, "diet": 1, "sleep": 2, "hba1c": 3, "hdl": 4, "unrelated": 5}

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(len(self._vocab))
            for w, i in self._vocab.items():
                if w in t.lower():
                    v[i] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.asarray(out)


_EVIDENCE = [
    EvidenceItem(id="E1", citation="NICE NG28", title="Diabetes",
                 text="Regular walking improves hba1c control."),
    EvidenceItem(id="E2", citation="NICE CG181", title="Lipids",
                 text="A healthy diet supports hdl cholesterol."),
]


def _report():
    return RecommendationReport(
        explanation="Higher-risk pattern; not a diagnosis.",
        lifestyle=[AdviceItem(advice="Build in daily walk", rationale="helps hba1c", evidence=["E1"])],
        diet=[AdviceItem(advice="Adopt a healthy diet", rationale="supports hdl", evidence=["E2"])],
        exercise=[AdviceItem(advice="Take an unrelated supplement",
                             rationale="unrelated claim", evidence=["E9"])],  # ungrounded
    )


def test_score_grounded_and_faithful():
    s = score_recommendation(_report(), _EVIDENCE, _FakeEmbedder(), tau=0.3)
    assert s["total_items"] == 3
    assert s["grounded_items"] == 2               # E1, E2 valid; E9 not
    assert s["faithful_items"] == 2               # both grounded items match their evidence
    assert abs(s["groundedness"] - 2 / 3) < 1e-9
    assert abs(s["hallucination_rate"] - 1 / 3) < 1e-9  # only the E9 item
    assert s["faithfulness_mean_cosine"] > 0.9


def test_ungrounded_counts_as_hallucination():
    rep = RecommendationReport(
        explanation="x",
        diet=[AdviceItem(advice="cite nothing valid", rationale="none", evidence=["E9"])])
    s = score_recommendation(rep, _EVIDENCE, _FakeEmbedder())
    assert s["groundedness"] == 0.0 and s["hallucination_rate"] == 1.0


def test_unfaithful_grounded_is_hallucination():
    # grounded (E1 exists) but text is unrelated to the evidence → low cosine
    rep = RecommendationReport(
        explanation="x",
        diet=[AdviceItem(advice="do something unrelated", rationale="unrelated", evidence=["E1"])])
    s = score_recommendation(rep, _EVIDENCE, _FakeEmbedder(), tau=0.3)
    assert s["grounded_items"] == 1 and s["faithful_items"] == 0
    assert s["hallucination_rate"] == 1.0


def test_aggregate():
    a = aggregate([score_recommendation(_report(), _EVIDENCE, _FakeEmbedder(), tau=0.3)])
    assert a["total_items"] == 3 and a["n_reports"] == 1
    assert 0 < a["groundedness"] <= 1 and 0 <= a["hallucination_rate"] < 1


def test_rag_eval_if_index_exists():
    from app.rag import config as RC
    if not RC.FAISS_PATH.exists():
        print("SKIP rag eval (index not built)")
        return
    from app.eval.rag_eval import evaluate_rag
    res = evaluate_rag(ks=(1, 3))
    agg = res["aggregate"]
    assert 0.0 <= agg["precision@1"] <= 1.0
    assert 0.0 <= agg["recall@3"] <= 1.0
    assert agg["precision@1"] > 0  # a tag-matched query should hit at rank 1
    print("rag P@1=%.2f R@3=%.2f ctx=%.2f" % (
        agg["precision@1"], agg["recall@3"], agg["context_relevance_score"]))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
