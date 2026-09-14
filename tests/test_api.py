"""API tests using FastAPI's TestClient.

Using the client as a context manager runs the lifespan, so the real rule
engine, model and index are loaded. There's normally no LLM running during
tests, so /recommend is expected to return a 502 rather than a server error.

Run:  python tests/test_api.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402

DEMO = {"age": 54, "sex": "male", "eth_code": 3, "pir": 2.5, "educ_code": 4}
BIO = {"hba1c_pct": 6.1, "total_chol_mgdl": 232, "hdl_mgdl": 34,
       "triglycerides_mgdl": 205, "alt": 46}


def test_health_and_model_info():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        info = c.get("/model-info").json()
        assert info["rule_engine"]["biomarker_count"] == 27
        assert len(info["biomarkers"]) == 27
        assert info["rag"]["index_size"] >= 25
        assert "random_forest" in info


def test_predict_ok():
    with TestClient(app) as c:
        r = c.post("/predict", json={"demographics": DEMO, "biomarkers": BIO})
        assert r.status_code == 200, r.text
        a = r.json()["assessment"]
        assert a["severity"] in {"borderline", "serious"}
        assert len(a["biomarkers"]) == len(BIO)
        assert any(f["code"] == "hba1c_pct" for f in a["flagged"])


def test_predict_validation_error_envelope():
    with TestClient(app) as c:
        # unknown biomarker code
        r = c.post("/predict", json={"demographics": DEMO, "biomarkers": {"blood_pressure": 120}})
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["type"] == "validation_error"
        assert "request_id" in body

        # under 18
        r2 = c.post("/predict", json={"demographics": {"age": 5, "sex": "male"}, "biomarkers": BIO})
        assert r2.status_code == 422


def test_retrieve_by_severity_and_flags():
    with TestClient(app) as c:
        r = c.post("/retrieve", json={
            "severity": "serious",
            "flagged": [{"code": "hba1c_pct", "name": "HbA1c", "status": "borderline"}],
            "k": 4,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 4 and len(body["passages"]) == 4
        assert body["passages"][0]["score"] >= body["passages"][-1]["score"]
        assert body["passages"][0]["id"] == "E1"


def test_retrieve_requires_a_signal():
    with TestClient(app) as c:
        r = c.post("/retrieve", json={"k": 3})
        assert r.status_code == 422


def test_upload_json():
    with TestClient(app) as c:
        payload = json.dumps({"biomarkers": {"HbA1c": 6.1, "HDL cholesterol": 34}}).encode()
        files = {"file": ("report.json", io.BytesIO(payload), "application/json")}
        r = c.post("/upload", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["recognised"] == 2
        assert body["biomarkers"]["hba1c_pct"] == 6.1


def test_upload_unsupported_type():
    with TestClient(app) as c:
        files = {"file": ("report.txt", io.BytesIO(b"hi"), "text/plain")}
        r = c.post("/upload", files=files)
        assert r.status_code == 422
        assert r.json()["error"]["type"] == "parsing_error"


def test_recommend_degrades_gracefully_without_llm():
    with TestClient(app) as c:
        r = c.post("/recommend", json={"demographics": DEMO, "biomarkers": BIO,
                                       "provider": "ollama", "k": 5})
        # 502 when no model is running
        assert r.status_code in (200, 502)
        if r.status_code == 502:
            assert r.json()["error"]["type"] == "llm_unavailable"
        else:
            body = r.json()
            assert body["disclaimer"] and "assessment" in body


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
