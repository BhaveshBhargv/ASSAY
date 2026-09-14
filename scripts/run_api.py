"""Start the FastAPI backend.

    python scripts/run_api.py                 # http://localhost:8000, docs at /docs
    python scripts/run_api.py --reload
    ASSAY_LOG_LEVEL=DEBUG python scripts/run_api.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Assay API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
