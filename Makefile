# Convenience commands. Usage: `make <target>`  (needs GNU make; on Windows use WSL/Git Bash)
.PHONY: help install test lint api dashboard build-index train eval

help:
	@echo "install       install Python dependencies"
	@echo "test          run all test suites"
	@echo "lint          ruff over src, streamlit_app, scripts, tests"
	@echo "api           run the FastAPI backend (:8000, docs at /docs)"
	@echo "dashboard     run the Streamlit dashboard (:8501)"
	@echo "build-index   (re)build the RAG FAISS index"
	@echo "train         train + evaluate the Random Forest"
	@echo "eval          run the Phase-9 system evaluation"

install:
	pip install -r requirements.txt

test:
	python tests/test_rule_engine.py
	python tests/test_parser.py
	python tests/test_rag.py
	python tests/test_ml.py
	python tests/test_recommend.py
	python tests/test_api.py
	python tests/test_eval.py

lint:
	ruff check src streamlit_app scripts tests

api:
	uvicorn app.api.main:app --app-dir src --reload

dashboard:
	streamlit run streamlit_app/dashboard.py

build-index:
	python scripts/build_rag_index.py

train:
	python scripts/train_model.py

eval:
	python scripts/evaluate_system.py
