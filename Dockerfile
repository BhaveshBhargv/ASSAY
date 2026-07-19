# Blood-test lifestyle recommender — one image runs both the API and the dashboard
# (the command is chosen per service in docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src:/app/streamlit_app

WORKDIR /app

# Build tools (some wheels) + curl for container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so sentence-transformers doesn't pull the ~2GB CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the embedding model in so the first request is fast and works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Application code + baked artefacts (RF model in src/app/ml/registry,
# FAISS index in src/app/rag/index, rules in config/, corpus in data/guidelines).
COPY src ./src
COPY streamlit_app ./streamlit_app
COPY scripts ./scripts
COPY config ./config
COPY data/guidelines ./data/guidelines
COPY .streamlit ./.streamlit

EXPOSE 8000 8502

# Default command runs the API; docker-compose overrides it for the dashboard.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
