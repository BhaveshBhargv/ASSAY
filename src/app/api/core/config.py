"""API settings, read from ASSAY_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(f"ASSAY_{key}", default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(f"ASSAY_{key}", str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = "Assay API"
    version: str = "1.0.0"
    description: str = (
        "Backend for the blood-test lifestyle recommender. Turns a blood panel into "
        "a clinical-rules + machine-learning risk assessment, retrieves NICE/NHS/WHO "
        "guideline passages, and generates grounded, non-diagnostic lifestyle "
        "recommendations. **Not a diagnostic device.**"
    )
    default_provider: str = field(default_factory=lambda: _env("PROVIDER", "ollama"))
    default_k: int = field(default_factory=lambda: _env_int("RETRIEVE_K", 6))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    # Comma-separated; defaults to the ports the Streamlit app runs on.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _env("CORS_ORIGINS",
                                    "http://localhost:8501,http://localhost:8502").split(",") if o.strip()
        )
    )


settings = Settings()
