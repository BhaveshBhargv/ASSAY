"""Settings for the recommendation engine."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase6"

# Pick up API keys from a .env file in the project root, if there is one. Real
# environment variables still take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv is optional
    pass

DEFAULT_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3.1"  # ollama pull llama3.1
ANTHROPIC_MODEL = "claude-sonnet-5"
# OpenRouter needs OPENROUTER_API_KEY. Free model names end in ":free" and change
# fairly often, so the model can be overridden with ASSAY_OPENROUTER_MODEL.
OPENROUTER_MODEL = os.getenv("ASSAY_OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# A low temperature keeps the output consistent between runs.
TEMPERATURE = 0.2
MAX_TOKENS = 2048

OPENROUTER_REASONING = {
    "effort": "none",
    "exclude": True,
}

TOP_K_EVIDENCE = 6  # guideline passages passed to the LLM

# The citation check always runs. The extra LLM call that checks each piece of
# advice is really supported by its evidence is slower, so it's off by default.
LLM_ENTAILMENT_RECHECK = False
# Share of advice items that need a valid citation for the report to count as passed.
MIN_GROUNDEDNESS = 0.80

# Always added by the app, never written by the LLM.
DISCLAIMER = (
    "This report provides general, evidence-based lifestyle information derived "
    "from your blood test results. It is NOT a medical diagnosis and does not "
    "replace advice from a qualified healthcare professional. Discuss any "
    "changes, and any results that concern you, with your GP or clinician."
)
