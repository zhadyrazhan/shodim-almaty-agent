"""Shared configuration. Everything tunable lives here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SXODIM_DATA = DATA_DIR / "sxodim_data.json"

# --- Scraping ---------------------------------------------------------------
# Jina Reader turns any page into clean markdown: https://r.jina.ai/<url>
JINA_READER = "https://r.jina.ai/"
SXODIM_PAGES = {
    "afisha": "https://sxodim.com/almaty/afisha",
    "weekend": "https://sxodim.com/almaty/events/weekend",
    "places": "https://sxodim.com/almaty/places",
    "main": "https://sxodim.com/almaty",
}

# --- LLM backends -----------------------------------------------------------
# "openai" (default) or "ollama" for a local Qwen with no API cost.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EXTRACT_MODEL = "gpt-5-mini"   # structuring scraped markdown -> typed records
AGENT_MODEL = "gpt-5-mini"     # answering user questions

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Agent ------------------------------------------------------------------
TOP_K = 6          # how many records the retriever feeds the LLM
REQUEST_TIMEOUT = 180


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY in .env (copy .env.example)")
    return OPENAI_API_KEY
