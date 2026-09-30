"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BROWSER_MODE = sys.platform == "emscripten" or os.environ.get("TRIAGE_BROWSER_MODE") == "1"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("TRIAGE_DB_PATH", DATA_DIR / "triage.db"))
SAMPLE_PAPERS_PATH = DATA_DIR / "sample_papers.json"

# --- Embeddings -----------------------------------------------------------
# "auto" uses sentence-transformers when installed, otherwise TF-IDF + LSA.
EMBEDDING_BACKEND = os.environ.get("TRIAGE_EMBEDDING_BACKEND", "auto")
SBERT_MODEL = os.environ.get("TRIAGE_SBERT_MODEL", "all-MiniLM-L6-v2")

# --- LLM explanations -----------------------------------------------------
LLM_MODEL = os.environ.get("TRIAGE_LLM_MODEL", "claude-opus-5")
LLM_EFFORT = os.environ.get("TRIAGE_LLM_EFFORT", "low")  # one-line reasons need little thinking
LLM_MAX_WORKERS = int(os.environ.get("TRIAGE_LLM_WORKERS", "4"))
LLM_TIMEOUT_S = float(os.environ.get("TRIAGE_LLM_TIMEOUT", "60"))

# --- External APIs --------------------------------------------------------
S2_API_KEY = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
HTTP_TIMEOUT_S = float(os.environ.get("TRIAGE_HTTP_TIMEOUT", "30"))
USER_AGENT = "paper-triage/1.0 (research prototype)"

# --- Triage defaults (human-set, tunable in the UI) -----------------------
DEFAULT_READ_CUTOFF = 0.62
DEFAULT_SKIM_CUTOFF = 0.40
MINUTES_PER_READ = 30  # used to cap READ count by the user's time budget
MINUTES_PER_SKIM = 5

LABELS = ("READ", "SKIM", "SKIP")
LABEL_VALUE = {"READ": 1.0, "SKIM": 0.5, "SKIP": 0.0}
