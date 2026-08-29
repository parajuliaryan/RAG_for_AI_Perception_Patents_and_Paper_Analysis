import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file automatically — safe to call even if .env doesn't exist
load_dotenv()

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_ARXIV_DIR = DATA_DIR / "raw" / "arxiv"
RAW_PATENTS_DIR = DATA_DIR / "raw" / "patents"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_TEXT_DIR = PROCESSED_DIR / "texts"
CHUNKS_DEBUG_DIR = PROCESSED_DIR / "chunks_debug"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = DATA_DIR / "logs"

# Ensure all data directories exist on import
for _dir in [RAW_ARXIV_DIR, RAW_PATENTS_DIR, RAW_PDF_DIR, PROCESSED_DIR, EXTRACTED_TEXT_DIR, CHUNKS_DEBUG_DIR, VECTOR_STORE_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama / Model Settings
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL: str = "llama3"
EMBED_MODEL: str = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Chunking Settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 2500
CHUNK_OVERLAP: int = 250

# ---------------------------------------------------------------------------
# Vector Store Settings
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "ai_perception_docs"
TOP_K_DEFAULT: int = 5

# ---------------------------------------------------------------------------
# External API Keys  (loaded from .env — never hardcode here)
# ---------------------------------------------------------------------------
EPO_CONSUMER_KEY: str = os.getenv("EPO_CONSUMER_KEY", "")
EPO_CONSUMER_SECRET: str = os.getenv("EPO_CONSUMER_SECRET", "")

OPIK_API_KEY: str = os.getenv("OPIK_API_KEY", "")
OPIK_WORKSPACE: str = os.getenv("OPIK_WORKSPACE", "")
OPIK_PROJECT_NAME: str = os.getenv("OPIK_PROJECT_NAME", "ai-perception-rag")
ENABLE_OPIK: bool = os.getenv("ENABLE_OPIK", "true").lower() == "true"

SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
