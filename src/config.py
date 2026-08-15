import os
from pathlib import Path

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_ARXIV_DIR = DATA_DIR / "raw" / "arxiv"
RAW_PATENTS_DIR = DATA_DIR / "raw" / "patents"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist upon initialization
for directory in [RAW_ARXIV_DIR, RAW_PATENTS_DIR, PROCESSED_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM and Pipeline Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "llama3.1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200