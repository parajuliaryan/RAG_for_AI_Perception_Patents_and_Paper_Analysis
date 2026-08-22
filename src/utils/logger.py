import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Resolve logs directory relative to this file so the logger is self-contained
# and does not create a circular import with config.py.
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOGS_DIR / "pipeline.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured with:
      - Console handler  → INFO  level (visible in terminal / Streamlit output)
      - Rotating file    → DEBUG level (persisted to data/logs/pipeline.log)

    Calling get_logger() multiple times with the same name is safe — handlers
    are only attached once.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    # Guard: only add handlers the first time this logger is created
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Allow all levels; handlers filter further

    # --- Console handler (INFO) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)

    # --- Rotating file handler (DEBUG, max 5 MB × 3 backups) ---
    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

