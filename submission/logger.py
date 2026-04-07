# =============================================================================
# logger.py — structured logging
# =============================================================================
"""
Usage
-----
    from logger import get_logger
    log = get_logger(__name__)

    log.debug("token count: %d", n)
    log.info("phase started: %s", phase)
    log.warning("empty batch at index %d — skipping", i)
    log.error("Qdrant upsert failed: %s", exc)

One logger per module keeps output attributable.
The root logger is configured once when this module is first imported.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_DIR, LOG_LEVEL

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"

    level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+ to keep terminal clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler (DEBUG+ for full trace; rotates at 10 MB, keeps 3 backups)
    fh = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(ch)
    root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call."""
    _configure_root()
    return logging.getLogger(name)
