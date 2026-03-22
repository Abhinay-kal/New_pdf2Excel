"""
Structured logging configuration for Electra-Core.

Outputs a key=value formatted line to both stdout and a rotating log file.
Call ``setup_logging()`` once at process start (in main.py).
"""
from __future__ import annotations

import logging
import logging.config
import logging.handlers

from config.settings import LOG_DIR


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with console + rotating-file handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": (
                    "%(asctime)s level=%(levelname)-8s name=%(name)s "
                    "pid=%(process)d %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "structured",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_DIR / "electra_core.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "formatter": "structured",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(config)
