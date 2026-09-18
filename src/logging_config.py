"""Application logging to the console and a rotating file."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import Settings, load_settings


class JsonFormatter(logging.Formatter):
    """Format logs as machine-readable JSON for later analysis."""

    EXTRA_FIELDS = (
        "request_id",
        "model_version",
        "latency_ms",
        "prediction",
        "probability",
        "payload",
    )

    def format(self, record: logging.LogRecord) -> str:
        document = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            if hasattr(record, field):
                document[field] = getattr(record, field)
        if record.exc_info:
            document["exception"] = self.formatException(record.exc_info)
        return json.dumps(document, default=str, ensure_ascii=False)


def setup_logging(
    settings: Optional[Settings] = None,
    log_file_path: Optional[str] = None,
) -> None:
    """Configure root logging once with console and rotating file handlers."""

    active_settings = settings or load_settings()
    log_path = Path(log_file_path) if log_file_path else active_settings.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(active_settings.log_level.upper())
    if any(
        getattr(handler, "_mlops_handler", False) for handler in root_logger.handlers
    ):
        return

    formatter = JsonFormatter()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler._mlops_handler = True  # type: ignore[attr-defined]

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler._mlops_handler = True  # type: ignore[attr-defined]

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.info("logging_configured")
