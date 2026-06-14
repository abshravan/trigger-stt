"""Structured logging configuration.

Provides a single :func:`configure_logging` entry point that sets up the root
logger with either a human-readable or JSON formatter. Each log record carries a
``component`` (the logger name) so output can be traced back to a module.

Log format fields: timestamp, level, component, message.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

_CONFIGURED = False


class HumanFormatter(logging.Formatter):
    """Compact, human-readable log formatter."""

    default_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.default_fmt, datefmt="%Y-%m-%d %H:%M:%S")


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Include any extra structured fields attached via `extra=`.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


# Attributes that are part of every LogRecord and should not be treated as
# user-supplied structured fields.
_RESERVED_RECORD_KEYS = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime"}


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the root logger once for the whole application.

    Args:
        level: Logging level name (e.g. ``"INFO"``).
        json_output: If ``True``, emit JSON log lines; otherwise human-readable.
    """
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove pre-existing handlers so re-configuration (e.g. in tests) is clean.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    root.addHandler(handler)

    # Quiet down noisy third-party libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a component.

    Args:
        name: Component name, conventionally ``__name__`` or a short label.
    """
    return logging.getLogger(name)
