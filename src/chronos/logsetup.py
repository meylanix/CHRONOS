"""Structured logging. JSON lines (for experiments/CI) or compact text (for humans).

Pass structured context with `extra={"ctx": {...}}`; it is emitted as JSON fields.
Logs go to stderr so that stdout stays machine-readable (reports, JSON).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            payload.update(ctx)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"[{record.levelname.lower()}] {record.name}: {record.getMessage()}"
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            base += " " + " ".join(f"{k}={v}" for k, v in ctx.items())
        return base


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    root = logging.getLogger("chronos")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("chronos") else f"chronos.{name}")
