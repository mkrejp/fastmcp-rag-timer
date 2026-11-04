"""Application logging with an in-memory ring buffer for the admin UI."""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

LOG_BUFFER_SIZE = 500
LOGGER_NAME = "rag"


class RingBufferHandler(logging.Handler):
    """Keep the newest log records in memory for HTTP inspection."""

    def __init__(self, capacity: int = LOG_BUFFER_SIZE) -> None:
        """Create a handler that stores up to ``capacity`` formatted records."""
        super().__init__()
        self._capacity = capacity
        self._entries: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Append one formatted log record to the ring buffer."""
        try:
            entry = {
                "ts": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
                "func": record.funcName,
            }
            with self._lock:
                self._entries.append(entry)
        except Exception:  # noqa: BLE001 - never break the app on logging
            self.handleError(record)

    def snapshot(
        self, *, limit: int = 100, level: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the newest ``limit`` entries, optionally filtered by level name."""
        with self._lock:
            items = list(self._entries)
        if level:
            wanted = level.upper()
            items = [item for item in items if item["level"] == wanted]
        if limit < 1:
            return []
        return items[-limit:]


_handler: RingBufferHandler | None = None
_configured = False


def configure_logging(level: int = logging.INFO) -> RingBufferHandler:
    """Configure the ``rag`` logger once and return the ring-buffer handler."""
    global _handler, _configured
    if _configured and _handler is not None:
        return _handler

    handler = RingBufferHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    handler.setLevel(level)

    root = logging.getLogger()
    if not any(isinstance(existing, RingBufferHandler) for existing in root.handlers):
        # Keep uvicorn/access logs visible; add our ring buffer alongside defaults.
        if not root.handlers:
            stream = logging.StreamHandler()
            stream.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            root.addHandler(stream)
        root.addHandler(handler)
        root.setLevel(level)

    logging.getLogger(LOGGER_NAME).setLevel(level)
    _handler = handler
    _configured = True
    return handler


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a child logger under the RAG namespace."""
    configure_logging()
    return logging.getLogger(name)


def get_log_handler() -> RingBufferHandler:
    """Return the shared ring-buffer handler, configuring logging if needed."""
    return configure_logging()
