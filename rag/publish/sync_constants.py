"""Byte budgets for future live diff-push HTTP chunking.

Library-only constants for PR1; no HTTP routes consume them yet.
"""

from __future__ import annotations

# Preferred payload size per sync chunk (32 MiB).
CHUNK_TARGET_BYTES: int = 32 * 1024 * 1024

# Soft ceiling before the client should split further (64 MiB).
CHUNK_SOFT_MAX_BYTES: int = 64 * 1024 * 1024

# Hard reject above this size (96 MiB).
CHUNK_HARD_MAX_BYTES: int = 96 * 1024 * 1024

__all__ = [
    "CHUNK_HARD_MAX_BYTES",
    "CHUNK_SOFT_MAX_BYTES",
    "CHUNK_TARGET_BYTES",
]
