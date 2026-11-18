"""Byte budgets and path names for live diff-push sync."""

from __future__ import annotations

# Marek-approved target; soft ceiling for packing; hard reject on the reader.
CHUNK_TARGET_BYTES = 32 * 1024 * 1024
CHUNK_SOFT_MAX_BYTES = 64 * 1024 * 1024
CHUNK_HARD_MAX_BYTES = 96 * 1024 * 1024

SYNC_DIR_NAME = ".sync"
INCOMING_DIR_NAME = "incoming"
ROLLBACK_DIR_NAME = "rollback"
SESSION_FILENAME = "session.json"
CHUNKS_DIR_NAME = "chunks"
ASSEMBLED_DIR_NAME = "assembled"
BLOBS_STAGE = "blobs"
POINTS_STAGE = "points"
STAGE_NAMES = frozenset({BLOBS_STAGE, POINTS_STAGE})
