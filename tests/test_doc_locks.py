"""Tests for in-process per-document locks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from rag.doc_locks import doc_lock


def test_doc_lock_sorted_order_does_not_deadlock() -> None:
    """Two threads locking the same pair in opposite order should both finish."""
    started = Event()
    release = Event()

    def hold_then_wait(first: str, second: str) -> str:
        with doc_lock(first, second):
            started.set()
            release.wait(timeout=2)
        return f"{first}:{second}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(hold_then_wait, "bbb", "aaa")
        assert started.wait(timeout=2)
        future_b = pool.submit(hold_then_wait, "aaa", "bbb")
        release.set()
        assert future_a.result(timeout=2) == "bbb:aaa"
        assert future_b.result(timeout=2) == "aaa:bbb"


def test_doc_lock_ignores_blank_and_duplicate_ids() -> None:
    """Blank ids are skipped; duplicates acquire a single lock."""
    with doc_lock("", "  ", "doc-a", "doc-a"):
        with doc_lock():
            pass
