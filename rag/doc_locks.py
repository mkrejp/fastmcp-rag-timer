"""In-process per-document locks for concurrent catalog/payload writes.

Acquire every involved ``doc_id`` in sorted order so supersede (two ids),
metadata PATCH, DELETE, and PUT /content cannot deadlock or interleave.
Per-doc locks are ``threading.RLock`` so nested ``doc_lock`` (for example
DELETE calling a locked helper, or restore wrapping replace) does not
self-deadlock.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

_registry_lock = RLock()
_locks: dict[str, RLock] = {}


def _lock_for(doc_id: str) -> RLock:
    """Return the process-wide reentrant lock for ``doc_id``, creating it if needed."""
    with _registry_lock:
        lock = _locks.get(doc_id)
        if lock is None:
            lock = RLock()
            _locks[doc_id] = lock
        return lock


@contextmanager
def doc_lock(*doc_ids: str) -> Iterator[None]:
    """Hold per-doc locks in sorted order until the ``with`` block exits.

    Empty, blank, and duplicate ids are ignored. Acquiring no ids is a no-op.
    Re-entering with an overlapping id on the same thread is safe.
    """
    unique = sorted({str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()})
    locks = [_lock_for(doc_id) for doc_id in unique]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()
