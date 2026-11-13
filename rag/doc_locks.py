"""In-process per-document locks for concurrent catalog/payload writes.

Acquire every involved ``doc_id`` in sorted order so supersede (two ids),
metadata PATCH, and DELETE cannot deadlock or interleave. A parallel
PUT /content path can reuse ``doc_lock`` after rebase.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

_registry_lock = Lock()
_locks: dict[str, Lock] = {}


def _lock_for(doc_id: str) -> Lock:
    """Return the process-wide lock for ``doc_id``, creating it if needed."""
    with _registry_lock:
        lock = _locks.get(doc_id)
        if lock is None:
            lock = Lock()
            _locks[doc_id] = lock
        return lock


@contextmanager
def doc_lock(*doc_ids: str) -> Iterator[None]:
    """Hold per-doc locks in sorted order until the ``with`` block exits.

    Empty, blank, and duplicate ids are ignored. Acquiring no ids is a no-op.
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
