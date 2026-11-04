"""Tests for durable per-day accounting under ``/data/accnt``."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from rag.accounting import (
    AccountingStore,
    accounting_source,
    set_global_store,
    utc_today,
)
from rag.embeddings import EMBEDDING_DIM, get_embedding
from tests.conftest import AUTH_HEADERS, patched_embeddings


@pytest.fixture()
def store(accnt_root: Path) -> AccountingStore:
    """Provide an initialized accounting store under a temp root."""
    accounting = AccountingStore(accnt_root)
    accounting.ensure_layout()
    set_global_store(accounting)
    yield accounting
    set_global_store(None)


def test_record_embedding_writes_day_rollup_and_event(store: AccountingStore) -> None:
    """Embedding records should update today's JSON and append a JSONL event."""
    with accounting_source("ingest_http"):
        store.record_embedding(
            ok=True,
            input_chars=12,
            prompt_tokens=4,
            total_tokens=4,
            tokens_from_api=True,
            latency_ms=15,
            model="baai/bge-m3",
        )
    today = utc_today()
    day = store.read_day(today)
    assert day["embeddings"]["calls"] == 1
    assert day["embeddings"]["prompt_tokens"] == 4
    assert day["embeddings"]["by_source"]["ingest_http"]["calls"] == 1
    assert day["outbound"]["novita_embeddings"]["calls"] == 1
    assert store.day_path(today).exists()
    events = store.read_events(today, limit=10)
    assert events
    assert events[0]["kind"] == "embedding"
    assert events[0]["source"] == "ingest_http"


def test_summarize_aggregates_multiple_days(
    store: AccountingStore, monkeypatch: MonkeyPatch
) -> None:
    """Summarize should total counters across listed day files."""
    for date, calls in (("2026-11-01", 2), ("2026-11-02", 3)):
        monkeypatch.setattr("rag.accounting.utc_today", lambda d=date: d)
        for _ in range(calls):
            store.record_mcp(tool="list_documents", ok=True, response_chars=10)
    summary = store.summarize(days=30)
    assert summary["totals"]["mcp"]["calls"] == 5
    assert len(summary["days"]) == 2


def test_get_embedding_records_usage(
    store: AccountingStore,
    monkeypatch: MonkeyPatch,
) -> None:
    """Successful embeddings should persist OpenAI-shaped usage when present."""
    vector = [0.1] * EMBEDDING_DIM
    response = Mock()
    response.json.return_value = {
        "data": [{"embedding": vector}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr("rag.embeddings.requests.post", Mock(return_value=response))

    with accounting_source("search_mcp"):
        result = get_embedding("hello world", api_key="test-key")

    assert result == vector
    day = store.read_day(utc_today())
    assert day["embeddings"]["calls"] == 1
    assert day["embeddings"]["prompt_tokens"] == 7
    assert day["embeddings"]["tokens_from_api"] == 1
    assert day["embeddings"]["by_source"]["search_mcp"]["calls"] == 1


def test_get_embedding_error_increments_errors(
    store: AccountingStore,
    monkeypatch: MonkeyPatch,
) -> None:
    """Failed embedding HTTP calls should increment error counters."""
    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("boom")
    monkeypatch.setattr("rag.embeddings.requests.post", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="boom"):
        get_embedding("hello", api_key="test-key")

    day = store.read_day(utc_today())
    assert day["embeddings"]["calls"] == 1
    assert day["embeddings"]["errors"] == 1


def test_upload_records_http_and_ingest_source(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """HTTP upload should meter the route and attribute embeddings to ingest_http."""
    from rag.accounting import AccountingStore, get_global_store
    from rag.app import create_app

    recorded: list[str] = []

    def tracking_embed(text: str) -> list[float]:
        """Capture the active accounting source while embedding."""
        from rag.accounting import get_accounting_source

        recorded.append(get_accounting_source())
        return fake_embed(text)  # type: ignore[operator]

    with patched_embeddings(tracking_embed):
        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "note.txt",
                        b"Accounting meters embeddings.",
                        "text/plain",
                    )
                },
            )
            assert response.status_code == 200

    store = get_global_store()
    assert isinstance(store, AccountingStore)
    day = store.read_day(utc_today())
    assert day["http"]["upload"]["calls"] == 1
    assert recorded
    assert all(source == "ingest_http" for source in recorded)


def test_admin_accounts_endpoints(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Admin accounts APIs should return rollups for signed-in Google users."""
    from rag.app import create_app
    from rag.google_oauth import require_google_user

    with patched_embeddings(fake_embed):
        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        app.dependency_overrides[require_google_user] = lambda: {
            "email": "ops@example.com"
        }
        with TestClient(app, headers=AUTH_HEADERS) as client:
            upload = client.post(
                "/upload",
                files={"file": ("a.txt", b"hello tokens", "text/plain")},
            )
            assert upload.status_code == 200
            listed = client.get("/api/admin/accounts?days=7")
            assert listed.status_code == 200
            body = listed.json()
            assert body["accnt_root"] == str(accnt_root)
            assert "totals" in body
            today = body["today"]
            detail = client.get(f"/api/admin/accounts/{today}")
            assert detail.status_code == 200
            assert detail.json()["day"]["http"]["upload"]["calls"] >= 1
            status_resp = client.get("/api/admin/status")
            assert status_resp.status_code == 200
            assert status_resp.json()["volume"]["accnt_root"] == str(accnt_root)
