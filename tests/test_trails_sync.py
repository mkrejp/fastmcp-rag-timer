"""TrailsAssemblage locality-link forward, retry, and durable outbox."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rag.payloads import chunk_text_sha256
from rag.trails_sync import (
    TrailsSyncError,
    drain_trails_outbox,
    enqueue_trails_outbox,
    forward_locality_links,
    locality_payload_to_trails_upsert,
    sync_links_to_trails,
)
from rag.vectorstore import open_qdrant, upsert_locality_links


def _sample_link(**overrides: object) -> dict[str, object]:
    """Return a valid locality-link mapping."""
    payload: dict[str, object] = {
        "obec_id": "554782",
        "name": "Praha",
        "doc_id": "doc-1",
        "chunk_index": 0,
        "matched_text": "Praha",
        "char_start": 12,
        "char_end": 17,
        "source_type": "pdf_chunk",
        "method": "exact",
        "chunk_sha256": "abc123",
    }
    payload.update(overrides)
    return payload


def test_payload_mapping_omits_link_id_and_fills_chunk_sha256() -> None:
    """Trails POST items must not include link_id; empty sha falls back."""
    payload = _sample_link(chunk_sha256="", link_id="should-not-send")
    body = locality_payload_to_trails_upsert(payload)
    assert body is not None
    assert "link_id" not in body
    assert body["chunk_sha256"] == chunk_text_sha256("Praha")
    assert body["cast_obce_kod"] is None
    assert body["locality_level"] == "obec"
    karlin = locality_payload_to_trails_upsert(
        _sample_link(
            locality_level="cast_obce",
            cast_obce="400637",
            cast_obce_name="Karlín",
        )
    )
    assert karlin is not None
    assert karlin["cast_obce_kod"] == "400637"
    assert karlin["cast_obce_name"] == "Karlín"


def test_forwarding_disabled_makes_no_http(tmp_path: Path) -> None:
    """Unset Trails env must not call HTTP."""
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upserted = upsert_locality_links(client, [_sample_link()])
        with patch("rag.trails_sync.requests.request") as mocked:
            result = forward_locality_links(
                client,
                upserted["ids"],
                outbox_path=tmp_path / "trails_outbox.json",
            )
        assert result["enabled"] is False
        mocked.assert_not_called()
    finally:
        client.close()


def test_forward_posts_batch_and_stores_trails_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful POST should map Trails ids onto local payloads."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upserted = upsert_locality_links(client, [_sample_link()])
        local_id = upserted["ids"][0]
        trails_id = "11111111-1111-4111-8111-111111111111"

        def fake_request(method: str, url: str, **kwargs: Any) -> Any:
            assert method == "POST"
            assert url == "https://trails.example/locality-links"
            headers = kwargs["headers"]
            assert headers["Authorization"] == "Bearer super-secret-token"
            body = kwargs["json"]
            assert "link_id" not in body["links"][0]
            assert body["links"][0]["obec_id"] == "554782"
            response = type("Resp", (), {})()
            response.status_code = 200
            response.content = b'{"inserted":1,"updated":0,"ids":["%s"]}' % trails_id.encode()
            response.text = response.content.decode()
            response.json = lambda: {
                "inserted": 1,
                "updated": 0,
                "ids": [trails_id],
            }
            return response

        with patch("rag.trails_sync.requests.request", side_effect=fake_request):
            result = forward_locality_links(
                client,
                [local_id],
                outbox_path=tmp_path / "trails_outbox.json",
            )
        assert result["enabled"] is True
        assert result["inserted"] == 1
        assert result["queued"] == 0
        stored = client.retrieve(
            collection_name="locality_links",
            ids=[local_id],
            with_payload=True,
        )
        assert stored[0].payload["trails_link_id"] == trails_id
    finally:
        client.close()


def test_http_500_retries_then_queues_outbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """5xx should retry with backoff, then persist the outbox without raising."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upserted = upsert_locality_links(client, [_sample_link()])
        local_id = upserted["ids"][0]
        calls = {"n": 0}

        def fake_request(*_args: object, **_kwargs: object) -> Any:
            calls["n"] += 1
            response = type("Resp", (), {})()
            response.status_code = 500
            response.content = b"nope"
            response.text = "nope"
            response.json = lambda: {}
            return response

        caplog.set_level(logging.WARNING)
        with (
            patch("rag.trails_sync.requests.request", side_effect=fake_request),
            patch("rag.trails_sync.time.sleep") as slept,
        ):
            result = forward_locality_links(
                client,
                [local_id],
                outbox_path=tmp_path / "outbox.json",
            )
        assert result["enabled"] is True
        assert result["queued"] == 1
        assert calls["n"] == 4
        assert slept.call_count == 3
        outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
        assert local_id in outbox["items"][0]["link_ids"]
        joined = " ".join(record.getMessage() for record in caplog.records)
        assert "super-secret-token" not in joined
        assert "super-secret-token" not in (tmp_path / "outbox.json").read_text()
    finally:
        client.close()


def test_outbox_drain_replays_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later drain should POST queued ids once Trails recovers."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
    client = open_qdrant(tmp_path / "qdrant")
    outbox = tmp_path / "outbox.json"
    try:
        upserted = upsert_locality_links(client, [_sample_link()])
        local_id = upserted["ids"][0]
        enqueue_trails_outbox(outbox, [local_id], error="Trails HTTP 500")
        posts = {"n": 0}

        def fake_request(method: str, url: str, **kwargs: Any) -> Any:
            posts["n"] += 1
            assert method == "POST"
            response = type("Resp", (), {})()
            response.status_code = 200
            response.content = b"{}"
            response.text = "{}"
            response.json = lambda: {
                "inserted": 0,
                "updated": 1,
                "ids": ["22222222-2222-4222-8222-222222222222"],
            }
            return response

        with (
            patch("rag.trails_sync.requests.request", side_effect=fake_request),
            patch("rag.trails_sync.time.sleep"),
        ):
            drained = drain_trails_outbox(client, outbox)
        assert drained["drained"] == 1
        assert drained["remaining"] == 0
        assert posts["n"] == 1
        leftover = json.loads(outbox.read_text(encoding="utf-8"))
        assert leftover["items"] == []
    finally:
        client.close()


def test_verify_forwards_patch_after_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified local links should POST then PATCH Trails."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upserted = upsert_locality_links(
            client, [_sample_link(verification_status="verified")]
        )
        local_id = upserted["ids"][0]
        trails_id = "33333333-3333-4333-8333-333333333333"
        methods: list[str] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> Any:
            methods.append(method)
            response = type("Resp", (), {})()
            response.status_code = 200
            if method == "POST":
                response.content = b"ids"
                response.text = ""
                response.json = lambda: {
                    "inserted": 1,
                    "updated": 0,
                    "ids": [trails_id],
                }
            else:
                assert method == "PATCH"
                assert url.endswith(f"/locality-links/{trails_id}/verify")
                assert kwargs["json"]["status"] == "verified"
                response.content = b"{}"
                response.text = "{}"
                response.json = lambda: {"link_id": trails_id}
            return response

        with (
            patch("rag.trails_sync.requests.request", side_effect=fake_request),
            patch("rag.trails_sync.time.sleep"),
        ):
            result = forward_locality_links(
                client,
                [local_id],
                outbox_path=tmp_path / "outbox.json",
            )
        assert methods == ["POST", "PATCH"]
        assert result["verified"] == 1
    finally:
        client.close()


def test_sync_links_to_trails_dry_run_and_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run counts without HTTP; live POST is idempotent."""
    client = open_qdrant(tmp_path / "qdrant")
    outbox = tmp_path / "outbox.json"
    try:
        upsert_locality_links(
            client,
            [
                _sample_link(),
                _sample_link(
                    obec_id="582786",
                    name="Brno",
                    char_start=40,
                    verification_status="rejected",
                ),
            ],
        )
        dry = sync_links_to_trails(client, outbox_path=outbox, dry_run=True)
        assert dry["dry_run"] is True
        assert dry["enabled"] is False
        assert dry["scanned"] == 2
        assert dry["would_upsert"] == 2
        assert dry["would_verify"] == 1

        monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
        monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")
        calls: list[tuple[str, str]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> Any:
            calls.append((method, url))
            response = type("Resp", (), {})()
            response.status_code = 200
            response.content = b"{}"
            response.text = "{}"
            if method == "POST":
                n = len(kwargs["json"]["links"])
                response.json = lambda: {
                    "inserted": n,
                    "updated": 0,
                    "ids": [f"id-{i}" for i in range(n)],
                }
            else:
                response.json = lambda: {}
            return response

        with (
            patch("rag.trails_sync.requests.request", side_effect=fake_request),
            patch("rag.trails_sync.time.sleep"),
        ):
            live = sync_links_to_trails(client, outbox_path=outbox, dry_run=False)
        assert live["status"] == "ok"
        assert live["forwarded"] == 2
        assert live["verified"] == 1
        assert any(method == "POST" for method, _url in calls)
        assert any(method == "PATCH" for method, _url in calls)
    finally:
        client.close()


def test_http_request_redacts_token_in_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4xx bodies that echo the token must not leak it in TrailsSyncError."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "super-secret-token")

    def fake_request(*_args: object, **_kwargs: object) -> Any:
        response = type("Resp", (), {})()
        response.status_code = 401
        response.content = b"token=super-secret-token"
        response.text = "token=super-secret-token"
        response.json = lambda: {}
        return response

    from rag.trails_sync import _http_request

    with patch("rag.trails_sync.requests.request", side_effect=fake_request):
        with pytest.raises(TrailsSyncError) as excinfo:
            _http_request(
                "POST",
                "https://trails.example/locality-links",
                json_body={"links": []},
                token="super-secret-token",
            )
    assert "super-secret-token" not in str(excinfo.value)
