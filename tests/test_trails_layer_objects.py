"""Trails curated layer_objects forward: qualify, fingerprint, outbox, hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag.storage import DocumentMeta, ImageMeta, RagStorage
from rag.trails_layer_objects import (
    PR_STAV_BUDOVA,
    PR_STAV_FASADA,
    ack_key,
    build_layer_object_ops,
    drain_objects_outbox,
    enqueue_objects_outbox,
    geo_qualifies,
    has_secese_signal,
    maybe_forward_layer_objects,
    object_fingerprint,
    sync_layer_objects_to_trails,
)


def _geo(**overrides: object) -> dict[str, Any]:
    """Valid high-confidence building geo_refined."""
    payload: dict[str, Any] = {
        "lat": 50.087,
        "lon": 14.421,
        "precision": "building",
        "precision_m": 15.0,
        "confidence": 0.91,
        "confidence_level": "high",
        "status": "overeno",
        "needs_refinement": False,
        "verified": True,
        "sources": [{"type": "manual"}],
    }
    payload.update(overrides)
    return payload


def _meta(
    tmp_path: Path,
    *,
    doc_id: str = "doc-1",
    tags: list[str] | None = None,
    geo: dict[str, Any] | None = None,
    images: list[ImageMeta] | None = None,
    doc_status: str = "active",
) -> DocumentMeta:
    """Build a DocumentMeta with optional secese tags and geo."""
    extra: dict[str, Any] = {"tags": tags or [], "doc_status": doc_status}
    return DocumentMeta(
        doc_id=doc_id,
        filename="secese-facade.pdf",
        mime="application/pdf",
        sha256="abc",
        uploaded_at="2026-01-01T00:00:00+00:00",
        chunk_count=1,
        status="ready",
        original_path=str(tmp_path / doc_id / "original.pdf"),
        fulltext_path=str(tmp_path / doc_id / "fulltext.txt"),
        chunks_path=str(tmp_path / doc_id / "chunks.json"),
        images=images or [],
        geo_refined=geo,
        extra=extra,
    )


def test_geo_qualifies_gate() -> None:
    """Status/confidence/precision gates for v1."""
    assert geo_qualifies(_geo())
    assert geo_qualifies(_geo(status="zpresneno", confidence=0.8))
    assert not geo_qualifies(_geo(status="priblizne"))
    assert not geo_qualifies(_geo(confidence=0.79))
    assert not geo_qualifies(_geo(precision="municipality"))
    assert not geo_qualifies(None)


def test_has_secese_signal() -> None:
    """Secese / art-nouveau tags must match."""
    assert has_secese_signal({"tags": ["secese", "praha"]})
    assert has_secese_signal({"tags": ["Art Nouveau"]})
    assert not has_secese_signal({"tags": ["funkcionalismus"]})


def test_build_ops_building_and_facade(tmp_path: Path) -> None:
    """Doc-level → budova; image-level → fasáda; both need secese."""
    image = ImageMeta(
        image_id="img-1",
        sha256="img",
        file_path="img.jpg",
        page_number=1,
        width=10,
        height=10,
        mime="image/jpeg",
        geo_refined=_geo(lat=50.1, lon=14.5),
        extra={"tags": ["secese"]},
    )
    meta = _meta(
        tmp_path,
        tags=["secese"],
        geo=_geo(),
        images=[image],
    )
    ops = build_layer_object_ops(meta, previously_acked={})
    kinds = {(op["op"], op["pr_stav_nazev"], op.get("rag_image_id")) for op in ops}
    assert ("upsert", PR_STAV_BUDOVA, None) in kinds
    assert ("upsert", PR_STAV_FASADA, "img-1") in kinds
    for op in ops:
        assert "fingerprint" in op
        assert op["fingerprint"] == object_fingerprint(op)


def test_build_ops_absent_when_hidden(tmp_path: Path) -> None:
    """Retracted docs soft-absent previously acked keys."""
    meta = _meta(tmp_path, tags=["secese"], geo=_geo(), doc_status="retracted")
    key = ack_key("doc-1")
    ops = build_layer_object_ops(meta, previously_acked={key: "oldfp"})
    assert len(ops) == 1
    assert ops[0]["op"] == "absent"
    assert ops[0]["rag_doc_id"] == "doc-1"


def test_maybe_forward_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag default off → no HTTP."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.delenv("TRAILS_LAYER_OBJECTS_FORWARD", raising=False)
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    with patch("rag.trails_layer_objects.requests.request") as mocked:
        result = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
    assert result["enabled"] is False
    mocked.assert_not_called()


def test_maybe_forward_runs_when_push_env_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer objects forward must not depend on Diff Push env."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    monkeypatch.delenv("RAG_SYNC_URL", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ENABLED", raising=False)
    monkeypatch.delenv("RAG_SYNC_PUSH_ON_INGEST", raising=False)
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    meta = _meta(tmp_path, tags=["secese"], geo=_geo())
    storage.processed_doc_dir(meta.doc_id).mkdir(parents=True, exist_ok=True)
    storage.write_meta(meta)

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        assert method == "POST"
        assert url == "https://trails.example/catalog/objects"
        body = kwargs["json"]
        assert body["objects"][0]["op"] == "upsert"
        assert body["objects"][0]["pr_stav_nazev"] == PR_STAV_BUDOVA
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"inserted":1,"updated":0}'
        resp.text = '{"inserted":1,"updated":0}'
        resp.json.return_value = {"inserted": 1, "updated": 0}
        return resp

    with (
        patch("rag.trails_layer_objects.requests.request", side_effect=fake_request),
        patch("rag.trails_layer_objects.time.sleep"),
    ):
        result = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
    assert result["enabled"] is True
    assert result.get("ok") is True
    assert result.get("forwarded") == 1


def test_maybe_forward_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP failures queue to outbox; wrapper returns ok=False."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    meta = _meta(tmp_path, tags=["secese"], geo=_geo())
    storage.processed_doc_dir(meta.doc_id).mkdir(parents=True, exist_ok=True)
    storage.write_meta(meta)

    def boom(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("down")

    with (
        patch("rag.trails_layer_objects.requests.request", side_effect=boom),
        patch("rag.trails_layer_objects.time.sleep"),
    ):
        result = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
    assert result["enabled"] is True
    assert result.get("ok") is False
    assert result.get("queued", 0) >= 1
    assert storage.trails_objects_outbox_path.is_file()


def test_fingerprint_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged fingerprint must not POST again."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    meta = _meta(tmp_path, tags=["secese"], geo=_geo())
    storage.processed_doc_dir(meta.doc_id).mkdir(parents=True, exist_ok=True)
    storage.write_meta(meta)

    calls = {"n": 0}

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        calls["n"] += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"inserted":1}'
        resp.text = '{"inserted":1}'
        resp.json.return_value = {"inserted": 1, "updated": 0}
        return resp

    with (
        patch("rag.trails_layer_objects.requests.request", side_effect=fake_request),
        patch("rag.trails_layer_objects.time.sleep"),
    ):
        first = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
        second = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
    assert first.get("forwarded") == 1
    assert second.get("forwarded") == 0
    assert calls["n"] == 1


def test_outbox_drain_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued ops drain on a later successful call."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    outbox = tmp_path / "trails_objects_outbox.json"
    ack = tmp_path / "trails_objects_ack.json"
    body = {
        "op": "upsert",
        "rag_doc_id": "doc-1",
        "rag_image_id": None,
        "lon": 14.4,
        "lat": 50.0,
        "hlavni_prvek": "x",
        "pr_stav_nazev": PR_STAV_BUDOVA,
        "geo_status": "overeno",
        "geo_confidence": 0.9,
        "geo_precision": "building",
    }
    body["fingerprint"] = object_fingerprint(body)
    enqueue_objects_outbox(outbox, [body], error="boom")

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"inserted":1}'
        resp.text = '{"inserted":1}'
        resp.json.return_value = {"inserted": 1, "updated": 0}
        return resp

    with (
        patch("rag.trails_layer_objects.requests.request", side_effect=fake_request),
        patch("rag.trails_layer_objects.time.sleep"),
    ):
        drained = drain_objects_outbox(outbox_path=outbox, ack_path=ack)
    assert drained["drained"] == 1
    assert drained["remaining"] == 0


def test_reader_mode_never_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """APP_MODE=reader disables forward even with flag on."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    storage = RagStorage(tmp_path)
    with patch("rag.trails_layer_objects.requests.request") as mocked:
        result = maybe_forward_layer_objects(storage, doc_ids=["doc-1"])
    assert result["enabled"] is False
    assert result.get("reason") == "reader_mode"
    mocked.assert_not_called()


def test_sync_dry_run_lists_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backfill dry_run reports would_forward without HTTP."""
    monkeypatch.delenv("TRAILS_LAYER_OBJECTS_FORWARD", raising=False)
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    meta = _meta(tmp_path, tags=["secese"], geo=_geo())
    storage.processed_doc_dir(meta.doc_id).mkdir(parents=True, exist_ok=True)
    storage.write_meta(meta)
    with patch("rag.trails_layer_objects.requests.request") as mocked:
        report = sync_layer_objects_to_trails(storage, dry_run=True)
    assert report["dry_run"] is True
    assert report["would_forward"] >= 1
    mocked.assert_not_called()


def test_post_mutate_hooks_parallel_to_failed_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routes helper still forwards objects when Diff Push raises."""
    monkeypatch.setenv("TRAILS_API_URL", "https://trails.example")
    monkeypatch.setenv("TRAILS_API_TOKEN", "secret")
    monkeypatch.setenv("TRAILS_LAYER_OBJECTS_FORWARD", "1")
    storage = RagStorage(tmp_path)
    storage.ensure_layout()
    meta = _meta(tmp_path, tags=["secese"], geo=_geo())
    storage.processed_doc_dir(meta.doc_id).mkdir(parents=True, exist_ok=True)
    storage.write_meta(meta)

    def boom_push(*_a: object, **_k: object) -> None:
        raise RuntimeError("push failed")

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"inserted":1}'
        resp.text = '{"inserted":1}'
        resp.json.return_value = {"inserted": 1, "updated": 0}
        return resp

    from rag.mcp_tools import _safe_post_mutate_hooks

    with (
        patch(
            "rag.publish.diff_push.maybe_push_after_ingest",
            side_effect=boom_push,
        ),
        patch("rag.trails_layer_objects.requests.request", side_effect=fake_request),
        patch("rag.trails_layer_objects.time.sleep"),
    ):
        out = _safe_post_mutate_hooks(
            storage, MagicMock(), doc_ids=["doc-1"]
        )
    assert out["sync_push"]["error"] == "sync_hook_crashed" or out[
        "sync_push"
    ].get("pushed") is False
    assert out["trails_objects"]["enabled"] is True
    assert out["trails_objects"].get("forwarded") == 1
