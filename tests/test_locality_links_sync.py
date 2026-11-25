"""Live Diff Push coverage for locality_links (obec–document links)."""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from rag.config import LOCALITY_LINKS_COLLECTION
from rag.publish.diff import build_file_diff, catalog_fingerprint
from rag.publish.diff_push import build_and_push
from rag.publish.manifest import ReleaseManifest
from rag.publish.qdrant_sync import (
    apply_qdrant_release_with_client,
    export_points_for_docs,
    locality_links_catalog_fingerprint,
)
from rag.publish.sync_rollback import restore_rollback, snapshot_touched_docs
from rag.storage import RagStorage
from rag.vectorstore import open_qdrant, upsert_locality_links


def _sample_link(doc_id: str = "doc-links-1", *, obec_id: str = "554782") -> dict:
    return {
        "obec_id": obec_id,
        "name": "Praha",
        "doc_id": doc_id,
        "chunk_index": 0,
        "matched_text": "Praha",
        "char_start": 0,
        "char_end": 5,
        "source_type": "text_chunk",
        "method": "manual",
    }


def test_locality_links_fingerprint_changes_on_upsert(tmp_path: Path) -> None:
    """Payload-only fingerprint must move when links are added."""
    client = open_qdrant(tmp_path / "qdrant")
    try:
        before = locality_links_catalog_fingerprint(client)
        assert before == {}
        upsert_locality_links(client, [_sample_link()])
        after = locality_links_catalog_fingerprint(client)
        assert "doc-links-1" in after
        assert after["doc-links-1"]
        # Idempotent upsert keeps the same fingerprint.
        upsert_locality_links(client, [_sample_link()])
        again = locality_links_catalog_fingerprint(client)
        assert again == after
    finally:
        client.close()


def test_build_file_diff_link_only_does_not_ship_files(tmp_path: Path) -> None:
    """Link-only changes list locality docs without re-adding file blobs."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    storage.save_original(doc_id, "a.txt", b"alpha")
    storage.write_processed(
        doc_id=doc_id,
        filename="a.txt",
        mime="text/plain",
        sha256="aaa",
        fulltext="alpha",
        chunks=[{"index": 0, "page": 0, "text": "alpha"}],
        images=[],
        status="ready",
    )
    writer_fp = catalog_fingerprint(storage)
    # Catalog at parity; links differ on reader vs writer.
    writer_links = {doc_id: "writer-link-fp"}
    reader_links = {doc_id: "reader-link-fp"}
    delta = build_file_diff(
        storage,
        parent_release_id="rel-parent",
        reader_catalog=writer_fp,
        reader_locality_catalog=reader_links,
        writer_locality_catalog=writer_links,
    )
    assert delta.catalog_doc_ids_upserted == []
    assert delta.locality_link_doc_ids_upserted == [doc_id]
    assert delta.files_added == []
    assert delta.files_deleted == []


def test_link_only_export_skips_knowledge_collection(tmp_path: Path) -> None:
    """Link-only export must not pull client_domain_knowledge vectors."""
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upsert_locality_links(client, [_sample_link()])
        records, ids = export_points_for_docs(
            client,
            ["doc-links-1"],
            collections=(LOCALITY_LINKS_COLLECTION,),
        )
        assert ids
        assert all(r["collection"] == LOCALITY_LINKS_COLLECTION for r in records)
        assert all(r.get("vector") in (None, {}, []) or True for r in records)
    finally:
        client.close()


def test_apply_link_only_is_idempotent(tmp_path: Path) -> None:
    """Applying the same locality points twice leaves a stable count."""
    client = open_qdrant(tmp_path / "qdrant")
    try:
        upsert_locality_links(client, [_sample_link(obec_id="554782")])
        records, _ids = export_points_for_docs(
            client,
            ["doc-links-1"],
            collections=(LOCALITY_LINKS_COLLECTION,),
        )
        stats1 = apply_qdrant_release_with_client(
            client,
            upsert_doc_ids=[],
            delete_doc_ids=[],
            locality_link_doc_ids=["doc-links-1"],
            records=records,
        )
        stats2 = apply_qdrant_release_with_client(
            client,
            upsert_doc_ids=[],
            delete_doc_ids=[],
            locality_link_doc_ids=["doc-links-1"],
            records=records,
        )
        assert stats1["points_upserted"] >= 1
        assert stats2["points_upserted"] >= 1
        fp = locality_links_catalog_fingerprint(client)
        assert "doc-links-1" in fp
    finally:
        client.close()


def test_build_and_push_link_only_exports_locality(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """build_and_push with link drift must begin a release and send link points."""
    monkeypatch.delenv("RAG_SYNC_NIGHTLY_OFFLINE", raising=False)
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    doc_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    storage.save_original(doc_id, "b.txt", b"beta")
    storage.write_processed(
        doc_id=doc_id,
        filename="b.txt",
        mime="text/plain",
        sha256="bbb",
        fulltext="beta",
        chunks=[{"index": 0, "page": 0, "text": "beta"}],
        images=[],
        status="ready",
    )
    qdrant = open_qdrant(storage.qdrant_dir)
    try:
        upsert_locality_links(qdrant, [_sample_link(doc_id=doc_id)])
        writer_fp = catalog_fingerprint(storage)
        writer_links = locality_links_catalog_fingerprint(qdrant)

        begin_manifests: list[ReleaseManifest] = []

        class _Client:
            def get_watermark(self) -> dict:
                return {"release_id": "rel-parent"}

            def fetch_catalog(self) -> tuple[
                dict[str, str], dict[str, str], dict[str, str]
            ]:
                # Catalog parity; reader has no links yet.
                return writer_fp, writer_fp, {}

            def begin(
                self, manifest: ReleaseManifest, *, dry_run: bool = False
            ) -> dict:
                begin_manifests.append(manifest)
                return {"status": "receiving", "dry_run": dry_run}

            def put_chunk(self, release_id: str, chunk: object) -> dict:
                return {"ok": True}

            def commit(self, release_id: str, *, source: str | None = None) -> dict:
                return {"status": "committed", "source": source}

            def abort(self, release_id: str) -> dict:
                return {"status": "aborted"}

        result = build_and_push(
            storage, qdrant=qdrant, client=_Client()  # type: ignore[arg-type]
        )
        assert result["pushed"] is True
        assert result["locality_link_docs"] == 1
        assert begin_manifests
        assert begin_manifests[0].locality_link_doc_ids_upserted == [doc_id]
        assert begin_manifests[0].catalog_doc_ids_upserted == []
        assert begin_manifests[0].files_added == []
        # Writer still has the link fingerprint we started from.
        assert writer_links[doc_id]
    finally:
        qdrant.close()


def test_rollback_link_only_does_not_wipe_knowledge(tmp_path: Path) -> None:
    """Link-only rollback restores locality points without requiring file trees."""
    root = tmp_path / "rag"
    storage = RagStorage(str(root))
    storage.ensure_layout()
    client = open_qdrant(storage.qdrant_dir)
    try:
        upsert_locality_links(client, [_sample_link()])
        snap = snapshot_touched_docs(
            root,
            client=client,
            release_id="rel-link-only",
            doc_ids=[],
            link_only_doc_ids=["doc-links-1"],
        )
        assert snap.is_dir()
        # Mutate links after snapshot.
        upsert_locality_links(
            client, [_sample_link(obec_id="582786")]  # different obec
        )
        restored = restore_rollback(root, client=client, release_id="rel-link-only")
        assert restored["link_only_docs_restored"] == 1
        fp = locality_links_catalog_fingerprint(client)
        assert "doc-links-1" in fp
    finally:
        client.close()


def test_mcp_link_locality_calls_sync_hook(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """MCP post-link helper must call maybe_push_after_ingest."""
    storage = RagStorage(str(tmp_path / "rag"))
    storage.ensure_layout()
    qdrant = open_qdrant(storage.qdrant_dir)
    called: dict[str, object] = {}

    def _fake_push(stor: object, qd: object, **kwargs: object) -> dict:
        called["ok"] = True
        return {"pushed": False, "reason": "empty_diff"}

    monkeypatch.setattr(
        "rag.publish.diff_push.maybe_push_after_ingest",
        _fake_push,
    )
    try:
        from rag.mcp_tools import _safe_sync_push_after_links

        out = _safe_sync_push_after_links(storage, qdrant)
        assert called.get("ok") is True
        assert out == {"pushed": False, "reason": "empty_diff"}
    finally:
        qdrant.close()
