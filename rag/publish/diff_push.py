"""Ingester-side live diff push client (Option A).

Uses the reader's sync API: watermark → catalog → begin → chunks → commit.
Vectors are exported from the already-open writer Qdrant client (no re-embed
on the reader).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode, urljoin

from qdrant_client import QdrantClient

from rag.config import (
    LOCALITY_LINKS_COLLECTION,
    resolve_sync_token,
    resolve_sync_url,
    sync_nightly_offline,
    sync_push_on_ingest_enabled,
)
from rag.logging_setup import get_logger
from rag.publish.diff import build_file_diff, catalog_fingerprint, summarize_diff
from rag.publish.manifest import ReleaseManifest
from rag.publish.qdrant_sync import (
    export_points_for_docs,
    locality_links_catalog_fingerprint,
)
from rag.publish.release_state import write_release_state
from rag.publish.sync_chunk import PackedChunk, pack_blob_chunks, pack_points_chunks
from rag.storage import RagStorage

logger = get_logger("rag.publish.diff_push")

_DRY_RUN_SAMPLE_SIZE = 10


class DiffPushError(RuntimeError):
    """Raised when a live sync push fails."""


def _sample_doc_ids(
    doc_ids: list[str], *, limit: int = _DRY_RUN_SAMPLE_SIZE
) -> list[str]:
    """Return the first ``limit`` doc ids for operator previews."""
    return list(doc_ids[:limit])


def sync_push_enabled() -> bool:
    """Return whether post-ingest auto-push should run.

    Default **off**. Requires ``RAG_SYNC_URL`` + ``RAG_SYNC_TOKEN``, an explicit
    ``RAG_SYNC_PUSH_ENABLED=1`` (or ``RAG_SYNC_PUSH_ON_INGEST=1``), and must not
    be in a nightly offline window.
    """
    if sync_nightly_offline():
        return False
    if resolve_sync_url() is None:
        return False
    if resolve_sync_token() is None:
        return False
    return sync_push_on_ingest_enabled()


def _request_json(
    method: str,
    url: str,
    *,
    token: str,
    body: Mapping[str, Any] | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 120.0,
) -> Any:
    """HTTP JSON helper with Bearer auth. Never logs the token."""
    data: bytes | None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if raw_body is not None:
        data = raw_body
        headers["Content-Type"] = content_type or "application/octet-stream"
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        data = None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if not payload:
                return {}
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DiffPushError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DiffPushError(f"{method} {url} failed: {exc}") from exc


class DiffPushClient:
    """Push incremental releases to a reader sync API."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        timeout: float = 120.0,
    ) -> None:
        """Configure reader origin and sync bearer token."""
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.timeout = timeout

    def _url(self, path: str, query: Mapping[str, str] | None = None) -> str:
        """Build an absolute URL under the sync API."""
        joined = urljoin(self.base_url, path.lstrip("/"))
        if query:
            return f"{joined}?{urlencode(query)}"
        return joined

    def get_watermark(self) -> dict[str, Any]:
        """GET /sync/v1/watermark."""
        return _request_json(
            "GET",
            self._url("sync/v1/watermark"),
            token=self.token,
            timeout=self.timeout,
        )

    def get_catalog(self) -> dict[str, str]:
        """GET /sync/v1/catalog → doc_id→fingerprint map (documents only)."""
        docs, _content, _links = self.fetch_catalog()
        return docs

    def fetch_catalog(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """GET /sync/v1/catalog → ``(documents, content, locality_links)``."""
        body = _request_json(
            "GET",
            self._url("sync/v1/catalog"),
            token=self.token,
            timeout=self.timeout,
        )
        docs = body.get("documents") or {}
        content = body.get("documents_content") or {}
        links = body.get("locality_links") or {}
        return (
            {str(k): str(v) for k, v in docs.items()},
            {str(k): str(v) for k, v in content.items()},
            {str(k): str(v) for k, v in links.items()},
        )

    def begin(
        self,
        manifest: ReleaseManifest,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """POST /sync/v1/push/begin."""
        return _request_json(
            "POST",
            self._url("sync/v1/push/begin"),
            token=self.token,
            body={"manifest": manifest.to_dict(), "dry_run": bool(dry_run)},
            timeout=self.timeout,
        )

    def validate(self, release_id: str) -> dict[str, Any]:
        """POST /sync/v1/push/validate (integrity only)."""
        return _request_json(
            "POST",
            self._url("sync/v1/push/validate"),
            token=self.token,
            body={"release_id": release_id},
            timeout=max(self.timeout, 300.0),
        )

    def put_chunk(self, release_id: str, chunk: PackedChunk) -> dict[str, Any]:
        """PUT /sync/v1/push/chunk?release_id=…"""
        return _request_json(
            "PUT",
            self._url("sync/v1/push/chunk", {"release_id": release_id}),
            token=self.token,
            raw_body=chunk.encode(),
            timeout=max(self.timeout, 300.0),
        )

    def commit(self, release_id: str, *, source: str | None = None) -> dict[str, Any]:
        """POST /sync/v1/push/commit."""
        body: dict[str, Any] = {"release_id": release_id}
        if source:
            body["source"] = source
        return _request_json(
            "POST",
            self._url("sync/v1/push/commit"),
            token=self.token,
            body=body,
            timeout=max(self.timeout, 300.0),
        )

    def abort(self, release_id: str) -> dict[str, Any]:
        """POST /sync/v1/push/abort."""
        return _request_json(
            "POST",
            self._url("sync/v1/push/abort"),
            token=self.token,
            body={"release_id": release_id},
            timeout=self.timeout,
        )

    def status(self, release_id: str | None = None) -> dict[str, Any]:
        """GET /sync/v1/push/status."""
        query = {"release_id": release_id} if release_id else None
        return _request_json(
            "GET",
            self._url("sync/v1/push/status", query),
            token=self.token,
            timeout=self.timeout,
        )


def build_and_push(
    storage: RagStorage,
    *,
    qdrant: QdrantClient,
    client: DiffPushClient | None = None,
    writer_host: str | None = None,
    mark_published: bool = True,
    force_baseline: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Diff against the reader, push chunks, commit. Returns a summary dict.

    No-op summary when there is nothing to publish. Raises ``DiffPushError`` on
    transport/API failures. Manual CLI may call this even when auto-push is off.
    """
    if sync_nightly_offline() and not dry_run:
        raise DiffPushError("RAG_SYNC_NIGHTLY_OFFLINE=1; refusing live push")

    push = client
    if push is None:
        url = resolve_sync_url()
        token = resolve_sync_token()
        if not url or not token:
            raise DiffPushError("RAG_SYNC_URL and RAG_SYNC_TOKEN are required")
        push = DiffPushClient(url, token=token)

    watermark = push.get_watermark()
    parent = watermark.get("release_id")
    reader_catalog, reader_content, reader_locality = push.fetch_catalog()
    writer_locality = locality_links_catalog_fingerprint(qdrant)
    manifest = build_file_diff(
        storage,
        parent_release_id=parent if isinstance(parent, str) else None,
        reader_catalog=reader_catalog,
        reader_content_catalog=reader_content or None,
        reader_locality_catalog=reader_locality,
        writer_locality_catalog=writer_locality,
        writer_host=writer_host,
        force_baseline=force_baseline,
    )

    summary = summarize_diff(manifest)
    # index.json alone is not a catalog change — ignore it for the no-op gate
    # so a parity/cron run never opens a release (V2).
    meaningful_files = [
        change for change in manifest.files_added if change.path != "index.json"
    ]
    if (
        not meaningful_files
        and not manifest.files_deleted
        and not manifest.catalog_doc_ids_upserted
        and not manifest.catalog_doc_ids_deleted
        and not manifest.locality_link_doc_ids_upserted
        and not manifest.catalog_doc_ids_metadata_only
    ):
        logger.info("Live sync: nothing to push (empty diff)")
        return {**summary, "pushed": False, "reason": "empty_diff"}

    metadata_only_set = set(manifest.catalog_doc_ids_metadata_only)
    full_upsert_ids = [
        doc_id
        for doc_id in manifest.catalog_doc_ids_upserted
        if doc_id not in metadata_only_set
    ]
    if dry_run:
        preview: dict[str, Any] = {
            **summary,
            "pushed": False,
            "dry_run": True,
            "reason": "dry_run_preview",
            "parent_release_id": parent,
            "docs_upserted_full": len(full_upsert_ids),
            "docs_upserted_metadata_only": len(metadata_only_set),
            "docs_deleted": len(manifest.catalog_doc_ids_deleted),
            "locality_link_docs": len(manifest.locality_link_doc_ids_upserted),
            "sample_doc_ids_upserted": _sample_doc_ids(
                manifest.catalog_doc_ids_upserted
            ),
            "sample_doc_ids_upserted_full": _sample_doc_ids(full_upsert_ids),
            "sample_doc_ids_metadata_only": _sample_doc_ids(
                manifest.catalog_doc_ids_metadata_only
            ),
            "sample_doc_ids_deleted": _sample_doc_ids(manifest.catalog_doc_ids_deleted),
            "writer_catalog_count": len(catalog_fingerprint(storage)),
        }
        logger.info(
            "Live sync dry-run preview upserts=%s metadata_only=%s full=%s "
            "deletes=%s files_added=%s",
            preview["docs_upserted"],
            preview["docs_upserted_metadata_only"],
            preview["docs_upserted_full"],
            preview["docs_deleted"],
            preview["files_added"],
        )
        return preview
    vector_doc_ids = full_upsert_ids
    records: list[dict[str, Any]] = []
    point_ids: list[str] = []
    if vector_doc_ids:
        exported, ids = export_points_for_docs(qdrant, vector_doc_ids)
        records.extend(exported)
        point_ids.extend(ids)
    if metadata_only_set:
        exported, ids = export_points_for_docs(
            qdrant,
            sorted(metadata_only_set),
            payload_only=True,
        )
        records.extend(exported)
        point_ids.extend(ids)
    manifest.qdrant_point_ids_deleted = list(manifest.catalog_doc_ids_deleted)
    # Link-only docs: payload-only export (no knowledge vectors) for memory safety.
    catalog_set = set(manifest.catalog_doc_ids_upserted)
    link_only = [
        doc_id
        for doc_id in manifest.locality_link_doc_ids_upserted
        if doc_id not in catalog_set
    ]
    if link_only:
        exported, ids = export_points_for_docs(
            qdrant,
            link_only,
            collections=(LOCALITY_LINKS_COLLECTION,),
        )
        records.extend(exported)
        point_ids.extend(ids)
    if point_ids:
        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for pid in point_ids:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        manifest.qdrant_point_ids_upserted = ordered

    push.begin(manifest, dry_run=False)

    root = Path(storage.root)
    blob_files: list[tuple[str, bytes, str]] = []
    for change in manifest.files_added:
        data = (root / change.path).read_bytes()
        blob_files.append((change.path, data, change.sha256))

    try:
        for chunk in pack_blob_chunks(manifest.release_id, blob_files):
            push.put_chunk(manifest.release_id, chunk)
        for chunk in pack_points_chunks(manifest.release_id, records):
            push.put_chunk(manifest.release_id, chunk)
        commit = push.commit(manifest.release_id, source=writer_host or "diff-push")
    except DiffPushError:
        try:
            push.abort(manifest.release_id)
        except DiffPushError:
            logger.warning(
                "Failed to abort release %s after push error",
                manifest.release_id,
                exc_info=True,
            )
        raise

    if mark_published and not dry_run and commit.get("status") == "committed":
        write_release_state(
            storage.root,
            release_id=manifest.release_id,
            parent_release_id=manifest.parent_release_id,
            source=writer_host or "diff-push",
            extra=summarize_diff(manifest),
        )

    logger.info(
        "Live sync pushed release=%s files=%s docs_upserted=%s status=%s",
        manifest.release_id,
        len(manifest.files_added),
        len(manifest.catalog_doc_ids_upserted),
        commit.get("status"),
    )
    return {
        **summarize_diff(manifest),
        "pushed": commit.get("status") == "committed",
        "commit": commit,
        "writer_catalog_count": len(catalog_fingerprint(storage)),
    }


def maybe_push_after_ingest(
    storage: RagStorage,
    qdrant: QdrantClient,
    *,
    writer_host: str | None = None,
) -> dict[str, Any] | None:
    """Best-effort post-ingest push when sync env is configured.

    Failures are logged and returned; they never raise into the ingest path.
    """
    if not sync_push_enabled():
        return None
    try:
        return build_and_push(
            storage,
            qdrant=qdrant,
            writer_host=writer_host,
            mark_published=True,
        )
    except Exception as exc:  # noqa: BLE001 - never fail ingest on sync
        logger.warning("Post-ingest live sync push failed: %s", exc, exc_info=True)
        return {"pushed": False, "error": str(exc)}
