#!/usr/bin/env python3
"""Build an incremental release manifest (+ optional blob/Qdrant staging).

Example::

    # Stop the writer first when using --stage-qdrant (embedded Qdrant lock).
    RAG_SKIP_DEFAULT_APP=1 python scripts/rag-publish.py \\
        --rag-root /data/rag --out /tmp/release --stage-blobs --stage-qdrant
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.publish.diff import build_file_diff, summarize_diff  # noqa: E402
from rag.publish.qdrant_sync import (  # noqa: E402
    POINTS_REL_PATH,
    export_points_for_docs,
    open_qdrant_for_sync,
    write_points_jsonl,
)
from rag.publish.release_state import (  # noqa: E402
    read_release_state,
    write_release_state,
)
from rag.storage import RagStorage  # noqa: E402


def main() -> int:
    """CLI entry: compute diff and write manifest (+ blobs / Qdrant points)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-root", required=True, help="Writer RAG_DATA_ROOT")
    parser.add_argument("--out", required=True, help="Output release directory")
    parser.add_argument(
        "--reader-catalog",
        default="",
        help="Optional JSON file of reader doc_id→sha256 (empty = baseline)",
    )
    parser.add_argument(
        "--parent-release-id",
        default="",
        help="Override parent release id (default: local release.json)",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Publish all files even when a reader catalog is provided",
    )
    parser.add_argument(
        "--stage-blobs",
        action="store_true",
        help="Copy changed files into out/blobs for transport",
    )
    parser.add_argument(
        "--stage-qdrant",
        action="store_true",
        help=(
            "Export Qdrant points for upserted docs into out/qdrant/points.jsonl "
            "(writer must be stopped; embedded Qdrant is single-writer)"
        ),
    )
    parser.add_argument(
        "--mark-published",
        action="store_true",
        help="Write release.json on the writer root after building",
    )
    parser.add_argument("--writer-host", default="", help="Optional writer hostname")
    args = parser.parse_args()

    rag_root = Path(args.rag_root)
    storage = RagStorage(str(rag_root))
    storage.ensure_layout()

    reader_catalog: dict[str, str] | None = None
    if args.reader_catalog:
        payload = json.loads(Path(args.reader_catalog).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("reader catalog must be a JSON object")
        reader_catalog = {str(k): str(v) for k, v in payload.items()}

    local = read_release_state(rag_root)
    parent = args.parent_release_id.strip() or (
        str(local["release_id"]) if local and local.get("release_id") else None
    )

    manifest = build_file_diff(
        storage,
        parent_release_id=parent,
        reader_catalog=reader_catalog,
        writer_host=args.writer_host.strip() or None,
        force_baseline=bool(args.force_baseline),
    )

    if args.stage_qdrant:
        # Deleted docs: record placeholder ids so apply knows to purge points.
        # Upserted docs: export live points (requires exclusive Qdrant access).
        try:
            client = open_qdrant_for_sync(storage.qdrant_dir)
        except Exception as exc:  # noqa: BLE001 - surface lock errors clearly
            raise SystemExit(
                f"Cannot open writer Qdrant at {storage.qdrant_dir}: {exc}. "
                "Stop the ingester process before --stage-qdrant."
            ) from exc
        try:
            records, point_ids = export_points_for_docs(
                client, manifest.catalog_doc_ids_upserted
            )
            manifest.qdrant_point_ids_upserted = point_ids
            # Apply deletes by doc_id filter; list deleted doc ids as markers.
            manifest.qdrant_point_ids_deleted = list(manifest.catalog_doc_ids_deleted)
            out_points = Path(args.out) / POINTS_REL_PATH
            write_points_jsonl(out_points, records)
        finally:
            client.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_json(out / "manifest.json")
    (out / "summary.json").write_text(
        json.dumps(summarize_diff(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.stage_blobs:
        blob_root = out / "blobs"
        for change in manifest.files_added:
            src = rag_root / change.path
            dest = blob_root / change.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    if args.mark_published:
        write_release_state(
            rag_root,
            release_id=manifest.release_id,
            parent_release_id=manifest.parent_release_id,
            source=args.writer_host.strip() or "rag-publish",
            extra=summarize_diff(manifest),
        )

    print(json.dumps(summarize_diff(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
