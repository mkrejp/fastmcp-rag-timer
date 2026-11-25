#!/usr/bin/env python3
"""Manually push an incremental live diff from ingester → reader.

Requires ``RAG_SYNC_URL`` (reader origin) and ``RAG_SYNC_TOKEN``. Uses the
writer's openable Qdrant path; prefer running while the ingester app is
stopped **or** call via the post-ingest hook which reuses the live client.

Example::

    RAG_SKIP_DEFAULT_APP=1 \\
    RAG_SYNC_URL=https://reader.example.com \\
    RAG_SYNC_TOKEN=… \\
    python scripts/rag-diff-push.py --rag-root /data/rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.publish.diff_push import DiffPushError, build_and_push  # noqa: E402
from rag.publish.qdrant_sync import open_qdrant_for_sync  # noqa: E402
from rag.storage import RagStorage  # noqa: E402


def main() -> int:
    """CLI entry for a one-shot live diff push."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-root", required=True, help="Writer RAG_DATA_ROOT")
    parser.add_argument(
        "--writer-host",
        default="",
        help="Optional writer hostname recorded in the release",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Force a full baseline publish against the reader catalog",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Begin/commit with dry_run (reader validates only, no mutations)",
    )
    parser.add_argument(
        "--no-mark-published",
        action="store_true",
        help="Do not write release.json on the writer after a successful push",
    )
    args = parser.parse_args()

    storage = RagStorage(str(Path(args.rag_root)))
    storage.ensure_layout()
    client = open_qdrant_for_sync(storage.qdrant_dir)
    try:
        result = build_and_push(
            storage,
            qdrant=client,
            writer_host=args.writer_host.strip() or None,
            mark_published=not args.no_mark_published,
            force_baseline=bool(args.force_baseline),
            dry_run=bool(args.dry_run),
        )
    except DiffPushError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        client.close()

    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
