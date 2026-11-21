#!/usr/bin/env python3
"""CLI backfill of curated Trails ``layer_objects`` (metadata only).

Examples::

    python scripts/rag-sync-layer-objects.py --dry-run
    python scripts/rag-sync-layer-objects.py

Requires ``RAG_DATA_ROOT`` (or default). Live mode also needs
``TRAILS_LAYER_OBJECTS_FORWARD=1`` and ``TRAILS_API_URL`` / ``TRAILS_API_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import resolve_rag_data_root  # noqa: E402
from rag.storage import RagStorage  # noqa: E402
from rag.trails_sync import sync_layer_objects_to_trails  # noqa: E402


def main() -> int:
    """Run dry-run or live layer_objects backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count/qualify without calling Trails HTTP",
    )
    parser.add_argument(
        "--rag-data-root",
        default=None,
        help="Override RAG_DATA_ROOT",
    )
    args = parser.parse_args()
    root = resolve_rag_data_root(args.rag_data_root)
    storage = RagStorage(root)
    storage.ensure_layout()
    report = sync_layer_objects_to_trails(storage, dry_run=bool(args.dry_run))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") == "disabled" and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
