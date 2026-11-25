#!/usr/bin/env python3
"""Apply image visibility restrictions from a JSONL list (default: dry-run)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_REASON = "Operator licence review"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from JSONL (``doc_id``, ``image_id`` per line)."""
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _group_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group ``image_id`` values by ``doc_id``."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        image_id = str(row.get("image_id", "")).strip()
        if not doc_id or not image_id:
            raise SystemExit("each row needs doc_id and image_id")
        grouped.setdefault(doc_id, []).append(image_id)
    return grouped


def _request(
    *,
    base_url: str,
    token: str,
    method: str,
    path: str,
    body: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """PATCH visibility with bearer auth."""
    url = f"{base_url.rstrip('/')}{path}"
    if dry_run:
        url = f"{url}?dry_run=true"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {path}: {detail}") from exc


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "list_file",
        type=Path,
        help="JSONL file with doc_id, image_id per line",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_INGESTER_URL"),
        help="Ingester RAG HTTP base (env RAG_INGESTER_URL); never default to reader",
    )
    parser.add_argument(
        "--token-env",
        default="MCP_API_TOKEN",
        help="Env var name holding the write bearer token",
    )
    parser.add_argument(
        "--restricted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict (default) or un-restrict",
    )
    parser.add_argument("--reason", default=DEFAULT_REASON)
    parser.add_argument("--decided-by", required=True, help="Operator id for audit")
    parser.add_argument(
        "--license-status",
        default=None,
        help="license_status when restricting (default restricted)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform writes (default is dry-run)",
    )
    args = parser.parse_args()
    if not args.base_url:
        raise SystemExit("--base-url or RAG_INGESTER_URL required")
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"{args.token_env} is not set")
    dry_run = not args.apply
    rows = _load_rows(args.list_file)
    grouped = _group_by_doc(rows)
    shared: dict[str, Any] = {
        "restricted": args.restricted,
        "reason": args.reason,
        "decided_by": args.decided_by,
    }
    if args.license_status:
        shared["license_status"] = args.license_status

    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"{mode}: {sum(len(v) for v in grouped.values())} images", file=sys.stderr)
    for doc_id, image_ids in grouped.items():
        body = {**shared, "image_ids": image_ids}
        result = _request(
            base_url=str(args.base_url),
            token=token,
            method="PATCH",
            path=f"/images/{doc_id}/visibility",
            body=body,
            dry_run=dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
