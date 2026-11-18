#!/usr/bin/env bash
# Apply an incremental release directory onto a reader RAG_DATA_ROOT.
# Stop the reader process before running (Qdrant path lock). Optional --reboot.
set -euo pipefail

RAG_ROOT="${RAG_DATA_ROOT:-/data/rag}"
RELEASE_DIR=""
REBOOT=0
SERVICE_NAME="${RAG_READER_SERVICE:-}"

usage() {
  echo "Usage: $0 --release-dir DIR [--rag-root DIR] [--reboot] [--service NAME]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-dir) RELEASE_DIR="$2"; shift 2 ;;
    --rag-root) RAG_ROOT="$2"; shift 2 ;;
    --reboot) REBOOT=1; shift ;;
    --service) SERVICE_NAME="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$RELEASE_DIR" ]] || usage
[[ -d "$RELEASE_DIR" ]] || { echo "release dir missing: $RELEASE_DIR" >&2; exit 1; }

if [[ -n "$SERVICE_NAME" ]] && command -v systemctl >/dev/null 2>&1; then
  systemctl stop "$SERVICE_NAME" || true
fi

export RAG_SKIP_DEFAULT_APP=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}")
from rag.publish.apply import apply_release_dir
state = apply_release_dir("${RAG_ROOT}", "${RELEASE_DIR}")
print(state)
PY

if [[ "$REBOOT" -eq 1 ]]; then
  echo "Rebooting host after release apply..."
  /sbin/reboot
elif [[ -n "$SERVICE_NAME" ]] && command -v systemctl >/dev/null 2>&1; then
  systemctl start "$SERVICE_NAME"
fi
