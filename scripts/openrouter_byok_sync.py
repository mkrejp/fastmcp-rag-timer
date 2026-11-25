"""Register NVIDIA and Novita keys as OpenRouter BYOK credentials.

Requires ``OPENROUTER_MANAGEMENT_KEY``. Never prints secret values. Production
embeddings stay on direct Novita HTTP, not this path.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rag.openrouter import (
    openrouter_management_configured,
    resolve_openrouter_novita_byok_key,
    resolve_openrouter_nvidia_byok_key,
    sync_openrouter_nvidia_novita_byok,
)


def main() -> int:
    """Sync BYOK credentials and print a secret-free status JSON."""
    payload: dict[str, Any] = {
        "management_configured": openrouter_management_configured(),
        "nvidia_key_present": bool(resolve_openrouter_nvidia_byok_key()),
        "novita_key_present": bool(resolve_openrouter_novita_byok_key()),
    }
    try:
        payload["byok"] = sync_openrouter_nvidia_novita_byok()
    except RuntimeError as exc:
        payload["error"] = str(exc)
        print(json.dumps(payload, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True))
    byok = payload["byok"]
    if byok.get("nvidia") == "skipped_no_management":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
