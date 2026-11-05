"""Tests for search_images MCP wiring and hybrid merge."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from rag.mcp_tools import _reciprocal_rank_fusion


class _FakePoint:
    """Minimal Qdrant scored point stand-in."""

    def __init__(self, point_id: str, payload: dict[str, Any] | None = None) -> None:
        """Store id and payload for RRF tests."""
        self.id = point_id
        self.payload = payload or {}


def test_reciprocal_rank_fusion_prefers_both_lists() -> None:
    """RRF promotes ids that appear in multiple ranked lists."""
    visual = [_FakePoint("a"), _FakePoint("b")]
    text = [_FakePoint("b"), _FakePoint("c")]
    merged = _reciprocal_rank_fusion([visual, text], limit=3)
    assert [point.id for point in merged] == ["b", "a", "c"]


def test_search_images_module_checks_flag() -> None:
    """Visual search helpers honor IMAGE_EMBEDDING_ENABLED."""
    from rag.image_embed import embed_image_query

    with patch("rag.image_embed.image_embedding_enabled", return_value=False):
        with pytest.raises(RuntimeError, match="IMAGE_EMBEDDING_ENABLED"):
            embed_image_query("sunset")
