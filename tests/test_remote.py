"""Tests for remote URL normalization and SSRF guards."""

from __future__ import annotations

import pytest

from rag.remote import (
    UnsafeRemoteUrlError,
    filename_from_url,
    fetch_remote_bytes,
    normalize_download_url,
)


def test_github_blob_urls_become_raw() -> None:
    """GitHub blob pages for kraje/obce should map to raw.githubusercontent.com."""
    assert (
        normalize_download_url(
            "https://github.com/siwekm/czech-geojson/blob/master/kraje.json"
        )
        == "https://raw.githubusercontent.com/siwekm/czech-geojson/master/kraje.json"
    )
    assert (
        normalize_download_url(
            "https://github.com/siwekm/czech-geojson/blob/master/obce.json"
        )
        == "https://raw.githubusercontent.com/siwekm/czech-geojson/master/obce.json"
    )
    already_raw = "https://raw.githubusercontent.com/siwekm/czech-geojson/master/okresy.json"
    assert normalize_download_url(already_raw) == already_raw


def test_filename_from_url() -> None:
    """Basenames should come from the URL path."""
    assert (
        filename_from_url(
            "https://raw.githubusercontent.com/siwekm/czech-geojson/master/obce.json"
        )
        == "obce.json"
    )


def test_localhost_is_rejected() -> None:
    """Loopback URLs must not be fetched."""
    with pytest.raises(UnsafeRemoteUrlError):
        fetch_remote_bytes("http://127.0.0.1/secret.json")
    with pytest.raises(UnsafeRemoteUrlError):
        fetch_remote_bytes("http://localhost/secret.json")


def test_non_http_scheme_rejected() -> None:
    """file:// and similar schemes are not allowed."""
    with pytest.raises(UnsafeRemoteUrlError):
        fetch_remote_bytes("file:///etc/passwd")
