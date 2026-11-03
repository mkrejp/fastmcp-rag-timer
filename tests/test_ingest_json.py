"""HTTP tests for bearer-authenticated remote GeoJSON ingest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


KRAJE_SAMPLE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "nationalCode": "19",
            "localId": "AU.2.19",
            "name": "Hlavní město Praha",
            "id": "CZ0100000000",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[14.0, 50.0], [14.1, 50.1]]],
            },
        }
    ],
}


@pytest.fixture()
def api_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a test bearer token."""
    token = "test-rag-token"
    monkeypatch.setenv("RAG_API_TOKEN", token)
    return token


@pytest.fixture()
def auth_client(
    rag_root: Path, fake_embed: Callable[[str], list[float]], api_token: str
) -> Iterator[TestClient]:
    """Build a TestClient with mocked embeddings."""
    with patch("rag.embeddings.get_embedding", side_effect=fake_embed), patch(
        "rag.ingest.get_embedding", side_effect=fake_embed
    ), patch("rag.mcp_tools.get_embedding", side_effect=fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root))
        with TestClient(app) as test_client:
            yield test_client


def test_ingest_json_requires_bearer(auth_client: TestClient) -> None:
    """Missing or wrong tokens should be rejected."""
    url = "https://github.com/siwekm/czech-geojson/blob/master/kraje.json"
    missing = auth_client.post("/ingest-json", json={"url": url})
    assert missing.status_code == 401
    wrong = auth_client.post(
        "/ingest-json",
        json={"url": url},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert wrong.status_code == 401


def test_ingest_json_from_github_blob_url(
    auth_client: TestClient, api_token: str
) -> None:
    """A GitHub blob URL should fetch raw GeoJSON and index feature names."""
    blob = "https://github.com/siwekm/czech-geojson/blob/master/kraje.json"
    raw = "https://raw.githubusercontent.com/siwekm/czech-geojson/master/kraje.json"
    body = json.dumps(KRAJE_SAMPLE).encode("utf-8")
    with patch("rag.routes.fetch_remote_bytes", return_value=(body, raw)):
        response = auth_client.post(
            "/ingest-json",
            json={"url": blob},
            headers={"Authorization": f"Bearer {api_token}"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Success"
    assert payload["filename"] == "kraje.json"
    assert payload["chunks_indexed"] >= 1
    assert payload["source_url"] == raw

    text_resp = auth_client.get(f"/documents/{payload['doc_id']}/text")
    assert text_resp.status_code == 200
    text = text_resp.json()["text"]
    assert "Hlavní město Praha" in text
    assert "14.0" not in text


def test_ingest_json_unconfigured_token(
    rag_root: Path, fake_embed: Callable[[str], list[float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint should fail closed when RAG_API_TOKEN is unset."""
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    with patch("rag.embeddings.get_embedding", side_effect=fake_embed), patch(
        "rag.ingest.get_embedding", side_effect=fake_embed
    ), patch("rag.mcp_tools.get_embedding", side_effect=fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root))
        with TestClient(app) as client:
            response = client.post(
                "/ingest-json",
                json={"url": "https://example.com/kraje.json"},
                headers={"Authorization": "Bearer anything"},
            )
    assert response.status_code == 503
