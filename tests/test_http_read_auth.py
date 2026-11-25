"""Bearer auth on catalog GET routes, OpenAPI surfaces, and upload-media."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, TEST_MCP_TOKEN, patched_embeddings

TEST_READ_TOKEN = "test-mcp-read-only"
READ_HEADERS = {"Authorization": f"Bearer {TEST_READ_TOKEN}"}
WRONG_HEADERS = {"Authorization": "Bearer wrong-token"}


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """TestClient with write bearer configured on default headers."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


@pytest.fixture()
def dual_token_client(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """App with distinct write and read tokens."""
    monkeypatch.setenv("MCP_API_TOKEN", TEST_MCP_TOKEN)
    monkeypatch.setenv("MCP_READ_TOKEN", TEST_READ_TOKEN)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            yield test_client


_PROTECTED_GET_ROUTES = (
    "/documents",
    "/openapi.json",
    "/docs",
    "/redoc",
)


@pytest.mark.parametrize("path", _PROTECTED_GET_ROUTES)
def test_protected_routes_reject_unauthenticated(
    dual_token_client: TestClient, path: str
) -> None:
    """Missing bearer must return 401 on protected read surfaces."""
    assert dual_token_client.get(path).status_code == 401


@pytest.mark.parametrize("path", _PROTECTED_GET_ROUTES)
def test_protected_routes_reject_wrong_token(
    dual_token_client: TestClient, path: str
) -> None:
    """Invalid bearer must return 401."""
    assert dual_token_client.get(path, headers=WRONG_HEADERS).status_code == 401


@pytest.mark.parametrize("path", _PROTECTED_GET_ROUTES)
def test_protected_routes_accept_write_token(
    client: TestClient, path: str
) -> None:
    """Write bearer must authorize protected GET routes."""
    response = client.get(path)
    assert response.status_code == 200


@pytest.mark.parametrize("path", _PROTECTED_GET_ROUTES)
def test_protected_routes_accept_read_token(
    dual_token_client: TestClient, path: str
) -> None:
    """Read-only bearer must authorize protected GET routes."""
    response = dual_token_client.get(path, headers=READ_HEADERS)
    assert response.status_code == 200


def test_catalog_get_rejects_unauthenticated_after_upload(client: TestClient) -> None:
    """Document detail endpoints require bearer once a doc exists."""
    upload = client.post(
        "/upload",
        files={"file": ("note.txt", b"secret catalog", "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["doc_id"]
    no_auth: dict[str, str] = {"Authorization": ""}
    assert client.get("/documents", headers=no_auth).status_code == 401
    assert client.get(f"/documents/{doc_id}", headers=no_auth).status_code == 401
    assert client.get(f"/documents/{doc_id}/text", headers=no_auth).status_code == 401


def test_health_and_node_info_stay_public(
    dual_token_client: TestClient,
) -> None:
    """Liveness and node catalog remain unauthenticated."""
    assert dual_token_client.get("/health").status_code == 200
    assert dual_token_client.get("/node-info").status_code == 200


def test_fail_closed_without_any_bearer_env(
    monkeypatch: MonkeyPatch,
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Unset MCP/RAG read tokens must yield 503 on protected routes."""
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.delenv("MCP_READ_TOKEN", raising=False)
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as test_client:
            assert test_client.get("/documents").status_code == 503
            assert test_client.get("/openapi.json").status_code == 503


def test_upload_media_requires_write_token_when_enabled(
    monkeypatch: MonkeyPatch,
    dual_token_client: TestClient,
) -> None:
    """Legacy upload-media (when re-enabled) must reject read-only bearer."""
    monkeypatch.setenv("RAG_UPLOAD_MEDIA_DEPRECATED", "0")
    unauth = dual_token_client.post(
        "/upload-media",
        files={"file": ("x.bin", b"x", "application/octet-stream")},
    )
    assert unauth.status_code == 401
    read_only = dual_token_client.post(
        "/upload-media",
        files={"file": ("x.bin", b"x", "application/octet-stream")},
        headers=READ_HEADERS,
    )
    assert read_only.status_code == 401
    write_ok = dual_token_client.post(
        "/upload-media",
        files={"file": ("x.bin", b"payload", "application/octet-stream")},
        headers={"Authorization": f"Bearer {TEST_MCP_TOKEN}"},
    )
    assert write_ok.status_code == 200