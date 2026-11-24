"""Reader chat helper: prepaid Novita rerank plus optional Gateway chat."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from rag.accounting import today_outbound_calls
from rag.gateway_chat import chat_configured, chat_enabled, parse_helper_json
from rag.node_info import is_http_write_request
from rag.novita_rerank import NOVITA_BILLING, rerank_configured
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.test_mcp_search import _call_tool, _initialize


@pytest.fixture(autouse=True)
def _scrub_live_keys(monkeypatch: MonkeyPatch) -> None:
    """Keep helper tests off live Novita / Gateway credentials."""
    for name in (
        "BGE_M3_NOVITA_API_KEY",
        "NOVITA_API_KEY",
        "OPENAI_TOKEN",
        "OPENAI_API_KEY",
        "AI_GATEWAY_API_KEY",
        "VERCEL_AI_GATEWAY_API_KEY",
        "AI_GATEWAY_CHAT_MODEL",
        "CHAT_HELPER_CHAT",
        "CHAT_HELPER_RERANK",
        "CHAT_HELPER_DAILY_CHAT_CALLS",
        "CHAT_HELPER_DAILY_RERANK_CALLS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """Build a TestClient with temp storage and mocked embeddings."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


def _ingest(client: TestClient, name: str, body: bytes) -> str:
    """Upload a text document and return its ``doc_id``."""
    response = client.post(
        "/upload",
        files={"file": (name, body, "text/plain")},
    )
    assert response.status_code == 200, response.text
    doc_id = response.json()["doc_id"]
    assert isinstance(doc_id, str)
    return doc_id


def _helper(
    client: TestClient,
    arguments: dict[str, Any],
    *,
    call_id: int = 2,
) -> dict[str, Any]:
    """Call MCP chat_helper and parse the JSON payload."""
    _initialize(client)
    text = _call_tool(client, "chat_helper", arguments, call_id=call_id)
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def test_health_reports_prepaid_novita_helper(client: TestClient) -> None:
    """Health exposes chat-helper flags without secrets."""
    response = client.get("/health")
    assert response.status_code == 200
    helper = response.json()["chat_helper"]
    assert helper["novita_billing"] == NOVITA_BILLING
    assert helper["rerank"]["model"] == "baai/bge-reranker-v2-m3"
    assert helper["rerank"]["configured"] is False
    assert helper["chat"]["model"] == "vmc/cheap"
    assert helper["chat"]["configured"] is False
    assert helper["chat"]["enabled"] is False


def test_chat_helper_has_no_rest_route(client: TestClient) -> None:
    """This slice keeps search on MCP; REST /chat/helper is absent."""
    response = client.post("/chat/helper", json={"query": "timer"})
    assert response.status_code in {404, 405}


def test_chat_helper_retrieve_without_llm(client: TestClient) -> None:
    """Missing Novita/Gateway keys still return vector hits."""
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    _ingest(client, "other.txt", b"Unrelated railway volume mounts at /data.")
    body = _helper(
        client,
        {"query": "Prague Art Nouveau timer", "limit": 1, "related_limit": 2},
    )
    assert body["status"] == "ok"
    assert body["novita_billing"] == "prepaid_credits"
    assert body["hits"]
    assert "content" in body["hits"][0]
    assert body["hits"][0].get("doc_id")
    assert body["rerank"]["status"] == "skipped_no_key"
    assert body["chat"]["status"] == "disabled"
    assert body["answer"] is None


def test_chat_helper_related_facts_are_neighbors(client: TestClient) -> None:
    """Leftover ranked hits become related facts, not supersession peers."""
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    _ingest(client, "other.txt", b"Unrelated railway volume mounts at /data.")
    body = _helper(
        client,
        {
            "query": "Prague Art Nouveau timer",
            "limit": 1,
            "related_limit": 2,
            "use_rerank": False,
        },
    )
    assert body["hits"]
    assert len(body["hits"]) == 1
    assert body["related_facts"]
    related = body["related_facts"][0]
    assert related["suggestion_kind"] == "neighbor_rank"
    assert related["cited"] is False
    assert related["doc_id"] != body["hits"][0]["doc_id"]
    assert "content" in related


def test_chat_helper_truncates_long_query(client: TestClient) -> None:
    """MCP queries longer than 1000 characters are truncated before retrieve."""
    body = _helper(
        client,
        {"query": "timer " * 400, "use_rerank": False, "use_chat": False},
    )
    assert len(body["query"]) == 1000


def test_chat_helper_chat_off_by_default(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """Gateway is not called unless both env and use_chat are on."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-test-key")
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    with patch("rag.gateway_chat.requests.post") as posted:
        body = _helper(
            client,
            {"query": "Prague timer", "use_rerank": False, "use_chat": True},
        )
    posted.assert_not_called()
    assert body["chat"]["status"] == "disabled"
    assert chat_enabled() is False


def test_chat_helper_reranks_best_fit(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """Prepaid Novita rerank scores documents by content, not vector order."""
    monkeypatch.setenv("NOVITA_API_KEY", "prepaid-test-key")

    def _rerank_by_prague(url: str, json: dict[str, Any], **_kwargs: object) -> Mock:
        """Score documents that mention Prague higher."""
        del url
        documents = list(json["documents"])
        ranked = sorted(
            enumerate(documents),
            key=lambda item: (0 if "Prague" in str(item[1]) else 1, item[0]),
        )
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(
            return_value={
                "results": [
                    {"index": index, "relevance_score": 0.99 - rank * 0.1}
                    for rank, (index, _text) in enumerate(ranked)
                ]
            }
        )
        return response

    _ingest(client, "first.txt", b"Alpha railway volume mounts at /data.")
    _ingest(client, "second.txt", b"Beta Prague Art Nouveau timer facade.")
    with patch(
        "rag.novita_rerank.requests.post", side_effect=_rerank_by_prague
    ) as posted:
        body = _helper(
            client,
            {"query": "Prague Art Nouveau", "limit": 2, "use_chat": False},
        )
    assert body["rerank"]["status"] == "applied"
    assert body["rerank"]["model"] == "baai/bge-reranker-v2-m3"
    assert "Prague" in body["hits"][0]["content"]
    assert body["hits"][0]["rerank_score"] == 0.99
    posted.assert_called_once()
    sent = posted.call_args.kwargs["json"]
    assert sent["model"] == "baai/bge-reranker-v2-m3"
    assert "return_documents" not in sent
    assert today_outbound_calls("novita_rerank") == 1


def test_chat_helper_llm_marks_cited_without_reorder(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """Gateway chat may cite hits but must not reorder them."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-test-key")
    monkeypatch.setenv("CHAT_HELPER_CHAT", "1")
    _ingest(client, "first.txt", b"Alpha railway volume mounts at /data.")
    _ingest(client, "second.txt", b"Beta Prague Art Nouveau timer facade.")
    first_hits = _helper(
        client,
        {
            "query": "Prague timer",
            "limit": 2,
            "related_limit": 0,
            "use_rerank": False,
            "use_chat": False,
        },
        call_id=2,
    )
    original_ids = [hit["doc_id"] for hit in first_hits["hits"]]
    gateway = Mock()
    gateway.raise_for_status = Mock()
    gateway.json = Mock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "Best fit is the Prague timer facade.",
                                "best_hit_indexes": [0],
                                "related_suggestions": [],
                            }
                        )
                    }
                }
            ]
        }
    )
    with patch("rag.gateway_chat.requests.post", return_value=gateway) as posted:
        body = _helper(
            client,
            {
                "query": "Prague timer",
                "limit": 2,
                "related_limit": 0,
                "use_rerank": False,
                "use_chat": True,
            },
            call_id=3,
        )
    assert body["chat"]["status"] == "applied"
    assert body["chat"]["model"] == "vmc/cheap"
    assert body["answer"] == "Best fit is the Prague timer facade."
    assert [hit["doc_id"] for hit in body["hits"]] == original_ids
    assert body["hits"][0]["cited"] is True
    sent = posted.call_args.kwargs["json"]
    assert sent["model"] == "vmc/cheap"
    assert sent["providerOptions"] == {"gateway": {"only": ["novita"]}}
    assert sent["response_format"] == {"type": "json_object"}


def test_chat_helper_reader_mcp_read_token(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
    monkeypatch: MonkeyPatch,
) -> None:
    """Reader mode advertises chat_helper to MCP_READ_TOKEN callers."""
    monkeypatch.setenv("APP_MODE", "reader")
    monkeypatch.setenv("MCP_READ_TOKEN", "test-mcp-read-token")
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        headers = {
            "Authorization": "Bearer test-mcp-read-token",
            "Accept": "application/json, text/event-stream",
            "Host": "localhost",
        }
        with TestClient(app, headers=headers) as client:
            listed = client.post(
                "/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "1"},
                    },
                },
            )
            assert listed.status_code == 200
            tools = client.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert tools.status_code == 200
            assert "chat_helper" in tools.text
            assert "ingest_text" not in tools.text


def test_chat_helper_is_not_an_http_write() -> None:
    """Reader nodes must keep helper search off the write-guard list."""
    assert is_http_write_request("POST", "/chat/helper") is False
    assert is_http_write_request("POST", "/upload") is True


def test_gateway_ignores_embedding_openai_key(monkeypatch: MonkeyPatch) -> None:
    """Fail-closed: embedding OPENAI_API_KEY is not a Gateway credential."""
    monkeypatch.setenv("OPENAI_API_KEY", "embedding-fallback")
    monkeypatch.setenv("NOVITA_API_KEY", "prepaid-novita")
    assert chat_configured() is False
    assert rerank_configured() is True


def test_openai_key_does_not_configure_rerank(monkeypatch: MonkeyPatch) -> None:
    """Rerank accepts only prepaid Novita-named keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "embedding-fallback")
    assert rerank_configured() is False


def test_parse_helper_json_strips_fences() -> None:
    """LLM markdown fences still parse into a JSON object."""
    payload = parse_helper_json(
        '```json\n{"answer": "ok", "best_hit_indexes": []}\n```'
    )
    assert payload["answer"] == "ok"


def test_parse_helper_json_strips_think_blocks() -> None:
    """DeepSeek-style think tags before JSON still parse."""
    wrapped = (
        "<think>plan citation indexes</think>"
        '{"answer": "ok", "best_hit_indexes": [0], "related_suggestions": []}'
    )
    payload = parse_helper_json(wrapped)
    assert payload["answer"] == "ok"
    assert payload["best_hit_indexes"] == [0]


def test_parse_helper_json_accepts_leading_prose() -> None:
    """Leading natural-language prose before the JSON object still parses."""
    wrapped = (
        "Here is the JSON response.\n"
        '{"answer": "facade note", "best_hit_indexes": [], '
        '"related_suggestions": []}'
    )
    payload = parse_helper_json(wrapped)
    assert payload["answer"] == "facade note"


def test_parse_helper_json_rejects_non_object() -> None:
    """Arrays are not accepted as helper payloads."""
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_helper_json("[1, 2, 3]")


def test_chat_helper_think_wrapped_gateway_json(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """Think-wrapped Gateway JSON still yields applied chat with citations."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-test-key")
    monkeypatch.setenv("CHAT_HELPER_CHAT", "1")
    _ingest(client, "facade.txt", b"Beta Prague Art Nouveau timer facade.")
    gateway = Mock()
    gateway.raise_for_status = Mock()
    inner = json.dumps(
        {
            "answer": "The timer facade matches Prague Art Nouveau.",
            "best_hit_indexes": [0],
            "related_suggestions": [],
        }
    )
    gateway.json = Mock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": ("<reasoning>match hit zero</reasoning>" + inner)
                    }
                }
            ]
        }
    )
    with patch("rag.gateway_chat.requests.post", return_value=gateway):
        body = _helper(
            client,
            {
                "query": "Prague timer",
                "limit": 1,
                "related_limit": 0,
                "use_rerank": False,
                "use_chat": True,
            },
        )
    assert body["chat"]["status"] == "applied"
    assert body["answer"] is not None
    assert body["chat"]["best_hit_indexes"] == [0]
    assert body["hits"][0]["cited"] is True


def test_chat_helper_skips_budget(client: TestClient, monkeypatch: MonkeyPatch) -> None:
    """Daily rerank cap returns skipped_budget and makes no HTTP call."""
    monkeypatch.setenv("NOVITA_API_KEY", "prepaid-test-key")
    monkeypatch.setenv("CHAT_HELPER_DAILY_RERANK_CALLS", "0")
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    with patch("rag.novita_rerank.requests.post") as posted:
        body = _helper(
            client,
            {"query": "Prague timer", "use_chat": False},
        )
    posted.assert_not_called()
    assert body["rerank"]["status"] == "skipped_budget"
    assert today_outbound_calls("novita_rerank") == 0


def test_chat_helper_rerank_failure_keeps_hits(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """A rerank 500 keeps vector hits and does not call Gateway."""
    monkeypatch.setenv("NOVITA_API_KEY", "prepaid-test-key")
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    response = Mock()
    error = __import__("requests").HTTPError("server error")
    error.response = Mock(status_code=500)
    response.raise_for_status.side_effect = error
    with (
        patch("rag.novita_rerank.requests.post", return_value=response),
        patch("rag.gateway_chat.complete_chat") as complete,
    ):
        body = _helper(
            client,
            {"query": "Prague timer", "use_chat": False},
        )
    complete.assert_not_called()
    assert body["hits"]
    assert body["rerank"]["status"] == "http_5xx"


def test_chat_helper_model_pin_rejects_override(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """Non-allowlisted AI_GATEWAY_CHAT_MODEL is not called."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-test-key")
    monkeypatch.setenv("CHAT_HELPER_CHAT", "1")
    monkeypatch.setenv("AI_GATEWAY_CHAT_MODEL", "openai/gpt-4o-mini")
    _ingest(client, "facade.txt", b"Beta Prague Art Nouveau timer facade.")
    with patch("rag.gateway_chat.complete_chat") as complete:
        body = _helper(
            client,
            {
                "query": "Art Nouveau facade",
                "limit": 1,
                "use_rerank": False,
                "use_chat": True,
            },
        )
    complete.assert_not_called()
    assert body["chat"]["status"] == "skipped_bad_model"
    assert body["hits"]


def test_health_stays_up_during_slow_helper(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    """MCP chat_helper must not block the FastAPI event loop."""
    monkeypatch.setenv("NOVITA_API_KEY", "prepaid-test-key")
    _ingest(client, "prague.txt", b"Prague Art Nouveau timer facade on the square.")
    _initialize(client)

    def _slow(*_args: object, **_kwargs: object) -> dict[str, Any]:
        """Block a worker thread, not the event loop."""
        time.sleep(1.2)
        return {
            "status": "ok",
            "query": "timer",
            "novita_billing": NOVITA_BILLING,
            "hits": [],
            "related_facts": [],
            "answer": None,
            "rerank": {"status": "disabled", "model": "baai/bge-reranker-v2-m3"},
            "chat": {
                "status": "disabled",
                "model": "vmc/cheap",
                "best_hit_indexes": [],
                "related_suggestions": [],
            },
        }

    health_ms: list[float] = []

    def _probe() -> None:
        """Hit /health while MCP helper is in flight."""
        time.sleep(0.2)
        started = time.perf_counter()
        response = client.get("/health")
        health_ms.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200

    with patch("rag.mcp_tools.run_chat_helper", side_effect=_slow):
        probe = threading.Thread(target=_probe)
        probe.start()
        text = _call_tool(
            client,
            "chat_helper",
            {"query": "timer", "use_rerank": False, "use_chat": False},
            call_id=9,
        )
        probe.join(timeout=5)
    assert json.loads(text)["status"] == "ok"
    assert health_ms, "health probe did not run"
    assert health_ms[0] < 400
