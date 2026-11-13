"""Tests for document supersession links (S1–S12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.document_supersede import supersede_document, unsupersede_document
from rag.embeddings import EMBEDDING_DIM
from rag.ingest import ingest_bytes
from rag.storage import RagStorage
from rag.verification_meta import MAX_CORRECTION_NOTE_LEN
from tests.conftest import AUTH_HEADERS, patched_embeddings
from tests.fixtures.pdf_factory import build_image_rich_pdf
from tests.fixtures.pdf_figure_factory import build_figure_caption_pdf
from tests.test_mcp_search import _call_tool, _initialize

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

COLLECTION = "client_domain_knowledge"
MCP_HEADERS = {
    **AUTH_HEADERS,
    "Accept": "application/json, text/event-stream",
    "Host": "localhost",
}
MISSING_ID = "00000000-0000-0000-0000-000000000000"


def _scroll_doc(qdrant_client: QdrantClient, doc_id: str) -> list[Any]:
    """Return knowledge points for ``doc_id``."""
    points, _offset = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=64,
        with_payload=True,
        with_vectors=True,
    )
    return list(points)


def _item() -> dict[str, Any]:
    """Return a valid verification[] row."""
    return {
        "agent": "TesterVerifier",
        "check": "gps",
        "tool": "qgate.py",
        "date": "2026-11-12T12:00:00+01:00",
        "result": "potvrzeno-wiki",
        "source_url": "https://example.test/wiki",
        "quote": "Souřadnice u náměstí.",
        "independent": False,
        "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    }


def _upload(client: TestClient, name: str, body: bytes) -> str:
    """Upload bytes and return the new ``doc_id``."""
    response = client.post(
        "/upload",
        files={"file": (name, body, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["doc_id"])


def _search_text(
    client: TestClient,
    query: str,
    *,
    include_superseded: bool = False,
    limit: int = 8,
) -> str:
    """Call MCP search_domain_knowledge and return joined text."""
    _initialize(client)
    return _call_tool(
        client,
        "search_domain_knowledge",
        {
            "query": query,
            "limit": limit,
            "include_superseded": include_superseded,
        },
    )


@pytest.fixture()
def client(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> Iterator[TestClient]:
    """Authenticated TestClient with temp storage."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app, headers=AUTH_HEADERS) as test_client:
            yield test_client


def test_s1_invalid_replacement_ids_are_4xx(client: TestClient) -> None:
    """S1: missing / non-UUID / empty replacement ids fail and leave metadata."""
    old_id = _upload(client, "old.txt", b"Vadin pazderna 1923 original fact.")
    before = client.get(f"/documents/{old_id}").json()
    missing = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": MISSING_ID},
    )
    assert missing.status_code == 404
    assert "new_doc_id not found" in missing.json()["detail"]
    invalid = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": "not-a-uuid"},
    )
    assert invalid.status_code == 400
    empty = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": "   "},
    )
    assert empty.status_code in {400, 422}
    patched = client.patch(
        f"/documents/{old_id}/metadata",
        json={"superseded_by": MISSING_ID},
    )
    assert patched.status_code == 400
    assert "POST /documents/{doc_id}/supersede" in patched.json()["detail"]
    after = client.get(f"/documents/{old_id}").json()
    assert after.get("superseded_by") is None
    assert after.get("doc_status") is None
    assert after["sha256"] == before["sha256"]


def test_s2_self_reference_is_400(client: TestClient) -> None:
    """S2: a document may not supersede itself via POST or PATCH."""
    doc_id = _upload(client, "self.txt", b"Self-link must be rejected.")
    response = client.post(
        f"/documents/{doc_id}/supersede",
        json={"new_doc_id": doc_id},
    )
    assert response.status_code == 400
    assert "itself" in response.json()["detail"]
    patched = client.patch(
        f"/documents/{doc_id}/metadata",
        json={"supersedes": [doc_id]},
    )
    assert patched.status_code == 400
    fetched = client.get(f"/documents/{doc_id}").json()
    assert "superseded_by" not in fetched or fetched.get("superseded_by") in (None, "")


def test_s3_cycle_is_400(client: TestClient) -> None:
    """S3: A→B then B→A (and longer chains) are rejected."""
    doc_a = _upload(client, "a.txt", b"Document A fact about Pazderna 1923.")
    doc_b = _upload(client, "b.txt", b"Document B fact about Pazderna 1920.")
    doc_c = _upload(client, "c.txt", b"Document C fact about Lom 1921.")
    first = client.post(
        f"/documents/{doc_a}/supersede",
        json={"new_doc_id": doc_b, "correction_note": "year was 1920"},
    )
    assert first.status_code == 200
    cycle = client.post(
        f"/documents/{doc_b}/supersede",
        json={"new_doc_id": doc_a},
    )
    assert cycle.status_code == 400
    assert "cycle" in cycle.json()["detail"]
    second = client.post(
        f"/documents/{doc_b}/supersede",
        json={"new_doc_id": doc_c},
    )
    assert second.status_code == 200
    long_cycle = client.post(
        f"/documents/{doc_c}/supersede",
        json={"new_doc_id": doc_a},
    )
    assert long_cycle.status_code == 400
    assert "cycle" in long_cycle.json()["detail"]
    b_meta = client.get(f"/documents/{doc_b}").json()
    assert b_meta["superseded_by"] == doc_c
    assert doc_a in b_meta["supersedes"]


def test_s4_atomic_supersede_writes_both_sides(client: TestClient) -> None:
    """S4: A.superseded_by=B always pairs with B.supersedes containing A."""
    doc_a = _upload(client, "old-fact.txt", b"UniqueOldPhrase 1923.")
    doc_b = _upload(client, "new-fact.txt", b"UniqueNewPhrase 1920.")
    doc_d = _upload(client, "other-old.txt", b"SecondOldPhrase also replaced.")
    response = client.post(
        f"/documents/{doc_a}/supersede",
        json={"new_doc_id": doc_b, "correction_note": "corrected to 1920"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["doc_status"] == "superseded"
    assert body["superseded_by"] == doc_b
    assert doc_a in body["supersedes"]
    old = client.get(f"/documents/{doc_a}").json()
    new = client.get(f"/documents/{doc_b}").json()
    assert old["doc_status"] == "superseded"
    assert old["superseded_by"] == doc_b
    assert old["correction_note"] == "corrected to 1920"
    assert old["status"] == "ready"
    assert new["supersedes"] == [doc_a]
    assert new.get("doc_status") in (None, "active")
    second = client.post(
        f"/documents/{doc_d}/supersede",
        json={"new_doc_id": doc_b},
    )
    assert second.status_code == 200
    new_after = client.get(f"/documents/{doc_b}").json()
    assert new_after["supersedes"] == [doc_a, doc_d]


def test_s5_doc_status_enum_and_invariants(client: TestClient) -> None:
    """S5: enum plus superseded-without-link and retracted-without-note."""
    doc_id = _upload(client, "status.txt", b"Lifecycle enum checks.")
    bad = client.patch(f"/documents/{doc_id}/metadata", json={"doc_status": "archived"})
    assert bad.status_code == 400
    alias = client.patch(f"/documents/{doc_id}/metadata", json={"status": "superseded"})
    assert alias.status_code == 400
    missing_link = client.patch(
        f"/documents/{doc_id}/metadata", json={"doc_status": "superseded"}
    )
    assert missing_link.status_code == 400
    assert "superseded_by" in missing_link.json()["detail"]
    missing_note = client.patch(
        f"/documents/{doc_id}/metadata", json={"doc_status": "retracted"}
    )
    assert missing_note.status_code == 400
    assert "correction_note" in missing_note.json()["detail"]
    retracted = client.patch(
        f"/documents/{doc_id}/metadata",
        json={"doc_status": "retracted", "correction_note": "withdrawn"},
    )
    assert retracted.status_code == 200
    fetched = client.get(f"/documents/{doc_id}").json()
    assert fetched["doc_status"] == "retracted"
    assert fetched["correction_note"] == "withdrawn"


def test_s6_search_and_listings_hide_superseded(
    client: TestClient,
) -> None:
    """S6: search and list_image_records omit superseded unless opted in."""
    old_id = _upload(
        client,
        "old-search.txt",
        b"SharedSearchPhrase UniqueOldYear1923.",
    )
    new_id = _upload(
        client,
        "new-search.txt",
        b"SharedSearchPhrase UniqueNewYear1920.",
    )

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
        """Return deterministic vectors for figure-index embedding."""
        return [[0.01] * EMBEDDING_DIM for _ in texts]

    with (
        patch("rag.figure_index.get_ingest_embeddings", side_effect=fake_many),
        patch(
            "rag.ingest.ocr_image_file",
            return_value=("indexed ocr phrase long enough", 90.0),
        ),
        patch(
            "rag.pdf_figures.ocr_pil_image",
            return_value=("figure ocr", "figure ocr", 80.0, 4, "tesseract"),
        ),
    ):
        pdf_old = client.post(
            "/upload",
            files={
                "file": (
                    "old-fig.pdf",
                    build_figure_caption_pdf(),
                    "application/pdf",
                )
            },
        )
        pdf_new = client.post(
            "/upload",
            files={
                "file": (
                    "new-fig.pdf",
                    build_image_rich_pdf(),
                    "application/pdf",
                )
            },
        )
    assert pdf_old.status_code == 200
    assert pdf_new.status_code == 200
    pdf_old_id = str(pdf_old.json()["doc_id"])
    pdf_new_id = str(pdf_new.json()["doc_id"])

    linked = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "1920"},
    )
    assert linked.status_code == 200
    fig_linked = client.post(
        f"/documents/{pdf_old_id}/supersede",
        json={"new_doc_id": pdf_new_id},
    )
    assert fig_linked.status_code == 200

    hidden = _search_text(client, "SharedSearchPhrase")
    assert "UniqueOldYear1923" not in hidden
    assert old_id not in hidden
    assert "UniqueNewYear1920" in hidden or new_id in hidden
    shown = _search_text(client, "SharedSearchPhrase", include_superseded=True)
    assert old_id in shown
    assert new_id in shown

    _initialize(client)
    default_records = json.loads(
        _call_tool(
            client,
            "list_image_records",
            {"doc_id": pdf_old_id, "limit": 20},
            call_id=3,
        )
    )
    assert default_records["returned"] == 0
    included = json.loads(
        _call_tool(
            client,
            "list_image_records",
            {
                "doc_id": pdf_old_id,
                "limit": 20,
                "include_superseded": True,
            },
            call_id=4,
        )
    )
    assert included["returned"] >= 1
    new_records = json.loads(
        _call_tool(
            client,
            "list_image_records",
            {"doc_id": pdf_new_id, "limit": 20},
            call_id=5,
        )
    )
    assert new_records["returned"] >= 0


def test_s6_search_images_hides_superseded(
    client: TestClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """S6: search_images default excludes superseded pdf_image/figure hits."""
    from rag.image_embed import IMAGE_EMBEDDING_DIM

    image_vector = [0.05] * IMAGE_EMBEDDING_DIM

    def _image_paths(paths: list[Any], **_kwargs: object) -> list[list[float]]:
        return [image_vector for _ in paths]

    def fake_many(texts: list[str], **_kwargs: object) -> list[list[float]]:
        """Return deterministic vectors for figure-index embedding."""
        return [[0.01] * EMBEDDING_DIM for _ in texts]

    with (
        patch("rag.mcp_tools.image_embedding_enabled", return_value=True),
        patch("rag.mcp_tools.embed_image_query", return_value=image_vector),
        patch("rag.ingest.image_embedding_enabled", return_value=True),
        patch("rag.ingest.embed_image_paths", side_effect=_image_paths),
        patch("rag.figure_index.get_ingest_embeddings", side_effect=fake_many),
        patch("rag.figure_index.embed_image_paths", side_effect=_image_paths),
        patch(
            "rag.ingest.ocr_image_file",
            return_value=("indexed ocr phrase long enough", 90.0),
        ),
        patch(
            "rag.pdf_figures.ocr_pil_image",
            return_value=("figure ocr", "figure ocr", 80.0, 4, "tesseract"),
        ),
    ):
        old_pdf = client.post(
            "/upload",
            files={
                "file": (
                    "img-old.pdf",
                    build_image_rich_pdf(),
                    "application/pdf",
                )
            },
        )
        new_pdf = client.post(
            "/upload",
            files={
                "file": (
                    "img-new.pdf",
                    build_image_rich_pdf(),
                    "application/pdf",
                )
            },
        )
        assert old_pdf.status_code == 200
        assert new_pdf.status_code == 200
        old_id = str(old_pdf.json()["doc_id"])
        new_id = str(new_pdf.json()["doc_id"])
        linked = client.post(
            f"/documents/{old_id}/supersede",
            json={"new_doc_id": new_id},
        )
        assert linked.status_code == 200
        _initialize(client)
        hidden = _call_tool(
            client,
            "search_images",
            {"query": "figure", "limit": 8, "format": "json"},
            call_id=3,
        )
        assert "Error executing search_images" not in hidden
        hidden_hits = json.loads(hidden) if hidden.strip().startswith("[") else []
        hidden_ids = {hit.get("doc_id") for hit in hidden_hits}
        assert old_id not in hidden_ids
        shown = _call_tool(
            client,
            "search_images",
            {
                "query": "figure",
                "limit": 8,
                "format": "json",
                "include_superseded": True,
            },
            call_id=4,
        )
        assert "Error executing search_images" not in shown
        shown_hits = json.loads(shown) if shown.strip().startswith("[") else []
        shown_ids = {hit.get("doc_id") for hit in shown_hits}
        assert old_id in shown_ids


def test_s7_direct_get_and_text_still_work(client: TestClient) -> None:
    """S7: GET by id and get_document_text keep working after supersede."""
    old_id = _upload(client, "keep.txt", b"Direct read of superseded still works.")
    new_id = _upload(client, "keep-new.txt", b"Replacement document text.")
    client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "see replacement"},
    )
    fetched = client.get(f"/documents/{old_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["doc_status"] == "superseded"
    assert body["superseded_by"] == new_id
    assert body["correction_note"] == "see replacement"
    text = client.get(f"/documents/{old_id}/text")
    assert text.status_code == 200
    assert "superseded still works" in text.json()["text"]
    _initialize(client)
    mcp_text = _call_tool(client, "get_document_text", {"doc_id": old_id}, call_id=3)
    assert "superseded still works" in mcp_text


def test_s8_unlink_restores_search_and_null_is_targeted(
    client: TestClient,
) -> None:
    """S8: unlink clears link fields only; PATCH null of the link is 4xx."""
    old_id = _upload(client, "del.txt", b"SharedUnlinkPhrase UniqueGone1923.")
    new_id = _upload(client, "del-new.txt", b"SharedUnlinkPhrase UniqueKept1920.")
    seeded = client.patch(
        f"/documents/{old_id}/metadata",
        json={
            "reliability": 2,
            "source_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "verification": [_item()],
        },
    )
    assert seeded.status_code == 200
    client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "fix year"},
    )
    reject_null = client.patch(
        f"/documents/{old_id}/metadata",
        json={"superseded_by": None, "doc_status": None},
    )
    assert reject_null.status_code == 400
    unlinked = client.post(f"/documents/{old_id}/unsupersede")
    assert unlinked.status_code == 200
    fetched = client.get(f"/documents/{old_id}").json()
    assert "superseded_by" not in fetched
    assert "doc_status" not in fetched
    assert fetched["correction_note"] == "fix year"
    assert fetched["reliability"] == 2
    assert fetched["verification"][0]["check"] == "gps"
    assert fetched["source_key"] == "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    peer = client.get(f"/documents/{new_id}").json()
    assert "supersedes" not in peer
    restored = _search_text(client, "SharedUnlinkPhrase")
    assert old_id in restored
    note_cleared = client.patch(
        f"/documents/{old_id}/metadata",
        json={"correction_note": None},
    )
    assert note_cleared.status_code == 200
    after_note = client.get(f"/documents/{old_id}").json()
    assert "correction_note" not in after_note
    assert after_note["reliability"] == 2
    assert after_note["verification"][0]["quote"] == "Souřadnice u náměstí."


def test_s9_vectors_and_doc_id_unchanged(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """S9: supersede does not re-embed, change doc_id, or add chunks."""
    old = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="vec-old.txt",
        data=b"Vector stability old.",
        embed_fn=fake_embed,
    )
    new = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="vec-new.txt",
        data=b"Vector stability new.",
        embed_fn=fake_embed,
    )
    old_id = str(old["doc_id"])
    new_id = str(new["doc_id"])
    before = _scroll_doc(qdrant_client, old_id)
    before_vectors = {point.id: point.vector for point in before}
    before_count = len(before)
    result = supersede_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        old_doc_id=old_id,
        new_doc_id=new_id,
        correction_note="no re-embed",
    )
    assert result["old_doc_id"] == old_id
    after = _scroll_doc(qdrant_client, old_id)
    assert len(after) == before_count
    assert {point.id: point.vector for point in after} == before_vectors
    payload = after[0].payload or {}
    assert payload["doc_id"] == old_id
    assert payload["doc_status"] == "superseded"
    assert payload["superseded_by"] == new_id
    assert payload["text"] == (before[0].payload or {}).get("text")


def test_s10_supersede_is_idempotent(client: TestClient) -> None:
    """S10: repeating the same supersede does not duplicate supersedes."""
    old_id = _upload(client, "idem-old.txt", b"Idempotent old fact.")
    new_id = _upload(client, "idem-new.txt", b"Idempotent new fact.")
    first = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "once"},
    )
    second = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "twice"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    new = client.get(f"/documents/{new_id}").json()
    assert new["supersedes"] == [old_id]
    old = client.get(f"/documents/{old_id}").json()
    assert old["correction_note"] == "twice"
    again = client.post(f"/documents/{old_id}/unsupersede")
    assert again.status_code == 200
    noop = client.post(f"/documents/{old_id}/unsupersede")
    assert noop.status_code == 200
    assert client.get(f"/documents/{old_id}").json().get("superseded_by") in (
        None,
        "",
    )


def test_s11_supersede_requires_bearer(
    rag_root: Path,
    accnt_root: Path,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """S11: missing bearer token is 401."""
    with patched_embeddings(fake_embed):
        from rag.app import create_app

        app = create_app(str(rag_root), accnt_data_root=str(accnt_root))
        with TestClient(app) as unauth:
            response = unauth.post(
                f"/documents/{MISSING_ID}/supersede",
                json={"new_doc_id": MISSING_ID},
            )
            assert response.status_code == 401
            unlinked = unauth.post(f"/documents/{MISSING_ID}/unsupersede")
            assert unlinked.status_code == 401


def test_s12_audit_log_is_out_of_band(client: TestClient, rag_root: Path) -> None:
    """S12: rag-changes.jsonl is a RAGloader convention, not written here."""
    old_id = _upload(client, "audit-old.txt", b"Audit stays with RAGloader.")
    new_id = _upload(client, "audit-new.txt", b"Replacement for audit check.")
    client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "loader logs this"},
    )
    assert not (rag_root / "rag-changes.jsonl").exists()


def test_patch_rejects_link_writes_pointing_to_endpoint(
    client: TestClient,
) -> None:
    """PATCH /metadata must not set or null supersession links."""
    doc_id = _upload(client, "patch-link.txt", b"PATCH cannot write links.")
    other = _upload(client, "patch-other.txt", b"Peer document.")
    for body in (
        {"superseded_by": other},
        {"supersedes": [other]},
        {"superseded_by": None},
        {"supersedes": None},
    ):
        response = client.patch(f"/documents/{doc_id}/metadata", json=body)
        assert response.status_code == 400, body
        assert "supersede" in response.json()["detail"]


def test_correction_note_truncation_reported(client: TestClient) -> None:
    """Long correction_note is truncated and listed in truncated[]."""
    old_id = _upload(client, "trunc-old.txt", b"Note truncation old.")
    new_id = _upload(client, "trunc-new.txt", b"Note truncation new.")
    long_note = "x" * (MAX_CORRECTION_NOTE_LEN + 20)
    response = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": long_note},
    )
    assert response.status_code == 200
    assert response.json()["truncated"] == ["correction_note"]
    fetched = client.get(f"/documents/{old_id}").json()
    assert fetched["correction_note"] == "x" * MAX_CORRECTION_NOTE_LEN


def test_cannot_overwrite_existing_supersession_silently(
    client: TestClient,
) -> None:
    """A second replacement must unlink first; other docs are not rewritten."""
    old_id = _upload(client, "ow-old.txt", b"Already superseded old.")
    first = _upload(client, "ow-first.txt", b"First replacement.")
    second = _upload(client, "ow-second.txt", b"Second replacement.")
    client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": first},
    )
    refused = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": second},
    )
    assert refused.status_code == 400
    assert first in refused.json()["detail"]
    assert client.get(f"/documents/{first}").json()["supersedes"] == [old_id]
    assert "supersedes" not in client.get(f"/documents/{second}").json()


def test_figure_points_receive_lifecycle_fields(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """Supersede writes doc_status onto pdf_image/pdf_figure points."""
    pdf_bytes = build_image_rich_pdf()
    replacement = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="fig-new.txt",
        data=b"Replacement for figure source.",
        embed_fn=fake_embed,
    )
    with patch(
        "rag.ingest.ocr_image_file",
        return_value=("indexed ocr phrase long enough", 90.0),
    ):
        ingested = ingest_bytes(
            storage=rag_storage,
            qdrant=qdrant_client,
            collection_name=COLLECTION,
            filename="fig-old.pdf",
            data=pdf_bytes,
            embed_fn=fake_embed,
        )
    old_id = str(ingested["doc_id"])
    new_id = str(replacement["doc_id"])
    before_vectors = {
        point.id: point.vector for point in _scroll_doc(qdrant_client, old_id)
    }
    supersede_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        old_doc_id=old_id,
        new_doc_id=new_id,
        correction_note="figure source replaced",
    )
    after = _scroll_doc(qdrant_client, old_id)
    assert {point.id: point.vector for point in after} == before_vectors
    figure_like = [
        point.payload or {}
        for point in after
        if (point.payload or {}).get("source_type") in {"pdf_image", "pdf_figure"}
    ]
    assert figure_like
    for payload in figure_like:
        assert payload["doc_status"] == "superseded"
        assert payload["superseded_by"] == new_id
        assert payload["correction_note"] == "figure source replaced"
        assert payload["doc_id"] == old_id
    unsupersede_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        old_doc_id=old_id,
    )
    cleared = _scroll_doc(qdrant_client, old_id)
    for point in cleared:
        payload = point.payload or {}
        assert "doc_status" not in payload
        assert "superseded_by" not in payload
        assert payload.get("correction_note") == "figure source replaced"


def test_mcp_supersede_and_unsupersede_tools(client: TestClient) -> None:
    """MCP tools should set and clear both sides of the link."""
    old_id = _upload(client, "mcp-old.txt", b"MCP supersede old.")
    new_id = _upload(client, "mcp-new.txt", b"MCP supersede new.")
    _initialize(client)
    linked = _call_tool(
        client,
        "supersede_document",
        {
            "old_doc_id": old_id,
            "new_doc_id": new_id,
            "correction_note": "mcp",
        },
        call_id=3,
    )
    assert "superseded" in linked
    assert client.get(f"/documents/{old_id}").json()["superseded_by"] == new_id
    cleared = _call_tool(
        client,
        "unsupersede_document",
        {"old_doc_id": old_id},
        call_id=4,
    )
    assert "unlinked" in cleared
    assert "superseded_by" not in client.get(f"/documents/{old_id}").json()


def test_retracted_is_hidden_from_search(client: TestClient) -> None:
    """Retracted documents are omitted from search unless include_superseded."""
    doc_id = _upload(client, "ret.txt", b"RetractedSearchPhrase unique body.")
    other = _upload(client, "ret-keep.txt", b"RetractedSearchPhrase keeper body.")
    patched = client.patch(
        f"/documents/{doc_id}/metadata",
        json={"doc_status": "retracted", "correction_note": "withdrawn"},
    )
    assert patched.status_code == 200
    hidden = _search_text(client, "RetractedSearchPhrase")
    assert doc_id not in hidden
    assert other in hidden
    shown = _search_text(client, "RetractedSearchPhrase", include_superseded=True)
    assert doc_id in shown


def test_openapi_supersede_routes(client: TestClient) -> None:
    """OpenAPI should advertise the atomic supersede and unlink endpoints."""
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    payload = spec.json()
    paths = payload["paths"]
    assert "/documents/{doc_id}/supersede" in paths
    assert "/documents/{doc_id}/unsupersede" in paths
    assert "post" in paths["/documents/{doc_id}/supersede"]
    assert "post" in paths["/documents/{doc_id}/unsupersede"]
    schemas = payload["components"]["schemas"]
    assert "SupersedeDocumentRequest" in schemas
    assert "SupersedeDocumentResponse" in schemas
    assert "UnsupersedeDocumentResponse" in schemas


def test_supersede_rolls_back_when_second_write_fails(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """If the replacement write fails, the old document is restored."""
    from rag import document_supersede as supersede_mod

    old = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="rb-old.txt",
        data=b"Rollback old fact.",
        embed_fn=fake_embed,
    )
    new = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="rb-new.txt",
        data=b"Rollback new fact.",
        embed_fn=fake_embed,
    )
    old_id = str(old["doc_id"])
    new_id = str(new["doc_id"])
    real = supersede_mod._apply_doc_fields
    calls = {"n": 0}

    def wrapper(**kwargs: Any) -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated second write failure")
        return real(**kwargs)

    with patch.object(supersede_mod, "_apply_doc_fields", side_effect=wrapper):
        with pytest.raises(RuntimeError, match="simulated second write"):
            supersede_document(
                storage=rag_storage,
                qdrant=qdrant_client,
                collection_name=COLLECTION,
                old_doc_id=old_id,
                new_doc_id=new_id,
                correction_note="should not stick",
            )
    old_meta = rag_storage.read_meta(old_id)
    new_meta = rag_storage.read_meta(new_id)
    assert old_meta is not None
    assert new_meta is not None
    assert "superseded_by" not in old_meta.extra
    assert "doc_status" not in old_meta.extra
    assert "correction_note" not in old_meta.extra
    assert "supersedes" not in new_meta.extra
    for point in _scroll_doc(qdrant_client, old_id):
        payload = point.payload or {}
        assert "superseded_by" not in payload
        assert "doc_status" not in payload
    assert calls["n"] >= 3


def test_unsupersede_rolls_back_when_second_write_fails(
    rag_storage: RagStorage,
    qdrant_client: QdrantClient,
    fake_embed: Callable[[str], list[float]],
) -> None:
    """If the old-doc unlink write fails, the replacement supersedes list is restored."""
    from rag import document_supersede as supersede_mod

    old = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="urb-old.txt",
        data=b"Unlink rollback old.",
        embed_fn=fake_embed,
    )
    new = ingest_bytes(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        filename="urb-new.txt",
        data=b"Unlink rollback new.",
        embed_fn=fake_embed,
    )
    old_id = str(old["doc_id"])
    new_id = str(new["doc_id"])
    supersede_document(
        storage=rag_storage,
        qdrant=qdrant_client,
        collection_name=COLLECTION,
        old_doc_id=old_id,
        new_doc_id=new_id,
    )
    real = supersede_mod._apply_doc_fields
    calls = {"n": 0}

    def wrapper(**kwargs: Any) -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated unlink second write failure")
        return real(**kwargs)

    with patch.object(supersede_mod, "_apply_doc_fields", side_effect=wrapper):
        with pytest.raises(RuntimeError, match="simulated unlink second write"):
            unsupersede_document(
                storage=rag_storage,
                qdrant=qdrant_client,
                collection_name=COLLECTION,
                old_doc_id=old_id,
            )
    old_meta = rag_storage.read_meta(old_id)
    new_meta = rag_storage.read_meta(new_id)
    assert old_meta is not None
    assert new_meta is not None
    assert old_meta.extra.get("superseded_by") == new_id
    assert old_meta.extra.get("doc_status") == "superseded"
    assert new_meta.extra.get("supersedes") == [old_id]


def test_hidden_replacement_is_400(client: TestClient) -> None:
    """A superseded or retracted replacement cannot be the target of a new link."""
    old_id = _upload(client, "hid-old.txt", b"Hidden replacement old.")
    new_id = _upload(client, "hid-new.txt", b"Hidden replacement new.")
    third_id = _upload(client, "hid-third.txt", b"Hidden replacement third.")
    retracted = client.patch(
        f"/documents/{new_id}/metadata",
        json={"doc_status": "retracted", "correction_note": "withdrawn"},
    )
    assert retracted.status_code == 200
    onto_retracted = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id},
    )
    assert onto_retracted.status_code == 400
    assert "active replacement" in onto_retracted.json()["detail"]
    client.patch(
        f"/documents/{new_id}/metadata",
        json={"doc_status": None, "correction_note": None},
    )
    linked = client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id},
    )
    assert linked.status_code == 200
    onto_superseded = client.post(
        f"/documents/{third_id}/supersede",
        json={"new_doc_id": old_id},
    )
    assert onto_superseded.status_code == 400
    assert "active replacement" in onto_superseded.json()["detail"]


def test_patch_cannot_clear_doc_status_while_linked(client: TestClient) -> None:
    """PATCH null/active doc_status while superseded_by is set must use unsupersede."""
    old_id = _upload(client, "clr-old.txt", b"Clear status old.")
    new_id = _upload(client, "clr-new.txt", b"Clear status new.")
    client.post(f"/documents/{old_id}/supersede", json={"new_doc_id": new_id})
    for body in ({"doc_status": None}, {"doc_status": "active"}):
        response = client.patch(f"/documents/{old_id}/metadata", json=body)
        assert response.status_code == 400, body
        assert "unsupersede" in response.json()["detail"]
    fetched = client.get(f"/documents/{old_id}").json()
    assert fetched["doc_status"] == "superseded"
    assert fetched["superseded_by"] == new_id


def test_unsupersede_keeps_retracted_doc_status(client: TestClient) -> None:
    """Unlink deletes doc_status only when it is superseded."""
    old_id = _upload(client, "ret-old.txt", b"Keep retracted old.")
    new_id = _upload(client, "ret-new.txt", b"Keep retracted new.")
    client.post(
        f"/documents/{old_id}/supersede",
        json={"new_doc_id": new_id, "correction_note": "later withdrawn"},
    )
    patched = client.patch(
        f"/documents/{old_id}/metadata",
        json={"doc_status": "retracted"},
    )
    assert patched.status_code == 200
    unlinked = client.post(f"/documents/{old_id}/unsupersede")
    assert unlinked.status_code == 200
    fetched = client.get(f"/documents/{old_id}").json()
    assert "superseded_by" not in fetched
    assert fetched["doc_status"] == "retracted"
    assert fetched["correction_note"] == "later withdrawn"
    peer = client.get(f"/documents/{new_id}").json()
    assert "supersedes" not in peer


def test_delete_linked_document_returns_409(client: TestClient) -> None:
    """DELETE of a linked document lists peer ids and leaves both docs."""
    old_id = _upload(client, "del-link-old.txt", b"Linked delete old.")
    new_id = _upload(client, "del-link-new.txt", b"Linked delete new.")
    client.post(f"/documents/{old_id}/supersede", json={"new_doc_id": new_id})
    blocked_old = client.delete(f"/documents/{old_id}")
    assert blocked_old.status_code == 409
    assert new_id in blocked_old.json()["detail"]["linked_ids"]
    blocked_new = client.delete(f"/documents/{new_id}")
    assert blocked_new.status_code == 409
    assert old_id in blocked_new.json()["detail"]["linked_ids"]
    assert client.get(f"/documents/{old_id}").status_code == 200
    assert client.get(f"/documents/{new_id}").status_code == 200
    client.post(f"/documents/{old_id}/unsupersede")
    assert client.delete(f"/documents/{old_id}").status_code == 200
    assert client.delete(f"/documents/{new_id}").status_code == 200
