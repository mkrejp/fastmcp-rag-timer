# Architecture

## Purpose and scope

This document describes **fastmcp-rag-timer** as implemented in this repository.

The service is a single-process Python application that:

- Accepts PDF, TXT, Markdown, and JSON/GeoJSON over HTTP (FastAPI), remote GeoJSON by URL via bearer-authenticated `POST /ingest-json`, and optional plain-text ingest over MCP.
- Persists originals and processed fulltext/chunks under a data root (default `/data/rag`).
- Embeds chunks with Novita `baai/bge-m3` by default (`EMBEDDING_PROVIDER` unset/`novita`), or locally with FastEmbed `BAAI/bge-m3` when `EMBEDDING_PROVIDER=fastembed`, and indexes them in an embedded Qdrant collection.
- Exposes catalog/text HTTP APIs and Streamable HTTP MCP tools (including similarity search) from the same uvicorn process.

**Sources of truth:** `main.py`, `rag/` (`app.py`, `config.py`, `embeddings.py`, `ingest.py`, `document_delete.py`, `mcp_tools.py`, `processing.py`, `routes.py`, `storage.py`, `vectorstore.py`), `Dockerfile`, `README.md`, `docs/embedding-provider.md`, and `tests/`.
`pyproject.toml` lists runtime dependencies (including `fastembed`).
There is no `requirements.txt`, lockfile, compose file, or platform config (for example `railway.toml`).
CI lives under `.github/workflows/`.

Statements below are taken from those files unless labeled **Assumption**.

Embedding request details (endpoint, auth, timeouts, privacy, cost) live in [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity) and [Embedding provider](./embedding-provider.md).

## System context

```mermaid
flowchart LR
  httpClient["HTTP client<br/>upload, catalog, health"]
  mcpClient["MCP client / agent<br/>Streamable HTTP /mcp"]
  this["Lightweight RAG Engine<br/>FastAPI + FastMCP + embedded Qdrant"]
  novita["api.novita.ai<br/>POST /openai/v1/embeddings"]
  disk[("/data/rag<br/>originals, processed, qdrant, index.json")]

  httpClient -->|"HTTP :8080"| this
  mcpClient -->|"HTTP :8080 /mcp"| this
  this -->|"HTTPS Bearer BGE_M3_NOVITA_API_KEY or fallbacks"| novita
  this -->|"read/write files and vector index"| disk
```

**Assumption:** An HTTPS edge (Railway or similar reverse proxy) sits in front of port 8080 in production. Nothing in the repository terminates TLS. README says to mount a Railway volume at `/data`; there is no Railway config file in the repository.

## Component and module overview

```text
main.py                 # uvicorn entrypoint; module-level app = create_app()
rag/
  app.py                # FastAPI factory, FastMCP mount, shared storage + Qdrant
  routes.py             # HTTP API
  mcp_tools.py          # FastMCP tools
  storage.py            # originals + processed + index.json
  processing.py         # PDF/text/GeoJSON extract + chunk
  geojson.py            # generic GeoJSON → RAG text
  remote.py             # fetch remote JSON (GitHub blob → raw)
  auth.py               # inbound bearer token for /ingest-json
  embeddings.py         # Novita HTTP or local FastEmbed BGE-M3
  ingest.py             # persist + embed + upsert pipeline
  document_delete.py    # remove one document from disk + Qdrant
  vectorstore.py        # local Qdrant open/create
  config.py             # constants and env helpers
```

| Component | Where | Role |
| --- | --- | --- |
| FastAPI app `Lightweight RAG Engine` | `rag.app.create_app()` | HTTP API (see [HTTP API](#http-api)). FastAPI also exposes default `/docs`, `/redoc`, `/openapi.json` (not disabled). |
| FastMCP server `Client RAG Node` | `rag.app.create_app()` | `stateless_http=True`, `streamable_http_path="/"`, mounted at `/mcp` with the MCP app lifespan entered from FastAPI. |
| `get_embedding()` / `get_embeddings()` | `rag/embeddings.py` | Default: `POST https://api.novita.ai/openai/v1/embeddings` with model `baai/bge-m3`. `EMBEDDING_PROVIDER=fastembed`: local ONNX `BAAI/bge-m3`, lazy-loaded. |
| `RagStorage` | `rag/storage.py` | Filesystem catalog under `RAG_DATA_ROOT`. |
| Embedded Qdrant | `rag/vectorstore.py` | `QdrantClient(path=storage.qdrant_dir)`, collection `client_domain_knowledge`, cosine, size from `embedding_dim()` (1024 for both providers). |
| Uvicorn | `main.py` `__main__` and Docker `CMD` | `host=HOST` (`0.0.0.0`), `port=PORT` (`8080`). |
| Tests | `tests/` | pytest; `RAG_SKIP_DEFAULT_APP=1` so `main.py` does not lock the production Qdrant path. |
| Container image | `Dockerfile` | Python 3.11 slim, `uv` installer, `mcp[cli]<2`, `EXPOSE 8080`, `ENV RAG_DATA_ROOT=/data/rag`. No `VOLUME` instruction. |

```mermaid
flowchart TB
  subgraph process["OS process: python main.py"]
    uvicorn["uvicorn 0.0.0.0:8080"]
    fastapi["FastAPI: Lightweight RAG Engine"]
    mcp["FastMCP: Client RAG Node<br/>mounted at /mcp"]
    routes["rag.routes"]
    tools["rag.mcp_tools"]
    ingest["rag.ingest"]
    deleteDoc["rag.document_delete"]
    proc["rag.processing"]
    embed["rag.embeddings.get_embedding"]
    storage["rag.storage.RagStorage"]
    qdrant["QdrantClient(path=.../qdrant)"]
    uvicorn --> fastapi
    fastapi --> routes
    fastapi --> mcp
    mcp --> tools
    routes --> ingest
    routes --> deleteDoc
    deleteDoc --> storage
    deleteDoc --> qdrant
    tools --> ingest
    tools --> storage
    tools --> embed
    tools --> qdrant
    ingest --> proc
    ingest --> storage
    ingest --> embed
    ingest --> qdrant
  end
  disk[("/data/rag")]
  novita["https://api.novita.ai/openai/v1/embeddings"]
  storage --> disk
  qdrant --> disk
  embed -->|"HTTPS POST + Bearer"| novita
```

`mcp` and FastAPI share one process, one `RagStorage`, and one Qdrant client created in `create_app()`. There is no separate database container. Qdrant is opened once at app construction so uvicorn does not take the local-path lock twice (`main.py` comment).

## HTTP API

From `rag/routes.py`:

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok", "rag_data_root": ...}` |
| `POST` | `/upload` | PDF, TXT, Markdown, JSON, or GeoJSON → persist + index |
| `POST` | `/upload-pdf` | PDF-only alias |
| `POST` | `/upload-media` | Store bytes under `originals/{uuid}/`; **not** extracted or indexed |
| `POST` | `/ingest-json` | Bearer `RAG_API_TOKEN`; body `{"url": "..."}`; fetch remote GeoJSON/JSON and index feature attributes (coordinates skipped) |
| `GET` | `/documents` | Catalog from `index.json` |
| `GET` | `/documents/{doc_id}` | Metadata |
| `GET` | `/documents/{doc_id}/text` | Processed full text from disk (no re-parse, no embedding call) |
| `DELETE` | `/documents/{doc_id}` | Bearer or Google session (same as `/upload`); remove files (including `images/`), catalog row, and Qdrant points. `200` `{"status":"deleted","doc_id","filename","chunks_removed"}`; `404` if missing; `401`/`503` if unauthorized |
| `GET` | `/images/{doc_id}/{image_id}` | Extracted PDF image bytes (same auth as `/upload`) |

There is **no REST search** endpoint. Similarity search is MCP-only (`search_domain_knowledge`). There is no HTTP reindex route. `rag.document_delete.delete_document` deletes Qdrant points by `payload.doc_id` then calls `RagStorage.remove_document_files`.

## Exposed MCP tools, resources, and prompts

From `rag/mcp_tools.py`, registered in `create_app()` and served at **`/mcp`** (Streamable HTTP). `streamable_http_path="/"` so the MCP endpoint is the mount root `/mcp`, not `/mcp/mcp`.

| Kind | Name | Present? |
| --- | --- | --- |
| Tool | `list_documents() -> str` | Yes |
| Tool | `get_document_text(doc_id: str) -> str` | Yes |
| Tool | `get_document_chunks(doc_id: str) -> str` | Yes |
| Tool | `ingest_text(filename: str, content: str) -> str` | Yes |
| Tool | `search_domain_knowledge(query: str, limit: int = 3) -> str` | Yes |
| Resource | — | None |
| Prompt | — | None |

Tool behavior:

- `list_documents`: JSON of `storage.list_documents()`, or `"No documents stored."`
- `get_document_text` / `get_document_chunks`: disk reads of processed artifacts; missing ID returns a not-found string
- `ingest_text`: if the filename is not a supported suffix, appends `.txt`; calls `ingest_bytes`; returns JSON or `"Error ingesting text: ..."`
- `search_domain_knowledge`: embed query → `search_similar` (`qdrant.query_points`) → format `Source: {source} (doc_id=...)` + `Content:`; empty → `"No relevant facts found."`; exceptions → `"Error executing RAG search: ..."`

No stdio MCP entrypoint (`mcp.run()` is not called). Tests assert that `/mcp` is mounted; they do not exercise a live MCP session handshake.

## RAG pipeline

There is **retrieval + embedding**, not generation. No chat/completions call exists. See [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity).

### Ingestion

Shared path: `rag.ingest.ingest_bytes` (HTTP `/upload`, `/upload-pdf`, `/ingest-json`, and MCP `ingest_text`).

1. Require a filename; reject unsupported types (`ValueError` → HTTP 400 on REST).
2. SHA-256 the bytes. If `skip_duplicate` (default true) and the digest is already in the catalog, return `status: exists` with `reused: true` and **do not** re-embed.
3. Allocate `doc_id = uuid4()`.
4. `save_original`: write under `originals/{doc_id}/{Path(filename).name}` (basename only).
5. `process_document`: extract + chunk (~1000 characters, page-aware for PDFs).
6. Write `processed/{doc_id}/fulltext.txt`, `chunks.json`, `meta.json`; upsert `index.json`.
7. For each chunk, `get_embedding(text)` and `qdrant.upsert` a `PointStruct`.
8. Point id: `uuid5(NAMESPACE_URL, f"{doc_id}:{chunk_index}")`.
9. Payload: `text`, `source` (filename), `doc_id`, `page`, `chunk_index`.

Supported types (`rag/processing.py`): `.pdf`; `.txt`, `.md`, `.markdown`, `.text`; `.json`, `.geojson`. PDFs use `pypdf.PdfReader` over in-memory bytes. Text is decoded as utf-8, utf-8-sig, then latin-1, then utf-8 with replacement. JSON/GeoJSON is turned into per-feature text records via `rag.geojson` (attributes only; coordinate rings are omitted).

Chunking: page segments are concatenated until over `DEFAULT_CHUNK_SIZE` (1000 characters); oversize segments are sliced at 1000 chars. No overlap, no token counting. Each chunk stores the page index of the segment currently being accumulated. Empty extractable text yields zero chunks and no upsert (HTTP still `status: Success`).

`POST /upload-media` assigns a new `doc_id`, saves the original, and returns path + ID. It does not call `ingest_bytes`.

### Embeddings

See [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity).
Collection size follows `embedding_dim()` (1024 for Novita and FastEmbed BGE-M3).
Missing Novita credentials raise `ValueError` naming `BGE_M3_NOVITA_API_KEY`.
FastEmbed needs no API key.
HTTP `_ingest_upload` maps any `ValueError` to **400** (filename/type checks and missing credentials).
Other embedding failures (`HTTPError`, JSON errors, unexpected dimension) become HTTP **500** `"Ingest failed: ..."`.
MCP search and ingest tools catch exceptions and return error strings.

### Vector store

- `QdrantClient(path=str(storage.qdrant_dir))` — embedded on-disk, not a remote Qdrant URL.
- Default directory: `{RAG_DATA_ROOT}/qdrant` → `/data/rag/qdrant` in the image.
- Collection name (hardcoded): `client_domain_knowledge`.
- Created in `open_qdrant()` if missing. If the existing collection’s vector size is not `embedding_dim()`, or `{RAG_DATA_ROOT}/embedding_backend.json` records a different provider/model, that collection is **deleted and recreated** (vectors in it are lost; files on disk are not). See [Embedding provider](./embedding-provider.md).
- In-memory mode (`:memory:`) is not used.

### Retrieval

Only `search_domain_knowledge`: embed query → cosine search with `limit` (default 3) → concatenate payload text. No metadata filters, no rerank, no hybrid/keyword search. Citations are `source` plus `doc_id` when present.

### Timer / scheduling

**None.** There is no scheduler, cron, background task, or interval job. The repository name includes `timer`; that behavior is not implemented.

Dockerfile installs `curl` (comment: healthchecks). `GET /health` exists. There is no Docker `HEALTHCHECK` instruction.

## Runtime and request flow

Startup (`python main.py` or Docker `CMD`):

1. Unless `RAG_SKIP_DEFAULT_APP=1`, `main.py` calls `create_app()`.
2. Resolve `RAG_DATA_ROOT` (default `/data/rag`; Dockerfile sets the same).
3. `RagStorage.ensure_layout()` creates `originals/`, `processed/`, `qdrant/`, and empty `index.json`.
4. Open embedded Qdrant; create or recreate the collection if needed.
5. Build FastMCP, register tools, `mcp.streamable_http_app()`, FastAPI lifespan enters the MCP app’s lifespan.
6. Include HTTP router; `app.mount("/mcp", mcp_app)`.
7. Uvicorn binds `0.0.0.0:8080`.

`uvicorn main:app` uses the module-level `app` so Qdrant is not opened a second time.

### HTTP: document ingest

```mermaid
sequenceDiagram
  actor Client
  participant Uvicorn
  participant Routes as rag.routes
  participant Ingest as ingest_bytes
  participant Storage as RagStorage
  participant Embed as get_embedding
  participant Novita as api.novita.ai
  participant Qdrant

  Client->>Uvicorn: POST /upload (multipart)
  Uvicorn->>Routes: upload_document()
  alt unsupported filename
    Routes-->>Client: 400
  else supported
    Routes->>Ingest: ingest_bytes(filename, data)
    Ingest->>Storage: sha256 lookup
    alt duplicate digest
      Ingest-->>Client: 200 status exists reused true
    else new document
      Ingest->>Storage: save_original + write_processed
      loop each chunk
        Ingest->>Embed: get_embedding(text)
        Embed->>Novita: POST /openai/v1/embeddings
        Novita-->>Embed: data[0].embedding length 1024
        Ingest->>Qdrant: upsert PointStruct
      end
      Ingest-->>Client: 200 status Success doc_id chunks_indexed
    end
  end
```

### Typical MCP call

```mermaid
sequenceDiagram
  actor Agent as MCP client
  participant MCP as FastMCP /mcp
  participant Tool as search_domain_knowledge
  participant Embed as get_embedding
  participant Novita as api.novita.ai
  participant Qdrant

  Agent->>MCP: Streamable HTTP tools/call search_domain_knowledge
  MCP->>Tool: invoke query, limit=3
  Tool->>Embed: get_embedding(query)
  Embed->>Novita: POST /openai/v1/embeddings Bearer token
  Novita-->>Embed: embedding vector
  Tool->>Qdrant: query_points client_domain_knowledge
  Qdrant-->>Tool: points with payload
  alt hits
    Tool-->>Agent: Source/Content blocks joined by ---
  else no hits
    Tool-->>Agent: No relevant facts found.
  else exception
    Tool-->>Agent: Error executing RAG search: ...
  end
```

`list_documents`, `get_document_text`, and `get_document_chunks` do not call the embedding API; they only read disk.

## Configuration and environment variables

| Name | Default / value | Used for |
| --- | --- | --- |
| `RAG_DATA_ROOT` | `/data/rag` (`DEFAULT_RAG_DATA_ROOT`; Dockerfile `ENV`) | Originals, processed artifacts, Qdrant, `index.json` |
| `EMBEDDING_PROVIDER` | unset → `novita` | `novita` (HTTP) or `fastembed` (local ONNX BGE-M3) |
| `FASTEMBED_CACHE_PATH` | `{RAG_DATA_ROOT}/fastembed` | FastEmbed ONNX cache |
| `BGE_M3_NOVITA_API_KEY` | unset | Preferred Bearer token for embeddings (Novita only) |
| `NOVITA_API_KEY` | unset | Embedding credential if `BGE_M3_NOVITA_API_KEY` is empty |
| `OPENAI_TOKEN` | unset | Embedding credential if both Novita-named vars are empty |
| `OPENAI_API_KEY` | unset | Embedding credential checked last |
| `RAG_SKIP_DEFAULT_APP` | unset; tests set `"1"` | Skip module-level `create_app()` in `main.py` |
| `RAG_API_TOKEN` | unset | Required Bearer token for `POST /ingest-json` (503 if unset) |
| `RAG_MAX_REMOTE_JSON_BYTES` | `134217728` (128 MiB) | Max remote JSON download size |
| `RAG_REMOTE_FETCH_TIMEOUT` | `120` | Remote fetch timeout (seconds) |

Not configurable without code changes:

| Item | Value |
| --- | --- |
| HTTP bind | `HOST = "0.0.0.0"`, `PORT = 8080` (not `os.getenv("PORT")`) |
| Collection | `client_domain_knowledge` |
| Vector size / metric | `embedding_dim()` (1024 for both backends) / cosine |
| Embed model | Novita `baai/bge-m3` or FastEmbed `BAAI/bge-m3` |
| Embed URL | Novita URL, or none when `EMBEDDING_PROVIDER=fastembed` |
| Embed timeout | 60 seconds (Novita HTTP only) |
| Chunk size | `DEFAULT_CHUNK_SIZE = 1000` characters |
| MCP server name | `Client RAG Node` |
| FastAPI title | `Lightweight RAG Engine` |
| MCP mount | `/mcp` |

There is no `.env` file, `.env.example`, or secret manager integration in the repository.

**Assumption:** If deployed on Railway, `BGE_M3_NOVITA_API_KEY` is a service variable, a volume is mounted at `/data`, and public HTTPS is mapped to 8080. Railway often injects `PORT`; this app ignores it and always listens on 8080.

## Build and deployment

`Dockerfile`:

- Base: `python:3.11-slim-bookworm`
- Copies `uv` / `uvx` from `ghcr.io/astral-sh/uv:latest`
- `WORKDIR /app`
- `apt-get install curl`
- `uv pip install --system`: `fastapi`, `uvicorn`, `"mcp[cli]<2"` (comment: FastMCP lives in mcp 1.x; 2.x renames it to MCPServer), `pypdf`, `"qdrant-client>=1.12,<2"`, `requests`, `python-multipart`, `itsdangerous`, `fastembed`
- `COPY main.py ./` and `COPY rag ./rag`
- `EXPOSE 8080`
- No `VOLUME` instruction (comment: Railway rejects Dockerfile `VOLUME`; use the service volume mount)
- `ENV RAG_DATA_ROOT=/data/rag`
- `CMD ["python", "main.py"]`

No multi-stage build, no non-root user, no image pin for `uv:latest`. Aside from `mcp[cli]<2`, Python packages are unpinned.

Local run (from README):

```bash
export RAG_DATA_ROOT=./data/rag
export BGE_M3_NOVITA_API_KEY=your-novita-key
python main.py
```

Tests (from README):

```bash
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf qdrant-client requests python-multipart
pytest -q
```

There is no GitHub Actions (or other) CI workflow in the repository.

**Assumption:** Production is this Docker image on Railway with a volume at `/data` and HTTP routed to 8080.

## Security

| Topic | What the code does |
| --- | --- |
| HTTP auth | None on most REST routes and FastAPI docs. `POST /ingest-json` requires `Authorization: Bearer` matching `RAG_API_TOKEN`. `POST /upload`, `POST /upload-pdf`, and `DELETE /documents/{doc_id}` require a bearer (`MCP_API_TOKEN` or `RAG_API_TOKEN`) or a signed-in Google session. |
| MCP auth | None on `/mcp`. |
| Secrets | Embedding key from `BGE_M3_NOVITA_API_KEY`, then `NOVITA_API_KEY`, `OPENAI_TOKEN`, `OPENAI_API_KEY`; sent as `Authorization: Bearer`. Not read from a file or vault. See [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity). |
| TLS | Not implemented in-app. |
| Upload path | Originals use `Path(filename).name` (directory components stripped). |
| Upload size / type | `/upload` allows PDF/TXT/Markdown/JSON/GeoJSON. `/upload-pdf` requires `.pdf`. `/upload-media` accepts any upload. `/ingest-json` caps remote body size via `RAG_MAX_REMOTE_JSON_BYTES` and rejects private/loopback URLs. No multipart size quota in code. |
| Bind address | `0.0.0.0`. |
| Container user | Default root of `python:3.11-slim-bookworm`. |
| Dependency supply chain | Unpinned packages except `mcp[cli]<2`; `uv:latest`. |

Anyone who can reach port 8080 can ingest documents (which triggers paid embedding calls) and read the catalog/fulltext.

## Observability

- No application logger, metrics, tracing, or request IDs in `rag/` or `main.py`.
- Uvicorn default access/error logs only (not configured in code).
- `GET /health` returns liveness + `rag_data_root`; no readiness check of Qdrant or Novita.
- MCP tool failures are returned as strings, not logged.
- Embedding HTTP failures: `raise_for_status()` then JSON parse and dimension check; HTTP ingest → 400 or 500; MCP tools → error string.

## Limitations

Current gaps (not a history of old bugs):

1. **No timer/scheduler** despite the repository name.
2. **No MCP resources or prompts.**
3. **No REST search** — similarity search is MCP-only.
4. **No generator LLM** — retrieval only; no answer synthesis.
5. **Most routes have no inbound auth**; `/ingest-json` uses `RAG_API_TOKEN`, and `/upload`, `/upload-pdf`, and `DELETE /documents/{doc_id}` use bearer or Google session. No upload size quotas on multipart routes.
6. **Hardcoded port 8080** — `PORT` is not read.
7. **`/upload-media`** stores files but does not index them or add them to `index.json`.
8. **No HTTP reindex** — delete a `doc_id` then re-upload to replace an obsolete file.
9. **No CI workflow** and no lockfile; most Docker deps unpinned.
10. **No Docker HEALTHCHECK** despite `curl` and `/health`.
11. Embeddings are **one HTTP call per chunk/query**, with no batch `input` array, retries, or 429 handling.
12. Recreating a Qdrant collection on dimension mismatch **drops existing vectors**.
