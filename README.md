# fastmcp-rag-timer

Combines FastAPI (for web endpoints), FastMCP (to natively expose tools over the Model Context Protocol), and Qdrant Client (local-disk vector engine). It handles PDF and text extraction, stores originals and processed artifacts on disk, and indexes chunks for semantic retrieval.

## Persistent storage

All RAG server data lives under **`/data/rag`** (override with `RAG_DATA_ROOT`):

```text
/data/rag/
  originals/{doc_id}/...
  processed/{doc_id}/meta.json, fulltext.txt, chunks.json
  qdrant/
  index.json

/data/accnt/
  YYYY-MM-DD.json          # daily rollups (embeddings, MCP, HTTP, outbound)
  events/YYYY-MM-DD.jsonl  # per-call audit lines
```

On Railway, mount a volume at `/data`. The app creates `/data/rag` and `/data/accnt` at startup.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_ROOT` | `/data/rag` | Persistent originals, processed text, Qdrant |
| `ACCNT_DATA_ROOT` | `/data/accnt` | Per-day token/comms accounting rollups and event logs |
| `EMBEDDING_PROVIDER` | `novita` (when unset) | `novita` (HTTP API) or `fastembed` (local ONNX BGE-M3) |
| `FASTEMBED_CACHE_PATH` | `{RAG_DATA_ROOT}/fastembed` | ONNX model cache when `EMBEDDING_PROVIDER=fastembed` |
| `BGE_M3_NOVITA_API_KEY` | unset | Preferred Novita API credential for embeddings |
| `NOVITA_API_KEY` | unset | Embedding credential if `BGE_M3_NOVITA_API_KEY` is empty |
| `OPENAI_TOKEN` | unset | Embedding credential if both Novita-named vars are empty |
| `OPENAI_API_KEY` | unset | Embedding credential checked last |
| `RAG_SKIP_DEFAULT_APP` | unset | Set to `1` to skip module-level `create_app()` (used by tests) |
| `RAG_API_TOKEN` | unset | Bearer token required by `POST /ingest-json` |
| `RAG_MAX_REMOTE_JSON_BYTES` | `134217728` (128 MiB) | Remote GeoJSON download cap (`obce.json` is ~95 MiB) |
| `RAG_REMOTE_FETCH_TIMEOUT` | `120` | Remote fetch timeout in seconds |

Embeddings default to Novita's OpenAI-compatible API (`baai/bge-m3`, 1024 dimensions).
Set `EMBEDDING_PROVIDER=fastembed` to embed locally with FastEmbed (`BAAI/bge-m3`, also 1024-dim).
The ONNX model is downloaded on first embed (~2.3GB) and needs roughly **4GB+ RAM** (8GB preferred).
Switching providers recreates the Qdrant collection; re-ingest after a switch.
See [embedding provider](docs/embedding-provider.md).
Keep Novita keys in runtime variables; never commit them.
Do not change production Railway variables unless you intend to switch backends.

## HTTP API

- `POST /upload` — PDF, TXT, Markdown, JSON, or GeoJSON → persist + index
- `POST /upload-pdf` — PDF-only alias
- `POST /upload-media` — store bytes under `originals/` (not indexed)
- `POST /ingest-json` — Bearer-auth fetch of remote GeoJSON/JSON by URL → persist + index
- `GET /documents` — catalog
- `GET /documents/{doc_id}` — metadata
- `GET /documents/{doc_id}/text` — processed full text (no re-parse)
- `DELETE /documents/{doc_id}` — remove originals, processed files, catalog row, and Qdrant points
- `GET /health` — liveness

Listen address is `0.0.0.0:8080` (not configurable via `PORT`). Most routes have no inbound auth.
`POST /ingest-json` requires `Authorization: Bearer $RAG_API_TOKEN`.
`POST /upload`, `POST /upload-pdf`, and `DELETE /documents/{doc_id}` require a bearer token (`MCP_API_TOKEN`, else `RAG_API_TOKEN`) or a signed-in Google session.

### Remote GeoJSON (`POST /ingest-json`)

Body is `{"url": "<http(s) URL>"}`. GitHub blob pages are rewritten to `raw.githubusercontent.com`. Any similar GeoJSON is accepted (FeatureCollection, Feature, geometry, or a list of those). Feature attributes (`name`, `id`, `nationalCode`, `properties`, …) are indexed; coordinate rings are not.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X POST http://127.0.0.1:8080/ingest-json \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://github.com/siwekm/czech-geojson/blob/master/kraje.json"}'
```

`obce.json` is large (~95 MiB); ingest embeds attribute chunks and may take several minutes.

### Delete a document (`DELETE /documents/{doc_id}`)

Removes the original file, processed artifacts, catalog entry, and Qdrant vectors for one `doc_id`. Auth matches `POST /upload`: `Authorization: Bearer $MCP_API_TOKEN` (or `$RAG_API_TOKEN` if MCP token is unset), or a signed-in Google session.

Success: `{"status":"deleted","doc_id":"...","filename":"...","chunks_removed":N}`.
Unknown id: `404` `{"detail":"Document not found"}`. Missing/invalid token: `401`.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X DELETE \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  http://127.0.0.1:8080/documents/{doc_id}
```

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`**. Tools:

- `list_documents`
- `get_document_text`
- `get_document_chunks`
- `ingest_text`
- `search_domain_knowledge`

No MCP resources or prompts. No inbound MCP authentication.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system context, components, MCP surface, RAG pipeline, runtime, configuration, security.
- [Connectivity](docs/CONNECTIVITY.md) — trust boundaries, inbound/outbound paths, ports, auth points, connection inventory.
- [Embedding connectivity](docs/CONNECTIVITY.md#embedding-connectivity) — Novita embeddings API, auth, egress, timeouts, privacy, and cost.
- [Embedding provider](docs/embedding-provider.md) — Novita BGE-M3 configuration and Qdrant dimension handling.

## Package layout

```text
main.py                 # entrypoint (uvicorn / python main.py)
rag/
  app.py                # create_app()
  routes.py             # HTTP API
  mcp_tools.py          # FastMCP tools
  storage.py            # /data/rag originals + processed + index
  processing.py         # PDF/text/GeoJSON extract + chunk
  geojson.py            # generic GeoJSON → RAG text
  remote.py             # fetch remote JSON (GitHub blob → raw)
  auth.py               # inbound bearer token for /ingest-json
  embeddings.py         # Novita or local FastEmbed BGE-M3 embeddings
  accounting.py         # per-day token/comms accounting under /data/accnt
  ingest.py             # persist + index pipeline
  document_delete.py    # remove one document from disk + Qdrant
  vectorstore.py        # local Qdrant
  config.py             # env defaults
```

## Local run

```bash
export RAG_DATA_ROOT=./data/rag
export BGE_M3_NOVITA_API_KEY=your-novita-key
export RAG_API_TOKEN=your-api-token
python main.py

# Local ONNX instead of Novita (first embed downloads ~2.3GB):
# export EMBEDDING_PROVIDER=fastembed
```

## Tests

```bash
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf 'qdrant-client>=1.12,<2' requests python-multipart itsdangerous fastembed
pytest -q
```

## Connecting MCP clients

Endpoint: `https://rag-timer-00001-production.up.railway.app/mcp/` (Streamable HTTP, stateless; keep the trailing slash).

Auth: the **whole `/mcp` endpoint** requires `Authorization: Bearer <token>`.
The token comes from `MCP_API_TOKEN`; if that is unset, `RAG_API_TOKEN` is used.
If neither is set, `/mcp` and the embedding REST routes `/upload`, `/upload-pdf`, and `DELETE /documents/{doc_id}` fail closed with 503 and log a warning.
`/upload`, `/upload-pdf`, and `DELETE /documents/{doc_id}` also accept a signed-in Google session, which the console uses.
`/ingest-json` still requires `RAG_API_TOKEN`. Tokens are compared in constant time.

Host allow-list (MCP DNS-rebinding protection): `MCP_ALLOWED_HOSTS` takes a comma-separated list. The default is `rag-timer-00001-production.up.railway.app` plus localhost/127.0.0.1. Set `MCP_DISABLE_DNS_REBINDING_PROTECTION=1` to accept any Host. The server honors `PORT` (default 8080) and trusts `X-Forwarded-*` headers.

Cursor `mcp.json`:

```json
{ "mcpServers": { "rag-timer": {
  "url": "https://rag-timer-00001-production.up.railway.app/mcp/",
  "headers": { "Authorization": "Bearer ${env:MCP_API_TOKEN}" } } } }
```

Claude Desktop (via mcp-remote):

```json
{ "mcpServers": { "rag-timer": {
  "command": "npx",
  "args": ["-y", "mcp-remote", "https://rag-timer-00001-production.up.railway.app/mcp/",
           "--header", "Authorization:Bearer ${MCP_API_TOKEN}"],
  "env": { "MCP_API_TOKEN": "<token>" } } } }
```
