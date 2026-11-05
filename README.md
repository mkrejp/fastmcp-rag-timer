# fastmcp-rag-timer

Combines FastAPI (for web endpoints), FastMCP (to natively expose tools over the Model Context Protocol), and Qdrant Client (local-disk vector engine). It handles PDF and text extraction, stores originals and processed artifacts on disk, and indexes chunks for semantic retrieval.

## Persistent storage

All RAG server data lives under **`/data/rag`** (override with `RAG_DATA_ROOT`):

```text
/data/rag/
  originals/{doc_id}/...
  processed/{doc_id}/meta.json, fulltext.txt, chunks.json, images/{image_id}.{ext}
  qdrant/
  index.json
  embedding_backend.json

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
| `OCR_ENABLED` | `true` | Run local Tesseract OCR on PDF images during ingest |
| `OCR_LANGS` | `ces+eng` | Tesseract language pack string |
| `OCR_MIN_CONFIDENCE` | `60` | Drop OCR words below this confidence (0–100) |
| `OCR_MIN_CHARS` | `20` | Minimum cleaned OCR length to index a `pdf_image` point |
| `PUBLIC_BASE_URL` | unset | Public base URL for image links in MCP search (`/images/...`) |

Embeddings default to Novita's OpenAI-compatible API (`baai/bge-m3`, 1024 dimensions).
Set `EMBEDDING_PROVIDER=fastembed` to embed locally with FastEmbed (`BAAI/bge-m3`, also 1024-dim).
The ONNX model is downloaded on first embed (~2.3GB) and needs roughly **4GB+ RAM** (8GB preferred).
Switching providers recreates the Qdrant collection; re-ingest after a switch.
See [embedding provider](docs/embedding-provider.md).
Keep Novita keys in runtime variables; never commit them.
Do not change production Railway variables unless you intend to switch backends.

### PDF images and OCR

During PDF ingest, embedded images are extracted with PyMuPDF, deduplicated by SHA-256, and stored under `processed/{doc_id}/images/`. Images smaller than 64×64 px or 4 KiB are skipped. Local Tesseract OCR (`ces+eng` in Docker) fills `DocumentMeta.images[]` with optional `ocr_text` / `ocr_confidence`. When OCR text is long enough, a **`pdf_image`** vector point is upserted using the same BGE-M3 text embedding backend as body chunks (`pdf_chunk`). Qdrant payload fields include `source_type`, `doc_id`, `parent_document`, `page_number`, `file_path`, `image_id`, `sha256`, and `content_summary` (no image blobs in Qdrant).

**Re-ingest:** Existing deployments do **not** need a Qdrant collection recreate for this feature; re-upload PDFs to populate images and `pdf_image` points. A future phase with CLIP/`image_vector` named vectors **will** require collection recreate.

**Delete:** Removing a document (via `delete_document` helper used in tests/admin flows) deletes its `images/` directory and all Qdrant points with matching `doc_id` (chunks and images).

## HTTP API

- `POST /upload` — PDF, TXT, Markdown, JSON, or GeoJSON → persist + index
- `POST /upload-pdf` — PDF-only alias
- `POST /upload-media` — store bytes under `originals/` (not indexed)
- `POST /ingest-json` — Bearer-auth fetch of remote GeoJSON/JSON by URL → persist + index
- `GET /documents` — catalog
- `GET /documents/{doc_id}` — metadata
- `GET /documents/{doc_id}/text` — processed full text (no re-parse)
- `GET /images/{doc_id}/{image_id}` — extracted PDF image bytes (same auth as `POST /upload`)
- `GET /health` — liveness

Listen address is `0.0.0.0:8080` (not configurable via `PORT`). Most routes have no inbound auth; `POST /ingest-json` requires `Authorization: Bearer $RAG_API_TOKEN`. `GET /images/...` uses the same bearer or Google session as upload routes.

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

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`**. Tools:

- `list_documents`
- `get_document_text`
- `get_document_chunks`
- `get_document_images`
- `get_image` (inline image up to 5 MiB, otherwise a URL hint)
- `ingest_text`
- `search_domain_knowledge` (includes `source_type`, `page_number`, and `image_id`/URL for image hits)

No MCP resources or prompts. `/mcp` requires bearer auth (see below).

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
  pdf_images.py         # PyMuPDF embedded image extraction
  ocr.py                # local Tesseract OCR for extracted images
  payloads.py           # Qdrant payload builders
  document_delete.py    # delete doc files + Qdrant points
  geojson.py            # generic GeoJSON → RAG text
  remote.py             # fetch remote JSON (GitHub blob → raw)
  auth.py               # inbound bearer token for /ingest-json
  embeddings.py         # Novita or local FastEmbed BGE-M3 embeddings
  accounting.py         # per-day token/comms accounting under /data/accnt
  ingest.py             # persist + index pipeline
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
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf pymupdf pillow pytesseract 'qdrant-client>=1.12,<2' requests python-multipart itsdangerous fastembed
pytest -q
```

Optional integration tests (phrase-only-in-image search) require the Tesseract binary (`tesseract-ocr`, `tesseract-ocr-ces`, `tesseract-ocr-eng`). They are marked `integration` and skip when Tesseract is missing.

## Connecting MCP clients

Endpoint: `https://rag-timer-00001-production.up.railway.app/mcp/` (Streamable HTTP, stateless; keep the trailing slash).

Auth: the **whole `/mcp` endpoint** requires `Authorization: Bearer <token>`.
The token comes from `MCP_API_TOKEN`; if that is unset, `RAG_API_TOKEN` is used.
If neither is set, `/mcp` and the embedding REST routes `/upload` and `/upload-pdf` fail closed with 503 and log a warning.
`/upload` and `/upload-pdf` also accept a signed-in Google session, which the console uses.
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
