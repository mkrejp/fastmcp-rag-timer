# fastmcp-rag-timer

Combines FastAPI (web endpoints and a Google-authenticated console), FastMCP (tools over Streamable HTTP), and an on-disk Qdrant Client. It extracts PDF/text/GeoJSON, stores originals and processed artifacts, and indexes named vectors for semantic retrieval. Search is retrieval-only (no generator LLM). There is no scheduler despite the repository name.

## Persistent storage

All RAG server data lives under **`/data/rag`** (override with `RAG_DATA_ROOT`):

```text
/data/rag/
  originals/{doc_id}/...
  processed/{doc_id}/meta.json, fulltext.txt, chunks.json, images/{image_id}.{ext}
  uploads/{upload_id}/          # chunked-upload staging (TTL)
  qdrant/
  index.json
  embedding_backend.json
  fastembed/                    # BGE-M3 ONNX cache (or FASTEMBED_CACHE_PATH)
  clip/                         # CLIP ONNX cache when visual search is on

/data/accnt/
  YYYY-MM-DD.json          # daily rollups (embeddings, MCP, HTTP, outbound)
  events/YYYY-MM-DD.jsonl  # per-call audit lines
```

On Railway, mount a volume at `/data`. The app creates `/data/rag` and `/data/accnt` at startup. Dockerfile does not declare `VOLUME`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | uvicorn listen port (`HOST` is always `0.0.0.0`) |
| `RAG_DATA_ROOT` | `/data/rag` | Persistent originals, processed text, Qdrant, upload staging |
| `ACCNT_DATA_ROOT` | `/data/accnt` | Per-day token/comms accounting rollups and event logs |
| `EMBEDDING_PROVIDER` | `novita` (when unset) | Default for ingest/query when the path-specific vars are unset |
| `EMBEDDING_PROVIDER_INGEST` | falls back to `EMBEDDING_PROVIDER` | Ingest / OCR indexing (`novita` or `fastembed`, BGE-M3 1024) |
| `EMBEDDING_PROVIDER_QUERY` | falls back to `EMBEDDING_PROVIDER` | Search paths only; **must be `fastembed`** (alias: `EMBEDDING_PROVIDER_SEARCH`) |
| `FASTEMBED_CACHE_PATH` | `{RAG_DATA_ROOT}/fastembed` | ONNX cache for local BGE-M3 (query always uses this backend) |
| `IMAGE_EMBEDDING_ENABLED` | `false` | Load CLIP and index/search `image_vector` (adds RAM beside BGE-M3) |
| `IMAGE_EMBEDDING_CACHE_PATH` | `{RAG_DATA_ROOT}/clip` | ONNX cache for CLIP ViT-B/32 vision + text towers |
| `BGE_M3_NOVITA_API_KEY` | unset | Preferred Novita API credential for ingest embeddings |
| `NOVITA_API_KEY` | unset | Embedding credential if `BGE_M3_NOVITA_API_KEY` is empty |
| `OPENAI_TOKEN` | unset | Embedding credential if both Novita-named vars are empty |
| `OPENAI_API_KEY` | unset | Embedding credential checked last |
| `RAG_SKIP_DEFAULT_APP` | unset | Set to `1` to skip module-level `create_app()` (used by tests) |
| `RAG_API_TOKEN` | unset | Required Bearer token for `POST /ingest-json`; fallback token for `/mcp` and embedding REST routes |
| `MCP_API_TOKEN` | unset | Preferred Bearer token for `/mcp` and embedding REST routes (wins over `RAG_API_TOKEN`) |
| `MCP_ALLOWED_HOSTS` | production Railway host + localhost | Comma-separated Host allow-list for MCP DNS-rebinding protection |
| `MCP_DISABLE_DNS_REBINDING_PROTECTION` | unset | Set to `1`/`true`/`yes` to accept any Host |
| `RAG_MAX_REMOTE_JSON_BYTES` | `134217728` (128 MiB) | Remote GeoJSON download cap (`obce.json` is ~95 MiB) |
| `RAG_REMOTE_FETCH_TIMEOUT` | `120` | Remote fetch timeout in seconds |
| `RAG_UPLOAD_CHUNK_SIZE` | `52428800` (50 MiB) | Default chunk size for `POST /upload/init` |
| `RAG_UPLOAD_MAX_CHUNK_SIZE` | `52428800` (50 MiB) | Maximum accepted per-chunk body |
| `RAG_UPLOAD_MAX_BYTES` | `536870912` (512 MiB) | Maximum whole-file size for chunked uploads |
| `RAG_UPLOAD_TTL_SECONDS` | `86400` | Staging lifetime for chunked-upload sessions |
| `OCR_ENABLED` | `true` | Run local Tesseract OCR on PDF images during ingest |
| `OCR_LANGS` | `ces+eng` | Tesseract language pack string |
| `OCR_MIN_CONFIDENCE` | `60` | Drop OCR words below this confidence (0–100) |
| `OCR_MIN_CHARS` | `20` | Minimum cleaned OCR length to index a `pdf_image` text vector |
| `PUBLIC_BASE_URL` | unset | Public origin for Google OAuth redirects and MCP image URLs (`/images/...`) |
| `SESSION_SECRET` | ephemeral per process | Cookie session secret; regenerate on restart if unset |
| `GOOGLE_CLIENT_ID` | unset | Google OAuth client id (required with secret to enable `/ui`) |
| `GOOGLE_CLIENT_SECRET` | unset | Google OAuth client secret |
| `GOOGLE_ALLOWED_EMAILS` | empty (allow all) | Comma-separated Google emails allowed into the console |
| `GOOGLE_ALLOWED_DOMAINS` | empty (allow all) | Comma-separated email domains allowed into the console |

**Split ingest vs query:** Ingest may use Novita (`baai/bge-m3`, 1024-dim) while every search path uses **local** FastEmbed (`BAAI/bge-m3`, 1024-dim). Search never calls Novita. When `EMBEDDING_PROVIDER` is `novita` (the default), set `EMBEDDING_PROVIDER_QUERY=fastembed` or startup refuses to start. Novita HTTP vs local ONNX can differ slightly; re-check recall after switching ingest provider (optional `scripts/golden_recall_smoke.py`).

**Named vectors:** Qdrant collection `client_domain_knowledge` uses `text_vector` (BGE-M3 1024) and `image_vector` (CLIP `Qdrant/clip-ViT-B-32-vision`, 512). Collections with a single unnamed vector are recreated on startup; **re-ingest all documents** after that recreate. Enable visual search with `IMAGE_EMBEDDING_ENABLED=true` (lazy CLIP load; extra RAM on top of BGE-M3 — risky on 8GB hosts).

The FastEmbed BGE-M3 ONNX download is ~2.3GB on first query or FastEmbed ingest (~4GB+ RAM peak; 8GB preferred for BGE-M3 alone). Query always uses FastEmbed, so the first search downloads the model even when ingest stays on Novita.
See [embedding provider](docs/embedding-provider.md).
Keep Novita keys in runtime variables; never commit them.

### PDF images and OCR

During PDF ingest, embedded images are extracted with PyMuPDF, deduplicated by SHA-256, and stored under `processed/{doc_id}/images/`. Images smaller than 64×64 px or 4 KiB are skipped. Local Tesseract OCR (`ces+eng` in Docker) fills `DocumentMeta.images[]` with optional `ocr_text` / `ocr_confidence`. When OCR text is long enough, a **`pdf_image`** point gets a `text_vector` (BGE-M3 ingest backend). With `IMAGE_EMBEDDING_ENABLED=true`, each kept image also receives an `image_vector` (CLIP vision), including images without OCR. Payloads include `source_type`, `doc_id`, `parent_document`, `page_number`, `file_path`, `image_id`, `sha256`, and `content_summary` (no image blobs in Qdrant).

Toggling CLIP does not recreate the collection; re-ingest PDFs to backfill `image_vector`.

**Delete:** `DELETE /documents/{doc_id}` (and `rag.document_delete.delete_document`) removes originals, processed files including `images/`, the catalog row, and all Qdrant points with matching `doc_id` (chunks and images).

## HTTP API

Embedding REST routes (multipart upload, chunked upload, PDF alias, images, delete) require `Authorization: Bearer $MCP_API_TOKEN` (or `$RAG_API_TOKEN` if MCP token is unset) or a signed-in Google session. Missing both tokens fail closed with **503**.

- `POST /upload` — PDF, TXT, Markdown, JSON, or GeoJSON → persist + index (small files)
- `POST /upload/init` — start a chunked upload (large files; same auth as `/upload`)
- `PUT /upload/{upload_id}/chunks/{index}` — one raw chunk body (optional `X-Chunk-SHA256`)
- `POST /upload/{upload_id}/complete` — verify SHA-256, assemble on disk, ingest, remove staging
- `GET /upload/{upload_id}` — chunk progress
- `DELETE /upload/{upload_id}` — abort and delete staging (`204`; `404` if unknown)
- `POST /upload-pdf` — PDF-only alias
- `POST /upload-media` — store bytes under `originals/` (not indexed; **no inbound auth**)
- `POST /ingest-json` — Bearer `RAG_API_TOKEN` only; fetch remote GeoJSON/JSON by URL → persist + index
- `GET /documents` — catalog (**no inbound auth**)
- `GET /documents/{doc_id}` — metadata (**no inbound auth**)
- `GET /documents/{doc_id}/text` — processed full text, no re-parse (**no inbound auth**)
- `DELETE /documents/{doc_id}` — remove originals, processed files (including `images/`), catalog row, and Qdrant points
- `GET /images/{doc_id}/{image_id}` — extracted PDF image bytes (same auth as `POST /upload`; `413` if larger than 50 MiB)
- `GET /health` — liveness (`status`, `rag_data_root`, ingest/query embedding providers, `image_embedding_enabled`)
- `GET /` — redirect to `/ui/`
- `GET /ui/` — Google-authenticated RAG console (static shell under `static/`)
- `GET /auth/login`, `/auth/google`, `/auth/callback`, `/auth/logout`, `/auth/me`, `/auth/status` — Google OAuth
- `GET /api/admin/status`, `/documents`, `/documents/{doc_id}`, `/volume`, `/mcp`, `/logs`, `/accounts`, `/accounts/{date}` — console APIs (Google session required)

Listen address is `0.0.0.0` and port `PORT` (default 8080). uvicorn trusts `X-Forwarded-*` (`proxy_headers=True`, `forwarded_allow_ips=*`) and uses `timeout_graceful_shutdown=15`. FastAPI’s `/docs`, `/redoc`, and `/openapi.json` are enabled and unauthenticated.

### Chunked upload (large files)

Use when a single `POST /upload` would exceed proxy or worker limits. Compute the whole-file SHA-256 client-side, split into fixed-size chunks, then complete to run the same ingest path as `/upload`. Staging lives under `uploads/{upload_id}/` inside `RAG_DATA_ROOT`. Expired sessions return **410**; a resent chunk with different bytes returns **409**. Complete loads the assembled file into memory before ingest.

```bash
export MCP_API_TOKEN=your-token
FILE=large.pdf
SIZE=$(wc -c < "$FILE" | tr -d ' ')
SHA=$(sha256sum "$FILE" | awk '{print $1}')
CHUNK=52428800

INIT=$(curl -sS -X POST http://127.0.0.1:8080/upload/init \
  -H "Authorization: Bearer $MCP_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"$(basename "$FILE")\",\"size\":$SIZE,\"sha256\":\"$SHA\",\"chunk_size\":$CHUNK}")
UPLOAD_ID=$(echo "$INIT" | python3 -c "import sys,json; print(json.load(sys.stdin)['upload_id'])")

INDEX=0
OFFSET=0
while [ "$OFFSET" -lt "$SIZE" ]; do
  COUNT=$(( SIZE - OFFSET < CHUNK ? SIZE - OFFSET : CHUNK ))
  dd if="$FILE" bs=1 skip="$OFFSET" count="$COUNT" 2>/dev/null | \
    curl -sS -X PUT "http://127.0.0.1:8080/upload/$UPLOAD_ID/chunks/$INDEX" \
      -H "Authorization: Bearer $MCP_API_TOKEN" \
      --data-binary @-
  INDEX=$((INDEX + 1))
  OFFSET=$((OFFSET + COUNT))
done

curl -sS -X POST "http://127.0.0.1:8080/upload/$UPLOAD_ID/complete" \
  -H "Authorization: Bearer $MCP_API_TOKEN"
```

### Remote GeoJSON (`POST /ingest-json`)

Body is `{"url": "<http(s) URL>"}`. GitHub blob pages are rewritten to `raw.githubusercontent.com`. Any similar GeoJSON is accepted (FeatureCollection, Feature, geometry, or a list of those). Feature attributes (`name`, `id`, `nationalCode`, `properties`, …) are indexed; coordinate rings are not. Private/loopback/metadata hosts are rejected.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X POST http://127.0.0.1:8080/ingest-json \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://github.com/siwekm/czech-geojson/blob/master/kraje.json"}'
```

`obce.json` is large (~95 MiB); ingest embeds attribute chunks and may take several minutes.

### Delete a document (`DELETE /documents/{doc_id}`)

Removes the original file, processed artifacts (including `images/`), catalog entry, and Qdrant vectors (chunks and image points) for one `doc_id`. Auth matches `POST /upload`.

Success: `{"status":"deleted","doc_id":"...","filename":"...","chunks_removed":N}`.
Unknown id: `404` `{"detail":"Document not found"}`. Missing/invalid token: `401`. Tokens unset: `503`.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X DELETE \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  http://127.0.0.1:8080/documents/{doc_id}
```

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`** (`stateless_http=True`, `streamable_http_path="/"` so the endpoint is `/mcp`, not `/mcp/mcp`). The mount is wrapped with bearer auth and an ASGI helper that finishes SSE responses on process shutdown. Tools:

- `list_documents` (optional `since=` ISO timestamp vs `uploaded_at`)
- `get_document_text`
- `get_document_chunks`
- `get_document_images`
- `get_image` (inline image up to 5 MiB, otherwise a URL hint)
- `ingest_text`
- `search_domain_knowledge` (text `text_vector` only; optional `obec_id` / `cast_obce` payload filter; includes image metadata on OCR hits)
- `search_images` (CLIP `image_vector`; optional `hybrid=true` merges with text OCR hits via RRF)
- `link_locality` (upsert document-municipality links into payload-only `locality_links`; no embeddings)
- `list_links` (filter `doc_id` / `obec_id` / `status` / `ambiguous` / `locality_level` / `cast_obce`, paginate with `limit` / `offset`)
- `verify_link` (set `verification_status` / verifier / notes on an existing locality_links point)
- `link_stats` (counts by status, method, obec, ambiguous, locality_level)

No MCP resources or prompts. `/mcp` requires bearer auth (see below). There is no stdio MCP entrypoint.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system context, components, MCP surface, RAG pipeline, runtime, configuration, security.
- [Connectivity](docs/CONNECTIVITY.md) — trust boundaries, inbound/outbound paths, ports, auth points, connection inventory.
- [Embedding connectivity](docs/CONNECTIVITY.md#embedding-connectivity) — Novita embeddings API, auth, egress, timeouts, privacy, and cost.
- [Embedding provider](docs/embedding-provider.md) — split ingest/query providers, named vectors, and Qdrant layout.

## Package layout

```text
main.py                 # entrypoint (uvicorn / python main.py)
rag/
  app.py                # create_app() — FastAPI, FastMCP mount, UI, lifespan
  routes.py             # HTTP API (upload, ingest-json, documents, images)
  admin_routes.py       # Google-gated /api/admin
  google_oauth.py       # /auth/* Google login
  mcp_tools.py          # FastMCP tools
  chunked_upload.py     # server-side chunk staging and reassembly
  asgi_shutdown.py      # finish MCP SSE responses on SIGTERM
  storage.py            # /data/rag originals + processed + index
  processing.py         # PDF/text/GeoJSON extract + chunk
  pdf_images.py         # PyMuPDF embedded image extraction
  ocr.py                # local Tesseract OCR for extracted images
  payloads.py           # Qdrant payload builders
  document_delete.py    # remove one document from disk + Qdrant
  geojson.py            # generic GeoJSON → RAG text
  remote.py             # fetch remote JSON (GitHub blob → raw, SSRF guards)
  auth.py               # inbound bearer for /ingest-json, /mcp, embedding REST
  embeddings.py         # Novita ingest and/or local FastEmbed BGE-M3
  image_embed.py        # local CLIP for image_vector
  accounting.py         # per-day token/comms accounting under /data/accnt
  ingest.py             # persist + index pipeline
  vectorstore.py        # local Qdrant named vectors
  logging_setup.py      # rag logger + in-memory ring buffer
  config.py             # env defaults
static/                 # RAG console (index.html, app.js, styles.css)
scripts/golden_recall_smoke.py
```

## Local run

Startup validates that search uses FastEmbed. With the default Novita ingest provider you must set `EMBEDDING_PROVIDER_QUERY=fastembed`.

```bash
export RAG_DATA_ROOT=./data/rag
export EMBEDDING_PROVIDER_QUERY=fastembed
export BGE_M3_NOVITA_API_KEY=your-novita-key
export RAG_API_TOKEN=your-api-token
python main.py

# Local ONNX for ingest as well as query (first embed downloads ~2.3GB):
# export EMBEDDING_PROVIDER=fastembed
```

## Tests

```bash
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf pymupdf pillow pytesseract 'qdrant-client>=1.12,<2' requests python-multipart itsdangerous fastembed
pytest -q
```

Optional integration tests (phrase-only-in-image search) require the Tesseract binary (`tesseract-ocr`, `tesseract-ocr-ces`, `tesseract-ocr-eng`). They are marked `integration` and skip when Tesseract is missing.

CI (`.github/workflows/`) runs Super Linter on changed files and builds the Docker image; it does not run pytest.

## Connecting MCP clients

Endpoint: `https://rag-timer-00001-production.up.railway.app/mcp/` (Streamable HTTP, stateless; keep the trailing slash).

Auth: the **whole `/mcp` endpoint** requires `Authorization: Bearer <token>`.
The token comes from `MCP_API_TOKEN`; if that is unset, `RAG_API_TOKEN` is used.
If neither is set, `/mcp` and the embedding REST routes (`/upload`, chunked upload, `/upload-pdf`, `GET /images/...`, `DELETE /documents/{doc_id}`) fail closed with 503 and log a warning.
Those embedding REST routes also accept a signed-in Google session, which the console uses.
`POST /ingest-json` requires `RAG_API_TOKEN` specifically (not the MCP token, unless they are the same value). Tokens are compared in constant time.

Host allow-list (MCP DNS-rebinding protection): `MCP_ALLOWED_HOSTS` takes a comma-separated list. The default is `rag-timer-00001-production.up.railway.app` plus localhost/127.0.0.1 (including `localhost:*` / `127.0.0.1:*`). Set `MCP_DISABLE_DNS_REBINDING_PROTECTION=1` to accept any Host.

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
