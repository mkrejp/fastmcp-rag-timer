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
| `EMBEDDING_PROVIDER` | `novita` (when unset) | Default for both paths when ingest/query overrides are unset |
| `EMBEDDING_PROVIDER_INGEST` | falls back to `EMBEDDING_PROVIDER` | Ingest / OCR / reindex only (`novita` or `fastembed`, BGE-M3 1024) |
| `EMBEDDING_PROVIDER_QUERY` | falls back to `EMBEDDING_PROVIDER` | Search paths only; **must be `fastembed`** (alias: `EMBEDDING_PROVIDER_SEARCH`) |
| `FASTEMBED_CACHE_PATH` | `{RAG_DATA_ROOT}/fastembed` | ONNX cache for local BGE-M3 (query path always uses this backend) |
| `IMAGE_EMBEDDING_ENABLED` | `false` | Load CLIP and index/search `image_vector` (adds RAM beside BGE-M3) |
| `IMAGE_EMBEDDING_CACHE_PATH` | `{RAG_DATA_ROOT}/clip` | ONNX cache for CLIP ViT-B/32 vision + text towers |
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

**Phase D — split providers:** Ingest may use Novita (`baai/bge-m3`, 1024-dim) while every search path uses **local** FastEmbed (`BAAI/bge-m3`, 1024-dim). Search never calls Novita (zero tokens on query). When the default is Novita, set `EMBEDDING_PROVIDER_QUERY=fastembed`. Startup refuses `EMBEDDING_PROVIDER_QUERY=novita`. Novita HTTP vs local ONNX can differ slightly; re-check recall after switching ingest provider (optional `scripts/golden_recall_smoke.py`).

**Phase C — named vectors:** Qdrant uses `text_vector` (BGE-M3 1024) and `image_vector` (CLIP `Qdrant/clip-ViT-B-32-vision`, 512). Legacy single-vector collections are recreated on startup; **re-ingest all documents** after upgrade. Enable visual search with `IMAGE_EMBEDDING_ENABLED=true` (lazy CLIP load; ~extra RAM on top of BGE-M3 — risky on 8GB Railway).

The FastEmbed BGE-M3 ONNX download is ~2.3GB on first query/ingest (~4GB+ RAM peak; 8GB preferred for BGE-M3 alone).
See [embedding provider](docs/embedding-provider.md).
Keep Novita keys in runtime variables; never commit them.

### PDF images and OCR

During PDF ingest, embedded images are extracted with PyMuPDF, deduplicated by SHA-256, and stored under `processed/{doc_id}/images/`. Images smaller than 64×64 px or 4 KiB are skipped. Local Tesseract OCR (`ces+eng` in Docker) fills `DocumentMeta.images[]` with optional `ocr_text` / `ocr_confidence`. When OCR text is long enough, a **`pdf_image`** point gets a `text_vector` (BGE-M3 ingest backend). With `IMAGE_EMBEDDING_ENABLED=true`, each kept image also receives an `image_vector` (CLIP vision), including images without OCR. Payloads include `source_type`, `doc_id`, `parent_document`, `page_number`, `file_path`, `image_id`, `sha256`, and `content_summary` (no image blobs in Qdrant).

**Re-ingest:** Named vectors require collection recreate on first deploy of phase C; re-upload/re-index all PDFs. Toggle CLIP without schema change does not recreate; new uploads need re-ingest to backfill `image_vector`.

**Delete:** `DELETE /documents/{doc_id}` (and the `delete_document` helper used in tests) removes originals, processed files including `images/`, the catalog row, and all Qdrant points with matching `doc_id` (chunks and images).

## HTTP API

- `POST /upload` — PDF, TXT, Markdown, JSON, or GeoJSON → persist + index (small files)
- `POST /upload/init` — start a chunked upload (large files; same auth as `/upload`)
- `PUT /upload/{upload_id}/chunks/{index}` — upload one raw chunk (~50 MiB default)
- `POST /upload/{upload_id}/complete` — verify, assemble on disk, ingest, remove staging
- `GET /upload/{upload_id}` — chunk progress (optional)
- `DELETE /upload/{upload_id}` — abort and delete staging
- `POST /upload-pdf` — PDF-only alias
- `POST /upload-media` — store bytes under `originals/` (not indexed)
- `POST /ingest-json` — Bearer-auth fetch of remote GeoJSON/JSON by URL → persist + index
- `GET /documents` — catalog
- `GET /documents/{doc_id}` — metadata
- `GET /documents/{doc_id}/text` — processed full text (no re-parse)
- `DELETE /documents/{doc_id}` — remove originals, processed files (including `images/`), catalog row, and Qdrant points
- `GET /images/{doc_id}/{image_id}` — extracted PDF image bytes (same auth as `POST /upload`)
- `GET /health` — liveness (+ ingest/query embedding providers, `IMAGE_EMBEDDING_ENABLED`)

Listen address is `0.0.0.0:8080` (not configurable via `PORT`). Most routes have no inbound auth.
`POST /ingest-json` requires `Authorization: Bearer $RAG_API_TOKEN`.
`POST /upload`, chunked upload routes, `POST /upload-pdf`, `GET /images/...`, and `DELETE /documents/{doc_id}` require a bearer token (`MCP_API_TOKEN`, else `RAG_API_TOKEN`) or a signed-in Google session.

Chunked staging lives under `uploads/{upload_id}/` inside `RAG_DATA_ROOT`. Configure `RAG_UPLOAD_CHUNK_SIZE` (default 52428800), `RAG_UPLOAD_MAX_BYTES`, and `RAG_UPLOAD_TTL_SECONDS` (default 86400).

### Chunked upload (large files)

Use when a single `POST /upload` would exceed proxy or worker limits. Compute the whole-file SHA-256 client-side, split into fixed-size chunks, then complete to run the same ingest path as `/upload`.

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

# Upload chunk indexes 0..N-1 (example: two-chunk split with dd)
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

Removes the original file, processed artifacts (including `images/`), catalog entry, and Qdrant vectors (chunks and image points) for one `doc_id`. Auth matches `POST /upload`: `Authorization: Bearer $MCP_API_TOKEN` (or `$RAG_API_TOKEN` if MCP token is unset), or a signed-in Google session.

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
- `get_document_images`
- `get_image` (inline image up to 5 MiB, otherwise a URL hint)
- `ingest_text`
- `search_domain_knowledge` (text `text_vector` only; includes image metadata on OCR hits)
- `search_images` (CLIP `image_vector`; optional `hybrid=true` merges with text OCR hits via RRF)

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
  document_delete.py    # remove one document from disk + Qdrant
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
If neither is set, `/mcp` and the embedding REST routes `/upload`, `/upload-pdf`, `GET /images/...`, and `DELETE /documents/{doc_id}` fail closed with 503 and log a warning.
`/upload`, `/upload-pdf`, `GET /images/...`, and `DELETE /documents/{doc_id}` also accept a signed-in Google session, which the console uses.
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
