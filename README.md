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
| `PUBLIC_BASE_URL` | unset | Public origin for Google OAuth redirects, MCP image URLs (`/images/...`), and the login-page node address |
| `RAG_NODES` | two production nodes (see below) | Catalog of RAG nodes: JSON array of `{name,url,role}` or a list of `name\|url\|role` entries. Roles: `ZÁPIS`, `ČTENÍ`, `STOP` (also `write`/`read`/`primary`/`replica`). Extra JSON keys are ignored |
| `RAG_NODE_ROLE` | inferred | Role of **this** process (`ZÁPIS+ČTENÍ`, `ČTENÍ`, `STOP`, …). Wins over the catalog match |
| `RAG_NODE_NAME` | inferred | Optional catalog name for this process when URL / Railway service name would not match |
| `RAG_ENFORCE_NODE_ROLE` | unset (off) | Set to `1`/`true`/`yes` to reject HTTP and MCP **writes** on a ČTENÍ or STOP node (403). Off by default |
| `SESSION_SECRET` | ephemeral per process | Cookie session secret; regenerate on restart if unset |
| `GOOGLE_CLIENT_ID` | unset | Google OAuth client ID (required with secret to enable `/ui`) |
| `GOOGLE_CLIENT_SECRET` | unset | Google OAuth client secret |
| `GOOGLE_ALLOWED_EMAILS` | empty (allow all) | Comma-separated Google emails allowed into the console |
| `GOOGLE_ALLOWED_DOMAINS` | empty (allow all) | Comma-separated email domains allowed into the console |

**RAG nodes (login page + `/node-info`):** When `RAG_NODES` is unset, the catalog is `rag-timer-00001`
(`https://rag-timer-00001-production.up.railway.app`, **ZÁPIS+ČTENÍ**) and `rag-timer-00002`
(`https://rag-artnouveau.up.railway.app`, **ČTENÍ**). This process is matched by `RAG_NODE_NAME`,
`RAILWAY_SERVICE_NAME` (including `fastmcp-rag-timer-00002` → `00002`), `PUBLIC_BASE_URL`, or the
request host. Railway-provided `RAILWAY_SERVICE_NAME`, `RAILWAY_REPLICA_REGION`, and
`RAILWAY_ENVIRONMENT_NAME` are shown when present. Other env values are never returned. Write
blocking stays off until `RAG_ENFORCE_NODE_ROLE=1`.

**Split ingest vs query:** Ingest may use Novita (`baai/bge-m3`, 1024-dim) while every search path uses
**local** FastEmbed (`BAAI/bge-m3`, 1024-dim). Search never calls Novita. When `EMBEDDING_PROVIDER` is
`novita` (the default), set `EMBEDDING_PROVIDER_QUERY=fastembed` or startup refuses to start. Novita
HTTP vs local ONNX can differ slightly; re-check recall after switching ingest provider (optional
`scripts/golden_recall_smoke.py`).

**Named vectors:** Qdrant collection `client_domain_knowledge` uses `text_vector` (BGE-M3 1024) and `image_vector` (CLIP `Qdrant/clip-ViT-B-32-vision`, 512). Collections with a single unnamed vector are recreated on startup; **re-ingest all documents** after that recreate. Enable visual search with `IMAGE_EMBEDDING_ENABLED=true` (lazy CLIP load; extra RAM on top of BGE-M3 — risky on 8GB hosts).

The FastEmbed BGE-M3 ONNX download is ~2.3GB on first query or FastEmbed ingest (~4GB+ RAM peak; 8GB preferred for BGE-M3 alone). Query always uses FastEmbed, so the first search downloads the model even when ingest stays on Novita.
See [embedding provider](docs/embedding-provider.md).
Keep Novita keys in runtime variables; never commit them.

### PDF images and OCR

During PDF ingest, embedded images are extracted with PyMuPDF, deduplicated by SHA-256, and stored
under `processed/{doc_id}/images/`. Images smaller than 64×64 px or 4 KiB are skipped. Local Tesseract
OCR (`ces+eng` in Docker) fills `DocumentMeta.images[]` with optional `ocr_text` / `ocr_confidence`.
When OCR text is long enough, a **`pdf_image`** point gets a `text_vector` (BGE-M3 ingest backend).
With `IMAGE_EMBEDDING_ENABLED=true`, each kept image also receives an `image_vector` (CLIP vision),
including images without OCR. Payloads include `source_type`, `doc_id`, `parent_document`,
`page_number`, `file_path`, `image_id`, `sha256`, and `content_summary` (no image blobs in Qdrant).

Toggling CLIP does not recreate the collection; re-ingest PDFs to backfill `image_vector`.

**Delete:** `DELETE /documents/{doc_id}` (and `rag.document_delete.delete_document`) removes originals, processed files including `images/`, the catalog row, and all Qdrant points with matching `doc_id` (chunks and images).

## HTTP API

Embedding REST routes (multipart upload, chunked upload, PDF alias, images, delete, metadata patch) require `Authorization: Bearer $MCP_API_TOKEN` (or `$RAG_API_TOKEN` if MCP token is unset) or a signed-in Google session. Missing both tokens fail closed with **503**.

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
- `PATCH /documents/{doc_id}` — update catalog metadata and matching Qdrant payloads in place (no re-embed; same auth as `DELETE`)
- `PATCH /documents/{doc_id}/metadata` — write or delete verification fields (`set_payload` / JSON
  `null` on whitelist keys; no re-embed; same auth as `DELETE`)
- `DELETE /documents/{doc_id}` — remove originals, processed files (including `images/`), catalog row, and Qdrant points
- `GET /images/{doc_id}/{image_id}` — extracted PDF image bytes (same auth as `POST /upload`; `413` if larger than 50 MiB)
- `GET /health` — liveness (`status`, `rag_data_root`, ingest/query embedding providers, `image_embedding_enabled`)
- `GET /node-info` — public JSON: this instance (name, public URL, Railway service/region when present), catalog of nodes with Czech roles (`ZÁPIS` / `ČTENÍ` / `STOP`). No secrets
- `GET /` — redirect to `/ui/`
- `GET /ui/` — Google-authenticated RAG console (static shell under `static/`)
- `GET /auth/login`, `/auth/google`, `/auth/callback`, `/auth/logout`, `/auth/me`, `/auth/status` — Google OAuth
- `GET /api/admin/status`, `/documents`, `/documents/{doc_id}`, `/volume`, `/mcp`, `/logs`, `/accounts`, `/accounts/{date}` — console APIs (Google session required)

Listen address is `0.0.0.0` and port `PORT` (default 8080). uvicorn trusts `X-Forwarded-*` (`proxy_headers=True`, `forwarded_allow_ips=*`) and uses `timeout_graceful_shutdown=15`. FastAPI’s `/docs`, `/redoc`, and `/openapi.json` are enabled and unauthenticated.

### Chunked upload (large files)

Use when a single `POST /upload` would exceed proxy or worker limits. Compute the whole-file SHA-256
client-side, split into fixed-size chunks, then complete to run the same ingest path as `/upload`.
Staging lives under `uploads/{upload_id}/` inside `RAG_DATA_ROOT`. Expired sessions return **410**; a
resent chunk with different bytes returns **409**. Complete loads the assembled file into memory
before ingest.

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
Unknown ID: `404` `{"detail":"Document not found"}`. Missing/invalid token: `401`. Tokens unset: `503`.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X DELETE \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  http://127.0.0.1:8080/documents/{doc_id}
```

### Patch document metadata (`PATCH /documents/{doc_id}`)

Rename a stored document or merge custom `extra` keys without re-uploading or re-embedding. Auth
matches `DELETE /documents/{doc_id}`. Allowed JSON fields: `filename` (same suffix as the stored file)
and `extra` (object; JSON `null` removes a key). Unknown fields are rejected. Vectors are left
untouched; Qdrant payloads are updated with `set_payload` (`parent_document` / `source` / nested
`extra`).

Success: `{"status":"updated","doc_id","filename","updated_fields","points_updated","locality_links_updated","meta"}`.
Unknown ID: `404` `{"detail":"Document not found"}`. Invalid body: `400` or `422`. Missing/invalid
token: `401`. Tokens unset: `503`.

```bash
export RAG_API_TOKEN=your-api-token

curl -sS -X PATCH \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"renamed.txt","extra":{"akon_id":"AK110_057"}}' \
  http://127.0.0.1:8080/documents/{doc_id}
```

### Patch verification fields (`PATCH /documents/{doc_id}/metadata`)

Write RAGloader verification fields onto catalog `extra` and matching Qdrant payloads (`set_payload`
for non-null values; JSON `null` deletes whitelist keys via `delete_payload`). No re-embed; `doc_id`,
stored text, embeddings, and Qdrant points are unchanged. Same auth as `DELETE` (Bearer
`MCP_API_TOKEN` / `RAG_API_TOKEN` or Google session). MCP `update_document_metadata` mirrors this
behavior.

Non-null patches apply full fields on text/image chunk points (including `verification[]`). Figure
and page-image points receive only `figure_payload_fields` (flat keys; nested `verification[]` is not
written there). New uploads also store these fields when the optional `metadata` JSON form field (or
MCP `ingest_text` `metadata`) is provided; `reliability` defaults to `1`. `overeni_souhrn` is
truncated to 300 characters and quotes to 500; the response includes `truncated` with every clipped
field path. Each `verification[]` item may include `method`, `compared_value`, `found_value`,
`threshold`, and `evidence_type`.

**Deleting fields with JSON `null`.** Only these keys accept `null` (defined in code as
`DELETABLE_VERIFICATION_FIELDS`): `verification`, `verification_status`, `verified_by`,
`verified_at`, `verified_source_url`, `verified_quote`, `reliability`, `source_axis`, `source_key`,
`content_hash`, `overeni_souhrn`, `qg_trida`, `qg_flagy`, `qg_verze`. `source_key` and `content_hash`
may be nulled only in the same request as `verification: null` (full verification rollback); otherwise
the API returns **400**. `qg_trida`, `qg_flagy`, and `qg_verze` are plain deletes. `null` on any other
field — for example `tags`, `obec_kod`, `gps_presnost`, `extrakce_model`, `fakt_ref` — returns **400**.
Successful responses list removals in `deleted_fields` alongside
`updated_fields`, `points_updated`, and `figures_updated`. Operating convention (outside this repo):
RAGloader logs each deletion to `rag-changes.jsonl` with action `metadata_unset` and the list of
fields.

```bash
curl -sS -X PATCH \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reliability":2,"qg_verze":"2026-11-12:abcd","overeni_souhrn":"wiki strop 2"}' \
  http://127.0.0.1:8080/documents/{doc_id}/metadata

curl -sS -X PATCH \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"verified_by":null}' \
  http://127.0.0.1:8080/documents/{doc_id}/metadata
# → {"status":"updated","deleted_fields":["verified_by"],…}
```

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`** (`stateless_http=True`, `streamable_http_path="/"` so the endpoint is `/mcp`, not `/mcp/mcp`). The mount is wrapped with bearer auth and an ASGI helper that finishes SSE responses on process shutdown. Tools:

- `list_documents` (optional `since=` ISO timestamp vs `uploaded_at`)
- `get_document_text`
- `get_document_chunks`
- `get_document_images`
- `get_image` (inline image up to 5 MiB, otherwise a URL hint; accepts extracted `image_id` or full-page ID `page:{doc_id}:{page}`)
- `get_page_image` (`doc_id`, 1-based `page`) — lazy JPEG render from stored PDF (`PAGE_RENDER_DPI`, `PAGE_RENDER_QUALITY`)
- `get_page_highlights` (`doc_id`, `page`, `query`) — JSON boxes `{page_width, page_height, highlights:[{x0,y0,x1,y1,text,method}]}` (normalized 0–1)
- `ingest_text`
- `update_document` (`doc_id`, optional `filename`, optional `extra`) — patch catalog + Qdrant payloads, no re-embed
- `update_document_metadata` (`doc_id`, optional `metadata`) — patch verification payload fields (`null` deletes whitelist keys), no re-embed
- `search_domain_knowledge` (text `text_vector` only; optional `obec_id` / `cast_obce` payload filter; PDF hits add `page_image_id`; optional `format=json`)
- `search_images` (CLIP `image_vector`; optional `hybrid=true`; PDF hits add `page_image_id`; optional `format=json`; optional `precision` / `needs_refinement` filters on `geo_refined`)
- `link_locality` (upsert document-municipality links into payload-only `locality_links`; forwards to Trails when configured; no embeddings)
- `list_links` (filter `doc_id` / `obec_id` / `status` / `ambiguous` / `locality_level` / `cast_obce`, paginate with `limit` / `offset`)
- `verify_link` (set `verification_status` / verifier / notes on an existing locality_links point; forwards to Trails when configured)
- `link_stats` (counts by status, method, obec, ambiguous, locality_level)
- `sync_links_to_trails` (idempotent backfill of `locality_links` to TrailsAssemblage; `dry_run` option)
- `refine_geo` (`doc_id`, optional `chunk_index` / `image_id`, `geo_refined`, `dry_run=true`) — store refined GPS **beside** original `lat`/`lon` (never overwritten); writes append `geo_refined_history` and are idempotent
- `refine_geo_batch` (`items: [{doc_id, chunk_index?, image_id?, geo_refined}]`, `dry_run=true`) — same guarantees, per-item `ok` / `would_change` / `changed` / `error`; max **200** items per call
- `list_geo_refinements` (`filter` or individual `precision` / `status` / `confidence_level` / `needs_refinement` / `verified` / `doc_id`, paginated)

No MCP resources or prompts. `/mcp` requires bearer auth (see below). There is no stdio MCP entrypoint.

### Refined GPS (`geo_refined`)

Original coordinates stay on the record (`lat`/`lon`, or a GeoJSON `geom` Point). An optional `geo_refined` object is stored on document metadata (`meta.json`), chunk JSON, and Qdrant image/chunk payloads via `set_payload` (no re-embed, no schema migration). Existing documents without the field keep working.

```json
{
  "lat": 49.1951,
  "lon": 16.6080,
  "precision": "building",
  "precision_m": 15,
  "confidence": 0.86,
  "confidence_level": "high",
  "status": "zpresneno",
  "sources": [{"type": "wikidata_p625", "url": "https://www.wikidata.org/wiki/Q123", "distance_m": 12.4}],
  "method": "wikidata_p625",
  "distance_m": 6120,
  "needs_refinement": false,
  "verified": false,
  "refined_by": "gps-review",
  "refined_at": "2026-11-09T18:00:00+00:00",
  "note": "Deutsches Haus"
}
```

`precision` is `municipality|cast_obce|street|building`. `status` is `overeno|zpresneno|priblizne|rozpor`. `confidence_level` is `high|medium|low`. Source `type` is `ruian_adresni_misto|wikidata_p625|commons_camera|nominatim|osm_building|manual`. `needs_refinement` defaults to true for medium/low or `rozpor`. `dry_run=true` (the default) returns old vs new without writing.

Search hits, `list_image_records`, and `get_document_images` / `get_document_chunks` prefer `geo_refined` when present, else original, and expose both plus `geo_source: refined|original`. `list_image_records` and `search_images` accept optional `precision` and `needs_refinement` filters.

Backfill after deploy: call `refine_geo_batch` in pages of ≤200 (try `dry_run=true` first). Same bearer auth as `link_locality`. Geo refinements are **not** forwarded to Trails.

### Frontend contract (`fastmcp-rag-timer-frontend`)

Search hits can show the PDF page with query highlights without re-ingest. Qdrant payloads still use **0-based** `page_number`; page image IDs use **1-based** human page numbers.

| Field | Meaning |
| --- | --- |
| `page_image_id` | `page:{doc_id}:{page}` where `{page}` is 1-based (e.g. payload `page_number=0` → `page:…:1`) |
| `page_url` | Text citations only: absolute URL when `PUBLIC_BASE_URL` is set |

**Page image HTTP (same Bearer as upload):**

```http
GET /images/{doc_id}/page:{doc_id}:1
Authorization: Bearer $RAG_API_TOKEN
```

The frontend may proxy this as `GET /api/images/{page_image_id}` (URL-encode colons).

**Highlights (MCP):**

```json
// tools/call get_page_highlights
{"doc_id": "…", "page": 1, "query": "Praha", "max_hits": 20}
```

**Structured search (optional):**

```json
// search_domain_knowledge format=json
[
  {
    "doc_id": "…",
    "parent_document": "report.pdf",
    "source_type": "pdf_chunk",
    "page_number": 0,
    "page_image_id": "page:…:1",
    "image_id": null,
    "url": null,
    "score": 0.82,
    "content": "…",
    "lat": 49.1951,
    "lon": 16.6080,
    "geo_source": "refined",
    "geo_original": {"lat": 49.2, "lon": 16.6},
    "geo_refined": {"lat": 49.1951, "lon": 16.6080, "precision": "building"}
  }
]
```

Default `format=text` leaves legacy `Source:` / `Content:` blocks unchanged and only appends `page_image_id` / `page_url` on PDF `pdf_chunk` / `pdf_figure` hits (`search_images`: `pdf_image` / `pdf_figure`). When coordinates exist, text headers also append `lat` / `lon` / `geo_source`.

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
  pdf_pages.py          # lazy full-page JPEG render + highlight boxes
  page_ids.py           # page:{doc_id}:{page} identifiers
  ocr.py                # local Tesseract OCR for extracted images
  payloads.py           # Qdrant payload builders
  geo_refined.py        # geo_refined schema, prefer-refined helpers
  geo_index.py          # refine_geo / refine_geo_batch / list_geo_refinements
  document_delete.py    # remove one document from disk + Qdrant
  geojson.py            # generic GeoJSON → RAG text
  remote.py             # fetch remote JSON (GitHub blob → raw, SSRF guards)
  auth.py               # inbound bearer for /ingest-json, /mcp, embedding REST
  embeddings.py         # Novita ingest and/or local FastEmbed BGE-M3
  image_embed.py        # local CLIP for image_vector
  accounting.py         # per-day token/comms accounting under /data/accnt
  ingest.py             # persist + index pipeline
  vectorstore.py        # local Qdrant named vectors
  trails_sync.py        # TrailsAssemblage locality-link HTTP forward + outbox
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

# Optional TrailsAssemblage locality-link forward (both required; unset = off):
# export TRAILS_API_URL=https://trail.animarium.ai
# export TRAILS_API_TOKEN=...
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
If neither is set, `/mcp` and the embedding REST routes (`/upload`, chunked upload, `/upload-pdf`, `GET /images/...`, `PATCH`/`DELETE /documents/{doc_id}`) fail closed with 503 and log a warning.
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
