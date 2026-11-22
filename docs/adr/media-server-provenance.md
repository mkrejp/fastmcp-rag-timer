# ADR: Media server as binary + provenance origin

- **Status:** Proposed (Marek approval required before any schema, API, or deploy work)
- **Date:** 2026-11-22
- **Deciders:** Marek
- **Policy approved:** 2026-11-22 (shared provenance block on every used document and image; same snake_case field names in RAG and the media server)
- **Scope of this PR:** this Markdown file only. No application code, no migrations, no Railway changes, no secrets.
- **Follow-up agents** start only after Marek’s word on this ADR. ControlApps merges after TesterVerifier GO and Marek’s word. Do not merge this PR from an agent.

## 1. Decision summary

1. Introduce a **dedicated media service** as the single source of **binaries** (original + thumbnail + large) and of the **canonical provenance record**.
2. **RAG (fastmcp-rag-timer ingester 00001 / reader 00002)** keeps text, OCR, embeddings, catalog rows, and a **copy of the same provenance fields**, plus locators (`media_asset_id`, `rag_doc_id`, `rag_image_id`, `sha256`). RAG stops being the public image origin.
3. **TrailsAssemblage / geobjectsdb / mobile-api** keep geoobject identity as `(rag_doc_id, rag_image_id)` and add `media_asset_id` plus the shared provenance fields on `layer_object_images`. The browser never sees a RAG origin URL or a RAG token.
4. **Every hosted image is resized at intake** into a **thumbnail** and a **large** derivative **before** the asset can leave `license_status=intake`. Clients never request “make me a thumb”. A batch job backfills the existing RAG image files.
5. **Hugging Face and sftpgo remain backups**, not origins. **`POST /upload-media` is closed** (auth + provenance required, or the route is removed once media exists).
6. **Web UI contract:** JSON for every published image includes `thumb_url`, `large_url`,
   `attribution_text`, pixel `width`/`height` per size, `license_id`, `alt`, and `title`.
   Clicking the thumbnail opens the **large** rendition in a lightbox **with attribution
   always visible**. TrailsAssemblage map popups, catalog search, recent list, and trail
   detail are the consumers. **No frontend code in this PR.**

This ADR names concrete paths, endpoints, field names, and mismatches in the current code. It does not implement them.

## 2. Policy (already approved)

Every document used (books, magazines, scans, texts, images) carries provenance + licence metadata, stored in **both** the RAG catalog/Qdrant payloads **and** the media server, with the **same snake_case names**.

### 2.1 Shared provenance block

| Field | Meaning |
| --- | --- |
| `content_type` | `image` \| `document` \| `page_scan` \| `magazine` \| `book` \| `text` |
| `source_url` | Canonical source page (Commons File: page, Kramerius UUID page, IA item, …) |
| `source_name` | e.g. Wikimedia Commons, Kramerius NKP, `archive.org`, ÖNB AKON |
| `source_library` | Holding library when distinct from `source_name` |
| `author` | Creator / photographer |
| `title` | Work title |
| `license_id` | `CC-BY-SA-4.0` \| `CC-BY-4.0` \| `CC0-1.0` \| `PD` \| `Copyright` \| `Permission` \| `Own` \| `Unknown` |
| `license_url` | SPDX / Commons license URL when applicable |
| `attribution_text` | Ready to display. Commons = TASL |
| `retrieved_at` | ISO-8601 UTC |
| `modifications` | What we changed (empty/`none` on original; `resized` on derivatives) |
| `verified_by` | Operator / agent ID who completed licence check |
| `license_status` | `intake` \| `license_check` \| `metadata_complete` \| `verified` \| `published` \| `quarantine` \| `takedown` \| `restricted` |
| `sha256` | Hex digest of the **original** bytes |
| `publication_year` | When known |
| `author_death_year` | When known (EU 70y PD) |
| `pd_status` | `pd_eu` \| `pd_us` \| `pd_both` \| `not_pd` \| `unknown` |
| `quote` | Required when hosting is refused and a short copyrighted excerpt is kept in RAG text |
| `permission_evidence_url` | Required when `license_id=Permission` |

### 2.2 Hosting vs link-only

| Host binaries (original + thumb + large) | Do **not** host binaries (`license_status=restricted`; link `source_url` only) |
| --- | --- |
| `PD` with `pd_status` in `{pd_eu, pd_both}` (EU 70y) | `NC` / `ND` variants (not in `license_id` enum — map to `Copyright` + `restricted`) |
| `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0` | `Copyright`, `Unknown` |
| `Own` | `PD` that is `pd_us` only (`pd_status=pd_us`) |
| `Permission` **with** `permission_evidence_url` | `Permission` without evidence |

**Kramerius `licence=public` is an access flag, not a licence.** Map it to `license_id=Unknown` (or a documented source note) until a real SPDX/PD decision exists. Do not treat it as `PD` or `CC0-1.0`.

**Takedown:** set `license_status=takedown`, hide all sizes, **archive, never delete**, append an audit log row. Same for `quarantine`.

**API rule:** every image JSON includes `attribution_text`. Binary responses for `quarantine` / `takedown` / `restricted` / `license_id` in `{Copyright, Unknown}` are refused (404 to anonymous clients; 403 to operators who are not in the takedown-audit role). Metadata may still describe why.

### 2.3 Status namespaces (must not clash)

Three different “status” words already exist in rag-timer. `license_status` is a fourth, on purpose.

| Name | Where today | Values today | Role after this ADR |
| --- | --- | --- | --- |
| `DocumentMeta.status` | `rag/storage.py`, `index.json` | ingest machine state, typically `ready` | **Unchanged.** Ingest pipeline only. |
| `doc_status` | `rag/verification_meta.py` | `active` \| `superseded` \| `retracted` | RAG document lifecycle. Unchanged. |
| `verification_status` (Czech, documents) | `verification_meta.py` | `overeno`, `potvrzeno-wiki`, `nejiste`, `rozpor`, `zamitnuto`, `nelze` | Fact-check of RAG text. Unchanged. |
| `verification_status` (figures) | `rag/payloads.py` `FIGURE_VERIFICATION_STATUSES` | `unverified` \| `auto_checked` \| `verified` \| `rejected` | Caption/figure QA. Unchanged. **Not** a licence. |
| `license_status` | **new**, both media + RAG | policy enum in §2.1 | Provenance workflow only. |

`PATCH /documents/{id}/metadata` already rejects a field named `status` (`STATUS_ALIAS_MESSAGE` in `verification_meta.py`). Do not add `license_status` as an alias of any of the above.

`verified_by` already exists on RAG verification and on figures (`verify_image_record`). After this ADR it is **also** a provenance field (who signed the licence). If both are present on one payload, they share the name and mean “who last signed this record”; figure caption QA continues to use figure `verification_status`.

---

## 3. Current state (verified against code)

The operator-side map is **mostly correct** for rag-timer 00001. Corrections and extra surfaces are marked below.

### 3.1 Repositories actually read

| Repository | Role in this design | Notes |
| --- | --- | --- |
| `mkrejp/fastmcp-rag-timer` (this repository, ingester 00001 / reader split) | RAG ingest, disk catalog, Qdrant, `GET /images/...` | Source of today’s binaries |
| `mkrejp/TrailsAssemblage` | PostGIS `layer_objects`, catalog HTTP, `mobile-api/` | geobjectsdb; no separate `mobile-api` GitHub repository |
| `mkrejp/sftpgo-geo-servus` | SFTP daemon wrapper | Desktop backup sink, not an HTTP media origin |
| `mkrejp/sftp-mirror-geo-server` | Nightly rclone pull from sftpgo | DR copy of yesterday’s `COMPLETE` generation |
| `mkrejp/sbdb` | Different PostGIS sample-db | **Not** geobjectsdb; ignore for this ADR |

**Not in GitHub (operator box / agents only):** FotoImporter, catalog-upload-tools, RAGloader, Hugging Face backup jobs. Behaviours below for those tools are taken from `docs/AI_AGENTS.md` plus the task brief; they are **not** verifiable from git.

### 3.2 Storage on the rag-timer Railway volume

Confirmed in `rag/storage.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/SPLIT_DEPLOY.md`:

```text
/data/rag/                          # RAG_DATA_ROOT
  originals/{doc_id}/{filename}     # raw upload (PDF/txt/json); also /upload-media
  processed/{doc_id}/
    meta.json, fulltext.txt, chunks.json
    images/{image_id}.{ext}         # pdf_image extracts + pdf_figure PNG clips
    pages/{n}.jpg                   # lazy full-page renders (page:{doc_id}:{n})
  versions/{doc_id}/                # text replace snapshots; not replicated
  uploads/{upload_id}/              # chunked-upload staging
  qdrant/                           # embedded Qdrant, same volume
  index.json
```

`RagStorage.ensure_layout()` creates `originals/`, `processed/`, `qdrant/`, `versions/`. Image files live under `processed/{doc_id}/images/`. `pdf_figure` PNGs are often **on disk but not listed** in `DocumentMeta.images` (`image_delivery.py`).

Split deploy: **two volumes**, not shared. Observed full tree ~**39 GB**. Railway volume storage is billed even when the service is stopped (`deploy/railway/README.md`: ~$0.15/GB/mo). Warm **reader** is always on; **ingester is cold**.

### 3.3 Image HTTP and URLs

| Item | Code | Correction vs the incoming map |
| --- | --- | --- |
| `GET /images/{doc_id}/{image_id}` | `rag/routes.py`, `require_embed_access` | Confirmed: **not public**. Bearer `MCP_API_TOKEN` / `RAG_API_TOKEN` **or** Google session. Cap 50 MiB (`MAX_IMAGE_HTTP_BYTES`). |
| `MCP_READ_TOKEN` | `rag/auth.py` (PR #53) | Read-only for **`/mcp` only**. It **does not** authorize `GET /images/...`. Trails `docs/images-geo-objects.md` still says the proxy uses `MCP_READ_TOKEN` against `{RAG_READER_URL}/images/{doc}/{id}` — that will **401** unless Trails actually sends `MCP_API_TOKEN`/`RAG_API_TOKEN`. Call this out in Trails follow-up. |
| `PUBLIC_BASE_URL` | `rag/config.py`, `rag/mcp_tools.py` `_image_public_url` | Confirmed: it only **prefixes** `{PUBLIC_BASE_URL}/images/{doc_id}/{image_id}` onto MCP JSON (`get_document_images`, `get_image` hint). It is **not** a CDN and does not make the GET public. Also used for Google OAuth redirects. |
| `POST /upload-media` | `rag/routes.py` | Confirmed: **no auth**, any file, `doc_id=uuid4()`, `save_original` only, **no** `index.json` row, **no** ingest, **no** metadata. Returns `{status, path, doc_id}`. |
| Catalog `GET /documents*` | `rag/routes.py` | Unauthenticated. `index.json` columns: `doc_id`, `filename`, `mime`, `sha256`, `uploaded_at`, `chunk_count`, `status`, `figure_*`. **No** licence, **no** `source_url`. |

### 3.4 Ingest pipeline today

Shared write path: `rag.ingest.ingest_bytes` from `POST /upload`, `POST /upload-pdf`, chunked `POST /upload/{id}/complete`, `POST /ingest-json`, MCP `ingest_text`.

```text
bytes → sha256 → skip_duplicate via catalog find_by_sha256
     → doc_id = uuid4()
     → originals/{doc_id}/
     → process_document (PDF text / txt / GeoJSON)
     → index_pdf_figures (pdf_figure clips + Qdrant points)          [PDF]
     → extract_pdf_images (skip tiny; skip figure xrefs) + OCR         [PDF]
     → processed/{doc_id}/ + index.json  (DocumentMeta.status=ready)
     → optional verification metadata onto extra + Qdrant
     → BGE-M3 text vectors; optional CLIP image_vector
```

**What reaches Qdrant**

- Chunks: `build_chunk_payload` — `source_url` **only if** `ingest_bytes(..., source_url=)` was passed. Today that happens solely on `POST /ingest-json` (`final_url`). `POST /upload` and chunked complete **do not** pass `source_url`.
- `pdf_image`: `build_image_payload` — **no** `source_url`. Extra keys come from `figure_payload_fields(verification_fields)`.
- `pdf_figure`: `build_figure_payload` — **no** `source_url`. `figure_payload_fields` **strips** `source_url`, `source`, `verification`, `verification_status`, `verified` (`FIGURE_EXCLUDED_KEYS` in `verification_meta.py`).

**`POST /upload` `metadata` form:** parsed JSON is run through `normalize_verification_metadata`. Unknown keys **400**. The approved provenance names (`license_id`, `attribution_text`, `content_type`, …) are **not** in `FLAT_PAYLOAD_KEYS` and would be rejected. `zdroj` (string or object) is allowed as a verification blob, not as `source_url`.

**Chunked upload:** `POST /upload/init` → `PUT .../chunks/{i}` → `POST .../complete` calls `_ingest_upload` **without** `metadata` or `source_url`. RAGloader’s chunked path therefore **cannot** attach licence even if catalog-upload-tools collected it.

**MCP:** no `add_image_record`. `verify_image_record` (`rag/figure_index.py`, `rag/mcp_tools.py`) updates figure `verification_status` / `verified_by` / notes / caption / summary and re-embeds text. It does not write licence.

**Duplicates:** same file sha256 returns `status: exists` and **does not** merge new provenance.

**Page scans:** `content_type` in the policy sense does not exist. Closest types: `source_type=pdf_figure` (clipped page regions, PNG on disk), `source_type=pdf_image` (embedded streams), lazy `page:{doc_id}:{page}` JPEGs under `pages/` (not a catalog image row).

### 3.5 ImageMeta / DocumentMeta today

`ImageMeta` (`rag/storage.py`): `image_id` (16 hex = `sha256[:16]`), `sha256`, `file_path`, `page_number`, `width`, `height`, `mime`, `ocr_text`, `ocr_confidence`, plus geo (`lat`/`lon`/`geo_refined`). Unknown keys fall into `extra`.

`DocumentMeta`: `filename`, `mime`, `sha256` (whole file), `uploaded_at`, `status` (ingest), paths, `images[]`, figure index counters, geo, `extra` (verification fields after ingest).

**None of the §2.1 provenance keys are first-class.** A catalog export of documents/images
therefore has zero `license_id` / `source_url` / `attribution_text`. The brief’s
“726 docs / 1659 images” is an operator catalog count, not stored in this git
repository; this ADR treats **~1659 existing on-disk images** as the backfill population.

### 3.6 Trails / geobjectsdb / mobile-api

Verified in `mkrejp/TrailsAssemblage` (commit `b4dfcea` on `main` when read):

- **Identity:** migration `008_rag_timer_recent_updates.sql` — `map_layers.source_key='rag-timer'`, unique `rag_timer_key` = `rag_doc_id || U+001F || COALESCE(rag_image_id,'')` on `layer_object_properties`. `POST /catalog/objects` upserts GPS + labels; **no URL, no thumbnail, no attribution** on that contract (`docs/catalog-rag-objects.md`).
- **Image locators:** migration `009_layer_object_images.sql` — table
  `public.layer_object_images` with `source_kind`
  `rag|commons|wikidata|url|upload`, `source_doc_id` / `source_image_id`,
  `source_url`, **`license` / `attribution` / `credit`** (not the approved
  names), optional `checksum`. **Bytes are not stored.** `ON DELETE CASCADE`
  from `layer_objects`; image edits must not bump `layer_objects.updated_at`.
- **Serving today:** Trails-relative `GET /catalog/objects/{id}/thumbnail` proxies Commons `Special:FilePath/{file}?width=480` or RAG `GET /images/{doc}/{id}` with a server-side token. In-process LRU (64) + optional `IMAGE_CACHE_DIR`. JSON may include `thumbnail_url` + `image.{license,attribution,credit,source_kind,source_page_url}` (`src/trails_assemblage/services/object_images.py`).
- **Web frontend (later than `docs/images-geo-objects.md` “no image on the map”):**
  shared helper `frontend/src/object-thumb.js` — `thumbSrc(item.thumbnail_url)`,
  `attributionText(image.attribution || image.credit, image.license)`, 48×48
  `<img alt="">`, placeholder `/catalog/images/placeholder`, `bindThumbError`.
  Used by `catalog-search-dialog.js` (search rows) and `recent-list.js`
  (latest-updates list). `recent-api.js` `popupHtml` prepends `popupThumbHtml`.
  **`recent-map.js` strips `thumbnail_url` and `image` when building GeoJSON
  `properties`**, so Leaflet popups call `popupHtml` without those fields and
  fall back to the placeholder. No `large_url`, no lightbox; list-item click
  focuses the marker.
- **Audit (Trails doc, 2026-11-22):** many rag-timer **buildings** have
  `images: []` on RAG, Commons filename only in Markdown
  (`nenahráváno v této dávce`), empty `rag_image_id`. Search used to advertise a
  RAG thumb URL for any MCP `image_id` and then 404.
- **mobile-api:** `TrailsAssemblage/mobile-api/` on host
  `https://api.animarium.ai`, same Railway PostGIS (**geobjectsdb**). Trails and
  owner-scoped `/v1/trails*` only. Docs: trail list/edit has **no catalog image
  field**; `TrailItem.thumbnail_url` is optional and **ignored until the Android
  UI uses it**. There is **no** `mkrejp/mobile-api` repository. Android UI lives
  in `TrailsAssemblage/mobile/` (Expo).

### 3.7 sftpgo / HF / backups

- `sftpgo-geo-servus`: `Dockerfile` `FROM drakkan/sftpgo:v2.7.6` + chown entrypoint. Not an image CDN; no provenance schema.
- `sftp-mirror-geo-server`: cron 19:45 Europe/Prague, copies newest `COMPLETE` desktop backup onto a **5 GB** `/mirror` volume (two generations). Restore docs live on the operator box, not in rag-timer.
- Hugging Face in **this** repository is **ONNX download** (FastEmbed BGE-M3, CLIP),
  not a media origin. HF as RAG/PostGIS **backup** is an HFHosts/operator practice,
  **not in git**. Treat HF as cold backup only.

### 3.8 Where metadata is dropped (trace)

```text
FotoImporter (operator)
  local fields: autor, obrazek_licence, zdroj, licence
  → not this repository; not Trails; not media
catalog-upload-tools (operator)
  POST /upload or chunked /upload/init → complete
  licence/title/source → local upload-result.json (often license=public)
  → HTTP never receives those keys (chunked complete has no metadata slot;
     multipart metadata would 400 on unknown keys)
RAGloader (sole live RAG writer, docs/AI_AGENTS.md)
  → ingest_bytes → DocumentMeta + Qdrant without licence
extracted PDF images / pdf_figure / page JPEGs
  → inherit nothing; parent also has no provenance
Trails POST /catalog/objects
  → (rag_doc_id, rag_image_id) only
Trails object_images_sync (Markdown Commons parse)
  → license/attribution/credit (different names); bytes still on Commons or RAG
```

`license=public` / Kramerius public **must not** be copied forward as `license_id`.

---

## 4. Target architecture

```text
FotoImporter / researchers / Kramerius / Commons
        │  provenance filled (TASL / PD years / permission URL)
        ▼
   licence check (RAGloader + TesterVerifier)
        │  refuse NC/ND/Copyright/Unknown/PD-US-only for hosting
        ▼
   MEDIA SERVICE  (canonical binaries + provenance + audit)
        │  original + thumb + large written; sha256 index
        │  license_status: intake → … → published | restricted
        ▼
   RAG ingester  (text/OCR/embeddings + copy of provenance + locators)
        │  no public GET of binaries
        ▼
   RAG reader MCP/catalog  (search; media URLs in payload)
        │
        ▼
   Trails layer_objects + layer_object_images
        │  rag_doc_id, rag_image_id, media_asset_id
        │  attribution_text; Trails-relative thumb/large proxy
        ▼
   web map / mobile-api TrailItem (attribution always)
```

**RAG is not the origin.** After cut-over, `GET /images/{doc_id}/{image_id}` on rag-timer either redirects to the media service (authenticated server-to-server) or is restricted to operator tokens for debug. Trails stops fetching RAG bytes for published images.

### 4.1 Stable IDs

Keep existing IDs; add one.

| ID | Format today | After |
| --- | --- | --- |
| `rag_doc_id` | UUID | Unchanged. RAG catalog / Qdrant `doc_id`. Trails `rag_doc_id`. |
| `rag_image_id` | 16 hex (`sha256[:16]`) or `page:{doc_id}:{n}` | Unchanged for existing rows. New hosted images still get a 16-hex ID **or** keep using `sha256[:16]` of the original so Trails keys stay stable. |
| `media_asset_id` | — | New UUID from the media service. Stored on RAG image/figure payload, Trails `layer_object_images`, and media metadata. |
| `sha256` | image bytes or whole PDF | **Original bitstream** on the media asset. Derivatives have their own digests in the derivative table, not as the asset’s `sha256`. |

**Dedup:** unique index on original `sha256`. A second upload of the same bytes returns the existing `media_asset_id` and **does not** clone files. Provenance is **not** silently overwritten on dedup (see open question Q1). Extracted PDF images that match an already-hosted Commons photo collapse to one asset; both `rag_doc_id`s may link to it.

**Do not replace** Trails `(rag_doc_id, rag_image_id)` as the geoobject key in this phase. Facades remain one object per image; buildings may have `rag_image_id=''` plus a `layer_object_images` primary that points at `media_asset_id`.

### 4.2 Where the media service runs (options)

Marek is the sole user; ingest compute should be **cold**. Trails home map is public and already always-on. Volume storage bills while stopped.

| Option | Bytes | Compute | Serve thumbs to the public map | Cost shape | Verdict |
| --- | --- | --- | --- | --- | --- |
| **A. Dedicated Railway volume + FastAPI, started with the cold ingester** | `/data/media` | Intake + batch only | Trails (warm) proxies and caches; 502 on cache miss if media is down | ~$0.15/GB/mo volume | **Phase 1:** Trails proxy + `IMAGE_CACHE_DIR` for thumb + large |
| **B. Same rag-timer ingester volume** | Mix Qdrant+PDFs+media on the ~39 GB tree | Ingester already cold | Reader would have to grow by every derivative or Trails keeps proxying a **stopped** origin | Avoid. Mixes embeddings lock with binary origin; reader volume already large | Rejected |
| **C. Object storage (Cloudflare R2 / Backblaze B2)** | Cheap persistent bytes, no Railway GB | Media worker cold; no CDN required (R2 has no egress fee) | Public object URLs **or** Trails proxy | Storage cents/GB-mo; no always-on CDN | **Recommended phase 2** if hosted originals exceed ~10 GB or Railway volume is painful. Not required for ~1659 images. |
| **D. Hugging Face dataset as origin** | Possible backup dump | Not an attribution/quarantine API | Hub URLs ignore `takedown` | Free-ish but wrong control plane | Backup only |
| **E. sftpgo as origin** | Desktop tar generations | SFTP, not HTTP image API | No | Backup only | Rejected as origin |

**Phase 1 recommendation:** new service repository or a tightly scoped module (follow-up),
Railway service **media** in the RAG project, volume at `/data/media`, **stopped** with
the ingester. Trails remains the **public HTTP facade**
(`/catalog/objects/{id}/thumbnail` and a new `/large`) and caches derivatives.
Media internal URL is private (`*.railway.internal`). No Cloudflare CDN, no
always-on extra container.

**Do not** put originals on the Trails volume (PostGIS host). Trails stores locators + cached derivatives only.

### 4.3 Media storage layout and naming

```text
/data/media/
  blobs/{sha256[0:2]}/{sha256}/
    original{ext}            # unmodified upload
    thumb-v{recipe}.webp     # 480px long-edge
    large-v{recipe}.jpg      # 1600px long-edge
  meta/{asset_id}.json       # provenance + derivative descriptors
  index/
    by-sha256.jsonl          # or sqlite/postgres — follow-up choice (Q2)
  audit.jsonl                # takedown/quarantine; never delete rows
  quarantine/                # optional move of published pointers; files stay under blobs/
```

Naming is content-addressed on the **original** digest so derivatives for the same photo are overwritten in place when the recipe version changes (`thumb-v2.webp` alongside `thumb-v1.webp` until sweep).

### 4.4 Media HTTP (proposed; not implemented)

Auth: service Bearer (`MEDIA_API_TOKEN`) for writes and original download. Public binary GET is **not** required in phase 1 (Trails proxies). If a public GET is added later, it still refuses hidden statuses (§2.2).

| Method | Path | Auth | Behaviour |
| --- | --- | --- | --- |
| `POST` | `/v1/assets` | Bearer | Multipart: file + provenance JSON. Sync derivatives. **201** only when `derivatives_status=ready`. Dedup by sha256 → **200** existing. |
| `GET` | `/v1/assets/{asset_id}` | Bearer or Trails server token | JSON: §4.5 image object plus operator fields (`original_url`, `sha256`, full provenance). |
| `GET` | `/v1/assets/{asset_id}/thumb` | Trails / Bearer | WebP bytes or **404/403** if hidden. Header `Link` / JSON sibling not required; JSON is the attribution channel. |
| `GET` | `/v1/assets/{asset_id}/large` | Trails / Bearer | JPEG bytes, same hide rules. |
| `GET` | `/v1/assets/{asset_id}/original` | Operator Bearer only | Original bytes; still hidden for quarantine/takedown/restricted. |
| `PATCH` | `/v1/assets/{asset_id}` | Bearer | Provenance / `license_status`. Takedown is append-only audit. |
| `POST` | `/v1/assets/{asset_id}/rederive` | Bearer | Rebuild thumb+large for a new `recipe_version`. |
| `POST` | `/v1/jobs/backfill` | Bearer | Batch from a manifest (RAG paths or sha256 list). |
| `GET` | `/v1/assets/by-sha256/{sha256}` | Bearer | Dedup lookup. |

Browser-facing JSON is the **frontend image object** in §4.5 (same field names on media `GET /v1/assets/{id}` and on Trails catalog payloads). `thumb_url` / `large_url` in the browser are Trails-relative (never a RAG host, never a token query param). Internal rag-timer payloads may store `media_asset_id` and let Trails resolve URLs.

Binary responses do not need to embed IPTC, but the follow-up **should** write `attribution_text` + `license_id` into WebP/JPEG XMP when cheap (Pillow). That does not replace JSON `attribution_text`.

### 4.5 Frontend image object and lightbox (Trails web)

Marek: **clicking a thumbnail opens the large version (lightbox), always with attribution shown.** This ADR defines the JSON the UI needs. It does **not** implement HTML/CSS/JS.

#### Consumers (no code here)

| Surface today (`mkrejp/TrailsAssemblage`) | Image behaviour today | After this contract |
| --- | --- | --- |
| Shared helper `frontend/src/object-thumb.js` | `thumbnail_url`; credit from `attribution`/`credit` + `license`; hardcoded 48×48; `alt=""`; placeholder + `onerror`; **no** `large_url` / lightbox | Read §4.5 fields (`thumb_url` with `thumbnail_url` alias, `attribution_text`, real `thumb_width`/`height`, non-empty `alt`); thumb click → lightbox |
| Catalogue search `frontend/src/catalog-search-dialog.js` | Uses `object-thumb.js`; credit under the row; thumb not clickable as zoom | Same; lightbox on thumb |
| Recent list `frontend/src/recent-list.js` | Same helper; row click focuses the map, does not open large | Keep focus-on-map on the **row**; **thumb** click → lightbox (stop propagation) |
| Leaflet popups `frontend/src/recent-map.js` + `recent-api.js` `popupHtml` | Intended 48×48 thumb in the popup, but GeoJSON `properties` **omit** `thumbnail_url` and `image` → placeholder only | Pass the §4.5 object through popup properties (or look up by `id`); popup shows thumb; thumb click → overlay lightbox (not a large JPEG inside Leaflet) |
| Trail detail | Label + catalog UUID; no image in `mobile/` screens | Same image object on the linked `layer_objects` row when the web trail UI shows it |
| `GET /catalog/search` | `thumbnail_url` RAG proxy when MCP had `image_id` (often 404) | Prefer stored `layer_object_images` / media object; no RAG origin |
| GeoServer WFS/WMS | Geometry only | Unchanged (out of scope) |
| `mobile-api` / `mobile/` Android | Optional `thumbnail_url` ignored until the app uses it | Same JSON fields; native lightbox later |

The **browser talks only to Trails** (`GET /catalog/objects`, `{id}`, `/recent`, `/catalog/search`, and binary `GET /catalog/objects/{id}/thumbnail` + **`/large`**). It never calls rag-timer or the cold media service.

#### Image object (embed on every catalog JSON that can show a photo)

One object per primary photo. Field names are snake_case and **match the provenance block** where they overlap. Dimensions are the **actual** derivative pixels (after no-upscale), not the 480/1600 caps.

```json
{
  "media_asset_id": "3d1c0e2a-7c4b-4f11-9c0e-0b1a2c3d4e5f",
  "license_id": "CC-BY-SA-4.0",
  "license_status": "published",
  "title": "Žatec, čp. 753",
  "alt": "Žatec, čp. 753 — SchiDD, CC BY-SA 4.0, via Wikimedia Commons",
  "attribution_text": "SchiDD, CC BY-SA 4.0, via Wikimedia Commons",
  "source_url": "https://commons.wikimedia.org/wiki/File:Saaz-Nr753-1257.jpg",
  "thumb_url": "/catalog/objects/11111111-2222-4333-8444-555555555555/thumbnail",
  "thumb_width": 480,
  "thumb_height": 318,
  "large_url": "/catalog/objects/11111111-2222-4333-8444-555555555555/large",
  "large_width": 1600,
  "large_height": 1061
}
```

| Field | Required for hosted/published | UI use |
| --- | --- | --- |
| `thumb_url` | yes (null if hidden/restricted) | `<img src>` in search, popup, lists. Trails-relative. |
| `large_url` | yes (null if hidden/restricted) | Lightbox `<img src>`. **Not** the original. Trails-relative `…/large`. |
| `attribution_text` | **always** (non-empty string; may be `"licence neuvedena"` only while `Unknown` and URLs are null) | Caption under the thumb **and** under the lightbox image. Never hover-only. Commons = TASL. |
| `thumb_width`, `thumb_height` | yes when `thumb_url` is set | `width`/`height` attrs, aspect-ratio, CLS. Integers, pixels of the WebP. |
| `large_width`, `large_height` | yes when `large_url` is set | Lightbox layout before the JPEG loads. |
| `license_id` | always | Optional badge (`CC-BY-SA-4.0`, `PD`, …). Not a substitute for `attribution_text`. |
| `title` | recommended; `""` allowed | Lightbox heading / `img` `title`. From provenance `title`, else object `hlavni_prvek`. |
| `alt` | yes | `<img alt>`. Default composition: `"{title} — {attribution_text}"` when title is set, else `attribution_text`. Must not be empty if a thumb is shown. |
| `media_asset_id` | yes when hosted | Debug / future gallery; not shown. |
| `license_status` | yes | If not `published`, URLs are null; UI shows attribution + `source_url` link only (no lightbox). |
| `source_url` | when known | “Zdroj” link in the lightbox (Commons File: page, not a binary). |

**Do not** send `original_url` to the browser in phase 1 (Q13). **Do not** send RAG `/images/{doc}/{id}` or bearer tokens. **Do not** use Trails’ current `image.license` / `image.attribution` / `image.credit` names on new payloads (alias during migration only).

List endpoints may return this object as `image` next to `thumbnail_url` for one release, then drop the old key. Prefer `thumb_url` as the canonical name (aligned with this ADR). `GET /catalog/objects/{id}` **must** include the full object so a map popup can open the lightbox **without a second metadata round-trip**.

#### Lightbox behaviour (contract for a later frontend change)

1. Thumb is visible wherever the object is shown (search hit, **map popup**, recent card, trail item) when `thumb_url` is non-null.
2. Click (or keyboard activate) on the thumb opens a lightbox whose image `src` is `large_url`.
3. **`attribution_text` is visible in the lightbox without hover or extra click** (caption bar under the large image). Same string as under the thumb.
4. If `large_url` is null, **do not** open a lightbox; the thumb is not clickable as a zoom (restricted / takedown / missing derivatives). Attribution + `source_url` may still show.
5. Lightbox does not fetch the original, does not hotlink Commons `Special:FilePath` at full resolution, and does not use RAG.
6. `alt` stays on both `<img>` tags. `title` may be the lightbox heading.
7. Thumb `onerror` → existing Trails placeholder (`/catalog/images/placeholder` / empty CSS box), never a broken-image icon (`docs/images-geo-objects.md`).
8. Multiple gallery images (later): same object shape in an array; lightbox can step through `large_url`s; each frame has its own `attribution_text` (do not reuse the primary’s TASL on a different work).

This is a **data contract**. Markup, CSS, and Leaflet wiring stay in a Trails frontend follow-up (extend `object-thumb.js`; do not add a second thumb helper). This rag-timer PR ships no UI.

---

## 5. Derivatives (Marek: generate at intake, not on demand)

For **every hosted image** the media server stores three renditions: **original** (tracked, operator-only HTTP), **thumbnail**, **large**. Restricted/link-only assets store **metadata + `source_url` only** (no thumb/large generated, no original kept except a hash if the operator uploaded bytes by mistake — those bytes go to quarantine, not `published`).

### 5.1 Size presets (`recipe_version = 1`)

| Role | Long-edge max | Format | Quality | No upscale | Typical bytes | Why this number |
| --- | --- | --- | --- | --- | --- | --- |
| `original` | as uploaded | as uploaded | — | — | 0.2–8 MiB | Evidence + future re-derive |
| `thumb` | **480 px** | **WebP** | q **80** | yes | 15–60 KiB | Matches Trails `COMMONS_THUMB_WIDTH` default 480 (map pins, search, mobile list) |
| `large` | **1600 px** | **JPEG** (progressive, 4:2:0) | q **85** | yes | 120–400 KiB | Phone/detail; below 2K to save volume; still sharp on a 6" screen |

If the original’s long edge is already ≤ the target, **re-encode** to the target format at the original pixel size (still write `thumb-v1.webp` / `large-v1.jpg`) so URLs are stable. Do not invent pixels.

Colour: convert to sRGB. Strip GPS EXIF from derivatives (keep on original only). Keep ICC when present.

`modifications` values:

| Rendition | `modifications` |
| --- | --- |
| original | `none` (or empty) |
| thumb | `resized; recipe=v1; long_edge=480; format=webp` |
| large | `resized; recipe=v1; long_edge=1600; format=jpeg` |

Derivatives **inherit** the full provenance block from the original except `modifications` and the per-file digest recorded on the derivative row. `license_id` does **not** change.

### 5.2 CC BY-SA (and CC BY) on derivatives

- A resize is an **adaptation**. Under **CC BY-SA 4.0**, the derivative **must** stay `CC-BY-SA-4.0` (share-alike). Do not relicense thumbs as CC0 or drop SA.
- **TASL** travels on every size: `attribution_text` is identical for thumb, large, and JSON.
- **CC BY** (not SA) still requires attribution on every size; licence stays `CC-BY-4.0`.
- **CC0 / PD:** attribution is still shown when `attribution_text` is non-empty (policy: always return the field; it may be `"Public domain"`).
- Do not generate derivatives for `restricted` / `Copyright` / `Unknown` even if someone POSTs bytes.

### 5.3 When they are generated

**At intake, automatically, inside the media upload pipeline.** Clients (FotoImporter, RAGloader, Trails, mobile, browsers) never call a “render thumb” endpoint to create files.

**Default: synchronous in `POST /v1/assets`.** The handler does not return 201 until both derivative files exist and have been hashed. RAGloader must not ingest the image into Qdrant until that 201/200.

**Overflow (same process, not a separate always-on worker):** if decode/resize exceeds **20 seconds** or original is **> 25 MiB** or pixels **> 40 MPx** (magazine `page_scan` / TIFF), the handler:

1. Stores original + metadata with `license_status=intake`, `derivatives_status=pending`.
2. Returns **202** `{asset_id, derivatives_status: pending}`.
3. Runs Pillow/libvips in a **process-local queue** (thread or `FastAPI` `BackgroundTasks` is too easy to lose on Railway SIGTERM — prefer a **directory queue** `jobs/pending/{asset_id}.json` drained by the same service loop).
4. Client polls `GET /v1/assets/{id}` until `derivatives_status=ready` or `failed`.

There is **no Redis/Celery**. Cost and always-on complexity are not justified for one operator.

### 5.4 Library

| Library | Use |
| --- | --- |
| **Pillow** (already in rag-timer `Dockerfile` via `pillow`) | **Default.** JPEG/PNG/WebP/GIF. Known in this codebase (`tests/test_image_http.py` already imports `PIL.Image`). |
| **pyvips / libvips** | Optional later for huge TIFF/JP2 `page_scan`s. Extra `apt` dependency; do not add in the first media image unless Pillow OOMs on real Kramerius scans (open question Q3). |
| ImageMagick / ffmpeg | Do not add. |

Resize: LANCZOS, long-edge cap, keep aspect ratio. Animated GIF: first frame only (document in `modifications`).

### 5.5 Idempotency (sha256)

1. Compute `sha256` of the uploaded original **before** any resize.
2. `GET by-sha256` / unique constraint: if an asset exists with that digest **and** `derivatives_status=ready` **and** `recipe_version` is current → return it (**200**), do not write blobs again.
3. If digest exists but derivatives are `pending`/`failed` → **retry resize**, do not create a second `media_asset_id`.
4. If digest exists with an **older** `recipe_version` → do not block upload; return the asset and enqueue `rederive` (published thumbs stay until vN files are ready).
5. RAG `find_by_sha256` today is **whole-document** only and skips re-ingest. Media dedup is **per original image bytes**, independent of `doc_id`.

### 5.6 Failure handling

The asset **must not** leave `license_status=intake` until:

- original is on disk,
- `thumb` and `large` exist,
- both derivative sha256s are recorded,
- `derivatives_status=ready`.

On failure:

- Keep original bytes (needed for retry).
- `derivatives_status=failed`, `derivatives_error` (short, no secrets).
- Do **not** set `published`.
- Do **not** call rag-timer ingest for that image.
- Do **not** upsert Trails `layer_object_images` as `source_kind=rag` / media.
- A startup sweeper retries `pending`/`failed` younger than N days.

Railway deploy kill: directory queue + original on volume is the recovery. No silent 201.

### 5.7 Re-generation when presets change

`recipe_version` is an integer on the asset (start at 1). Changing 480/1600/quality in config **without** bumping the version is forbidden in code comments; a preset change **is** a version bump.

`POST /v1/assets/{id}/rederive` and the batch job:

1. Write `thumb-v{n+1}.webp` and `large-v{n+1}.jpg` next to vN.
2. Point metadata at vN+1 only when both files verify.
3. Keep vN until a later sweep (takedown still hides all versions).
4. Never delete originals.

Trails cache keys must include `recipe_version` (today’s LRU key is `kind:locator:width` — extend with recipe).

### 5.8 Batch job for ~1659 existing RAG images

Input (read-only on rag-timer volume or a copied manifest; **delete nothing**):

- `processed/{doc_id}/images/{image_id}.*` (`ImageMeta` + unlisted figure PNGs via `locate_image_file`)
- Optionally skip lazy `pages/*.jpg` until Marek says page renders are first-class media (`content_type=page_scan`) — default **include figure PNGs, exclude on-demand page JPEGs** (they can be re-rendered; they are not originals)

For each file:

1. Hash bytes; skip if already in media with ready derivatives.
2. Build provenance from parent `DocumentMeta.extra` if any; else `license_id=Unknown`, `license_status=restricted`, **do not publish binaries** (still store original in quarantine/intake for operator completion — open question Q4).
3. Inherit parent provenance onto extracted images and figure clips (§6.4).
4. Generate thumb+large when hosting is allowed; else metadata-only.
5. Write a mapping file `rag_doc_id, rag_image_id, sha256, media_asset_id` for the RAG follow-up agent to `set_payload`.

Default `--dry-run`. `--apply` is Marek-only. Never `DELETE` RAG files in this job (RAG volume stays until a later, approved cut-over).

### 5.9 Hide all sizes on quarantine / takedown / restricted

A single `license_status` gates **original, thumb, large, and every `recipe_version`**. Trails cache **must** be purged on status change (call Trails or short Cache-Control). JSON may return `attribution_text` + `source_url` + `license_status` without URLs (`thumb_url`/`large_url` = `null`).

---

## 6. End-to-end ingest redesign

### 6.1 Target flow (happy path, hosted photo)

1. **Intake:** operator/FotoImporter gathers bytes + provenance (TASL for Commons).
2. **Licence check:** map local `licence` / Kramerius public → `license_id` + `pd_status`. If not hostable → media metadata `restricted`, RAG text may keep `quote` + `source_url`, **no** binaries.
3. **POST media `/v1/assets`:** original stored, **sync thumb+large**, `license_status` at least `metadata_complete` (or `intake` until humans fill years).
4. **RAG ingest:** RAGloader uploads the **document** (PDF/md) via existing
   `/upload` **after** media returns IDs for each image. New ingest fields
   (follow-up): copy of provenance on `DocumentMeta.extra` and on every
   chunk/image/figure payload; `media_asset_id` on image/figure points.
   Extracted PDF images: if sha256 already in media, **link**, do not write a
   second original to `/data/rag/processed/.../images` (RAG may keep a local
   copy only while reader CLIP still needs a file — see Q5).
5. **Trails:** `POST /catalog/objects` unchanged for GPS; `layer_object_images` upsert with `media_asset_id`, approved field names, Trails proxy pointing at **media** not RAG.

### 6.2 FotoImporter

Operator agent (`docs/AI_AGENTS.md`: Commons, ÖNB AKON CC0; does not upload books). Changes:

- Stop treating `licence=public` / free-text `obrazek_licence` as hostable.
- Emit the §2.1 block (Commons TASL → `attribution_text`, SPDX `license_id`, `source_url` = File: page, `source_name=Wikimedia Commons` or `ÖNB AKON`).
- Upload **to media first**, not to rag-timer `/upload-media`.
- Hand `media_asset_id` + provenance to RAGloader’s queue (`rag-queue/inbox/` on the operator box).

### 6.3 catalog-upload-tools + RAGloader

- Chunked `/upload/init…complete` must grow a **metadata slot** (JSON on complete, or `POST /upload/init` body). Until then, **do not** use chunked upload for licensed works.
- Stop writing licence only to `upload-result.json`.
- Stop hardcoding `license=public`.
- RAGloader remains the **only live RAG writer** (`docs/AI_AGENTS.md`). It becomes a **media client** as well: no RAG image ingest without `media_asset_id` and `derivatives_status=ready` (hosted) or explicit `restricted` (link-only).
- `PATCH /documents/{id}/metadata` follow-up: allow the provenance keys (today they 400 as unknown). Keep them distinct from Czech `verification_status`.

### 6.4 rag-timer ingest / MCP

Follow-up (after approval), not this PR:

- Accept provenance on `/upload`, `/upload-pdf`, chunked complete, `ingest_text`.
- Persist on `DocumentMeta` (first-class or `extra` with a frozen key set) **and** Qdrant chunk/image/figure payloads. **Stop excluding `source_url` from figures** for provenance (today `FIGURE_EXCLUDED_KEYS`); keep excluding Czech document `verification[]` from figures if still desired.
- **Extracted `pdf_image` and `pdf_figure` inherit the parent document’s provenance**, then may override `content_type` (`image` vs `page_scan`) and `modifications` (e.g. `extracted from PDF; cropped`). Child `sha256` is the **image** digest; parent PDF digest stays on document `sha256`.
- `page:{doc_id}:{page}` renders inherit the parent too; they are derivatives of the PDF, not new works. Prefer not to publish page JPEGs until the parent is `published`.
- New MCP: `add_image_record` (link `media_asset_id` + provenance). Extend `verify_image_record` **or** add `set_image_provenance` so licence is not confused with figure `verification_status`.
- `GET /images/...`: deprecate as public-ish URL; operator-only; Trails uses media.
- **Close `POST /upload-media`:** require the same auth as `/upload` **and** provenance, or remove the route and return 410 with a pointer to media. Existing unindexed `originals/{uuid}/` files are swept into media as `intake`/`Unknown` (no delete).

### 6.5 TrailsAssemblage + mobile-api

Follow-up in that repository (after this ADR):

- Align `layer_object_images` columns with §2.1 (`license_id`, `attribution_text`, …). Today: `license`, `attribution`, `credit` — **mismatch**.
- Store `media_asset_id`. Proxy thumb **and large** from media. Keep Trails-relative URLs. Add public `GET /catalog/objects/{id}/large` next to today’s `/thumbnail`.
- Catalog JSON (`GET /catalog/objects`, `{id}`, `/recent`, search, trail items) embeds the §4.5 image object (`thumb_url`, `large_url`, `attribution_text`, `thumb_width`/`thumb_height`, `large_width`/`large_height`, `license_id`, `alt`, `title`). Null URLs when restricted.
- **Frontend consumer (follow-up, not this PR):** Trails web — `object-thumb.js`, `catalog-search-dialog.js`, `recent-list.js`, `recent-map.js` / `recent-api.js` popups. Thumb click → lightbox of `large_url` with `attribution_text` always visible. Fix GeoJSON property strip so popups receive `thumb_url`/`image`. No rag-timer UI work.
- Purge `IMAGE_CACHE_DIR` on takedown (thumb **and** large cache keys).
- **mobile-api:** once Trails `TrailItem` carries the same image object, Android can show thumb + attribution. Do not point the app at RAG. Do not add a second schema. Native lightbox is optional later.

### 6.6 Hugging Face / sftpgo

- After media exists: include `/data/media` (or R2 snapshot) in ControlApps backup; HF dataset as **encrypted/cold dump**, not live origin.
- sftpgo desktop backup: add a `media/` tree next to current `media && sha256sum` notes in operator RESTORE.md (operator box). Mirror 5 GB volume **cannot** hold originals+RAG; do not pretend it can.

---

## 7. Cost estimate (sole user, cold compute)

Railway volume list price used in this repository: **~$0.15/GB/month**, billed
while the service is stopped (`deploy/railway/README.md`).

Assumptions for **~1659 existing images** (operator count; not in git): mixed postcard/PDF extracts. If mean original is **400 KiB** (many JPEG extracts are smaller; some scans larger):

| Layer | Per image | ×1659 | Notes |
| --- | --- | --- | --- |
| Original | ~400 KiB | ~0.65 GiB | |
| Thumb WebP 480 | ~35 KiB | ~0.06 GiB | |
| Large JPEG 1600 | ~220 KiB | ~0.35 GiB | |
| **Total binaries** | | **~1.1 GiB** | Metadata negligible |
| Volume $ | | **~$0.17/mo** | Dwarfed by the **~39 GB RAG tree (~$6/mo)** already attached |

Growth: 5 000 hosted photos ≈ **~3–4 GiB** ≈ **~$0.5/mo**. Magazine `page_scan` TIFFs would dominate if originals are kept; then prefer **R2** for originals and keep only thumb+large on a small Railway volume (Q6).

**Always-on CDN:** not justified. Trails already warms the map. Cache thumb+large on Trails (`IMAGE_CACHE_DIR`). Media CPU runs during intake only.

**Do not** duplicate the 39 GB RAG originals onto a second 39 GB media volume. Media holds **images + optional extracted clips**, not a second copy of every book PDF, unless Marek later wants PDFs as `content_type=book` originals (Q7).

---

## 8. Migration path (existing 1659 + `/upload-media`)

Ordered, each step Marek-gated. **Delete nothing.**

1. **Approve this ADR.**
2. Follow-up: media service + volume (empty).
3. Backfill job `--dry-run` then `--apply` from RAG `processed/**/images` (§5.8). Unhosted → `restricted`.
4. Sweep `originals/*` that are **not** in `index.json` (`/upload-media` orphans) into media `intake` / `Unknown`.
5. RAG follow-up: write `media_asset_id` + provenance onto payloads; extend upload metadata; auth `/upload-media`.
6. Trails follow-up: field rename + media proxy + `large` + cache purge; stop advertising RAG `/images` as browser URLs.
7. Only after TesterVerifier GO: rag-timer `GET /images` operator-only; Trails uses media. **Leave RAG files on disk** until a later ADR (reader CLIP / OCR still reads local paths today).

Rollback: Trails can keep Commons FilePath and RAG proxy; media is additive.

---

## 9. Field-name mismatches (must fix in follow-ups)

| Current | Where | Approved name |
| --- | --- | --- |
| `licence` / `obrazek_licence` / `license=public` | FotoImporter, upload-result.json, Kramerius | `license_id` + `license_status`; public ≠ licence |
| `autor` | FotoImporter | `author` |
| `zdroj` | FotoImporter; RAG `verification_meta.py` (string/object) | `source_url` / `source_name` — keep `zdroj` as verification-only, do not overload |
| `DocumentMeta.status` | rag-timer catalog | stays ingest-only; **not** `license_status` |
| `doc_status` | RAG lifecycle | stays |
| figure `verification_status` | `unverified/verified/…` | stays; **not** `license_status` |
| Trails `license`, `attribution`, `credit` | `layer_object_images` + `object-thumb.js` `attributionText` | `license_id`, `attribution_text`; drop or alias `credit` |
| Trails / frontend `thumbnail_url` | catalog JSON, `thumbSrc` | `thumb_url` (keep `thumbnail_url` alias for one release) |
| Trails `source_kind` | rag/commons/… | keep as Trails locator; not a provenance substitute |
| `PUBLIC_BASE_URL` “public” image URL | MCP JSON | replace with media/Trails URLs + `attribution_text` |
| `sha256` on `pdf_figure` payload | document digest in `build_figure_payload` (`sha256` = `doc_sha256`) vs `image_sha256` | provenance `sha256` = **image original**; do not copy PDF digest into the media asset |

---

## 10. Out of scope (this ADR / this PR)

- Implementing the media service, RAG schema, or Trails migration 010+.
- **Any frontend / Leaflet / lightbox / CSS / Android UI** (contract only, §4.5).
- Railway provisioning, DNS, tokens.
- Merging, deleting RAG files, rewriting Qdrant.
- Choosing a new GitHub repository name for the media service (Q8).
- GeoServer WMS photos (`docs/images-geo-objects.md` already defers this).

---

## 11. Open questions for Marek

**Q1. Dedup provenance conflict.** Same `sha256` uploaded twice with different `author` / `license_id`. Keep first and reject, or require explicit overwrite with audit?

**Q2. Media metadata store.** JSON files on the volume (fits cold Railway, matches rag-timer `index.json`) vs SQLite on the volume vs a small table on geobjectsdb. Recommendation: **SQLite or JSON+sha256 index on the media volume** so PostGIS is not the binary catalog. Confirm.

**Q3. Pillow vs libvips for Kramerius page scans.** Start Pillow-only?

**Q4. Backfill of 1659 with empty licence.** Store originals as `restricted`/`Unknown` (no public thumb) until FotoImporter/RAGloader fills TASL, or skip files with no parent provenance?

**Q5. Keep extracted images on the RAG volume** for CLIP/`get_image` until reader can pull from media? Recommendation: **keep files during dual-run**; do not delete.

**Q6. When to move originals to R2.** Recommendation: only if originals exceed ~10 GiB or Railway volume resize is annoying.

**Q7. Host book PDFs on media** (`content_type=book`) or leave PDFs on RAG `originals/` and only move **images**? Recommendation: **images + page_scan clips on media; PDFs stay on RAG** until a later ADR (39 GB copy is expensive).

**Q8. Media code location.** New `mkrejp/geo-media` repository vs a `media/` tree in
rag-timer vs inside Trails. Recommendation: **new small repository**, so Trails/RAG
deploys stay independent and the cold service can stop without stopping Trails.

**Q9. Public thumbs without Trails.** Should media ever serve anonymous `GET /thumb` (home map if Trails is down), or is Trails-only facade enough?

**Q10. `verified_by` overlap.** One field for licence signer and RAG fact-check signer, or `license_verified_by` despite the approved block? Recommendation: **keep `verified_by` as approved** and record the agent in audit.jsonl with `check=licence`.

**Q11. Trails `GET /images` token.** After rag-timer PR #53, `MCP_READ_TOKEN` cannot fetch binaries. Use a dedicated `MEDIA_READ_TOKEN` / Trails→media, and stop using RAG as the proxy backend?

**Q12. Recipe: WebP thumb in all clients.** Android/old WebView WebP is fine in 2026; confirm no need for JPEG thumb fallback.

**Q13. Original download for published CC BY-SA.** Operator-only (recommended, smaller legal surface) vs public original URL with attribution?

**Q14. `alt` language.** Default `"{title} — {attribution_text}"` (often Czech title + English Commons TASL). Prefer Czech `alt` composed by RAGloader, or keep this mechanical default?

**Q15. Map popup vs lightbox.** Confirm the popup shows only the **thumb** (click → overlay lightbox) rather than loading `large_url` inside the Leaflet popup (large JPEG would be heavy on mobile). Recommendation: **thumb in popup, large in lightbox**. Today popups are 48×48 placeholders because `recent-map.js` drops image fields from GeoJSON properties.

---

## 12. Follow-up agents (after Marek’s word)

1. **Media implementer** — service, layout, Pillow pipeline, auth, audit, batch backfill `--dry-run`.
2. **RAG schema/API** — provenance on ingest/MCP, inherit to extracts, close `/upload-media`, payload `media_asset_id`.
3. **Trails/mobile** — `layer_object_images` field alignment, large URL, media proxy, cache purge, catalog JSON §4.5 image object, TrailItem attribution.
4. **Trails frontend (later)** — extend `frontend/src/object-thumb.js`; search / recent list / **map popups** (`recent-map.js` must pass image fields into GeoJSON): thumb click → lightbox of `large_url` with visible `attribution_text`. No work in this rag-timer PR.
5. **TesterVerifier** — licence_status vs doc_status clash tests; hide binaries; CC BY-SA TASL on JSON; lightbox fields present; dedup; takedown audit.
6. **ControlApps** — backup `/data/media` (or R2); do not merge until GO.

---

## 13. References (code)

- `rag/storage.py` — `ImageMeta`, `DocumentMeta`, `/data/rag` layout, `save_original`, `find_by_sha256`
- `rag/routes.py` — `/upload`, chunked upload, `/upload-media`, `/images/{doc_id}/{image_id}`
- `rag/auth.py` — `require_embed_access`, `MCP_READ_TOKEN` vs HTTP images
- `rag/ingest.py`, `rag/payloads.py`, `rag/pdf_images.py`, `rag/pdf_figures.py`, `rag/image_delivery.py`, `rag/mcp_tools.py`, `rag/figure_index.py` `verify_image_record`
- `rag/verification_meta.py` — `status` vs `doc_status` vs figure verification; `FIGURE_EXCLUDED_KEYS`
- `docs/ARCHITECTURE.md`, `docs/SPLIT_DEPLOY.md`, `docs/AI_AGENTS.md`, `deploy/railway/README.md`
- Trails: `migrations/008_rag_timer_recent_updates.sql`, `migrations/009_layer_object_images.sql`, `docs/catalog-rag-objects.md`, `docs/images-geo-objects.md`, `src/trails_assemblage/services/object_images.py`, `importers/object_images_sync.py`, `frontend/src/object-thumb.js`, `catalog-search-dialog.js`, `recent-list.js`, `recent-map.js`, `recent-api.js`, `mobile-api/README.md`, `mobile/` Expo app
- `mkrejp/sftpgo-geo-servus`, `mkrejp/sftp-mirror-geo-server`
