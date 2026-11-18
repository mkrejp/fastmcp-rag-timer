# Reader / ingester split deploy

One repository, two roles. Either role can run on **Railway, Coolify, or a raw VPS**.

| Role | `APP_MODE` | Typical RAM | DNS (example) |
| --- | --- | --- | --- |
| Reader (ČTENÍ) | `reader` | ~0.5–1.5 GB (Novita query) | `art-nouveau-rag.animarium.ai` |
| Ingester (ZÁPIS) | `ingester` | ~1.5–3 GB (Novita ingest + OCR) | `art-nouveau-rag-ingest.animarium.ai` |
| Monolith | `all` (default) | as before | existing hosts |

Aliases: `APP_MODE` or `RAG_APP_MODE`. Shortcuts: `readonly`→reader, `ingest`→ingester.

## Decided topology (2026-11-18)

**Warm reader + cold ingester.**

| Piece | Where (now) | State | Notes |
| --- | --- | --- | --- |
| Reader | **Railway** | Always running (warm) | Public read/MCP; ~4 GB+ plan with Novita query; custom domain when cut over |
| Ingester | **Railway** | **Stopped** most of the time (cold) | Start only to ingest → publish → apply on reader → stop again; volume kept |
| Hetzner / Coolify | — | **Deferred** | No VPS, no Coolify apps, no Hostinger DNS until Marek calls the system stable / release-ready |

Coolify/VPS contract below stays valid for later migration; **do not** provision it yet.

---

## Behaviour by mode

**Reader**

- Forces write enforcement (HTTP 403 on mutating routes; write MCP tools not registered).
- Defaults `EMBEDDING_PROVIDER_QUERY` to `novita` when unset (no local BGE-M3).
- Prefer `IMAGE_EMBEDDING_ENABLED=false`.
- Keeps serving while the ingester is stopped.

**Ingester**

- Defaults role to ZÁPIS+ČTENÍ when `RAG_NODE_ROLE` unset.
- Runs upload/ingest/OCR; builds incremental releases via `scripts/rag-publish.py`.

**All**

- Historical single-process behaviour (`main.py` / root `Dockerfile`).

## Images

| Mode | Dockerfile | Build context | Entrypoint / CMD | Listen port | Health |
| --- | --- | --- | --- | --- | --- |
| Reader | `Dockerfile.reader` | repository root (`.`) | `python reader_main.py` | `8080` (`PORT` override) | `GET /health` |
| Ingester | `Dockerfile.ingester` | repository root (`.`) | `python ingester_main.py` | `8080` (`PORT` override) | `GET /health` |
| Monolith | `Dockerfile` | repository root (`.`) | `python main.py` | `8080` (`PORT` override) | `GET /health` |

Examples: `deploy/vps/docker-compose.yml`, `deploy/railway/README.md`.

Orchestrator healthcheck: path `/health`, port matching the container listen port (default **8080**). Images do not bake a Docker `HEALTHCHECK`.

---

## Railway (current priority)

Same images and env contract as Coolify; two Railway services, **two volumes** (never shared).

### Services

| Service | Image | `APP_MODE` | Domain (when configured) | Runtime |
| --- | --- | --- | --- | --- |
| Reader | `Dockerfile.reader` | `reader` | `art-nouveau-rag.animarium.ai` (later) / Railway URL now | Always on |
| Ingester | `Dockerfile.ingester` | `ingester` | `art-nouveau-rag-ingest.animarium.ai` (optional) / Railway URL | Start on demand, then stop |

Volume each: mount at `/data`, `RAG_DATA_ROOT=/data/rag`. Observed size ~**39 GB** per full copy → plan **≥ 50–60 GB** volume capacity on the reader (smallest sensible: enough for current seed + ~20% headroom; do not shrink below used bytes).

Memory: reader **≥ 4 GB** with Novita + CLIP off (2 GB previously OOMed under load). Ingester **≥ 4 GB** while running (Novita + Tesseract); no need to keep it allocated when stopped.

### (a) Cold ingester: start / stop / where it lives

**Recommendation: keep the cold ingester on Railway** (volume attached, service stopped). That matches Marek’s cost goal and avoids a second always-on host. CoolifyHoster does **not** need to manage this unless Marek asks.

Operator loop (Railway dashboard or API — whoever Marek designates):

1. **Start** the ingester service; wait until `GET /health` on the ingest URL returns `status=ok`, `app_mode=ingester`.
2. Perform uploads / OCR / ingest against the **ingester** URL only (Bearer `MCP_API_TOKEN` / `RAG_API_TOKEN`).
3. **Live push** to the reader (post-ingest hook or `scripts/rag-diff-push.py`) when `RAG_SYNC_URL` + `RAG_SYNC_TOKEN` are set. Fall back to CLI stop-apply only for DR (see below).
4. Confirm reader `/health` and a sample search.
5. **Stop** the ingester service again; leave its volume attached (billed storage only).

While cold: clients must not send writes to the reader (they get 403). Writes to the ingest hostname fail at the edge (connection refused / Railway offline) until the service is started — that is expected.

### (b) Incremental publish (ingester → reader) in practice

Volumes are independent. There is **no** shared live Qdrant between services. Vectors travel with each publish (reader does **not** re-embed).

#### Happy path: live diff push (Option A)

After a successful ingest batch, the **ingester** pushes a delta to the warm **reader** over an authenticated sync API. Apply runs **in-process** on the reader using the already-open `QdrantClient` (never open a second `QdrantClient(path=)` beside the app).

| Piece | Detail |
| --- | --- |
| Chunk target | **32 MiB** (soft max 64 MiB, hard reject > 96 MiB) |
| Staging | Reader writes under `.sync/incoming/<release_id>/` |
| Watermark | Reader `release.json` (`rag.publish.release_state`) |
| Diff | `build_file_diff` + `catalog_fingerprint` |
| Auth | Bearer **`RAG_SYNC_TOKEN`** (separate from MCP / `RAG_API_TOKEN`) |
| Triggers | Post-ingest hook on ingester; optional `scripts/rag-diff-push.py`; cron as future safety-net |
| Soft rollback | Before apply, snapshot touched docs under `.sync/rollback/<release_id>/`; `POST /sync/v1/rollback/{release_id}` |

**Reader sync routes** (only useful when `APP_MODE=reader` + token set):

- `GET /sync/v1/watermark`
- `GET /sync/v1/catalog` (`doc_id→sha256`)
- `POST /sync/v1/push/begin` (manifest; `parent_release_id` must match watermark)
- `PUT /sync/v1/push/chunk?release_id=…` (idempotent `(release_id, stage, seq)`)
- `POST /sync/v1/push/commit` / `abort`
- `GET /sync/v1/push/status`

Normal write routes (`/upload`, `/ingest-json`, document mutate, …) stay **403** on the reader. Sync is excluded from that write guard and uses `RAG_SYNC_TOKEN` only.

**Env names (values never in docs/chat):**

| Name | Where | Role |
| --- | --- | --- |
| `RAG_SYNC_TOKEN` | Reader **and** ingester | Shared Bearer for `/sync/v1/*` |
| `RAG_SYNC_URL` | Ingester | Reader public origin (e.g. `https://reader.example.com`) |
| `RAG_SYNC_PUSH_ENABLED` | Ingester | Default on when URL+token set; set `0` to disable auto-push |
| `RAG_DATA_ROOT` | Both | Default `/data/rag` |
| `MCP_API_TOKEN` / `RAG_API_TOKEN` | Ingester (and reader MCP) | **Not** used for sync |
| `BGE_M3_NOVITA_API_KEY` (or fallbacks) | Ingester | Ingest embeddings while ingester runs |

Operator loop with live push:

1. **Start** ingester → ingest against the ingest URL.
2. On success, the post-ingest hook (or `RAG_SKIP_DEFAULT_APP=1 python scripts/rag-diff-push.py --rag-root "$RAG_DATA_ROOT"`) pushes to `RAG_SYNC_URL`.
3. Confirm reader `/health` (`release_id` advanced) and a sample search.
4. **Stop** the ingester again.

**Cron (future / safety-net):** a scheduled `rag-diff-push.py` (or equivalent) can catch missed hooks; not required for the happy path.

#### DR escape hatch: CLI stop-apply

Keep `scripts/rag-publish.py` + `scripts/rag-apply-release.sh` for disaster recovery, first-time seed helpers, or when the sync API is unavailable. That path still needs the reader process **stopped** (embedded Qdrant single-writer lock).

**Env for CLI path:**

| Name | Role |
| --- | --- |
| `RAG_SKIP_DEFAULT_APP` | Set `1` when running publish/apply scripts so they do not boot the HTTP app |

**Qdrant sync design (verified against Qdrant 1.20 / qdrant-client 1.20 local mode):**

- Snapshots are **not** available for `QdrantClient(path=…)` (`NotImplementedError`).
- No WAL / incremental replication between two embedded stores — scroll+filter+upsert (our JSONL by `doc_id`) is the intended pattern.
- Baseline / disaster recovery: **stop both processes**, then **rsync** the data dir (`meta.json`, `.lock`, `collection/*/storage.sqlite`, …). Never copy while a client is open.
- Routine deltas: live push (above) or CLI JSONL (`--stage-qdrant`). Upserts must resend every named vector present on the point (omitted named vector ⇒ nulled).
- Lock: second opener gets `RuntimeError` … already accessed…; lockfile `{qdrant_dir}/.lock`. Live push avoids a second opener by reusing the reader's client.

CLI stop-apply sequence (DR only):

1. On **ingester**, free the Qdrant lock (HTTP app must not hold `path=`). Then:  
   `RAG_SKIP_DEFAULT_APP=1 python scripts/rag-publish.py --rag-root "$RAG_DATA_ROOT" --out /tmp/release --reader-catalog /path/to/reader-fp.json --stage-blobs --stage-qdrant`  
   (`reader-fp.json` = reader `doc_id→sha256` catalog snapshot; baseline if omitted / empty.)
2. **Transport** `/tmp/release` (`manifest.json`, `blobs/`, `qdrant/points.jsonl`) to the reader volume. Prefer object store or `scp` between hosts. Auth is infrastructure (SSH keys / bucket credentials), **not** app Bearer tokens.
3. On **reader**: free Qdrant lock →  
   `scripts/rag-apply-release.sh --release-dir /path/to/release --rag-root "$RAG_DATA_ROOT"` → start reader again.
4. Optional on writer: `--mark-published` after a successful apply.

**Railway volume limits (verified 2026-11-18):**

- Hobby volume max is **5 GB** — a ~39 GB tree needs **Pro** (default 50 GB, resize up to 1 TB).
- Volume storage bills ~$0.15/GB/mo even when the service is stopped.
- One volume ↔ one service; replicas cannot use volumes.
- Does **not** work for `/data` scripts: `railway run` (local env only), pre-deploy,
  `railway ssh` / `railway volume files` against a **stopped** deployment.

**Preferred ops recipe: temporary `sleep infinity` + `railway ssh`**

1. Clear / disable the HTTP healthcheck first (`sleep` never returns 2xx; default
   healthcheck timeout is 300 s). Ensure `railway.json` does not pin
   `deploy.startCommand` / `healthcheckPath` (config-as-code overrides the UI).
2. Set Custom Start Command to `sleep infinity` (UI Settings, or
   `railway environment edit --service-config <svc> deploy.startCommand "sleep infinity"`)
   and redeploy. There is no `RAILWAY_START_COMMAND` env var.
3. `railway ssh --service <svc> -- <cmd>` with `RAG_SKIP_DEFAULT_APP=1` for
   publish/apply (Qdrant unlocked because the HTTP app is not running).
4. Restore the normal start command + healthcheck and redeploy.
5. For the cold ingester, finish with `railway down --service <ingester> --yes`
   (volume stays attached and billed).

Caveat: volume services take cutover downtime on every deploy, so the **warm reader
is down for the whole sleep window**. Keep that window short.

**Alternatives (ranked):**

| Rank | Approach | When |
| --- | --- | --- |
| 2 | One-shot start command = the script; restart policy not Always | Ingester publish when inputs already on `/data` |
| 3 | Cron service (`railway cron run`) with volume | Not on the warm reader |
| 4 | Detach/attach volume to a temporary ops service | Same region; original service loses the volume |
| 5 | Backups/restore | Recovery only; manual backup cap is 50% of volume size |

**Suggested publish/apply runbook (deltas):**

1. Put **both** services on `sleep infinity` (healthchecks cleared) so volumes mount
   and Qdrant locks are free.
2. On ingester: run `rag-publish.py --stage-blobs --stage-qdrant`.
3. Copy the release dir to the reader (`scp` via `ssh.railway.com`, or
   `railway volume files` download/upload through local disk).
4. On reader: run `rag-apply-release.sh`.
5. Restore reader start command + healthcheck → redeploy (warm again).
6. `railway down` the ingester (cold).

First-time seed (or disaster recovery): stop source → copy entire `/data/rag`
(including `qdrant/`) onto the reader volume → start reader only.
Prefer that over `--force-baseline` JSONL for ~39 GB trees.

### (c) Reader survival across restarts

| Need | Value |
| --- | --- |
| Volume mount | `/data` |
| Data path | `/data/rag` (`RAG_DATA_ROOT`) |
| Must persist | `originals/`, `processed/`, `versions/`, `index.json`, `qdrant/`, `release.json`, optional caches |
| Smallest sensible disk | **≥ used size of current RAG tree + ~20%** (today that means planning **~50–60 GB** if the seed is ~39 GB). Do not provision a volume smaller than the seed. |
| RAM | **4 GB** minimum for warm Novita reader in production |

Ephemeral container filesystem is not enough; without the volume, Qdrant and catalog are lost.

### (d) When the ingester is stopped

| Path | Behaviour |
| --- | --- |
| Reader searches / MCP read tools / `/health` | Keep working |
| HTTP writes on **reader** (`POST /upload`, `/ingest-json`, document mutate, …) | **403** JSON: `detail` like `Writes are disabled on this node (ČTENÍ).`, plus `role`, `write_nodes` from `RAG_NODES` |
| MCP write tools on **reader** | Not registered / blocked with a clear error string |
| Calls to **ingester** URL while service stopped | Fail at network / platform layer (not a RAG 403) until started |

Set `RAG_NODES` so `write_nodes` in the 403 body points at the ingest URL (even when cold), so clients know where writes belong.

### Railway env (names only)

Same tables as Coolify §4 below. Reader must have: `APP_MODE=reader`, `RAG_DATA_ROOT`, `MCP_API_TOKEN` or `RAG_API_TOKEN`, Novita key if query is Novita, `PUBLIC_BASE_URL`, `MCP_ALLOWED_HOSTS`, `RAG_NODES`, `IMAGE_EMBEDDING_ENABLED=false`, `EMBEDDING_PROVIDER_QUERY=novita`.

### Railway cutover / standby notes

- Custom Hostinger DNS for `art-nouveau-rag.animarium.ai` waits until Marek moves off Railway or wants the Railway service behind that name.
- Google OAuth redirect: `{PUBLIC_BASE_URL}/auth/callback`.
- Do **not** delete volumes when stopping the ingester or the old monolith.
- CoolifyHoster: **no Railway action required** from the Coolify side unless Marek asks.

### Post-deploy validation (Railway)

1. Reader `/health` → `status=ok`, `app_mode=reader`.
2. Write to reader → **403** with writes-disabled detail.
3. MCP on reader → read tools work; no write tools (or writes error clearly).
4. Start ingester → `/health` `app_mode=ingester` → ingest one doc → live push (or CLI publish/apply) → reader search sees it → stop ingester → reader still healthy.
5. With `RAG_SYNC_TOKEN` set on reader: `GET /sync/v1/watermark` with Bearer returns JSON (401 without token).

---

## Coolify / Hetzner contract (deferred — documentation only)

**Do not** create the VPS, Coolify apps, or Hostinger DNS until Marek says the Railway split is stable and release-ready. The following is the migration contract when that happens.

### Topology then

| Service | Where | When | RAM | Disk |
| --- | --- | --- | --- | --- |
| **Reader** | Coolify on Hetzner VPS (public IPv4) | Always on (warm) | **4 GB** with Novita + CLIP off | **≥ 60 GB** for ~39 GB seed + headroom |
| **Ingester** | Prefer remain **cold on Railway** (volume only) **or** second Coolify app stopped when idle | Start only to ingest + publish | **4 GB** while running | Separate volume **≥ 60 GB** if full copy kept |

Public hostname: **`art-nouveau-rag.animarium.ai`** → Coolify reader. Ingester may keep Railway URL or get `art-nouveau-rag-ingest.animarium.ai`.

**Not recommended:** shared volume between reader and ingester; monolith 8 GB public box once the split is proven.

### Per-mode Coolify app settings

| Setting | Reader | Ingester (if on Coolify) |
| --- | --- | --- |
| Dockerfile | `Dockerfile.reader` | `Dockerfile.ingester` |
| Build context | repository root | repository root |
| Start command | `python reader_main.py` | `python ingester_main.py` |
| Port | **8080** (`PORT`) | **8080** |
| Healthcheck | `GET /health` | `GET /health` |
| Domain | `art-nouveau-rag.animarium.ai` | optional ingest hostname |
| Volume | Coolify volume → `/data` | **different** volume → `/data` |
| `RAG_DATA_ROOT` | `/data/rag` | `/data/rag` on its volume |

### Volumes and baseline seed (Coolify reader)

1. Stop Railway source process that owns the canonical `/data/rag`.
2. Stop Coolify reader.
3. `rsync -aH` entire `/data/rag/` (including `qdrant/`) onto Coolify `/data/rag/`.
4. Start Coolify reader only; check `/health`.
5. Later deltas: same publish/apply flow as Railway §(b).

### Environment variable names

**Secrets** (masked collection only; never commit values):

| Name | Required? | Secret? | Notes |
| --- | --- | --- | --- |
| `MCP_API_TOKEN` | **yes** (or `RAG_API_TOKEN`) | **yes** | Preferred Bearer for `/mcp` + embed REST |
| `RAG_API_TOKEN` | **yes** if MCP unset | **yes** | `POST /ingest-json`; MCP fallback |
| `RAG_SYNC_TOKEN` | **yes** for live push | **yes** | Bearer for reader `/sync/v1/*` (not MCP) |
| `BGE_M3_NOVITA_API_KEY` | **yes** when Novita used | **yes** | Preferred Novita key |
| `NOVITA_API_KEY` | optional | **yes** | Fallback |
| `OPENAI_TOKEN` / `OPENAI_API_KEY` | optional | **yes** | Last-resort embedding fallbacks |
| `SESSION_SECRET` | recommended | **yes** | Cookie sessions |
| `GOOGLE_CLIENT_SECRET` | if Google UI | **yes** | With `GOOGLE_CLIENT_ID` |
| `TRAILS_API_TOKEN` | if Trails sync | **yes** | With `TRAILS_API_URL` |

**Non-secret:**

| Name | Required? | Reader default | Notes |
| --- | --- | --- | --- |
| `APP_MODE` | yes | `reader` | |
| `RAG_DATA_ROOT` | yes | `/data/rag` | |
| `ACCNT_DATA_ROOT` | optional | `/data/accnt` | |
| `PORT` | optional | `8080` | |
| `PUBLIC_BASE_URL` | **yes** on public host | `https://art-nouveau-rag.animarium.ai` | |
| `MCP_ALLOWED_HOSTS` | **yes** | hostname + localhost variants | Host allow-list (**not** CORS) |
| `RAG_NODES` | **yes** after catalog cutover | JSON write+read URLs | |
| `RAG_NODE_NAME` | recommended | stable name | |
| `RAG_NODE_ROLE` | optional | omit on reader | |
| `EMBEDDING_PROVIDER` / `_QUERY` / `_INGEST` | as needed | query `novita` | |
| `IMAGE_EMBEDDING_ENABLED` | recommended | `false` | |
| `OCR_ENABLED` / `OCR_LANGS` | ingester | `true` / `ces+eng` | |
| `GOOGLE_CLIENT_ID` / `GOOGLE_ALLOWED_*` | optional | | |
| `TRAILS_API_URL` | optional | | |
| `FASTEMBED_CACHE_PATH` / `IMAGE_EMBEDDING_CACHE_PATH` | only if local models | | |
| `MCP_DISABLE_DNS_REBINDING_PROTECTION` | avoid in prod | unset | |
| `RAG_SKIP_DEFAULT_APP` | scripts only | unset in prod | |
| `RAG_SYNC_URL` | ingester (live push) | reader origin URL | |
| `RAG_SYNC_PUSH_ENABLED` | ingester optional | `1` when URL+token set | |

### Domain / clients (when DNS is created)

No CORS middleware. Set:

```text
PUBLIC_BASE_URL=https://art-nouveau-rag.animarium.ai
MCP_ALLOWED_HOSTS=art-nouveau-rag.animarium.ai,localhost,localhost:*,127.0.0.1,127.0.0.1:*
```

OAuth redirect: `https://art-nouveau-rag.animarium.ai/auth/callback`.

Hostinger `A` → Hetzner IPv4 only after the VPS exists. Clients (e.g. **mcp-chat-tester**)
must switch MCP base URL to the public reader; this repository does not own those clients.

### CoolifyHoster action now

**None.** No Coolify server, no apps, no DNS. Wait for Marek. Railway service changes are
owned by the repository/Railway operator (Marek / this project agent), not CoolifyHoster,
unless Marek explicitly asks.

---

## Incremental publish (summary)

**Happy path (live push):** ingest on ingester → auto/manual `diff_push` → reader `/sync/v1` commit (in-process) → `/health` shows new `release_id`.

**DR escape hatch (CLI):**

1. Stop **ingester** (Qdrant lock) → `rag-publish.py --stage-blobs --stage-qdrant`
2. Transport release dir
3. Stop **reader** → `rag-apply-release.sh` → start reader
4. Health: `app_mode`, `release_id`

Apply deletes then upserts points for touched `doc_id`s in `client_domain_knowledge` and `locality_links`. After baseline, do not rsync whole `qdrant/` for routine sync.

## Query embeddings

| `EMBEDDING_PROVIDER_QUERY` | Use when |
| --- | --- |
| `novita` | Low-RAM readers / portable deploys (default in reader mode) |
| `fastembed` | Local ONNX (~4GB+ peak); 8GB preferred |

Ingest may stay on Novita independently via `EMBEDDING_PROVIDER_INGEST`.

**Consistency:** prefer **Novita ingest + Novita query** on the warm reader. Mixing Novita ingest with local FastEmbed query is **unsafe until measured** (BGE-M3 is 1024-d, but bit-exact / cross-runtime parity is not guaranteed; FastEmbed’s built-in catalog may not list `BAAI/bge-m3`). Point sync copies existing vectors — it does not re-embed.

## Reader context search tools

On reader hosts, prefer tools that reuse disk/catalog data instead of loading more models:

| Tool | Embeddings? | Attaches |
| --- | --- | --- |
| `search_with_context` | yes (query provider only) | catalog, versions index, locality links, related docs, neighbor chunks |
| `get_document_context` | no | same context bundle for one `doc_id` |
| `search_catalog_documents` | no | filename/doc_id/sha256 substring; optional full context |

Keep `IMAGE_EMBEDDING_ENABLED=false` on ≤4GB readers. Context attachment is payload + filesystem only.
