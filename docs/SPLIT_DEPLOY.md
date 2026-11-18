# Reader / ingester split deploy

One repository, two roles. Either role can run on **Railway or a VPS**.

| Role | `APP_MODE` | Typical RAM | DNS (example) |
| --- | --- | --- | --- |
| Reader (ČTENÍ) | `reader` | ~0.5–1.5 GB (Novita query) | `art-nouveau-rag.animarium.ai` **(Assumption — not live until deployed)** |
| Ingester (ZÁPIS) | `ingester` | ~1.5–3 GB (Novita ingest + OCR) | `art-nouveau-rag-ingest.animarium.ai` **(Assumption)** |
| Monolith | `all` (default) | as before | existing Railway hosts (`rag-timer-00001`, etc.) |

Aliases: `APP_MODE` or `RAG_APP_MODE`. Shortcuts: `readonly`→reader, `ingest`→ingester.

## Behaviour by mode

**Reader**

- `enforce_node_role()` is always on: `NodeWriteGuard` returns **403** on HTTP write paths; mutating MCP tools are **not registered** (`mcp_write_tools=false` on `/health`).
- `resolve_node_role_override()` defaults this instance to **ČTENÍ** when `RAG_NODE_ROLE` is unset.
- Defaults `EMBEDDING_PROVIDER_QUERY` to `novita` when unset (no local BGE-M3 in `Dockerfile.reader`).
- Prefer `IMAGE_EMBEDDING_ENABLED=false` (CLIP is heavy).

**Ingester**

- Defaults role to ZÁPIS+ČTENÍ when `RAG_NODE_ROLE` unset.
- Runs upload/ingest/OCR; builds incremental releases via `scripts/rag-publish.py`.

**All**

- Historical single-process behaviour (`main.py` / root `Dockerfile`).

## Images

- `Dockerfile` — monolith (`APP_MODE=all`)
- `Dockerfile.reader` — `reader_main.py`
- `Dockerfile.ingester` — `ingester_main.py`

Examples: `deploy/vps/docker-compose.yml`, `deploy/railway/README.md`.

## Incremental publish (diffs only)

Volumes are **not** live-shared across hosts. The writer pushes **changed** catalog docs/files:

1. `python scripts/rag-publish.py --rag-root … --out … --reader-catalog reader-fp.json --stage-blobs`
2. Transport blobs (rsync/SSH or object store).
3. On the reader (app stopped): `scripts/rag-apply-release.sh --release-dir … [--reboot]`
4. Health exposes `app_mode` and `release_id`.

**Follow-up:** Qdrant point-level sync is not in `rag-apply-release.sh` — it copies catalog/processed files only (`rag/publish/apply.py`). After apply, operators still need a Qdrant strategy (full `qdrant/` directory transport, re-index on reader, or future tooling). `--force-baseline` on the publisher treats the diff as a full file baseline when needed.

## Environment highlights by mode

| Variable | `reader` | `ingester` | `all` |
| --- | --- | --- | --- |
| `APP_MODE` | `reader` | `ingester` | `all` (default) |
| `EMBEDDING_PROVIDER_QUERY` | `novita` (default if unset) | often `novita` in image | set explicitly if ingest is Novita |
| `EMBEDDING_PROVIDER_INGEST` | unused (no ingest) | `novita` (image default) | optional split |
| `IMAGE_EMBEDDING_ENABLED` | `false` recommended | usually `false` on small VMs | optional |
| `BGE_M3_NOVITA_API_KEY` | required when query=novita | required when ingest/query=novita | as configured |
| `MCP_API_TOKEN` / `RAG_API_TOKEN` | MCP + optional Google UI | ingest + MCP writes | both |
| `RAG_ENFORCE_NODE_ROLE` | implicit (always on) | optional extra guard | optional |
| `PUBLIC_BASE_URL` / `MCP_ALLOWED_HOSTS` | reader hostname | ingester hostname | service URL |

Shared: `RAG_DATA_ROOT`, volume at `/data`, `RAG_NODES` catalog (writer + reader URLs).

## Query embeddings

| `EMBEDDING_PROVIDER_QUERY` | Use when |
| --- | --- |
| `novita` | Low-RAM readers / portable deploys (default in reader mode) |
| `fastembed` | Local ONNX (~4GB+ peak); 8GB preferred |

Ingest may stay on Novita independently via `EMBEDDING_PROVIDER_INGEST`.
