# Railway deploy notes

Same images run on Railway or a VPS. Use one service per role (separate volumes). Monolith: root `Dockerfile` + `APP_MODE=all` (or omit).

Typical split:

| Service | Image / start | Domain |
| --- | --- | --- |
| Ingester | `Dockerfile.ingester` → `python ingester_main.py` | `art-nouveau-rag-ingest.animarium.ai` |
| Reader (optional on Railway) | `Dockerfile.reader` → `python reader_main.py` | custom domain |

## Required variables

Shared:

- `MCP_API_TOKEN` / `RAG_API_TOKEN`
- `MCP_READ_TOKEN` (optional; read-only `/mcp`)
- `BGE_M3_NOVITA_API_KEY` (when query or ingest uses Novita; prepaid Model
  API, not Coding Plan)
- `RAG_DATA_ROOT=/data/rag` + volume mount at `/data`
- `RAG_NODES` JSON with write + read URLs
- `PUBLIC_BASE_URL` matching the service domain
- `MCP_ALLOWED_HOSTS` including that hostname

Reader:

- `APP_MODE=reader`
- `EMBEDDING_PROVIDER_QUERY=novita`
- `IMAGE_EMBEDDING_ENABLED=false`
- Optional helper phrasing only: dedicated `AI_GATEWAY_API_KEY`,
  `CHAT_HELPER_CHAT=1`, and the `CHAT_HELPER_DAILY_*` caps. Leave chat off
  unless Marek enables it. Rerank uses the existing Novita key.

Ingester:

- `APP_MODE=ingester`
- `EMBEDDING_PROVIDER_INGEST=novita`

## Volumes and publish/apply

Volumes are **1:1** with a service (no sharing; replicas cannot use volumes).
Hobby max is **5 GB** — a ~39 GB RAG tree needs **Pro** (~50 GB+). Storage bills
~$0.15/GB/mo even when the service is stopped.

`railway run` does **not** mount volumes. `railway ssh` and `railway volume files`
require a **running** deployment.

To free the embedded Qdrant lock while keeping `/data` mounted:

1. Disable the HTTP healthcheck (`sleep` never returns 2xx).
2. Set start command to `sleep infinity`, redeploy.
3. `railway ssh --service <svc> -- …` with `RAG_SKIP_DEFAULT_APP=1` for
   `scripts/rag-publish.py --stage-blobs --stage-qdrant` or
   `scripts/rag-apply-release.sh`.
4. Restore start command + healthcheck; `railway down` the cold ingester.

The warm reader is down for that sleep window (volume deploy cutover). Copy release
dirs with `scp` via `ssh.railway.com` or `railway volume files` — not app Bearer
tokens. Baseline seed: copy `/data/rag` after processes are stopped (embedded
snapshots are unavailable). Full runbook: `docs/SPLIT_DEPLOY.md`.

**Happy path:** live diff-push (`RAG_SYNC_URL` + `RAG_SYNC_TOKEN`, optional `RAG_SYNC_PUSH_ENABLED=1`) — see [SPLIT_DEPLOY.md](../docs/SPLIT_DEPLOY.md).

`GET /health` should report `app_mode`, `writes_enabled`, `release_id`, and sync fields (`sync_enabled`, `catalog_doc_count`, …) when configured.

**Assumption:** Custom domains in the table are deployment targets; wire DNS and `PUBLIC_BASE_URL` when going live.
