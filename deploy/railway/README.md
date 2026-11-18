# Railway deploy notes

Same images run on Railway or a VPS. Typical split:

| Service | Image / start | Domain |
| --- | --- | --- |
| Ingester | `Dockerfile.ingester` → `python ingester_main.py` | `art-nouveau-rag-ingest.animarium.ai` |
| Reader (optional on Railway) | `Dockerfile.reader` → `python reader_main.py` | custom domain |

## Required variables

Shared:

- `MCP_API_TOKEN` / `RAG_API_TOKEN`
- `BGE_M3_NOVITA_API_KEY` (when query or ingest uses Novita)
- `RAG_DATA_ROOT=/data/rag` + volume mount at `/data`
- `RAG_NODES` JSON with write + read URLs
- `PUBLIC_BASE_URL` matching the service domain
- `MCP_ALLOWED_HOSTS` including that hostname

Reader:

- `APP_MODE=reader`
- `EMBEDDING_PROVIDER_QUERY=novita`
- `IMAGE_EMBEDDING_ENABLED=false`

Ingester:

- `APP_MODE=ingester`
- `EMBEDDING_PROVIDER_INGEST=novita`

Volumes are **not** shared across Railway services (1 volume : 1 service).
`railway run` does **not** mount volumes. For publish/apply, free the embedded
Qdrant lock (app must not hold `path=`), then run
`scripts/rag-publish.py --stage-blobs --stage-qdrant` and
`scripts/rag-apply-release.sh` via `railway ssh` into a container that still
mounts `/data` (e.g. temporary `sleep infinity` entrypoint) or another documented
volume-access path. Transport release dirs with object store / `scp` — not app
Bearer tokens. Baseline seed: stop processes and rsync `/data/rag` (snapshots
are unavailable in local/embedded mode).
