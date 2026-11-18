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

Volumes are **not** shared across Railway services. Use incremental
`scripts/rag-publish.py --stage-blobs --stage-qdrant` (stop the writer first)
plus transport (SSH/object store) and `scripts/rag-apply-release.sh` on the
reader (stop the reader first) to push file diffs and Qdrant points.
