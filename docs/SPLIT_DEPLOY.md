# Reader / ingester split deploy

One repository, two roles. Either role can run on **Railway or a VPS**.

| Role | `APP_MODE` | Typical RAM | DNS (example) |
| --- | --- | --- | --- |
| Reader (ČTENÍ) | `reader` | ~0.5–1.5 GB (Novita query) | `art-nouveau-rag.animarium.ai` |
| Ingester (ZÁPIS) | `ingester` | ~1.5–3 GB (Novita ingest + OCR) | `art-nouveau-rag-ingest.animarium.ai` |
| Monolith | `all` (default) | as before | existing hosts |

Aliases: `APP_MODE` or `RAG_APP_MODE`. Shortcuts: `readonly`→reader, `ingest`→ingester.

## Behaviour by mode

**Reader**

- Forces write enforcement (HTTP 403 on mutating routes; write MCP tools not registered).
- Defaults `EMBEDDING_PROVIDER_QUERY` to `novita` when unset (no local BGE-M3).
- Prefer `IMAGE_EMBEDDING_ENABLED=false`.

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

Volumes are **not** live-shared across hosts. The writer pushes **changed** catalog docs/files **and** matching Qdrant points:

1. Stop the **ingester** (embedded Qdrant is single-writer).
2. `python scripts/rag-publish.py --rag-root … --out … --reader-catalog reader-fp.json --stage-blobs --stage-qdrant`
3. Transport the release dir (rsync/SSH or object store) — includes `blobs/` and `qdrant/points.jsonl`.
4. Stop the **reader**, then: `scripts/rag-apply-release.sh --release-dir … [--reboot]`
5. Health exposes `app_mode` and `release_id`.

Apply deletes then upserts points for touched `doc_id`s in `client_domain_knowledge` and `locality_links`. The on-disk `qdrant/` directory is never rsynced as a whole; only JSONL point payloads move. `--force-baseline` still means a full file catalog dump plus all points for every catalog doc when `--stage-qdrant` is set.

## Query embeddings

| `EMBEDDING_PROVIDER_QUERY` | Use when |
| --- | --- |
| `novita` | Low-RAM readers / portable deploys (default in reader mode) |
| `fastembed` | Local ONNX (~4GB+ peak); 8GB preferred |

Ingest may stay on Novita independently via `EMBEDDING_PROVIDER_INGEST`.
