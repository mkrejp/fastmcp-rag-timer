# Embedding provider and Qdrant index

Runtime defaults: **`EMBEDDING_PROVIDER`** is `novita` when unset. **`EMBEDDING_PROVIDER_INGEST`** and **`EMBEDDING_PROVIDER_QUERY`** (alias **`EMBEDDING_PROVIDER_SEARCH`**) override ingest vs search independently so ingest can stay on Novita while search uses local FastEmbed only.

## Split ingest vs query

| Path | Env | Allowed values | Model family |
| --- | --- | --- | --- |
| Ingest / OCR indexing | `EMBEDDING_PROVIDER_INGEST` → else `EMBEDDING_PROVIDER` | `novita`, `fastembed` | BGE-M3, 1024-dim |
| Search (`search_domain_knowledge`, hybrid text leg of `search_images`) | `EMBEDDING_PROVIDER_QUERY` → else `EMBEDDING_PROVIDER` | **`fastembed` only** | BGE-M3, 1024-dim |

Startup calls `validate_embedding_configuration()`: a query provider that is not FastEmbed (including an unset default of `novita` without `EMBEDDING_PROVIDER_QUERY=fastembed`) **refuses to start**. Typical production split:

```bash
EMBEDDING_PROVIDER=novita
EMBEDDING_PROVIDER_INGEST=novita
EMBEDDING_PROVIDER_QUERY=fastembed
```

Search never issues Novita HTTP requests (zero embedding tokens on query). Ingest may still use Novita when configured.

Novita HTTP vs local ONNX BGE-M3 vectors can **drift slightly**. After changing ingest provider, re-ingest and optionally run `scripts/golden_recall_smoke.py` on golden phrases.

`GET /health` and `GET /api/admin/status` expose ingest vs query providers (no secrets).

## Provider backends

| Backend | Model | Dimension | Used on |
| --- | --- | --- | --- |
| Novita HTTP | `baai/bge-m3` | 1024 | Ingest when `EMBEDDING_PROVIDER_INGEST=novita` |
| FastEmbed ONNX | `BAAI/bge-m3` | 1024 | Query (required) and ingest when set to `fastembed` |

Configured in `rag/embeddings.py`. FastEmbed BGE-M3 loads on the **first** query or FastEmbed ingest call. Query always uses FastEmbed, so the first search downloads ONNX even when ingest stays on Novita.

### Novita (ingest)

| Item | Value |
| --- | --- |
| API | `https://api.novita.ai/openai/v1/embeddings` |
| Model | `baai/bge-m3` |
| Timeout | 60 seconds |
| Credentials | `BGE_M3_NOVITA_API_KEY`, then fallbacks (see README) |

Full connectivity notes: [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity).

### FastEmbed (query + optional ingest)

Registers Hugging Face ONNX `BAAI/bge-m3` (~2.3GB cache under `FASTEMBED_CACHE_PATH` or `{RAG_DATA_ROOT}/fastembed`). Plan **4GB+ RAM**, **8GB preferred** for BGE-M3 alone. FastEmbed does not ship BGE-M3 in its built-in dense catalog; the app registers the Hugging Face ONNX export (`onnx/model.onnx`).

## Named vectors

Qdrant collection `client_domain_knowledge` uses **named vectors**:

| Name | Model | Dim | Purpose |
| --- | --- | --- | --- |
| `text_vector` | BGE-M3 (ingest backend) | 1024 | Chunks, OCR text |
| `image_vector` | `Qdrant/clip-ViT-B-32-vision` | 512 | PDF image pixels |

Text queries for images use CLIP text tower `Qdrant/clip-ViT-B-32-text` (`rag/image_embed.py`). Enable with `IMAGE_EMBEDDING_ENABLED=true` (default **false**). CLIP loads lazily; cache under `{RAG_DATA_ROOT}/clip` or `IMAGE_EMBEDDING_CACHE_PATH`. Loading CLIP beside BGE-M3 can OOM on **8GB** hosts — increase RAM or leave visual search disabled.

Collections that use a **single unnamed vector** are recreated on startup; **re-ingest** is required after that recreate. Marker file `{RAG_DATA_ROOT}/embedding_backend.json` records schema version 2 and the named-vector layout.

Switching ingest provider (`novita` ↔ `fastembed`) at the same 1024-dim BGE-M3 family does **not** recreate the collection; recall still depends on re-embedding via re-ingest. Toggling `IMAGE_EMBEDDING_ENABLED` also does not recreate; re-ingest PDFs to backfill `image_vector`.

## Verification

```bash
pytest -q
```

Upload a document with `EMBEDDING_PROVIDER_QUERY=fastembed`, search via MCP `search_domain_knowledge`. With `IMAGE_EMBEDDING_ENABLED=true`, re-ingest a PDF and try `search_images`.
