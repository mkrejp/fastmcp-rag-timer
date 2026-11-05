# Embedding provider and Qdrant index

Runtime switch: **`EMBEDDING_PROVIDER`**. Unset or `novita` keeps the production Novita HTTP API. `fastembed` embeds locally with ONNX BGE-M3. Do not change production variables unless you intend to switch backends and re-ingest.

## Provider

| `EMBEDDING_PROVIDER` | Backend | Model | Vector dimension | Credentials |
| --- | --- | --- | --- | --- |
| unset / `novita` (default) | Novita OpenAI-compatible HTTP | `baai/bge-m3` | `1024` | `BGE_M3_NOVITA_API_KEY`, then `NOVITA_API_KEY`, `OPENAI_TOKEN`, `OPENAI_API_KEY` |
| `fastembed` | Local FastEmbed (lazy-loaded) | `BAAI/bge-m3` | `1024` | none |

Configured in `rag/embeddings.py`. The model is loaded on the **first** `get_embedding` / `get_embeddings` call, not at process import or `/health`. Cold start without search or ingest does not download or load ONNX weights.

### Novita (default)

| Item | Value |
| --- | --- |
| API | `https://api.novita.ai/openai/v1/embeddings` |
| Model | `baai/bge-m3` |
| Vector dimension | `1024` |
| Timeout | 60 seconds |
| Batching | one HTTP request per string (unchanged) |
| Retries | none (unchanged) |
| Preferred credential | `BGE_M3_NOVITA_API_KEY` |
| Other credentials (first non-empty wins) | `NOVITA_API_KEY`, `OPENAI_TOKEN`, `OPENAI_API_KEY` |

The key is read at runtime and sent as a Bearer token. Do not put its value in source, docs, or logs.

Request shape is OpenAI-compatible (`{"input": text, "model": "baai/bge-m3"}`). The host is Novita. There is no chat/completions client.

Full request/response, privacy, and cost notes: [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity).

### FastEmbed (local)

FastEmbed's built-in dense catalog does not include BGE-M3 (`qdrant/fastembed#107`). This service registers the official Hugging Face ONNX export (`BAAI/bge-m3`, `onnx/model.onnx` + `onnx/model.onnx_data`) with CLS pooling and L2 normalization so the local path stays **1024-dim multilingual BGE-M3**.

| Item | Value |
| --- | --- |
| Package | `fastembed` (Dockerfile / `pyproject.toml`) |
| Model ID | `BAAI/bge-m3` |
| Source | Hugging Face `BAAI/bge-m3` ONNX files |
| Vector dimension | `1024` (same as Novita; Qdrant size stays 1024) |
| First call | downloads ~2.3GB of ONNX weights unless already cached |
| Cache | `FASTEMBED_CACHE_PATH`, else `{RAG_DATA_ROOT}/fastembed` |
| RAM | plan **at least 4GB**, prefer 8GB, for ONNX Runtime + Qdrant + the app |
| Hugging Face egress | required on first embed (and after cache eviction) |

Local vectors are **not guaranteed bit-identical** to Novita's hosted BGE-M3 (different runtime/quantization). Switching providers requires a re-ingest even though the collection dimension does not change. `open_qdrant()` recreates `client_domain_knowledge` when the recorded provider/model/dim in `{RAG_DATA_ROOT}/embedding_backend.json` does not match the active backend.

## Collection dimension

Qdrant requires every vector in a collection to match that collection’s configured size. `open_qdrant()` (`rag/vectorstore.py`):

1. Opens the on-disk client at `{RAG_DATA_ROOT}/qdrant`.
2. Reads `{RAG_DATA_ROOT}/embedding_backend.json` (written on each successful open).
3. If collection `client_domain_knowledge` exists and `vectors.size` is not the **active** `embedding_dim()`, **deletes** that collection.
4. If the marker's provider/model/dim does not match the active backend, **deletes** that collection (covers Novita → FastEmbed at 1024-dim).
5. If the collection is missing, creates it with `VectorParams(size=embedding_dim(), distance=COSINE)`.
6. Writes the marker for the active provider.

A missing marker is treated as the historical Novita index so an unchanged production deploy does not wipe vectors.

Deleting a mismatched collection **removes its vectors**. Originals and processed text under `{RAG_DATA_ROOT}` are not deleted. Semantic search is empty until documents are ingested again.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `novita` (when unset) | `novita` or `fastembed` |
| `FASTEMBED_CACHE_PATH` | `{RAG_DATA_ROOT}/fastembed` | ONNX model cache for the local backend |
| `BGE_M3_NOVITA_API_KEY` (and fallbacks) | unset | Required only for `novita` |

## Verification

```bash
pytest -q
```

Upload a document, then search for content from that document via MCP `search_domain_knowledge`.
