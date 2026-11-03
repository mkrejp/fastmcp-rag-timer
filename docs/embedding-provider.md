# Embedding provider and Qdrant index

## Provider

Configured in `rag/embeddings.py`:

| Item | Value |
| --- | --- |
| API | `https://api.novita.ai/openai/v1/embeddings` |
| Model | `baai/bge-m3` |
| Vector dimension | `1024` |
| Timeout | 60 seconds |
| Preferred credential | `BGE_M3_NOVITA_API_KEY` |
| Other credentials (first non-empty wins) | `NOVITA_API_KEY`, `OPENAI_TOKEN`, `OPENAI_API_KEY` |

The key is read at runtime and sent as a Bearer token. Do not put its value in source, docs, or logs.

Request shape is OpenAI-compatible (`{"input": text, "model": "baai/bge-m3"}`). The host is Novita. There is no chat/completions client.

Full request/response, privacy, and cost notes: [Embedding connectivity](./CONNECTIVITY.md#embedding-connectivity).

## Collection dimension

Qdrant requires every vector in a collection to match that collection’s configured size. `open_qdrant()` (`rag/vectorstore.py`):

1. Opens the on-disk client at `{RAG_DATA_ROOT}/qdrant`.
2. If collection `client_domain_knowledge` exists and `vectors.size != 1024`, **deletes** that collection.
3. If the collection is missing, creates it with `VectorParams(size=1024, distance=COSINE)`.

Deleting a mismatched collection **removes its vectors**. Originals and processed text under `{RAG_DATA_ROOT}` are not deleted. Semantic search is empty until documents are ingested again.

## Verification

```bash
pytest -q
```

Upload a document, then search for content from that document via MCP `search_domain_knowledge`.
