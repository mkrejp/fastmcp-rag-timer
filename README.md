# fastmcp-rag-timer

Combines FastAPI (for web endpoints), FastMCP (to natively expose tools over the Model Context Protocol), and Qdrant Client (local-disk vector engine). It handles PDF and text extraction, stores originals and processed artifacts on disk, and indexes chunks for semantic retrieval.

## Persistent storage

All RAG server data lives under **`/data/rag`** (override with `RAG_DATA_ROOT`):

```text
/data/rag/
  originals/{doc_id}/...
  processed/{doc_id}/meta.json, fulltext.txt, chunks.json
  qdrant/
  index.json
```

On Railway, mount a volume at `/data`. The app creates `/data/rag` at startup.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_ROOT` | `/data/rag` | Persistent originals, processed text, Qdrant |
| `BGE_M3_NOVITA_API_KEY` | unset | Preferred Novita API credential for embeddings |
| `NOVITA_API_KEY` | unset | Embedding credential if `BGE_M3_NOVITA_API_KEY` is empty |
| `OPENAI_TOKEN` | unset | Embedding credential if both Novita-named vars are empty |
| `OPENAI_API_KEY` | unset | Embedding credential checked last |
| `RAG_SKIP_DEFAULT_APP` | unset | Set to `1` to skip module-level `create_app()` (used by tests) |

Embeddings use Novita's OpenAI-compatible API (`baai/bge-m3`, 1024 dimensions). See [embedding provider](docs/embedding-provider.md). Keep the key in runtime variables; never commit it.

## HTTP API

- `POST /upload` — PDF, TXT, or Markdown → persist + index
- `POST /upload-pdf` — PDF-only alias
- `POST /upload-media` — store bytes under `originals/` (not indexed)
- `GET /documents` — catalog
- `GET /documents/{doc_id}` — metadata
- `GET /documents/{doc_id}/text` — processed full text (no re-parse)
- `GET /health` — liveness

Listen address is `0.0.0.0:8080` (not configurable via `PORT`). No inbound HTTP authentication.

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`**. Tools:

- `list_documents`
- `get_document_text`
- `get_document_chunks`
- `ingest_text`
- `search_domain_knowledge`

No MCP resources or prompts. No inbound MCP authentication.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system context, components, MCP surface, RAG pipeline, runtime, configuration, security.
- [Connectivity](docs/CONNECTIVITY.md) — trust boundaries, inbound/outbound paths, ports, auth points, connection inventory.
- [Embedding connectivity](docs/CONNECTIVITY.md#embedding-connectivity) — Novita embeddings API, auth, egress, timeouts, privacy, and cost.
- [Embedding provider](docs/embedding-provider.md) — Novita BGE-M3 configuration and Qdrant dimension handling.

## Package layout

```text
main.py                 # entrypoint (uvicorn / python main.py)
rag/
  app.py                # create_app()
  routes.py             # HTTP API
  mcp_tools.py          # FastMCP tools
  storage.py            # /data/rag originals + processed + index
  processing.py         # PDF/text extract + chunk
  embeddings.py         # Novita BGE-M3 embeddings
  ingest.py             # persist + index pipeline
  vectorstore.py        # local Qdrant
  config.py             # env defaults
```

## Local run

```bash
export RAG_DATA_ROOT=./data/rag
export BGE_M3_NOVITA_API_KEY=your-novita-key
python main.py
```

## Tests

```bash
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf qdrant-client requests python-multipart
pytest -q
```
