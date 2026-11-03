# fastmcp-rag-timer

Combines FastAPI (for web endpoints), FastMCP (to natively expose tools over the Model Context Protocol), and Qdrant Client (local-disk vector engine). It handles PDF and text extraction, stores originals plus processed artifacts on disk, and indexes chunks for semantic search.

## Persistent storage

All RAG server data lives under **`/data/rag`** (override with `RAG_DATA_ROOT`):

```text
/data/rag/
  originals/{doc_id}/...
  processed/{doc_id}/meta.json, fulltext.txt, chunks.json
  qdrant/
  index.json
```

On Railway, mount a volume at `/data` (service `rag-timer-00001` already does). The app creates `/data/rag` at startup.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_ROOT` | `/data/rag` | Persistent originals, processed text, Qdrant |
| `OPENAI_TOKEN` | unset | Preferred OpenAI API credential for embeddings |
| `OPENAI_API_KEY` | unset | Fallback embedding credential |

## HTTP API

- `POST /upload` — PDF, TXT, or Markdown → persist + index
- `POST /upload-pdf` — PDF-only compatibility alias
- `GET /documents` — catalog
- `GET /documents/{doc_id}` — metadata
- `GET /documents/{doc_id}/text` — processed full text (no re-parse)
- `GET /health` — liveness

## MCP (AI clients)

Streamable HTTP MCP is mounted at **`/mcp`**. Tools:

- `list_documents`
- `get_document_text`
- `get_document_chunks`
- `ingest_text`
- `search_domain_knowledge`

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system context, components, MCP surface, RAG pipeline, runtime, configuration, security.
- [Connectivity](docs/CONNECTIVITY.md) — trust boundaries, inbound/outbound paths, ports, auth points, connection inventory.

## Package layout

```text
main.py                 # entrypoint (uvicorn / python main.py)
rag/
  app.py                # create_app()
  routes.py             # HTTP API
  mcp_tools.py          # FastMCP tools
  storage.py            # /data/rag originals + processed + index
  processing.py         # PDF/text extract + chunk
  embeddings.py         # OpenAI embeddings
  ingest.py             # persist + index pipeline
  vectorstore.py        # local Qdrant
  config.py             # env defaults
```

## Local run

```bash
export RAG_DATA_ROOT=./data/rag
export OPENAI_TOKEN=sk-...
python main.py
```

## Tests

```bash
pip install pytest pytest-mock 'mcp[cli]<2' fastapi uvicorn pypdf qdrant-client requests python-multipart
pytest -q
```
