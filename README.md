# fastmcp-rag-timer

combines FastAPI (for web endpoints), FastMCP (to natively expose tools over the Model Context Protocol), and Qdrant Client (using an in-memory or local-disk vector engine to eliminate the need for a separate database container). It handles PDF text extraction via pypdf and stores files directly on the local filesystem.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system context, components, MCP surface, RAG pipeline, runtime, configuration, security.
- [Connectivity](docs/CONNECTIVITY.md) — trust boundaries, inbound/outbound paths, ports, auth points, connection inventory.

These docs are based on `main.py` and `Dockerfile` as they exist in the tree. Notable gaps versus the repo name and README (no timer, MCP not mounted on the uvicorn entrypoint, on-disk Qdrant rather than in-memory) are called out there.
