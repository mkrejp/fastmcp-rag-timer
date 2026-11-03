# Use a highly optimized, modern python base image
FROM python:3.11-slim-bookworm

# Install UV for lightning fast, cached dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install basic system curl requirements for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Pin mcp to 1.x — FastMCP lives there; mcp 2.x renames it to MCPServer
RUN uv pip install --system \
    fastapi \
    uvicorn \
    "mcp[cli]<2" \
    pypdf \
    qdrant-client \
    requests \
    python-multipart \
    itsdangerous

COPY main.py ./
COPY rag ./rag
COPY static ./static

# Expose Web Interface Port
EXPOSE 8080

# Railway mounts a volume at /data; RAG artifacts live under /data/rag at runtime.
# Do not declare Docker VOLUME — Railway rejects it; use the service volume mount instead.
ENV RAG_DATA_ROOT=/data/rag

CMD ["python", "main.py"]
