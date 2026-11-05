# python:3.11-slim-bookworm — runtime matches requires-python >=3.11
FROM python:3.11-slim-bookworm

# uv installs unpinned PyPI deps into the system interpreter
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# curl for operator healthchecks (no Docker HEALTHCHECK instruction)
# Tesseract ces+eng for PDF image OCR during ingest
# hadolint ignore=DL3008 -- apt package versions float with the slim base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-ces \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Pin mcp to 1.x — FastMCP lives there; mcp 2.x renames it to MCPServer
# fastembed: local BGE-M3 ONNX (~2.3GB download on first embed; plan ~4GB+ RAM)
RUN uv pip install --system \
    fastapi \
    uvicorn \
    "mcp[cli]<2" \
    pypdf \
    pymupdf \
    pillow \
    pytesseract \
    "qdrant-client>=1.12,<2" \
    requests \
    python-multipart \
    itsdangerous \
    fastembed

COPY main.py ./
COPY rag ./rag
COPY static ./static

# HTTP + MCP listen port metadata (runtime PORT defaults to 8080)
EXPOSE 8080

# Railway mounts a volume at /data; RAG artifacts live under /data/rag at runtime.
# Do not declare Docker VOLUME — Railway rejects it; use the service volume mount instead.
ENV RAG_DATA_ROOT=/data/rag

CMD ["python", "main.py"]
