# Use a highly optimized, modern python base image
FROM python:3.11-slim-bookworm

# Install UV for lightning fast, cached dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install basic system curl requirements for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install specific Python requirements explicitly to prevent bloat
RUN uv pip install --system \
    fastapi \
    uvicorn \
    mcp[cli] \
    pypdf \
    qdrant-client \
    requests \
    python-multipart

COPY main.py .

# Expose Web Interface Port
EXPOSE 8080

# Setup Persistent Storage Volume Target for Railway
VOLUME ["/app/storage"]

CMD ["python", "main.py"]
