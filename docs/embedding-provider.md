# Embedding provider and Qdrant index

## Provider

- API: `https://api.novita.ai/openai/v1/embeddings`
- Model: `baai/bge-m3`
- Vector dimension: `1024`
- Railway credential: `BGE_M3_NOVITA_API_KEY`
- Legacy aliases: `NOVITA_API_KEY`, `OPENAI_TOKEN`, `OPENAI_API_KEY`

The API key is read at runtime and sent as a Bearer token. Never put its value in source, docs, logs, or chat.

## Collection dimension

Qdrant requires every vector in a collection to have the collection's configured dimension. `open_qdrant()` creates the configured collection at 1024 dimensions. If an existing collection has a different dimension, startup deletes and recreates that collection to match the active model.

**This replacement removes vectors in the mismatched collection.** The app's file originals and processed text remain on disk, but semantic search has no vectors until documents are indexed again. For a populated deployment, back up the Qdrant volume and reindex documents before switching models. The current deployment has no uploaded documents, so recreating an old empty 1536-dimensional collection is safe.

## Verification

- Run `pytest -q`.
- Upload a document and confirm it is indexed.
- Search for content from that document.
