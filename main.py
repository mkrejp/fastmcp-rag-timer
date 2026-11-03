import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests

# --- Configuration ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./storage/files")
DB_PATH = os.getenv("DB_PATH", "./storage/qdrant")
EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "client_domain_knowledge"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Initialize Apps
app = FastAPI(title="Lightweight RAG Engine")
mcp = FastMCP("Client RAG Node")

# Initialize Local Vector Store
qdrant = QdrantClient(path=DB_PATH)
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE) # text-embedding-3-small
    )

# --- Helper: Quick Vector Generation via OpenAI API ---
def get_embedding(text: str) -> list:
    if not EMBEDDING_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": text, "model": "text-embedding-3-small"}
    res = requests.post("https://openai.com", json=payload, headers=headers)
    return res.json()["data"][0]["embedding"]

# --- FastAPI Endpoints (Data Ingestion & Media Upload) ---

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported via this endpoint.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Extract Text & Chunk Document
    reader = PdfReader(file_path)
    chunks = []
    current_chunk = ""
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        current_chunk += text + "\n"
        if len(current_chunk) > 1000:  # ~1000 characters chunk sizing
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Ingest into Local Vector Store
    points = []
    for idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        points.append(PointStruct(
            id=hash(f"{file.filename}_{idx}") & 0xFFFFFFFFFFFFFFFF, # Keep ID within unsigned 64bit integer bounds
            vector=vector,
            payload={"text": chunk, "source": file.filename, "page": page_num}
        ))
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "Success", "filename": file.filename, "chunks_indexed": len(chunks)}

@app.post("/upload-media")
async def upload_media(file: UploadFile = File(...)):
    # Local filesystem storage endpoint for image/audio asset references
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "Saved", "path": file_path}

# --- Native MCP Tools (Exposed directly to your Client Agents) ---

@mcp.tool()
def search_domain_knowledge(query: str, limit: int = 3) -> str:
    """Queries the local client RAG vector storage for contextual facts."""
    try:
        query_vector = get_embedding(query)
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        context_blocks = []
        for res in results:
            context_blocks.append(f"Source: {res.payload['source']}\nContent: {res.payload['text']}\n")
            
        return "\n---\n".join(context_blocks) if context_blocks else "No relevant facts found."
    except Exception as e:
        return f"Error executing RAG search: {str(e)}"

# Boilerplate alignment for running via uvicorn wrapper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
