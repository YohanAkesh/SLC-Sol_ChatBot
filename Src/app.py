import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from Src.rag import RAGChatbot
from Src.ingestion import ingest_data, DATA_DIR, CHROMA_DIR

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "Src" / "static"
TEMPLATES_DIR = BASE_DIR / "Src" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="S.L.C Solutions RAG Chatbot",
    description="Intelligent AI Customer Support Assistant powered by Groq & ChromaDB",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Chatbot singleton lazily or on startup
chatbot_instance: Optional[RAGChatbot] = None

def get_chatbot() -> RAGChatbot:
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = RAGChatbot()
    return chatbot_instance

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str

class SourceItem(BaseModel):
    file: str
    page: int
    content_snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

class IngestResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return FileResponse(str(index_file))

@app.get("/api/health")
async def health_check():
    chroma_exists = CHROMA_DIR.exists()
    return {
        "status": "healthy",
        "chroma_db_initialized": chroma_exists,
        "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        bot = get_chatbot()
        result = bot.query(request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceItem(
                    file=s["file"],
                    page=int(s["page"]) if s["page"] is not None else 1,
                    content_snippet=s["content_snippet"]
                )
                for s in result["sources"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse)
async def trigger_ingest():
    try:
        global chatbot_instance
        ingest_data(DATA_DIR, CHROMA_DIR)
        chatbot_instance = RAGChatbot()  # Refresh vectorstore connection
        return IngestResponse(status="success", message="Knowledge Base successfully re-indexed into ChromaDB.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Src.app:app", host="127.0.0.1", port=8000, reload=True)
