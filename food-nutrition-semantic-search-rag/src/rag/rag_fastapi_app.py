# src/rag/rag_fastapi_app.py
"""FastAPI endpoints for RAG system."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import logging
import tempfile
import os

from rag import RAGSystem

logger = logging.getLogger(__name__)

# Global RAG instance (lazy loaded)
_rag_system = None

def get_rag_system() -> RAGSystem:
    """Lazy load RAG system."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system

def _check_db_connection(rag: RAGSystem) -> tuple[bool, str]:
    """
    Check the database connectivity by performing a minimal search.
    Returns (is_ok, message).
    """
    try:
        # Force a search to trigger the connection and a simple query.
        # Use top_k=1 to minimise overhead.
        rag.retriever.search("test", top_k=1)
        return True, "connected"
    except Exception as e:
        return False, f"Database connection failed: {e}"

def _close_rag_system(rag: RAGSystem):
    """
    Safely close the RAG system resources (database connection, etc.).
    Handles cases where the close method might not exist.
    """
    if rag is None:
        return
    
    # First, try to close the RAGSystem itself (if it has a close method)
    if hasattr(rag, 'close') and callable(rag.close):
        rag.close()
        logger.info("Closed RAGSystem.")
        return
    
    # Otherwise, try to close the retriever
    if hasattr(rag, 'retriever'):
        retriever = rag.retriever
        if hasattr(retriever, 'close') and callable(retriever.close):
            retriever.close()
            logger.info("Closed FoodRetriever database connection.")
            return
    
    # If nothing else, log a warning (the connection will be closed on process exit)
    logger.warning("No close method found for RAGSystem or its retriever. Skipping explicit cleanup.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check database connectivity
    logger.info("Starting up RAG system...")
    rag = get_rag_system()
    ok, msg = _check_db_connection(rag)
    if ok:
        logger.info("Database connection established successfully.")
    else:
        logger.error(f"Database connection failed on startup: {msg}")
        # Optionally raise to prevent startup:
        # raise RuntimeError(f"Database connection failed: {msg}")
    yield
    # Shutdown: safely close resources
    _close_rag_system(_rag_system)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    model: Optional[str] = None
    temperature: float = 0.7

class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[dict]
    model: str

# FastAPI app with lifespan
app = FastAPI(
    title="Food Nutrition RAG API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "Food Nutrition RAG API",
        "endpoints": {
            "/health": "Health check",
            "/rag/query": "POST text-based RAG system.",
            "/rag/multimodal": "POST multimodal RAG system using text and images.",
            "/rag/generate-image": "POST food image generation.",
            "/rag/stream": "POST stream RAG."
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies database connectivity."""
    rag = get_rag_system()
    ok, msg = _check_db_connection(rag)
    return {
        "status": "ok" if ok else "degraded",
        "database": msg
    }

@app.post("/rag/query")
async def rag_query(request: QueryRequest):
    """Query the RAG system with text."""
    rag = get_rag_system()
    response = rag.query(
        question=request.query,
        top_k=request.top_k,
        **{"model": request.model, "temperature": request.temperature}
    )
    
    return QueryResponse(
        query=response.query,
        answer=response.llm_response,
        documents=[{
            "food_en": doc.metadata.get("alim_nom_eng", "Unknown"),
            "food_fr": doc.metadata.get("alim_nom_fr", "Unknown"),
            "score": doc.score,
            "content": doc.content[:500] + "...",
            "image_url": doc.image_url
        } for doc in response.retrieved_documents],
        model=response.model_used
    )

@app.post("/rag/multimodal")
async def rag_multimodal(
    query: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(5),
    model: Optional[str] = Form(None),
    temperature: float = Form(0.7)
):
    """Query the RAG system with text and an image."""
    # Validate image
    '''When a user uploads a file in a framework like FastAPI, the file object contains a content_type attribute.
    Possible values: image/jpeg, image/png, image/webp, image/gif, application/pdf, text/plain
    '''
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Load image
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    rag = get_rag_system()
    response = rag.query_multimodal(
        question=query,
        images=[img],
        top_k=top_k,
        **{"model": model, "temperature": temperature}
    )
    
    return QueryResponse(
        query=response.query,
        answer=response.llm_response,
        documents=[{
            "food_en": doc.metadata.get("alim_nom_eng", "Unknown"),
            "food_fr": doc.metadata.get("alim_nom_fr", "Unknown"),
            "score": doc.score,
            "content": doc.content[:500] + "...",
            "image_url": doc.image_url
        } for doc in response.retrieved_documents],
        model=response.model_used
    )
    
@app.post("/rag/generate-image")
async def generate_food_image(
    description: str = Form(...),
    model: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Generate a food image from text description.
    Returns the generated image as a PNG file.
    """
    rag = get_rag_system()
    
    # Create a temporary file for the output image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Call the RAG system's image generation method
        # The method expects out_file as a positional argument
        rag.generate_food_image(description, out_file=tmp_path, model=model)
        
        # Schedule deletion of the temporary file after the response
        background_tasks.add_task(os.unlink, tmp_path)
        
        # Return the image file
        return FileResponse(
            tmp_path,
            media_type="image/png",
            filename="food_image.png"
        )
    except Exception as e:
        # Clean up the temporary file if an error occurs
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"Image generation failed: {str(e)}")

@app.post("/rag/stream")
async def rag_stream(request: QueryRequest):
    """Stream RAG response token by token."""
    rag = get_rag_system()
    
    def generate():
        for token in rag.stream_query(
            question=request.query,
            top_k=request.top_k,
            **{"model": request.model, "temperature": request.temperature}
        ):
            yield token
    
    return StreamingResponse(generate(), media_type="text/plain")