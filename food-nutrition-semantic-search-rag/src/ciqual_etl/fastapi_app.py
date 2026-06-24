#!/usr/bin/env python3
"""
FastAPI application for the food composition search engine.

Provides a REST endpoint to query the embedding-based food retriever.
Useful for RAG systems that need to fetch relevant food data before feeding it to an LLM.

This app uses Pydantic models to define the structure of request and response data for the FastAPI endpoints. 
They ensure that data sent to or returned from the API follows a predictable format, with automatic validation, serialization, and documentation.
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

from ciqual_etl import FoodRetriever

logger = logging.getLogger(__name__)

# Global retriever instance (reused across requests)
_retriever = None

def get_retriever() -> FoodRetriever:
    """Lazily initialise and return a singleton FoodRetriever."""
    global _retriever
    if _retriever is None:
        _retriever = FoodRetriever()
    return _retriever

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing needed here because retriever is lazy
    yield
    # Shutdown: close the retriever if it was created
    if _retriever is not None:
        _retriever.close()
        logger.info("Closed FoodRetriever database connection.")

# Pydantic models for request and response
class SearchRequest(BaseModel):
    """This model describes the expected JSON body when making a POST request to /search.
    query: a required string (... means required). The description is used in the generated OpenAPI documentation.
    top_k: an integer with a default value of 5, must be between 1 (greater than or equal to, ge) and 50 (less than or equal to, le). This controls how many similar foods to return.
    If the client sends invalid data (e.g., top_k = 0 or missing query), FastAPI automatically returns a clear error (422 Unprocessable Entity).
    """
    
    query: str = Field(..., description="Search query in French or English")
    top_k: int = Field(5, ge=1, le=50, description="Number of top results to return")

class SearchResultItem(BaseModel):
    """
    This model represents one item in the list of search results.
    - alim_code: the unique food code (can be None if missing in the source).
    - alim_nom_fr: French name of the food.
    - alim_nom_eng: English name of the food.
    - composition_text: the concatenated text (all nutrient values) that was used to generate the embedding.
    - metadata: a JSON object containing the original CSV row (e.g., all nutrient values, identifiers). Dict[str, Any] means any JSON‑compatible structure.
    - similarity: a float between -1 and 1, where 1 means the food is a perfect semantic match to the query.
    """
    
    alim_code: Optional[int]
    alim_nom_fr: Optional[str]
    alim_nom_eng: Optional[str]
    composition_text: str
    metadata: Dict[str, Any]
    similarity: float

class SearchResponse(BaseModel):
    """
    This model defines the structure of the API response.
    - query: echoes back the original search query (useful for logging/debugging).
    - top_k: echoes back the requested number of results.
    - results: a list of SearchResultItem objects (the actual matching foods).
    """
    
    query: str
    top_k: int
    results: List[SearchResultItem]

# Create FastAPI app with the lifespan manager
app = FastAPI(
    title="Ciqual Food Search Engine",
    description="Semantic search over French food composition data using vector embeddings.",
    version="1.0.0",
    lifespan=lifespan,   
)

@app.get("/")
async def root():
    return {
        "message": "Ciqual Food Search Engine API",
        "endpoints": {
            "/health": "Health check.",
            "/search": "POST or GET semantic search."
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_foods(request: SearchRequest):
    """
    Retrieve top‑k foods semantically similar to the query text.

    The search is performed using a multilingual sentence transformer,
    so queries in French or English both work.
    """
    retriever = get_retriever()
    try:
        results = retriever.search(query=request.query, top_k=request.top_k)
        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            results=[SearchResultItem(**item) for item in results]
        )
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_foods_get(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results")
):
    """GET version of the search endpoint (easier for testing)."""
    retriever = get_retriever()
    try:
        results = retriever.search(query=query, top_k=top_k)
        return SearchResponse(
            query=query,
            top_k=top_k,
            results=[SearchResultItem(**item) for item in results]
        )
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))
