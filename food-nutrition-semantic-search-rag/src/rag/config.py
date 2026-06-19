# src/rag/config.py
"""Configuration for RAG system."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    
    # Embedding
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Retrieval
    top_k: int = 5
    table_name: str = "food_composition_embeddings"
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    text_model: str = "llama3.2"
    vision_model: str = "llava"
    image_model: str = "x/flux2-klein"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Image handling
    enable_images: bool = True
    image_cache_dir: Optional[str] = "data/images/cache"
    
    # Prompt
    system_prompt: Optional[str] = None
    include_metadata: bool = True

# Default configuration
default_config = RAGConfig()