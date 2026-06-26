# src/rag/config.py
"""Configuration for RAG system."""

import os
from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class RAGConfig:
    # Embedding & vector DB
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    vector_dimension: int = 384  # Depends on the model above; change these dimensions if the model changes.
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    
    # PostgreSQL / pgvector
    pg_host: str = "localhost"
    pg_port: int = 5431
    pg_db: str = "ciqual_db"
    pg_user: str = "ciqual"
    pg_password: str = "ciqual"
    pg_table: str = "food_composition_embeddings"
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    text_model: str = "llama3.2"
    vision_model: str = "llava"
    image_model: str = "x/flux2-klein"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Image handling
    enable_images: bool = True
    image_cache_dir: Optional[str] = "images/cache"
    
    # Prompt
    system_prompt: Optional[str] = None
    include_metadata: bool = True
    
    # Misc
    device: str = "cpu"  # or "cuda"
    
    @property
    def pg_connection_string(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load config from environment variables (prefixed with RAG_)."""
        return cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            vector_dimension=os.getenv("RAG_VECTOR_DIMENSION", cls.vector_dimension),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("RAG_TOP_K", cls.top_k)),
            pg_host=os.getenv("RAG_PG_HOST", cls.pg_host),
            pg_port=int(os.getenv("RAG_PG_PORT", cls.pg_port)),
            pg_db=os.getenv("RAG_PG_DB", cls.pg_db),
            pg_user=os.getenv("RAG_PG_USER", cls.pg_user),
            pg_password=os.getenv("RAG_PG_PASSWORD", cls.pg_password),
            pg_table=os.getenv("RAG_PG_TABLE", cls.pg_table),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", cls.ollama_base_url),
            text_model=os.getenv("RAG_TEXT_MODEL", cls.text_model),
            vision_model=os.getenv("RAG_VISION_MODEL", cls.vision_model),
            image_model=os.getenv("RAG_IMAGE_MODEL", cls.image_model),
            temperature=float(os.getenv("RAG_EMPERATURE", cls.temperature)),
            max_tokens=os.getenv("RAG_MAX_TOKENS", cls.max_tokens),
            enable_images=os.getenv("RAG_MAX_TOKENS", cls.enable_images),
            image_cache_dir=os.getenv("RAG_MAX_TOKENS", cls.image_cache_dir),
            device=os.getenv("RAG_DEVICE", cls.device),
        )    
    
    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Default configuration
default_config = RAGConfig()