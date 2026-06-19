#!/usr/bin/env python3
"""RAG package for food nutrition search.
Lazy imports to avoid circular dependencies and speed up package loading.
"""

__version__ = "1.0.0"

def __getattr__(name):
    """Lazy import of submodules to avoid circular dependencies."""
    if name == "RAGSystem":
        from .rag_system import RAGSystem
        return globals().setdefault(name, RAGSystem)
    elif name == "RAGConfig":
        from .config import RAGConfig
        return globals().setdefault(name, RAGConfig)
    elif name == "OllamaClient":
        from .ollama_client import OllamaClient
        return globals().setdefault(name, OllamaClient)
    elif name == "FoodPromptBuilder":
        from .prompt_builder import FoodPromptBuilder
        return globals().setdefault(name, FoodPromptBuilder)
    elif name == "BaseRetriever":
        from .rag_base import BaseRetriever
        return globals().setdefault(name, BaseRetriever)
    elif name == "BasePromptBuilder":
        from .rag_base import BasePromptBuilder
        return globals().setdefault(name, BasePromptBuilder)
    elif name == "BaseLLMClient":
        from .rag_base import BaseLLMClient
        return globals().setdefault(name, BaseLLMClient)
    elif name == "RetrievedDocument":
        from .rag_base import RetrievedDocument
        return globals().setdefault(name, RetrievedDocument)
    elif name == "RAGResponse":
        from .rag_base import RAGResponse
        return globals().setdefault(name, RAGResponse)
    elif name == "ImageAwareFoodRetriever":
        from .food_image_retriever import ImageAwareFoodRetriever
        return globals().setdefault(name, ImageAwareFoodRetriever)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "RAGSystem",
    "RAGConfig",
    "OllamaClient",
    "FoodPromptBuilder",
    "ImageAwareFoodRetriever",
    "BaseRetriever",
    "BasePromptBuilder",
    "BaseLLMClient",
    "RetrievedDocument",
    "RAGResponse",
]