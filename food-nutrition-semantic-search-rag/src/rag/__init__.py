#!/usr/bin/env python3
"""RAG package for food nutrition search.
Lazy imports to avoid circular dependencies and speed up package loading.
"""

__version__ = "1.0.0"

import importlib

_imported = {}

def __getattr__(name):
    if name in _imported:
        return _imported[name]

    module_map = {
        "RAGSystem": "rag_system",
        "RAGConfig": "config",
        "OllamaClient": "ollama_client",
        "FoodPromptBuilder": "prompt_builder",
        "ImageAwareFoodRetriever": "food_image_retriever",
        "BaseRetriever": "rag_base",
        "BasePromptBuilder": "rag_base",
        "BaseLLMClient": "rag_base",
        "RetrievedDocument": "rag_base",
        "RAGResponse": "rag_base",
    }
    if name not in module_map:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_name = module_map[name]
    module = importlib.import_module(f".{module_name}", package=__name__)
    attr = getattr(module, name)
    _imported[name] = attr
    return attr

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