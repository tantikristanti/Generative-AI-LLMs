# src/rag/rag_base.py
"""Abstract base classes for RAG components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from PIL import Image

@dataclass
class RetrievedDocument:
    """A document retrieved from the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    score: float
    image_url: Optional[str] = None
    image: Optional[Image.Image] = None

@dataclass
class RAGResponse:
    """Complete RAG response."""
    query: str
    retrieved_documents: List[RetrievedDocument]
    llm_response: str
    model_used: str
    tokens_used: Optional[int] = None

class BaseRetriever(ABC):
    """Abstract retriever for knowledge base search."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Retrieve relevant documents from the knowledge base."""
        pass

    @abstractmethod
    def search_with_image(self, image: Image.Image, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Retrieve relevant documents using an image query (multimodal)."""
        pass

class BasePromptBuilder(ABC):
    """Abstract prompt builder for RAG."""

    @abstractmethod
    def build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build the system prompt."""
        pass

    @abstractmethod
    def build_user_prompt(self, query: str, documents: List[RetrievedDocument]) -> str:
        """Build the user prompt with retrieved context."""
        pass

    @abstractmethod
    def build_multimodal_user_prompt(self, query: str, documents: List[RetrievedDocument], 
                                     image: Optional[Image.Image] = None) -> Union[str, Dict]:
        """Build multimodal user prompt (text + optional image)."""
        pass

class BaseLLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 **kwargs) -> str:
        """Generate a response from text prompt."""
        pass

    @abstractmethod
    def generate_multimodal(self, prompt: str, images: List[Image.Image],
                           system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from text + images."""
        pass

    @abstractmethod
    def generate_image(self, prompt: str, out_file: str, **kwargs) -> Image.Image:
        """Generate an image from text prompt."""
        pass