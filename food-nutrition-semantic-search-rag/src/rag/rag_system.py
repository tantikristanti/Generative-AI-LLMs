# src/rag/rag_system.py
"""Main RAG system orchestrator."""

import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from rag import BaseRetriever, BasePromptBuilder, BaseLLMClient, \
    RetrievedDocument, RAGResponse, ImageAwareFoodRetriever, \
    FoodPromptBuilder, OllamaClient

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Main RAG system orchestrator.
    
    All components are injectable, making the system highly modular.
    """

    def __init__(self,
                 retriever: Optional[BaseRetriever] = None,
                 prompt_builder: Optional[BasePromptBuilder] = None,
                 llm_client: Optional[BaseLLMClient] = None,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 top_k: int = 5):
        """
        Initialize RAG system.
        
        Args:
            retriever: Retriever instance. If None, creates default FoodRetriever.
            prompt_builder: Prompt builder instance. If None, creates default.
            llm_client: LLM client. If None, creates default OllamaClient.
            embedding_model: Model name for embeddings.
            top_k: Default number of documents to retrieve.
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Initialize components with defaults if not provided
        self.retriever = retriever or ImageAwareFoodRetriever(model_name=embedding_model)
        self.prompt_builder = prompt_builder or FoodPromptBuilder()
        self.llm_client = llm_client or OllamaClient()
        
        logger.info(f"RAGSystem initialized with retriever={type(self.retriever).__name__}, "
                   f"prompt_builder={type(self.prompt_builder).__name__}, "
                   f"llm_client={type(self.llm_client).__name__}")

    def query(self, 
              question: str, 
              top_k: Optional[int] = None,
              system_prompt: Optional[str] = None,
              include_images: bool = False,
              **llm_kwargs) -> RAGResponse:
        """
        Execute RAG pipeline for a text query.
        
        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            system_prompt: Optional custom system prompt.
            include_images: Whether to include images in response.
            **llm_kwargs: Additional arguments for LLM.
        
        Returns:
            RAGResponse with retrieved documents and LLM answer.
        """
        top_k = top_k or self.top_k
        
        # Step 1: Retrieve relevant documents
        documents = self.retriever.search(question, top_k=top_k)
        
        # Step 2: Build prompts
        sys_prompt = system_prompt or self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(question, documents)
        
        # Step 3: Generate LLM response
        llm_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            **llm_kwargs
        )
        
        return RAGResponse(
            query=question,
            retrieved_documents=documents,
            llm_response=llm_response,
            model_used=llm_kwargs.get('model', 'default')
        )

    def query_multimodal(self, 
                         question: str,
                         images: List[Image.Image],
                         top_k: Optional[int] = None,
                         system_prompt: Optional[str] = None,
                         **llm_kwargs) -> RAGResponse:
        """
        Execute RAG pipeline with images (multimodal query).
        
        Args:
            question: User's question.
            images: List of images to include.
            top_k: Number of documents to retrieve.
            system_prompt: Optional custom system prompt.
            **llm_kwargs: Additional arguments for LLM.
        
        Returns:
            RAGResponse with retrieved documents and LLM answer.
        """
        top_k = top_k or self.top_k
        
        # Step 1: Retrieve documents (text-based, optionally with image)
        documents = self.retriever.search(question, top_k=top_k)
        
        # Step 2: Build multimodal prompt
        sys_prompt = system_prompt or self.prompt_builder.build_system_prompt()
        user_prompt_data = self.prompt_builder.build_multimodal_user_prompt(
            question, documents, images[0] if images else None
        )
        
        # Step 3: Generate multimodal LLM response
        if isinstance(user_prompt_data, dict):
            # Prompt with images
            llm_response = self.llm_client.generate_multimodal(
                prompt=user_prompt_data.get('prompt', question),
                images=user_prompt_data.get('images', images),
                system_prompt=sys_prompt,
                **llm_kwargs
            )
        else:
            # Text-only fallback
            llm_response = self.llm_client.generate(
                prompt=user_prompt_data,
                system_prompt=sys_prompt,
                **llm_kwargs
            )
        
        return RAGResponse(
            query=question,
            retrieved_documents=documents,
            llm_response=llm_response,
            model_used=llm_kwargs.get('model', 'default')
        )

    def generate_food_image(self, food_description: str, out_file: str, **kwargs) -> str:
        """
        Generate an image of a food based on description.
        
        Args:
            food_description: Description of the food.
            **kwargs: Additional arguments for image generation.
        
        Returns:
            Generated image.
        """
        return self.llm_client.generate_image(food_description, out_file, **kwargs)    

    def stream_query(self, question: str, top_k: Optional[int] = None,
                     system_prompt: Optional[str] = None, **llm_kwargs):
        """
        Stream RAG response token by token.
        
        Yields:
            Tokens from the LLM response.
        """
        top_k = top_k or self.top_k
        
        # Step 1: Retrieve
        documents = self.retriever.search(question, top_k=top_k)
        
        # Step 2: Build prompts
        sys_prompt = system_prompt or self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(question, documents)
        
        # Step 3: Stream LLM response
        for token in self.llm_client.stream_generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            **llm_kwargs
        ):
            yield token