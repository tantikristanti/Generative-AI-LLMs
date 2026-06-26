# src/rag/rag_system.py
"""Main RAG system orchestrator."""

import logging
from typing import List, Optional
from PIL import Image

# Sibling module imports
from .rag_base import BaseRetriever, BasePromptBuilder, BaseLLMClient, RAGResponse
from .food_image_retriever import ImageAwareFoodRetriever
from .prompt_builder import FoodPromptBuilder
from .ollama_client import OllamaClient

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
        
    def query_multimodal(
        self,
        question: str,
        images: List[Image.Image],
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **llm_kwargs
    ) -> RAGResponse:
        """
        Execute RAG pipeline with images (multimodal query).
        Improved: generate image description and use it to enhance retrieval.
        """
        top_k = top_k or self.top_k
        image_description = ""

        # ----- Step 1: Generate image description (if images are provided) -----
        if images:
            # Get the description prompt from the PromptBuilder
            desc_prompt = self.prompt_builder.build_image_description_prompt()
            try:
                # Use the multimodal LLM to describe the image
                # We use the same llm_client but with a specific system prompt for description
                desc_system_prompt = "You are a helpful assistant that accurately describes food images."
                desc_response = self.llm_client.generate_multimodal(
                    prompt=desc_prompt,
                    images=[images[0]],  # Use the first image
                    system_prompt=desc_system_prompt,
                    **llm_kwargs
                )
                image_description = desc_response.strip()
            except Exception as e:
                # Log and continue without description (fail gracefully)
                print(f"Warning: could not generate image description: {e}")
                image_description = ""

        # ----- Step 2: Build combined query for retrieval -----
        # Combine the user's question with the image description (if available)
        combined_query = question
        if image_description:
            combined_query = f"{question} Image description: {image_description}"

        # ----- Step 3: Retrieve documents using the combined query -----
        documents = self.retriever.search(combined_query, top_k=top_k)

        # ----- Step 4: Build multimodal prompt for final answer -----
        sys_prompt = system_prompt or self.prompt_builder.build_system_prompt()

        # Use the prompt builder to create the user prompt,
        # now passing the image_description as extra context
        user_prompt_data = self.prompt_builder.build_multimodal_user_prompt(
            query=question,          # original question (not combined)
            documents=documents,
            image=images[0] if images else None,
            image_description=image_description,   # pass the description
        )

        # ----- Step 5: Generate final multimodal response -----
        # The prompt builder returns a dict with 'prompt' and 'images'
        llm_response = self.llm_client.generate_multimodal(
            prompt=user_prompt_data.get('prompt', question),
            images=user_prompt_data.get('images', images),
            system_prompt=sys_prompt,
            **llm_kwargs
        )

        # ----- Step 6: Return the result -----
        return RAGResponse(
            query=question,
            retrieved_documents=documents,
            llm_response=llm_response,
            model_used=llm_kwargs.get('model', 'default'),
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