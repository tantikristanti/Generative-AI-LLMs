# src/rag/prompt_builder.py
"""Prompt builder for RAG system."""

import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from rag import BasePromptBuilder, RetrievedDocument
import re

logger = logging.getLogger(__name__)

class FoodPromptBuilder(BasePromptBuilder):
    """
    Prompt builder for food nutrition RAG.
    Supports both text-only and multimodal prompts.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a nutrition expert assistant. Your role is to:
1. Answer questions about food nutrition based on the provided data.
2. Provide accurate, evidence-based responses.
3. Cite specific foods and their nutritional values when relevant.
4. When the user asks about images, describe the food visually.
5. Respond in the same language as the user's query (French or English).

If the information is not available in the retrieved data, say so clearly.
Do not make up nutritional values that are not supported by the data."""

    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 include_metadata: bool = True):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.include_metadata = include_metadata

    def build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build the system prompt with optional context."""
        if context:
            return f"{self.system_prompt}\n\nAdditional context: {context}"
        return self.system_prompt

    def build_user_prompt(self, query: str, documents: List[RetrievedDocument]) -> str:
        """Build user prompt with retrieved documents."""
        
        if not documents:
            return f"User question: {query}\n\nNo relevant food data found."
        
        context = self._format_documents(documents)
        user_prompt = f"""Based on the following food composition data, answer the user's question.
        
RETRIEVED FOODS:
{context}

USER QUESTION: {query}

Please provide a clear, accurate answer based only on the data above. Include specific food names and their nutritional values when relevant."""
        
        return user_prompt

    def build_multimodal_user_prompt(self, query: str, 
                                     documents: List[RetrievedDocument],
                                     image: Optional[Image.Image] = None) -> Union[str, Dict]:
        """
        Build multimodal user prompt.
        For Ollama multimodal, returns a dict with text and images.
        """
        text_prompt = self.build_user_prompt(query, documents)
        
        if image:
            return {
                "prompt": text_prompt,
                "images": [image]
            }
        return text_prompt

    def _format_documents(self, documents: List[RetrievedDocument]) -> str:
        formatted = []
        for i, doc in enumerate(documents, 1):
            food_name_en = doc.metadata.get('alim_nom_eng', 'Unknown').strip()
            food_name_fr = doc.metadata.get('alim_nom_fr', 'Unknown').strip()
            lines = [f"Food {i}.  Name (English): {food_name_en} | \n \
                        Name (French): {food_name_fr} | \n \
                        Nutritional Information -  "
                    ]
            
            # Parse the composition_text to extract key nutrients
            content = doc.content
            nutrient_patterns = {
                'Energy (kJ 100g)': r'Energy[^:]*:\s*([\d.]+)',
                'Protein (g 100g)': r'Protein[^:]*:\s*([\d.]+)',
                'Fat (g 100g)': r'Fat[^:]*:\s*([\d.]+)',
                'Carbohydrate (g 100g)': r'Carbohydrate[^:]*:\s*([\d.]+)',
                'Sugars (g 100g)': r'Sugars[^:]*:\s*([\d.]+)',
            }
            
            for nutrient, pattern in nutrient_patterns.items():
                match = re.search(pattern, content)
                if match:
                    lines.append(f"{nutrient}: {match.group(1)}")
            
            formatted.append("\n".join(lines))
        
        return "\n\n".join(formatted)