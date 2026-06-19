# src/rag/food_image_retriever.py
"""Retriever implementation using existing FoodRetriever from ciqual_etl."""

import logging
from typing import List, Dict, Any, Optional
from PIL import Image

from ciqual_etl import FoodRetriever as BaseFoodRetriever
from rag import BaseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)

# Expand the FoodRetriever class in the ciqual etl package
class ImageAwareFoodRetriever(BaseRetriever):
    """
    Extends the base FoodRetriever with image support for multimodal search.
    """

    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 table_name: str = "food_composition_embeddings",
                 image_handler: Optional['ImageHandler'] = None):
        """
        Initialize the retriever.
        
        Args:
            model_name: Name of the sentence transformer model.
            table_name: PostgreSQL table name for embeddings.
            image_handler: Optional handler for fetching food images.
        """
        self._retriever = BaseFoodRetriever(model_name=model_name) # FoodRetriever from etl_ciqual package
        self.table_name = table_name
        self.image_handler = image_handler
        self.model_name = model_name

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Search the knowledge base using text query."""
        results = self._retriever.search(query, top_k=top_k)
        
        documents = []
        for r in results:
            doc = RetrievedDocument(
                content=r.get("composition_text", ""),
                metadata=r.get("metadata", {}),
                score=r.get("similarity", 0.0),
            )
            # Fetch image if handler is available
            if self.image_handler:
                alim_code = r.get("alim_code")
                if alim_code:
                    doc.image_url = self.image_handler.get_image_url(alim_code)
                    doc.image = self.image_handler.fetch_image(alim_code)
            documents.append(doc)
        
        return documents

    def search_with_image(self, image: Image.Image, top_k: int = 5, 
                          **kwargs) -> List[RetrievedDocument]:
        """
        Retrieve using an image query.
        
        Note: This uses CLIP or similar multimodal model to encode the image.
        For now, falls back to text search with extracted metadata.
        """
        # TODO: Implement CLIP-based image search
        # For now, use a placeholder approach
        logger.warning("Image search not fully implemented. Falling back to text search.")
        return self.search("food image", top_k=top_k, **kwargs)

class ImageHandler:
    """
    Handler for fetching food images from Open Food Facts.
    """
    
    IMAGE_BASE_URL = "https://images.openfoodfacts.org/images/products/"
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._cache = {}
    
    def get_image_url(self, alim_code: int, size: str = "400") -> Optional[str]:
        """
        Build the Open Food Facts image URL for a given food code.
        
        Open Food Facts uses barcodes. Since CIQUAL codes are different,
        we need a mapping table (alim_code -> barcode).
        """
        # Placeholder: you need a mapping table
        # For now, return None or a placeholder
        return None
    
    def fetch_image(self, alim_code: int) -> Optional[Image.Image]:
        """Fetch and cache an image for a food."""
        url = self.get_image_url(alim_code)
        if not url:
            return None
        
        # Check cache
        if alim_code in self._cache:
            return self._cache[alim_code]
        
        # Download and cache
        try:
            import requests
            from io import BytesIO
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if self.cache_dir:
                    # Save to disk cache
                    pass
                self._cache[alim_code] = img
                return img
        except Exception as e:
            logger.error(f"Failed to fetch image for {alim_code}: {e}")
        
        return None