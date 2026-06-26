"""
Search tools for the agent, now powered by the QueryTools pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Cross‑package imports
from rag.rag_base import RetrievedDocument
from rag.food_image_retriever import ImageAwareFoodRetriever
from rag.ollama_client import OllamaClient

# Sibling module imports
from .query_tools import QueryTools, RefinedQuery

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result of a search operation."""
    query: str
    documents: List[RetrievedDocument]
    total_found: int
    search_type: str
    refined_query: Optional[RefinedQuery] = None

class SearchTools:
    """
    Collection of search-related tools for the agent.
    Uses the new QueryTools pipeline for refinement, retrieval, and reranking.
    """

    def __init__(
        self,
        retriever: ImageAwareFoodRetriever,
        llm_client: OllamaClient,
        embedding_model_name: Optional[str] = None,  # kept for future use
        reranker_model_name: str = "/Volumes/TantiK/Hugging_Face/cross-encoder/ms-marco-MiniLM-L-6-v2",
        spell_dictionary_path: str = "dictionary/SymSpell/frequency_dictionary_en_82_765.txt",
        top_k: int = 20,          # candidates before reranking
        final_top_k: int = 5,     # final returned documents
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        
        # Instantiate the pipeline
        self.query_tools = QueryTools(
            llm_client=llm_client,
            retriever=retriever,
            embedding_model=None,  # not needed for retrieval; kept for potential extensions
            reranker_model_name=reranker_model_name,
            spell_dictionary_path=spell_dictionary_path,
            top_k=top_k,
            final_top_k=final_top_k,
        )
        self._search_history: List[Dict] = []

    def refined_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """
        Perform a full refined search:
        - Spelling correction
        - Query rewriting
        - Multi‑variation generation
        - Embedding search on all variations
        - Cross‑encoder reranking
        
        Returns a SearchResult containing the final top documents.
        """
        logger.info(f"Refined search for: '{query}'")
        
        # If top_k is provided, override the pipeline's final_top_k temporarily
        if top_k is not None:
            original_final = self.query_tools.final_top_k
            self.query_tools.final_top_k = top_k
        
        try:
            docs, refined = self.query_tools.retrieve_and_rerank(query)
        finally:
            if top_k is not None:
                self.query_tools.final_top_k = original_final
        
        result = SearchResult(
            query=query,
            documents=docs,
            total_found=len(docs),
            search_type="refined",
            refined_query=refined,
        )
        
        self._search_history.append({
            "query": query,
            "results_count": len(docs),
            "search_type": "refined",
            "corrected": refined.corrected,
            "rewritten": refined.rewritten,
            "variations": refined.variations,
        })
        
        return result

    def raw_search(self, query: str, top_k: int = 5) -> SearchResult:
        """
        Fallback: perform a direct vector search without any refinement.
        Useful for comparison or when the refined pipeline fails.
        """
        logger.info(f"Raw vector search for: '{query}'")
        docs = self.retriever.search(query, top_k=top_k)
        
        result = SearchResult(
            query=query,
            documents=docs,
            total_found=len(docs),
            search_type="raw_vector",
            refined_query=None,
        )
        
        self._search_history.append({
            "query": query,
            "results_count": len(docs),
            "search_type": "raw_vector",
        })
        
        return result

    def get_search_history(self) -> List[Dict]:
        return self._search_history

    def clear_history(self):
        self._search_history = []