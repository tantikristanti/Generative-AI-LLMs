# src/agentic_rag/agents/search_agent.py

import logging

# Cross module import underr the same package (agentic_rag)
from ..tools import SearchTools
from ..agents import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

class SearchAgent(BaseAgent):
    """
    Agent responsible for executing searches.
    It leverages the refined search pipeline (correction, rewrite, variations, rerank).
    """

    def __init__(
        self,
        search_tools: SearchTools,
        min_results_threshold: int = 2,
        use_fallback: bool = True,
        **kwargs
    ):
        instructions = """
        You are a Search Agent responsible for finding relevant documents in a food
        nutrition database.
        
        You use a state‑of‑the‑art pipeline:
        - Spelling correction (SymSpell)
        - Query rewriting (LLM)
        - Query variation generation (LLM)
        - Embedding search on all variations
        - Cross‑encoder reranking
        
        Your primary tool is 'refined_search', which already handles all these steps.
        If you get too few results, you can try 'raw_search' as a fallback.
        """
        
        self.search_tools = search_tools
        self.min_results_threshold = min_results_threshold
        self.use_fallback = use_fallback
        
        # We define tools for the agent. 
        # But since we're not using a full tool‑calling loop in this simplified version, we use the methods directly.
        super().__init__(
            name="SearchAgent",
            instructions=instructions,
            tools=[],  # We'll not use tool‑calling schema here; we'll just call methods.
            **kwargs
        )
    
    def process(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Execute search with refinement, fallback to raw if necessary.
        """
        self.clear_memory()
        query = input_text
        strategies_used = []
        attempts = 0
        
        logger.info(f"SearchAgent processing: '{query}'")
        
        # Primary: refined search
        result = self.search_tools.refined_search(query)
        strategies_used.append("refined")
        attempts += 1
        
        docs = result.documents
        
        # If we have enough results, return immediately
        if len(docs) >= self.min_results_threshold:
            logger.info(f"Refined search returned {len(docs)} results, enough.")
            return self._create_response(docs, strategies_used, attempts, result.refined_query)
        
        # If not enough, optionally try raw search as fallback
        if self.use_fallback:
            logger.info(f"Refined search returned only {len(docs)} results, trying raw fallback.")
            raw_result = self.search_tools.raw_search(query, top_k=self.min_results_threshold)
            strategies_used.append("raw_fallback")
            attempts += 1
            
            # Merge results: prefer refined docs, then add raw docs if not already present
            existing_ids = {doc.metadata.get("alim_code", doc.content[:50]) for doc in docs}
            for doc in raw_result.documents:
                doc_id = doc.metadata.get("alim_code", doc.content[:50])
                if doc_id not in existing_ids:
                    docs.append(doc)
                    existing_ids.add(doc_id)
            
            # Re‑sort by score (we might want to keep refined ones on top, but we can sort by score)
            docs.sort(key=lambda d: d.score, reverse=True)
        
        return self._create_response(docs, strategies_used, attempts, result.refined_query)
    
    def _create_response(self, documents, strategies_used, attempts, refined_query=None):
        """Create a formatted AgentResponse."""
        content = {
            "total_found": len(documents),
            "strategies_used": strategies_used,
            "attempts": attempts,
            "documents": [
                {
                    "content": doc.content,
                    "score": getattr(doc, "rerank_score", doc.score),
                    "metadata": doc.metadata,
                }
                for doc in documents[:10]       # Limit
            ],
        }
        if refined_query:
            content["refined_query"] = {
                "original": refined_query.original,
                "corrected": refined_query.corrected,
                "rewritten": refined_query.rewritten,
                "variations": refined_query.variations,
            }
        
        return AgentResponse(
            success=len(documents) > 0,
            content=str(content),
            iterations=attempts,
            tool_calls_made=[], 
        )