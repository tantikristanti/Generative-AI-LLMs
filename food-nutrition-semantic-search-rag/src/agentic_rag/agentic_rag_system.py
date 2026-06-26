"""
Convenience wrapper that instantiates the entire agentic RAG pipeline from a single config.
"""

import logging
from typing import Optional, Dict, Any
import json

# Cross‑package imports
from rag.config import RAGConfig
from rag.food_image_retriever import ImageAwareFoodRetriever
from rag.ollama_client import OllamaClient

# Sibling module imports
from .tools import SearchTools, ResponseTools
from .agents import OrchestratorAgent
from .config import AgentConfig

logger = logging.getLogger(__name__)

class AgenticRAGSystem:
    """
    High‑level API for the agentic RAG system.
    """

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        llm_client = None,
        retriever = None,
        reranker_model_name: str = "/Volumes/TantiK/Hugging_Face/cross-encoder/ms-marco-MiniLM-L-6-v2",
        spell_dictionary_path: str = "dictionary/SymSpell/frequency_dictionary_en_82_765.txt",
    ):
        self.rag_config = rag_config or RAGConfig()
        self.agent_config = agent_config or AgentConfig.default()
        self.agent_config.rag_config = self.rag_config

        # Instantiate retriever and LLM if not provided
        self.retriever = retriever or ImageAwareFoodRetriever(config=self.rag_config)
        self.llm_client = llm_client or OllamaClient(config=self.rag_config)

        # Build tools
        self.search_tools = SearchTools(
            retriever=self.retriever,
            llm_client=self.llm_client,
            reranker_model_name=reranker_model_name,
            spell_dictionary_path=spell_dictionary_path,
        )
        self.response_tools = ResponseTools(llm_client=self.llm_client)

        # Build orchestrator
        self.orchestrator = OrchestratorAgent(
            search_tools=self.search_tools,
            response_tools=self.response_tools,
            config=self.agent_config,
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query and return a structured dictionary."""
        response = self.orchestrator.process(question)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "query": question,
                "answer": response.content,
                "sources": [],
                "error": "Failed to parse response"
            }

    def search_only(self, query: str) -> Dict[str, Any]:
        """Return only the search results (no LLM answer)."""
        search_result = self.search_tools.refined_search(query)
        return {
            "query": query,
            "documents": [
                {
                    "content": doc.content,
                    "score": getattr(doc, "rerank_score", doc.score),
                    "metadata": doc.metadata,
                }
                for doc in search_result.documents
            ],
            "total_found": len(search_result.documents),
        }