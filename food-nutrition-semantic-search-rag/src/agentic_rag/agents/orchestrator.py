"""
Main orchestrator agent that coordinates the entire RAG process.
It leverages the refined SearchTools, QueryTools, and ResponseTools pipeline.
"""

import logging
import json
from typing import Optional

# Cross module import underr the same package (agentic_rag)
from ..tools import SearchTools, ResponseTools, SearchResult, FormattedResponse
from ..config import AgentConfig
from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that:
    1. Uses SearchTools to retrieve and rerank documents.
    2. Uses ResponseTools to generate the final answer and follow‑ups.
    """

    def __init__(
        self,
        search_tools: SearchTools,
        response_tools: ResponseTools,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        self.search_tools = search_tools
        self.response_tools = response_tools
        self.config = config or AgentConfig.default()

        instructions = """
        You are a Food Nutrition Assistant that helps users find information about
        food composition, nutritional values, and dietary information.

        You have access to:
        - A refined search pipeline (spelling correction, query rewriting,
          variation generation, embedding search, cross‑encoder reranking)
        - An answer generation module that uses the retrieved documents

        Your job is to:
        1. Accept the user's question.
        2. Run the refined search.
        3. Generate a clear, factual answer with sources.
        4. Suggest follow‑up questions.
        """

        super().__init__(
            name="OrchestratorAgent",
            instructions=instructions,
            tools=[],  # No tool‑calling needed; we directly call methods
            **kwargs,
        )

    def process(self, input_text: str, **kwargs) -> AgentResponse:
        """
        End‑to‑end processing:
        - refined search → formatted response
        """
        self.clear_memory()
        logger.info(f"Orchestrator processing: '{input_text}'")

        # Step 1: Refined search
        search_result: SearchResult = self.search_tools.refined_search(input_text)

        # Step 2: Generate complete response
        formatted_response: FormattedResponse = self.response_tools.respond_from_search_result(
            search_result
        )

        # Step 3: Build a structured output 
        output = {
            "query": formatted_response.query,
            "answer": formatted_response.answer,
            "sources": formatted_response.sources,
            "follow_ups": formatted_response.follow_ups,
            "total_sources": len(search_result.documents),
            "strategies_used": ["refined_search"],
            "refined_query": {
                "original": search_result.refined_query.original if search_result.refined_query else None,
                "corrected": search_result.refined_query.corrected if search_result.refined_query else None,
                "rewritten": search_result.refined_query.rewritten if search_result.refined_query else None,
                "variations": search_result.refined_query.variations if search_result.refined_query else [],
            } if search_result.refined_query else None,
        }

        return AgentResponse(
            success=len(search_result.documents) > 0,
            content=json.dumps(output, ensure_ascii=False, indent=2),
            iterations=1,  # single shot
            tool_calls_made=[],
        )