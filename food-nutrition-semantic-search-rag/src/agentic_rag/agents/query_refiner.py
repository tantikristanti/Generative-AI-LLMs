"""
Query Refiner Agent that wraps QueryTools for agentic decision‑making.
Provides a standalone agent that can be used independently or by the Orchestrator.
"""

import logging
import json

# Cross module import underr the same package (agentic_rag)
from ..tools import QueryTools, RefinedQuery
from ..agents import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

class QueryRefinerAgent(BaseAgent):
    """
    Agent responsible for refining user queries.
    
    Uses QueryTools for:
    - Spelling correction (SymSpell)
    - Query rewriting (LLM)
    - Multiple variation generation (LLM)
    
    This agent can be used as a standalone preprocessing step or integrated into a larger agentic workflow.
    """

    def __init__(
        self,
        query_tools: QueryTools,
        always_refine: bool = True,
        **kwargs,
    ):
        """
        Args:
            query_tools: An instance of QueryTools.
            always_refine: If True, always run refinement. If False, the agent
                           could decide whether to refine based on confidence
                           (simplified here, always True).
        """
        self.query_tools = query_tools
        self.always_refine = always_refine

        instructions = """
        You are a Query Refiner Agent. Your job is to preprocess and refine user queries
        to improve search quality in a food nutrition database.

        Your refinement pipeline includes:
        1. Spelling correction using SymSpell.
        2. Query rewriting using an LLM to make the query clearer.
        3. Generation of multiple search variations (up to 3).

        You produce a RefinedQuery object containing the original, corrected,
        rewritten, and variation queries.
        """

        super().__init__(
            name="QueryRefinerAgent",
            instructions=instructions,
            tools=[],  # No external tools – we directly call query_tools methods
            **kwargs,
        )

    def process(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Refine the input query and return the results.

        Args:
            input_text: The raw user query.

        Returns:
            AgentResponse with content containing the RefinedQuery details.
        """
        self.clear_memory()
        logger.info(f"QueryRefinerAgent processing: '{input_text}'")

        # If always_refine is True, run the full pipeline.
        # In a more advanced version, the agent could decide based on
        # confidence scores, character‑level analysis, etc.
        if self.always_refine:
            refined: RefinedQuery = self.query_tools.refine(input_text)
        else:
            # Placeholder for a "decision" step – here we just pass through.
            refined = RefinedQuery(
                original=input_text,
                corrected=input_text,
                rewritten=input_text,
                variations=[input_text],
            )

        # Build a structured output
        output = {
            "original": refined.original,
            "corrected": refined.corrected,
            "rewritten": refined.rewritten,
            "variations": refined.variations,
            "refined": self.always_refine,
        }

        return AgentResponse(
            success=True,
            content=json.dumps(output, ensure_ascii=False, indent=2),
            iterations=1,
            tool_calls_made=[],
        )