"""
Response tools for the agent.
Handles answer generation, formatting, and follow‑up suggestions.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Cross‑package imports
from rag.rag_base import RetrievedDocument
from rag.ollama_client import OllamaClient

# Sibling module imports
from.query_tools import RefinedQuery

logger = logging.getLogger(__name__)

@dataclass
class FormattedResponse:
    """Complete response with answer, sources, and suggestions."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    follow_ups: List[str]
    refined_query: Optional[RefinedQuery] = None

class ResponseTools:
    """
    Tools for generating and formatting responses from retrieved documents.
    Uses the LLM for answer generation and follow‑up suggestions.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        max_sources: int = 5,
        include_scores: bool = True,
    ):
        self.llm = llm_client
        self.max_sources = max_sources
        self.include_scores = include_scores

    def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        refined_query: Optional[RefinedQuery] = None,
        max_sources: Optional[int] = None,
    ) -> str:
        """
        Generate a natural language answer from the retrieved documents.

        Args:
            query: Original user query.
            documents: List of retrieved documents (already reranked).
            refined_query: Optional refined query info for context.
            max_sources: Override for number of sources to include.

        Returns:
            Generated answer string.
        """
        if not documents:
            return "I couldn't find any relevant information in the database for your query."

        # Use provided max_sources or fallback to instance default
        sources_limit = max_sources or self.max_sources
        docs_to_use = documents[:sources_limit]

        # Build context
        context_parts = []
        for i, doc in enumerate(docs_to_use, 1):
            # Use rerank_score if available, else fallback to original score
            score = getattr(doc, "rerank_score", doc.score)
            context_parts.append(
                f"[Document {i}] (relevance: {score:.3f})\n{doc.content}\n"
            )
        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""
You are a food nutrition assistant. Use the following context from the Ciqual database to answer the user's question.

Question: {query}

Context:
{context}

Instructions:
- Answer based ONLY on the provided context.
- If the context does not contain enough information, say so clearly.
- Be concise, factual, and easy to read.
- When possible, include specific nutritional values.
- Do not mention the document numbers or relevance scores in your answer.
"""
        # If we have refined query info, optionally add it for better generation
        if refined_query:
            prompt += f"\nNote: The search was refined from '{refined_query.original}' to '{refined_query.rewritten}' with variations: {', '.join(refined_query.variations)}."

        return self.llm.generate(prompt).strip()

    def suggest_follow_ups(
        self,
        query: str,
        documents: List[RetrievedDocument],
        num_suggestions: int = 3,
    ) -> List[str]:
        """
        Suggest related questions based on the query and retrieved content.

        Args:
            query: Original user query.
            documents: Retrieved documents.
            num_suggestions: Number of follow‑up questions to generate.

        Returns:
            List of follow‑up question strings.
        """
        if not documents:
            return ["Try rephrasing your question.", "What food are you interested in?"]

        # Use top few documents to generate ideas
        top_docs = documents[:3]
        content_snippet = "\n".join([doc.content[:200] for doc in top_docs])

        prompt = f"""
Based on the user's question and the retrieved information about food nutrition, suggest up to {num_suggestions} natural follow‑up questions that the user might want to ask next.

User question: "{query}"

Information snippet:
{content_snippet}

Suggest {num_suggestions} follow‑up questions, one per line. Do not number them. Keep them short and relevant.
"""
        response = self.llm.generate(prompt).strip()
        # Split by newline and filter empty
        suggestions = [q.strip() for q in response.split("\n") if q.strip()]
        # If we got fewer than requested, pad with generic suggestions
        while len(suggestions) < num_suggestions:
            suggestions.append("Can you tell me more about that?")
        return suggestions[:num_suggestions]

    def format_response(
        self,
        query: str,
        documents: List[RetrievedDocument],
        refined_query: Optional[RefinedQuery] = None,
        include_follow_ups: bool = True,
    ) -> FormattedResponse:
        """
        Complete response generation: answer, sources, follow‑ups.

        Returns:
            FormattedResponse object with all components.
        """
        # Generate answer
        answer = self.generate_answer(query, documents, refined_query)

        # Build sources list
        sources = []
        for doc in documents[:self.max_sources]:
            source = {
                "content": doc.content,
                "metadata": doc.metadata,
            }
            if self.include_scores:
                source["score"] = getattr(doc, "rerank_score", doc.score)
            sources.append(source)

        # Generate follow‑ups if requested
        follow_ups = []
        if include_follow_ups:
            follow_ups = self.suggest_follow_ups(query, documents)

        return FormattedResponse(
            query=query,
            answer=answer,
            sources=sources,
            follow_ups=follow_ups,
            refined_query=refined_query,
        )

    # ----- end‑to‑end use with SearchResult -----
    def respond_from_search_result(self, search_result) -> FormattedResponse:
        """
        Given a SearchResult (from search_tools), produce a full response.
        """
        return self.format_response(
            query=search_result.query,
            documents=search_result.documents,
            refined_query=search_result.refined_query,
            include_follow_ups=True,
        )