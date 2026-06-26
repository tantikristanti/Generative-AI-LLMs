"""
Query refinement tools for the agent.

Pipeline:
1. SymSpell Correction (fast, dictionary-based)
2. LLM Query Rewriter (rephrase for clarity)
3. Multi-Query Generation (expand to multiple search variations)
4. Embedding Search (retrieve candidates from pgvector)
5. Cross-Encoder Reranking (reorder by relevance)
"""

import logging
import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Third‑party packages
from symspellpy import SymSpell, Verbosity
from sentence_transformers import SentenceTransformer, CrossEncoder

# Cross‑package imports
from rag.food_image_retriever import ImageAwareFoodRetriever
from rag.rag_base import RetrievedDocument
from rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

@dataclass
class RefinedQuery:
    """Result of query refinement."""
    original: str
    corrected: str
    rewritten: str
    variations: List[str]

class SpellingCorrector:
    """
    SymSpell-based spelling correction.
    Expects a frequency dictionary file (e.g., from SymSpell project).
    """
    def __init__(self, dictionary_path: str = "dictionary/SymSpell/frequency_dictionary_en_82_765.txt"):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        try:
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            logger.info(f"Spelling dictionary loaded from {dictionary_path}")
        except FileNotFoundError:
            logger.warning(
                f"Dictionary not found at {dictionary_path}. "
                "Download from https://github.com/wolfgarbe/SymSpell and place it there. "
                "Falling back to no correction."
            )
            self.sym_spell = None

    def correct(self, text: str) -> str:
        """Return corrected text (compound-aware)."""
        if self.sym_spell is None:
            return text
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        return text

class QueryTools:
    """
    Main query refinement and retrieval pipeline.

    Uses:
    - SymSpell for spelling correction
    - LLM (Ollama) for rewriting and variation generation
    - Embedding model (SentenceTransformer) – not directly used for search
      because the retriever already embeds, but kept for potential future use.
    - Vector retriever (ImageAwareFoodRetriever) for initial candidate search.
    - Cross-encoder for reranking.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        retriever: ImageAwareFoodRetriever,
        embedding_model: Optional[SentenceTransformer] = None,
        reranker_model_name: str = "/Volumes/TantiK/Hugging_Face/cross-encoder/ms-marco-MiniLM-L-6-v2",
        spell_dictionary_path: str = "dictionary/SymSpell/frequency_dictionary_en_82_765.txt",
        top_k: int = 20,          # candidates to retrieve before reranking
        final_top_k: int = 5,     # final number of documents after reranking
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.top_k = top_k
        self.final_top_k = final_top_k

        # Spelling corrector
        self.spell_corrector = SpellingCorrector(spell_dictionary_path)

        # Embedding model (optional, for potential expansion)
        self.embedding_model = embedding_model

        # Cross-encoder for reranking
        logger.info(f"Loading cross-encoder: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)

    # ---------- Core refinement steps ----------
    def correct_spelling(self, query: str) -> str:
        """Step 1: Spelling correction."""
        return self.spell_corrector.correct(query)

    def rewrite_query(self, query: str) -> str:
        """
        Step 2: LLM-based query rewriting.
        Rephrase for clarity and remove ambiguity.
        """
        prompt = f"""
You are a helpful assistant that rewrites user queries for a food nutrition search engine.
The goal is to make the query clearer, more specific, and easier to match against a database of food compositions.

Original query: "{query}"

Rewrite the query to be concise, natural, and well-formed. Only output the rewritten query, nothing else.
"""
        return self.llm.generate(prompt).strip()

    def generate_variations(self, query: str) -> List[str]:
        """
        Step 3: LLM-based generation of multiple query variations.
        """
        prompt = f"""
You are a search expert. Given a user query about food nutrition, generate up to 3 alternative search queries that cover different phrasings or aspects.
Separate each query with a newline. Do not number them.

Original query: "{query}"

Variations:
"""
        response = self.llm.generate(prompt).strip()
        # Split by newline and filter empty
        variations = [v.strip() for v in response.split("\n") if v.strip()]
        # Always include the original query
        if query not in variations:
            variations = [query] + variations
        return variations[:3]   # limit to 3

    def refine(self, query: str) -> RefinedQuery:
        """
        Run the full refinement pipeline: correction → rewrite → variations.
        Returns a RefinedQuery object.
        """
        corrected = self.correct_spelling(query)
        rewritten = self.rewrite_query(corrected)
        variations = self.generate_variations(rewritten)

        return RefinedQuery(
            original=query,
            corrected=corrected,
            rewritten=rewritten,
            variations=variations,
        )

    def retrieve_and_rerank(self, query: str) -> Tuple[List[RetrievedDocument], RefinedQuery]:
        """
        End-to-end:
        refine → search all variations → rerank with cross-encoder.

        Returns:
            (
                top reranked documents,
                refined query object
            )
        """
        refined = self.refine(query)

        # Collect candidates from all variations
        all_docs: List[RetrievedDocument] = []
        seen_ids = set()

        for q in refined.variations:
            docs = self.retriever.search(q, top_k=self.top_k)

            for doc in docs:
                # Deduplicate by alim_code (or content fallback)
                doc_id = doc.metadata.get("alim_code", doc.content[:50])

                if doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)

        if not all_docs:
            return [], refined

        # Rerank using cross-encoder
        pairs = [(refined.rewritten, doc.content) for doc in all_docs]
        scores = self.reranker.predict(pairs)

        # Attach rerank scores
        for doc, score in zip(all_docs, scores):
            doc.rerank_score = float(score)

        # Sort by rerank score (higher is better)
        all_docs.sort(
            key=lambda d: getattr(d, "rerank_score", 0.0),
            reverse=True,
        )

        # Return top final_top_k docs + refined query
        return all_docs[:self.final_top_k], refined

    # ---------- Async variants (if LLM supports async) ----------
    async def arefine(self, query: str) -> RefinedQuery:
        """Async version of refine (uses asyncio.to_thread for LLM calls)."""
        loop = asyncio.get_event_loop()
        corrected = await loop.run_in_executor(None, self.correct_spelling, query)
        rewritten = await loop.run_in_executor(None, self.rewrite_query, corrected)
        variations = await loop.run_in_executor(None, self.generate_variations, rewritten)
        return RefinedQuery(
            original=query,
            corrected=corrected,
            rewritten=rewritten,
            variations=variations,
        )

    async def aretrieve_and_rerank(self, query: str) -> Tuple[List[RetrievedDocument], RefinedQuery]:
        """Async version of retrieve_and_rerank."""
        refined = await self.arefine(query)

        all_docs = []
        seen_ids = set()
        for q in refined.variations:
            docs = await asyncio.to_thread(self.retriever.search, q, top_k=self.top_k)
            for doc in docs:
                doc_id = doc.metadata.get("alim_code", doc.content[:50])
                if doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)

        if not all_docs:
            return []

        pairs = [(refined.rewritten, doc.content) for doc in all_docs]
        scores = await asyncio.to_thread(self.reranker.predict, pairs)
        for doc, score in zip(all_docs, scores):
            doc.rerank_score = float(score)

        all_docs.sort(key=lambda d: getattr(d, "rerank_score", 0.0), reverse=True)
        return all_docs[:self.final_top_k], refined