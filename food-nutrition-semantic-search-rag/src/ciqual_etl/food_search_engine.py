#!/usr/bin/env python3
"""
- Query food composition embeddings using natural language.
- Supports French/English queries via a multilingual sentence transformer.
- Semantic vector search (FastAPI).
- RAG endpoint ready for Ollama multimodal LLMs
"""

import logging
import psycopg2
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer

from rag.config import RAGConfig

logger = logging.getLogger(__name__)

class FoodRetriever:
    """
    Retrieve top‑k foods most similar to a query text.
    Uses the same embedding model as FoodEmbeddingGenerator.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        model_name: Optional[str] = None,   # kept for backward compatibility
    ):
        # Use config if provided, else fallback to default_config
        self.config = config or RAGConfig()

        # Determine model name: explicit model_name overrides config
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = self.config.embedding_model

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Use the connection string from config
        self.conn_string = self.config.pg_connection_string
        self.table_name = self.config.pg_table

        self.conn = None
        self.cur = None
        self._connect_db()

    def _connect_db(self):
        """Establish a database connection using the config."""
        try:
            self.conn = psycopg2.connect(self.conn_string)
            self.cur = self.conn.cursor()
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents similar to the query.
        If top_k is not provided, uses config.top_k.
        """
        if top_k is None:
            top_k = self.config.top_k

        query_emb = self.model.encode([query])[0]
        query_vec = '[' + ','.join(str(x) for x in query_emb) + ']'

        ''' In PostgreSQL’s pgvector extension, the operator <=> computes cosine distance, not cosine similarity.
            Cosine distance = 1 - cosine_similarity
            Cosine similarity ranges from -1 (opposite) to +1 (identical), while distance ranges from 0 (identical) to 2 (opposite).
            `1 - (embedding <=> %s::vector) AS similarity` converts the cosine distance into cosine similarity, giving a value between -1 and +1 where 1 means perfect match.
        '''

        # SQL query using pgvector cosine distance
        sql = f"""
            SELECT
                alim_code,
                alim_nom_fr,
                alim_nom_eng,
                composition_text,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        self.cur.execute(sql, (query_vec, query_vec, top_k))
        rows = self.cur.fetchall()

        # Convert results to RetrievedDocument 
        results = []
        
        for row in rows:
            results.append({
                "alim_code": row[0],
                "alim_nom_fr": row[1],
                "alim_nom_eng": row[2],
                "composition_text": row[3],
                "metadata": row[4],          # already a dict (JSONB -> Python dict)
                "similarity": round(row[5], 4),
            })
        return results

    def close(self):
        """Close the database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()