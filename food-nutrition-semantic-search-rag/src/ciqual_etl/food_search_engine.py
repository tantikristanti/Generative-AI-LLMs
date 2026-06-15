#!/usr/bin/env python3
"""
- Query food composition embeddings using natural language.
- Supports French/English queries via a multilingual sentence transformer.
- Semantic vector search (FastAPI).
- RAG endpoint ready for Ollama multimodal LLMs
"""

import logging
import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from ciqual_etl import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, \
    FoodEmbeddingGenerator

logger = logging.getLogger(__name__)

class FoodRetriever:
    """
    Retrieve top‑k foods most similar to a query text.
    Uses the same embedding model as FoodEmbeddingGenerator.
    """

    def __init__(self, model_name: str = FoodEmbeddingGenerator.DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.conn = None
        self.cur = None
        self._connect_db()

    def _connect_db(self):
        self.conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def search(
        self,
        query: str,
        table_name: str = "food_composition_embeddings",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top‑k foods most similar to the query text.

        Returns a list of dicts with keys:
            alim_code, alim_nom_fr, alim_nom_eng, composition_text, metadata, similarity
        """
        query_emb = self.model.encode([query])[0]
        query_vec = '[' + ','.join(str(x) for x in query_emb) + ']'

        ''' In PostgreSQL’s pgvector extension, the operator <=> computes cosine distance, not cosine similarity.
            Cosine distance = 1 - cosine_similarity
            Cosine similarity ranges from -1 (opposite) to +1 (identical), while distance ranges from 0 (identical) to 2 (opposite).
            `1 - (embedding <=> %s::vector) AS similarity` converts the cosine distance into cosine similarity, giving a value between -1 and +1 where 1 means perfect match.
        '''
        sql = f"""
            SELECT
                alim_code,
                alim_nom_fr,
                alim_nom_eng,
                composition_text,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        self.cur.execute(sql, (query_vec, query_vec, top_k))
        rows = self.cur.fetchall()
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