#!/usr/bin/env python3
"""
Generate and store vector embeddings for food composition data.
Uses a multilingual sentence transformer to support French/English queries.
"""

import logging
import pandas as pd
import json
import numpy as np
import math 
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
from ciqual_etl import DatabaseConnection

logger = logging.getLogger(__name__)


class FoodEmbeddingGenerator:
    """
    Reads the Ciqual composition CSV, creates text representations,
    generates embeddings using a multilingual model, and stores them
    in a PostgreSQL table with pgvector support.
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim

    def __init__(
        self,
        csv_path: str,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        table_name: str = "food_composition_embeddings",
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.table_name = table_name
        self.model = None
        self.db = None
        self.conn = None
        self.cur = None

    def _connect_db(self):
        self.db = DatabaseConnection(exit_on_failure=False)
        self.db.connect()
        self.cur = self.db.cur
        self.conn = self.db.conn

    def _disconnect_db(self):
        if self.db:
            self.db.disconnect()

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded.")

    def _create_table(self, vector_dim: int):
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                alim_code INTEGER,
                alim_nom_fr TEXT,
                alim_nom_eng TEXT,
                composition_text TEXT,
                metadata JSONB,
                embedding vector({vector_dim})
            );
        """)
        
        '''IVFFlat (Inverted File Flat) is an indexing method provided by the pgvector extension. 
        It partitions the vector space into lists clusters (like a coarse quantizer).
        During a query, only the most relevant clusters are searched, speeding up approximate nearest neighbour (ANN) search.
        It is not exact but gives good recall with much lower latency than brute‑force.
        '''
        self.cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
            ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        self.conn.commit()
        logger.info(f"Table {self.table_name} ready (dim={vector_dim}).")
    
    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        parts = []
        for col, val in row.items():
            # Convert to string first
            val_str = str(val).strip()
            # Check for common NA representations
            if val_str == '' or val_str.lower() in ('nan', 'na', 'n/a', 'none', 'null'):
                val_str = "N/A"
            parts.append(f"{col.strip()}: {val_str}")
        return "\n".join(parts)

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def run(self, drop_existing: bool = False):
        logger.info("Reading CSV file...")
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns.")
        
        # Safely handle the possibility that the CSV might not contain these column names. It prevents a missing columns would crash the script.
        alim_code_col = 'alim_code' if 'alim_code' in df.columns else None
        nom_fr_col = 'alim_nom_fr' if 'alim_nom_fr' in df.columns else None
        nom_eng_col = 'alim_nom_eng' if 'alim_nom_eng' in df.columns else None

        logger.info("Building text representations and metadata...")
        texts = []
        metadata_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            texts.append(self._row_to_text(row))
            
            row_dict = row.to_dict()
            # Clean NaN values since PostgreSQL's JSONB parser will refuse "NaN" string.
            # NaN values (from pandas np.na in _row_to_text function) will be rejected by PostgreSQL's JSONB parser. 
            # So, we need to converts "NaN" string to None (compatible with JSONB parser)
            for k, v in row_dict.items():
                if isinstance(v, float) and math.isnan(v):
                    row_dict[k] = None
                # If you have string 'nan' or 'N/A', you can also convert:
                elif isinstance(v, str) and v.lower() in ('nan', 'na', 'n/a', 'none', 'null'):
                    row_dict[k] = None
            
            metadata_list.append({
                "alim_code": row.get('alim_code'),
                "alim_nom_fr": row.get('alim_nom_fr'),
                "alim_nom_eng": row.get('alim_nom_eng'),
                "row_data": row_dict
            })

        logger.info("Generating embeddings...")
        embeddings = self._generate_embeddings(texts)
        vector_dim = embeddings.shape[1]

        self._connect_db()
        try:
            if drop_existing:
                self.cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
                self.conn.commit()
            self._create_table(vector_dim)

            data = []
            for idx, row in df.iterrows():
                alim_code = row[alim_code_col] if alim_code_col else None
                nom_fr = row[nom_fr_col] if nom_fr_col else None
                nom_eng = row[nom_eng_col] if nom_eng_col else None
                emb_vec = '[' + ','.join(str(x) for x in embeddings[idx]) + ']'
                data.append((
                    int(alim_code) if pd.notna(alim_code) else None,
                    nom_fr,
                    nom_eng,
                    texts[idx],
                    json.dumps(metadata_list[idx]),
                    emb_vec,
                ))

            insert_sql = f"""
                INSERT INTO {self.table_name}
                (alim_code, alim_nom_fr, alim_nom_eng, composition_text, metadata, embedding)
                VALUES %s
            """
            execute_values(self.cur, insert_sql, data, page_size=1000)
            self.conn.commit()
            logger.info(f"Inserted {len(data)} rows into {self.table_name}.")
        finally:
            self._disconnect_db()

        logger.info("Embedding pipeline finished successfully.")