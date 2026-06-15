#!/usr/bin/env python3
"""
CLI for building Ciqual food embeddings.

Usage:
    # Import data from the specified Ciqual XML directory 
    uv run python -m ciqual_etl.run_ciqual_embeddings \
        --csv "data/ciqual/pre-processed/table-ciqual-2025-11-03-with-fr-food-name.csv" 
  
    # Drop existing table before creating
    uv run python -m ciqual_etl.run_ciqual_embeddings \
        --csv "data/ciqual/pre-processed/table-ciqual-2025-11-03-with-fr-food-name.csv" \
        --drop
"""

import argparse
import sys
import logging
from ciqual_etl import FoodEmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if __name__ == "__main__":
    """
    Main entry point: parse arguments, override configuration if needed,
    and generate Ciqual embeddings.
    """
    
    parser = argparse.ArgumentParser(description="Build food composition embeddings.")

    # Generate embeddings
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--drop", action="store_true", help="Drop existing table before creating")

    args = parser.parse_args()

    generator = FoodEmbeddingGenerator(csv_path=args.csv)
    generator.run(drop_existing=args.drop)
