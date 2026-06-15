#!/usr/bin/env python3
"""
CLI for food search engine.

Usage:
    # Search for food composition 
    uv run python -m ciqual_etl.run_food_search_engine \
        --query "poisson riche en oméga 3"
        
    # Adjust number of results
    uv run python -m ciqual_etl.run_ciqual_embeddings  \
        --query "high protein" --top-k 10 

"""

import argparse
import sys
import logging
from ciqual_etl import FoodRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Query examples
EXAMPLE_QUERIES = [
    "Aliment riche en protéines et faible en matières grasses",     # French
    "High calcium food for bone health",                            # English
    "Fruit avec beaucoup de vitamine C",                            # French
    "Low carbohydrate vegetables",                                  # English
    "Poisson gras oméga 3",                                         # French
    "Foods with high iron content",                                 # English
]

def run_example_queries(retriever: FoodRetriever, top_k: int = 3):
    """Run a set of predefined example queries and print their results."""
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"Example query {i}: \"{query}\"")
        print('='*60)
        results = retriever.search(query, top_k=top_k)
        if not results:
            print("No results found.")
            continue
        for res in results:
            print(f"  • Food: {res['alim_nom_fr']} (code {res['alim_code']}) | similarity: {res['similarity']}")
            print(f"    Snippet: {res['composition_text'][:150]}...")
            print()

if __name__ == "__main__":
    """
    Main entry point: parse arguments, override configuration if needed,
    and search for food information.
    """
    
    parser = argparse.ArgumentParser(
        description="Search for food information using semantic similarity."
    )

    # Query
    parser.add_argument(
        "--query",
        type=str,
        help="Search query (French/English). If omitted, a set of example queries will be run."
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)."
    )

    args = parser.parse_args()
    
    retriever = FoodRetriever()
    try:
        if args.query:
            # User‑provided query
            results = retriever.search(args.query, top_k=args.top_k)
            print(f"\n=== Search Results for: \"{args.query}\" ===")
            for res in results:
                print(f"Food: {res['alim_nom_fr']} (code {res['alim_code']}) | similarity: {res['similarity']}")
                print(f"Text snippet: {res['composition_text'][:200]}...\n")
        
        else:
            # No query provided → run example queries
            print("No query provided. Running default example queries (French/English).")
            # Use a smaller top_k for examples to keep output readable
            run_example_queries(retriever, top_k=min(3, args.top_k))
    finally:
        retriever.close()


