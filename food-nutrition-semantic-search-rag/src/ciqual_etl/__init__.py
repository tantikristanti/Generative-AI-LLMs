#!/usr/bin/env python3
"""Ciqual 2025 ETL package.
It uses lazy imports where modules are imported only when the attribute is actually accessed. This breaks the circular dependency and also speeds up package loading.
"""

__version__ = "1.0.0"

def __getattr__(name):
    """Lazy import of submodules to avoid circular dependencies."""
    if name in ("Food", "FoodGroup", "Component", "Composition", "DataSource"):
        from .ciqual_data import Food, FoodGroup, Component, Composition, DataSource
        return globals().setdefault(name, eval(name))
    elif name == "Ciqual_ETL_Pipeline":
        from .ciqual_etl_pipeline import Ciqual_ETL_Pipeline
        return globals().setdefault(name, Ciqual_ETL_Pipeline)
    elif name == "CiqualXMLParser":
        from .ciqual_xml_parser import CiqualXMLParser
        return globals().setdefault(name, CiqualXMLParser)
    elif name in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"):
        from .config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
        return globals().setdefault(name, eval(name))
    elif name == "DatabaseConnection":
        from .db_utils import DatabaseConnection
        return globals().setdefault(name, DatabaseConnection)
    elif name == "PostgresImporter":
        from .postgres_importer import PostgresImporter
        return globals().setdefault(name, PostgresImporter)
    elif name == "FoodEmbeddingGenerator":
        from .embedding_generator import FoodEmbeddingGenerator
        return globals().setdefault(name, FoodEmbeddingGenerator)
    elif name == "FoodRetriever":
        from .food_search_engine import FoodRetriever
        return globals().setdefault(name, FoodRetriever)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "Food", "FoodGroup", "Component", "Composition", "DataSource",
    "Ciqual_ETL_Pipeline", "CiqualXMLParser",
    "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT",
    "DatabaseConnection", "PostgresImporter",
    "FoodEmbeddingGenerator", "FoodRetriever",
]