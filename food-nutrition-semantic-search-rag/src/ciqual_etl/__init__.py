# src/ciqual_etl/__init__.py
#!/usr/bin/env python3
"""Ciqual 2025 ETL package.
Lazy imports to avoid circular dependencies and speed up package loading.
"""

__version__ = "1.0.0"

import importlib

_imported = {}

def __getattr__(name):
    if name in _imported:
        return _imported[name]

    module_map = {
        "Food": "ciqual_data",
        "FoodGroup": "ciqual_data",
        "Component": "ciqual_data",
        "Composition": "ciqual_data",
        "DataSource": "ciqual_data",
        "Ciqual_ETL_Pipeline": "ciqual_etl_pipeline",
        "CiqualXMLParser": "ciqual_xml_parser",
        "DB_NAME": "config",
        "DB_USER": "config",
        "DB_PASSWORD": "config",
        "DB_HOST": "config",
        "DB_PORT": "config",
        "DatabaseConnection": "db_utils",
        "PostgresImporter": "postgres_importer",
        "FoodEmbeddingGenerator": "embedding_generator",
        "FoodRetriever": "food_search_engine",
    }
    if name not in module_map:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_name = module_map[name]
    module = importlib.import_module(f".{module_name}", package=__name__)
    attr = getattr(module, name)
    _imported[name] = attr
    return attr

__all__ = [
    "Food", "FoodGroup", "Component", "Composition", "DataSource",
    "Ciqual_ETL_Pipeline", "CiqualXMLParser",
    "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT",
    "DatabaseConnection", "PostgresImporter",
    "FoodEmbeddingGenerator", "FoodRetriever",
]