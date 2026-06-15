#!/usr/bin/env python3
"""
Command-line interface for the Ciqual ETL pipeline.

This script parses command-line arguments, optionally overrides database
connection settings, and runs the ETL pipeline.

Usage:
    # Import data from the specified Ciqual XML directory 
    uv run python -m ciqual_etl.run_ciqual_etl \
        --xml-dir "data/ciqual" 
  
    # Clear existing table data before importing 
    uv run python -m ciqual_etl.run_ciqual_etl \
        --xml_dir "data/ciqual" \
        --clear
"""

import sys
import logging
import argparse
import src.ciqual_etl.config as config
from ciqual_etl import Ciqual_ETL_Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if __name__ == "__main__":
    """
    Main entry point: parse arguments, override configuration if needed,
    and execute the ETL pipeline.
    """
    
    parser = argparse.ArgumentParser(description='Import Ciqual 2025 XML data into PostgreSQL.')
    
    # Generate embedding and save the result into Postgres DB
    parser.add_argument('--xml_dir', required=True, help='Directory containing the Ciqual 2025 XML files')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before import')
    parser.add_argument('--db-name', help='PostgreSQL database name')
    parser.add_argument('--db-user', help='PostgreSQL user')
    parser.add_argument('--db-pass', help='PostgreSQL password')
    parser.add_argument('--db-host', help='PostgreSQL host')
    parser.add_argument('--db-port', help='PostgreSQL port')

    args = parser.parse_args()

    # Override configuration module attributes if provided on the command line.
    # This ensures that all modules (e.g., PostgresImporter) use the updated values.
    if args.db_name:
        config.DB_NAME = args.db_name
    if args.db_user:
        config.DB_USER = args.db_user
    if args.db_pass:
        config.DB_PASSWORD = args.db_pass
    if args.db_host:
        config.DB_HOST = args.db_host
    if args.db_port:
        config.DB_PORT = args.db_port

    # Run the pipeline
    Ciqual_ETL_Pipeline.run(args.xml_dir, clear_existing=args.clear)