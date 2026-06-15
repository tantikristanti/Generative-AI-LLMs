#!/usr/bin/env python3
"""Orchestrator for the full ETL pipeline."""


import logging
from ciqual_etl import CiqualXMLParser, PostgresImporter

logger = logging.getLogger(__name__)

class Ciqual_ETL_Pipeline:
    """
    Main orchestrator for the Ciqual 2025 ETL (Extract‑Transform‑Load) process.

    This class coordinates the entire pipeline:
        1. Parses Ciqual XML files using CiqualXMLParser.
        2. Imports parsed data into PostgreSQL using PostgresImporter.
        3. Handles table creation, staging, validation, orphan logging, and reporting.
    """

    @staticmethod
    def run(xml_dir: str, clear_existing: bool = False) -> None:
        """
        Execute the complete Ciqual ETL pipeline.

        The pipeline performs the following steps in order:
            - Parse all XML files (food groups, foods, components, sources, composition).
            - Connect to the PostgreSQL database.
            - Create staging, logging, and final tables if they do not exist.
            - Optionally clear all existing data (if `clear_existing` is True).
            - Insert all data into staging tables (raw load without constraints).
            - Insert a special 'UNKNOWN' food group to absorb placeholder references.
            - Validate foods against existing groups, normalise placeholder codes,
              log orphans, and insert valid foods.
            - Insert food groups, components, and data sources into clean tables.
            - Insert composition rows only for foods that were successfully loaded.
            - Generate a reconciliation report with charts and CSV exports.
            - Close the database connection.

        Args:
            xml_dir (str): Path to the directory containing the Ciqual 2025 XML files.
            clear_existing (bool, optional): If True, truncates all existing data
                (staging, logging, and final tables) before the import. Defaults to False.

        Returns:
            None

        Raises:
            SystemExit: If no food records are found after parsing (critical error).
            Any database or parsing exceptions are logged; the pipeline exits gracefully.

        Example:
            >>> Ciqual_ETL_Pipeline.run('/data/ciqual_xml', clear_existing=True)
        """
        
        logger.info("Starting Ciqual 2025 ETL Pipeline")

        # Step 1: Parse XML
        logger.info("Step 1: Parsing XML files...")
        parser = CiqualXMLParser(xml_dir)
        food_groups = parser.parse_food_groups()
        foods = parser.parse_foods()
        components = parser.parse_components()
        sources = parser.parse_data_sources()
        compositions = parser.parse_composition()

        if not foods:
            logger.error("No foods found. Check XML files.")
            return

        # Step 2: Import into PostgreSQL
        logger.info("Step 2: Importing data into PostgreSQL...")
        importer = PostgresImporter()
        
        # Connect to the database
        importer.connect()
        try:
            # Create tables
            importer.create_staging_tables()
            importer.create_log_tables()
            importer.create_tables()

            # Clear existing data (if users ask)
            if clear_existing:
                importer.clear_tables()

            # Load staging tables
            importer.insert_staging_food_groups(food_groups)
            importer.insert_staging_foods(foods)
            importer.insert_staging_components(components)
            importer.insert_staging_data_sources(sources)
            importer.insert_staging_composition(compositions)

            # Insert unknown data
            importer.insert_unknown_food_groups()
            
            # Insert real food groups FIRST
            importer.insert_food_groups(food_groups)
            
            # Insert foods after inserting food groups
            valid_foods = importer.insert_foods(food_groups, foods)
            
            # Insert data to other tables
            importer.insert_components(components)
            importer.insert_data_sources(sources)
            importer.insert_composition(valid_foods, compositions)

        finally:
            importer.generate_reconciliation_report(output_dir="reports")
            importer.disconnect()

        logger.info("Pipeline completed successfully")