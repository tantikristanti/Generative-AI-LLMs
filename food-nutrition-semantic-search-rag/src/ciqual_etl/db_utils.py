#!/usr/bin/env python3
"""Shared database connection utilities."""

import sys
import logging
import psycopg2
from ciqual_etl import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Context manager for database connections."""

    def __init__(self, exit_on_failure: bool = True):
        self.exit_on_failure = exit_on_failure
        self.conn = None
        self.cur = None

    def connect(self) -> None:
        """
        Establish a connection to the PostgreSQL database.

        Uses the configuration from environment variables (or defaults).
        Exits the program if the connection fails.

        Raises:
            SystemExit: If the database connection cannot be established.
        """
        
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
            )
            self.cur = self.conn.cursor()
            logger.info(f"Connected to database '{DB_NAME}'")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            if self.exit_on_failure:
                sys.exit(1)
            raise

    def disconnect(self) -> None:
        """Close the database connection and cursor if they are open."""
        
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def get_connection(exit_on_failure: bool = True) -> DatabaseConnection:
    """Return a new DatabaseConnection instance."""
    return DatabaseConnection(exit_on_failure)