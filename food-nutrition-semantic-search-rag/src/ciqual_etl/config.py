#!/usr/bin/env python3
"""Configuration settings from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# Required environment variables for the database connection
DB_USER = os.environ.get('POSTGRES_USER', 'ciqual')
DB_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'ciqual')
DB_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
DB_PORT = os.environ.get('POSTGRES_PORT', '5432')
DB_NAME = os.environ.get('POSTGRES_DB', 'ciqual_db')