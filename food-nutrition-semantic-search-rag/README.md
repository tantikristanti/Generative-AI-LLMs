# 🍳 CIQUAL Food Search Engine – RAG‑Powered Semantic Search 🍜

A complete food search engine combining **French CIQUAL nutritional data** with **Open Food Facts images**, stored in **PostgreSQL + pgvector**.
Provides **semantic vector search** and a **RAG endpoint** ready for multimodal LLMs (e.g., Ollama’s LLaVA).

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
  - [Excel Files Structure](#excel-files-structure)
  - [XML Files Structure](#xml-files-structure)
- [System Architecture](#system-architecture)
  - [Ciqual ETL & Vector Search Pipeline](#ciqual-etl--vector-search-pipeline)
  - [Components](#components)
  - [Features](#features)
- [PostgreSQL Database Schema](#postgresql-database-schema)
- [Requirements](#requirements)
- [Installation &amp; Setup](#installation--setup)
  - [1. Clone this repository ](#1-clone-this-repository)
  - [2. Create and activate a virtual environment](#2-create-and-activate-a-virtual-environment)
  - [3. Install the Ciqual ETL Package](#3-install-the-ciqual-etl-package)
  - [4. Start PostgreSQL with pgvector extension (Docker)](#4-start-postgresql-with-pgvector-extension-docker)
- [Usage Instructions](#usage-instructions)
  - [Load Ciqual Data into PostgreSQL](#load-ciqual-data-into-postgresql)
  - [Generate Text Embedding from Ciqual Data](#generate-text-embedding-from-ciqual-data)
  - [Food Search Engine (one‑time)](#food-search-engine-onetime)
  - [Start API Server](#start-api-server)
- [RAG with Ollama](#rag-with-ollama)
- [Performance & Indexing](#performance--indexing)
- [Future Extensions](#future-extensions)
- [Troubleshoot](#troubleshoot)
  - [Connect to PostgreSQL](#connect-to-postgresql)
  - [Enable pgVector Extension](#enable-pgvector-extension)
- [References](#references)

---

## Dataset Overview

This project uses nutritional data from [CIQUAL (ANSES)](https://ciqual.anses.fr/), French Food Composition Table.

- **Source**: [ANSES CIQUAL](https://ciqual.anses.fr/)
- **License**: [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) – distributed as Open Data via [data.gouv.fr](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FRDMHWY)[[1][1]]
- **Usage in this project**: Core nutritional data stored in PostgreSQL.

### Excel Files Structure

The Excel file Table `Ciqual 2025_ENG_2025_11_03.xls/.xlsx` contains two tabs:

1. `food composition` tab - Denormalized wide table with:

- 3,484 foods (rows) × 74 components (columns).
- Food metadata columns: `alim_code`, `alim_nom_eng`, `alim_nom_sci`, `alim_grp_code`, `alim_ssgrp_code`, `alim_ssssgrp_code`, `alim_grp_nom_eng`, `alim_ssgrp_nom_eng`, `alim_ssssgrp_nom_eng`, `Jones_factor`, `N(g/100g)*Jones_factor`
- 74 component columns (values + units in headers)

2. `INFOODS codes` tab - Component mapping table:

- `INFOODS_code`, `const_code`, `const_nom_eng`

### XML Files Structure

The XML files provide a normalized relational structure with 5 files:

| File                    | Content               | Key Fields                                                                                                                                           |
| ----------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| alim_2025_11_03.xml     | List of foods         | `alim_code`, `alim_nom_eng`, `alim_nom_fr`, `alim_nom_sci`, `alim_grp_code`, `alim_ssgrp_code`, `alim_ssssgrp_code`, `facteur_jones` |
| alim_grp_2025_11_03.xml | Food group hierarchy  | `alim_grp_code`, `alim_ssgrp_code`, `alim_ssssgrp_code`, `group names (EN/FR)`                                                               |
| compo_2025_11_03.xml    | Composition values    | `alim_code`, `const_code`, `teneur`, `min`, `max`, `code_confiance`, `source_code`                                                     |
| const_2025_11_03.xml    | Component definitions | `const_code`, `const_nom_eng`, `const_nom_fr`, `code_INFOODS`                                                                                |
| sources_2025_11_03.xml  | Data sources          | `source_code`, `ref_citation`                                                                                                                    |

This project uses normalized XML files as data sources for structured data.

---

## System Architecture

```
┌─────────────────────┐
│ CIQUAL Dataset      │
│ (.xls, .xml)        │
└──────────┬──────────┘
           │ download & parse
           ▼
┌─────────────────────┐
│ PostgreSQL          │
│ + pgvector          │ ◄─── embeddings stored as vector(384)
└──────────┬──────────┘
           │
           ├─── Open Food Facts API ───► enrich with images
           │
           ▼
┌─────────────────────┐
│ FastAPI Service     │
│ - /search (vector)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Multimodal LLM      │
└─────────────────────┘
```

---

### Ciqual ETL & Vector Search Pipeline

```
ciqual_etl/
├── __init__.py                     # A Python package marker allowing us to import modules from it 
├── ciqual_data.py                  # Dataclasses (FoodGroup, Food, Component, Composition, DataSource)
├── ciqual_etl_pipeline.py          # Pipeline orchestrator
├── ciqual_xml_parser.py            # CiqualXMLParser class
├── postgres_importer.py            # PostgresImporter class (including reporting)
├── config.py                       # Environment variables & settings
├── db_utils.py                     # Database connection utilities
├── embedding_generator.py          # Generate and store vector embeddings for food composition data
├── fastapi_app.py                  # FastAPI food search engine app
├── food_search_engine.py           # Food search engine
├── run_ciqual_embeddings.py        # CLI entry point for embedding generation
├── run_ciqual_etl.py               # CLI entry point for ETL pipeline
└── run_food_search_engine.py       # CLI entry point for food search engine

```

---

### Components

- **PostgreSQL + pgvector**[[2],[3]]: Stores foods, nutrition, image URLs, and 384‑dim vectors. Enables cosine similarity search via IVFFlat index.
- **Sentence‑Transformers** [[4]]: Converts text queries and food descriptions into vectors.
- **FastAPI** [[5]]: REST API for semantic search and RAG.
- **Ollama** [[6]]: Local LLM server. The `/rag` endpoint retrieves relevant foods, builds a context prompt, and asks the LLM.

---

### Features

- Semantic search using cosine similarity on pgvector.
- IVFFlat index for fast approximate nearest neighbour search.
- FastAPI endpoints with auto‑generated OpenAPI docs (`/docs`)
- RAG endpoint that integrates retrieved foods with Ollama (multimodal models like LLaVA).

---

## PostgreSQL Database Schema

The PostgreSQL database schema is described in [Figure 1](#fig1).

<figure id="fig1">
  <img src="images/ER-Diagram-CIQUAL.png" alt="CIQUAL" height="100%" weight="100%">
  <figcaption>Figure 1: Ciqual ER-Diagram.</figcaption>
</figure>

---

## Requirements

- **Python** 3.9+
- **PostgreSQL** 14+ with **pgvector** extension installed. `pgvector` is the PostgreSQL extension to perform vector search.
- **Docker** (recommended for pgvector)

Python packages (install via `pip` or `uv` – see [Installation and Setup](#installation--setup)).

---

## Installation & Setup

### 1. Clone this repository

```bash
git clone https://github.com/tantikristanti/Generative-AI-LLMs/tree/main/food-nutrition-semantic-search-rag

cd food-nutrition-semantic-search-rag
```

---

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# or with uv:
uv venv .venv

source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   on Windows
```

---

### 3. Install the Ciqual ETL Package

Install the Ciqual ETL package to use them cleanly in other modules.

- `-e` stands for editable mode. The package is installed in place, so changes to the source code are immediately reflected without reinstalling.
- `.` refers to the current directory (which should contain a pyproject.toml or setup.py).

```bash
# Install the package with pip
pip install -e .

# Or using uv
uv pip install -e .
```

---

### 4. Start PostgreSQL with pgvector extension (Docker)

You can run PostgreSQL with the pgvector extension either directly with Docker or by using Docker Compose.

**Option 1: Run PostgreSQL with Docker**

When starting the container:

- Use `--name` to assign a meaningful name to the container. This makes it easier to manage and reference the container later using commands such as docker start, docker stop, docker logs, and docker exec.
- Configure the database credentials and settings using environment variables (`-e`) or load them from a `.env` file with `--env-file`.
- Expose PostgreSQL by mapping a host port to the container port using the format `[HOST_PORT:CONTAINER_PORT]`. PostgreSQL listens on port `5432` inside the container by default. If port `5432` is already in use on the host, you can map it to another port such as `5431`.
- Mount a persistent volume using `-v` to ensure that database data is retained even if the container is stopped, recreated, or removed.

```bash
docker run -d --name pgvector \
  -e POSTGRES_USER=ciqual \
  -e POSTGRES_PASSWORD=ciqual \
  -e POSTGRES_DB=ciqual \
  -v ciqual_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  pgvector/pgvector:pg18
```

The `pgvector/pgvector:pg18` image includes both PostgreSQL and the `pgvector` extension, so no additional installation steps are required.

**Verify Running Containers**

List all running containers:

```bash
docker ps
```

---

## Usage Instructions

### Load Ciqual Data into PostgreSQL

This script parses the Ciqual XML files and loads the data into PostgreSQL tables.

```bash

# Display available command-line arguments 
uv run python -m ciqual_etl.run_ciqual_etl --help

# Import data from the specified Ciqual XML directory 
uv run python -m ciqual_etl.run_ciqual_etl \
  --xml-dir "data/ciqual" 
  
# Clear existing table data before importing 
uv run python -m ciqual_etl.run_ciqual_etl \
  --xml_dir "data/ciqual" \
  --clear
```

### Generate Text Embedding from Ciqual Data

This script parses the Ciqual CSV file and loads the data into PostgreSQL DB.

CLI for building Ciqual food embeddings.

```bash
# Import data from the specified pre-processed Ciqual CSV directory 
uv run python -m ciqual_etl.run_ciqual_embeddings \
    --csv "data/ciqual/pre-processed/table-ciqual-2025-11-03-with-fr-food-name.csv" 

# Drop existing table before creating
uv run python -m ciqual_etl.run_ciqual_embeddings \
    --csv "data/ciqual/pre-processed/table-ciqual-2025-11-03-with-fr-food-name.csv" \
    --drop
```

### Food Search Engine (one‑time)

```bash
uv run python -m ciqual_etl.run_food_search_engine \
  --query "poisson riche en oméga 3"
```

We will get the following results:

```
=== Search Results ===
Food: Accra de poisson, préemballé (code 25433) | similarity: 0.598
Text snippet: alim_grp_code: 4
alim_ssgrp_code: 409
alim_ssssgrp_code: 0
alim_grp_nom_eng: meat, egg and fish
alim_ssgrp_nom_eng: fish products
alim_ssssgrp_nom_eng: -
alim_code: 25433
alim_nom_eng: Caribbean-style...

Food: Carpaccio de saumon avec marinade, fait maison (code 25537) | similarity: 0.587
Text snippet: alim_grp_code: 4
alim_ssgrp_code: 409
alim_ssssgrp_code: 0
alim_grp_nom_eng: meat, egg and fish
alim_ssgrp_nom_eng: fish products
alim_ssssgrp_nom_eng: -
alim_code: 25537
alim_nom_eng: Salmon carpacci...

Food: Terrine de fruits de mer (par ex. de Saint-Jacques), avec ou sans poisson, préemballée (code 8292) | similarity: 0.5792
Text snippet: alim_grp_code: 4
alim_ssgrp_code: 409
alim_ssssgrp_code: 0
alim_grp_nom_eng: meat, egg and fish
alim_ssgrp_nom_eng: fish products
alim_ssssgrp_nom_eng: -
alim_code: 8292
alim_nom_eng: Seafood terrine,...

Food: Truite saumonée, sautée/poêlée (code 26998) | similarity: 0.5626
Text snippet: alim_grp_code: 4
alim_ssgrp_code: 405
alim_ssssgrp_code: 0
alim_grp_nom_eng: meat, egg and fish
alim_ssgrp_nom_eng: fish, cooked
alim_ssssgrp_nom_eng: -
alim_code: 26998
alim_nom_eng: Salmon trout, sa...

Food: Surimi, bâtonnets, tranche ou râpé saveur crabe (code 26046) | similarity: 0.5601
Text snippet: alim_grp_code: 4
alim_ssgrp_code: 409
alim_ssssgrp_code: 0
alim_grp_nom_eng: meat, egg and fish
alim_ssgrp_nom_eng: fish products
alim_ssssgrp_nom_eng: -
alim_code: 26046
alim_nom_eng: Surimi, on stic...
```

---

### Start API Server

```bash
uv run uvicorn ciqual_etl.fastapi_app:app --reload --port 8000
```

> Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

This will run server at http://localhost:8000 with interactive API documentation at http://localhost:8000/docs.

**API Endpoints**

| Method | Path    | Description                                       |
| ------ | ------- | ------------------------------------------------- |
| GET    | /health | Health check & API info.                          |
| GET    | /docs   | Auto‑generated Swagger UI documentation.          |
| GET    | /search | Semantic vector search (params:`query`, `top_k`). |
| GET    | /rag    | RAG pipeline: search + Ollama LLM.                |

***1. Health Check Endpoint (GET)***

- `GET` -> http://localhost:8000/health 
- Expected response (status 200), [Figure 2](#fig2):

```json
{ "status": "ok" }
```

<figure id="fig2">
  <img src="images/postman-health-check-endpoint.png" alt="health-check" height="100%" weight="100%">
  <figcaption>Figure 2: Health Check Endpoint (GET).</figcaption>
</figure>

***2. Search Endpoint – GET version***

- `GET` -> `http://localhost:8000/search?query=poisson%20gras%20om%C3%A9ga%203&top_k=3`
- `URL`: `http://localhost:8000/search` 
- Parameters: 
  - `query`: `poisson gras oméga 3`
  - `top_k`: `3
- Expected response (status 200), [Figure 3](#fig3):

```json

{
  "query": "poisson gras oméga 3",
  "top_k": 3,
  "results": [
    {
      "alim_code": 12345,
      "alim_nom_fr": "Saumon",
      "alim_nom_eng": "Salmon",
      "composition_text": "...",
      "metadata": { ... },
      "similarity": 0.92
    },
    ...
  ]
}
```

<figure id="fig3">
  <img src="images/postman-query-get.png" alt="query-get" height="100%" weight="100%">
  <figcaption>Figure 3: Search Endpoint (GET).</figcaption>
</figure>

***3. Search Endpoint – POST version***

- Purpose: Send the query in the request body (more structured, supports larger payloads), [Figure 4](#fig4).
- `URL` -> `http://localhost:8000/search`
- Headers:
  - `Key`: `Content-Type`
  - `Value`: `application/json`
- Body:
  - Raw → JSON

    ```json
    {
      "query": "poisson gras oméga 3",
      "top_k": 3
    }
    ```

<figure id="fig4">
  <img src="images/postman-query-post.png" alt="query-post" height="100%" weight="100%">
  <figcaption>Figure 4: Search Endpoint (POST).</figcaption>
</figure>

**Response for both query versions (GET/POST)**:

```json
{
  "query":"poisson gras oméga 3",
  "top_k":5,"results":
  [
    {"alim_code":25433,
    "alim_nom_fr":"Accra de poisson, préemballé",
    "alim_nom_eng":"Caribbean-style fish fritters, fish acras,prepacked",
    "composition_text":"alim_grp_code: 4\nalim_ssgrp_code: 409\nalim_ssssgrp_code: 0\nalim_grp_nom_eng: meat, egg and fish\nalim_ssgrp_nom_eng: fish products\nalim_ssssgrp_nom_eng: -\nalim_code: 25433\nalim_nom_eng: Caribbean-style fish fritters, fish acras, prepacked\nalim_nom_sci: (Genus and species unknown or multiple)\n...alim_nom_fr: Accra de poisson, préemballé\nalim_grp_nom_fr: viandes, oeufs, poissons\nalim_ssgrp_nom_fr: produits à base de poissons et produits de la mer",
    "metadata":{"alim_code":25433,"alim_nom_fr":"Accra de poisson, préemballé","alim_nom_eng":"Caribbean-style fish fritters, fish acras,\nprepacked"},
    "similarity":0.613},
    ...
  ]
}
```

**Error Handling Examples**

***Invalid top_k***

- POST request with top_k = 0 (invalid, must be ≥1):
```json
{ "query": "healthy", "top_k": 0 }

```

- Expected response (status 422):

```json
{
    "detail": [
        {
            "type": "greater_than_equal",
            "loc": [
                "body",
                "top_k"
            ],
            "msg": "Input should be greater than or equal to 1",
            "input": 0,
            "ctx": {
                "ge": 1
            }
        }
    ]
}
```

---

### RAG with Ollama

- **Endpoint**: `GET /rag?q=<question>&model=<ollama_model>&top_k=<n>`
- **Example**:

```bash
curl "http://localhost:8000/rag?q=What%20foods%20are%20rich%20in%20protein%20but%20low%20in%20fat?&model=llava&top_k=5"
```

- Response:

```json

{
  "query": "What foods are rich in protein but low in fat?",
  "retrieved_foods": [ ... ],
  "llm_answer": "Based on the data, skinless chicken breast, fat-free yogurt, and white fish are excellent choices..."
}
```

> Requires Ollama running on http://localhost:11434 (default).
> Pull a model first: `ollama pull llava`

---

## Performance & Indexing

- **Index type**: ***IVFFlat index*** on the `embedding` column with cosine distance operator (<=>).
- **Number of lists**: Automatically set to `sqrt(row_count)`.
- **For larger datasets (>100k rows), we need to consider switching to `HNSW index` (pgvector supports both).
- Example of creating an HNSW index manually:

```sql
CREATE INDEX ON foods USING hnsw (embedding vector_cosine_ops);
```

- **IVFFlat vs HNSW** – detailed study: [PGVector: HNSW vs IVFFlat](https://medium.com/@bavalpreetsinghh/pgvector-hnsw-vs-ivfflat-a-comprehensive-study-21ce0aaab931)[[4][4]]


---

## Future Extensions

- **User feedback loop**: Log user clicks to fine‑tune embeddings.
- **Full‑text + hybrid search**: Combine vector similarity with BM25 (`pgvector` sparse vectors or `pg_trgm`).
- **Frontend dashboard**: `Streamlit` or `Next.js` app to visualise results & images.
- **Periodic updates**: Automatically refresh `CIQUAL` when new versions are released.
- **Multilingual support**: Use a multilingual embedding model (e.g., `distiluse-base-multilingual-cased-v2`).

---

## Troubleshoot

### Connect to PostgreSQL

```bash
docker exec -it postgres psql -U db_user_name -d db_name
```

For example:

```bash
docker exec -it postgres psql -U ciqual -d ciqual_db
```

### Enable pgVector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Verify**

```sql
SELECT * FROM pg_extension;
```

**Exit from PostgreSQL**

```sql
\q
```

---

## References

1. Du Chaffaut, Laure; Oseredczuk, Marine; Gauvreau-Béziat, Julie, 2025, **Table de composition nutritionnelle des aliments Ciqual 2025**, https://doi.org/10.57745/RDMHWY, Recherche Data Gouv, V1.
2. [PostgreSQL: The World&#39;s Most Advanced Open Source Relational Database](https://www.postgresql.org/)
3. [pgvector](https://github.com/pgvector/pgvector)
4. [SentenceTransformers Documentation](https://sbert.net)
5. [fastapi](https://github.com/fastapi/fastapi)
6. [Ollama](https://ollama.com/)

---

[1]: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%253A10.57745%252FRDMHWY
[2]: https://www.postgresql.org/
[3]: https://github.com/pgvector/pgvector
[4]: https://sbert.net
[5]: https://github.com/fastapi/fastapi
[6]: https://ollama.com/
