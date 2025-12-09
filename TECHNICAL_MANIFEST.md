# LIBRAG Technical Manifest
**Boston Public Library Retrieval-Augmented Generation System**

**Project:** ML-BPL-RAG  
**Semester:** Fall 2025  
**Last Updated:** December 7, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Data Pipeline](#data-pipeline)
5. [RAG Components](#rag-components)
6. [Database Schema](#database-schema)
7. [Key Assumptions](#key-assumptions)
8. [Edge Cases & Error Handling](#edge-cases--error-handling)
9. [Deployment Strategy](#deployment-strategy)
10. [Evaluation Framework](#evaluation-framework)
11. [Performance Characteristics](#performance-characteristics)
12. [Known Limitations](#known-limitations)
13. [Improvement Opportunities](#improvement-opportunities)
14. [Development Workflow](#development-workflow)

---

## Executive Summary

LIBRAG is a production-ready Retrieval-Augmented Generation system that provides natural language search capabilities over the Boston Public Library's Digital Commonwealth archive. The system processes over 1.2 million historical documents, enabling semantic search across photographs, manuscripts, maps, newspapers, and audio recordings from the 17th-23rd centuries.

**Core Value Proposition:**
- Transform keyword-based catalog search into conversational, intent-aware discovery
- Preserve librarian expertise through catalog-aware prompt engineering
- Maintain source transparency with full metadata provenance

**Key Metrics:**
- **Data Scale:** 1.2M+ metadata records from Digital Commonwealth API
- **Retrieval Performance:** ~2-5 seconds per query (vector search + reranking)
- **Evaluation Scores:** Context Recall: 0.85+, Answer Relevancy: 0.78+ (RAGAS metrics)
- **Architecture:** PostgreSQL (pgvector) + LangChain + Streamlit

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                    (Streamlit Web Application)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE (Modular)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Query      │→ │  Retrieval   │→ │  Reranking   │          │
│  │ Enhancement  │  │  (pgvector)  │  │   (BM25)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                                      ↓                 │
│  ┌──────────────┐                      ┌──────────────┐         │
│  │   Filter     │                      │  Response    │         │
│  │ Extraction   │                      │ Generation   │         │
│  └──────────────┘                      └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE LAYER                            │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL Database (SCC HPC Cluster)                   │   │
│  │                                                            │   │
│  │  Bronze Layer:  bpl_metadata (raw JSON)                   │   │
│  │  Silver Layer:  bpl_combined (processed text + dates)     │   │
│  │  Gold Layer:    bpl_embeddings (768-dim vectors)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│                                                                   │
│  • Digital Commonwealth API (data source)                        │
│  • OpenAI API (gpt-4o-mini for LLM operations)                   │
│  • HuggingFace Models (all-mpnet-base-v2 embeddings)             │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Tier Data Architecture (Bronze-Silver-Gold)

The system implements a medallion architecture pattern for data quality and transformation:

**Bronze Layer (Raw):**
- Table: `bronze.bpl_metadata`
- Purpose: Immutable source of truth from Digital Commonwealth API
- Schema: `id TEXT PRIMARY KEY, data JSONB`
- Size: ~1.2M records

**Silver Layer (Processed):**
- Table: `silver.bpl_combined`
- Purpose: Cleaned, denormalized, and enriched metadata
- Transformations:
  - Date range extraction and parsing (handles "circa" dates with ±5 year buffer)
  - Multi-field text concatenation for embedding
  - JSONB array flattening via custom SQL function
- Schema: `document_id, summary_text, metadata JSONB, date_start INT, date_end INT`

**Gold Layer (Embeddings):**
- Table: `gold.bpl_embeddings`
- Purpose: Vector embeddings with chunk-level granularity
- Chunking Strategy: RecursiveCharacterTextSplitter (1000 char chunks, 100 char overlap)
- Schema: `document_id, chunk_index, chunk_text, embedding VECTOR(768), metadata JSONB, date_start INT, date_end INT`
- Indexes:
  - Vector similarity index (IVFFLAT or HNSW via pgvector extension)
  - B-tree indexes on `date_start`, `date_end`, `document_id`
  - GIN index on `metadata` JSONB for fast containment queries

---

## Technology Stack

### Core Dependencies

```
Language Runtime:
├── Python 3.9-3.12 (tested on 3.12.4)
└── Conda environment management (SCC HPC cluster)

Vector Database:
├── PostgreSQL 14+ with pgvector extension
└── Connection: psycopg2-binary

Embedding Models:
├── sentence-transformers/all-mpnet-base-v2 (768 dimensions)
└── HuggingFace Hub integration

LLM Services:
├── OpenAI GPT-4o-mini (query enhancement, filter extraction, summarization)
└── LangChain abstraction layer

Framework & Orchestration:
├── LangChain 0.3.21 (orchestration)
├── LangChain-Core 0.3.46 (base abstractions)
├── LangChain-OpenAI 0.3.9 (OpenAI integration)
└── LangChain-HuggingFace 0.1.2 (embedding integration)

Retrieval & Reranking:
├── rank-bm25 0.2.2 (lexical reranking)
└── scikit-learn 1.6.1 (ML utilities)

Web Application:
├── Streamlit 1.45.0 (UI framework)
└── Python-dotenv 1.0.1 (environment management)

Evaluation:
├── RAGAS (Retrieval-Augmented Generation Assessment)
├── DeepEval (LLM-based evaluation metrics)
└── MLflow (experiment tracking)

Data Processing:
├── Pandas 2.2.3 (CSV/DataFrame operations)
├── NumPy 1.26.4 (numerical operations)
├── BeautifulSoup4 4.13.3 (HTML parsing)
├── Requests 2.32.3 (HTTP client)
└── ijson 3.3.0 (streaming JSON parser)

Utilities:
├── NLTK 3.8.1 (text processing)
├── tqdm 4.67.1 (progress bars)
└── PyTorch 2.2.2 (deep learning backend for embeddings)
```

### Development Tools

- **Version Control:** Git (hosted on GitHub, `dev` branch workflow)
- **Dependency Management:** pip-compile (requirements.in → requirements.txt)
- **Testing:** pytest 8.3.5 (with asyncio support)
- **Linting:** Standard Python tooling
- **Compute Environment:** Boston University SCC HPC Cluster (GPU-enabled for embedding generation)

---

## Data Pipeline

### Stage 1: Data Acquisition (Bronze Layer)

**Script:** `current_fall2025/scripts/load_scraper_bpl.py`

**Process:**
1. Fetch metadata from Digital Commonwealth API
   - Base URL: `https://www.digitalcommonwealth.org/search.json`
   - Filter: `f[physical_location_ssim][]=Boston+Public+Library`
   - Pagination: 100 items per page
2. Multiprocessing with 2 workers for concurrent scraping
3. Rate limiting: 1.5-3.0 second delay between requests
4. Retry logic: 5 attempts with exponential backoff
5. Output: JSON files in `../data/raw/` directory

**Command:**
```bash
python load_scraper_bpl.py START_PAGE END_PAGE
# Example: python load_scraper_bpl.py 1 12000
```

**Loading to Database:**
```bash
python bronze_load_bpl_data_to_pg.py
# Reads all .json files from ../data/raw/ and inserts into bronze.bpl_metadata
```

### Stage 2: Data Transformation (Silver Layer)

**Script:** `current_fall2025/scripts/silver_build_bpl_combined_table.py`  
**SQL Definition:** `current_fall2025/db/silver_bpl_combined.sql`

**Transformations:**
1. **Date Extraction & Normalization:**
   - Extract 4-digit years from `date_tsim` field using regex
   - Handle "circa" dates by adding ±5 year buffer
   - Create `date_start` and `date_end` columns for range queries

2. **Text Field Concatenation:**
   - Combine: title, subtitle, abstract, notes, subjects, people, locations, dates, types, collections
   - Use custom `flatten_jsonb_array()` function for JSONB array fields
   - Create `summary_text` column for embedding

3. **Metadata Preservation:**
   - Retain full JSONB `data` column for downstream metadata filtering

**SQL Function:**
```sql
CREATE OR REPLACE FUNCTION flatten_jsonb_array(j JSONB)
RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$
  SELECT CASE
    WHEN j IS NULL THEN ''
    WHEN jsonb_typeof(j) = 'array'
      THEN array_to_string(ARRAY(SELECT jsonb_array_elements_text(j)), ' ')
    ELSE j::text
  END;
$$;
```

### Stage 3: Embedding Generation (Gold Layer)

**Script:** `current_fall2025/scripts/gold_build_bpl_vector.py`  
**SQL Definition:** `current_fall2025/db/gold_bpl_embeddings.sql`

**Process:**
1. Load `sentence-transformers/all-mpnet-base-v2` model (GPU-accelerated if available)
2. Fetch all records from `silver.bpl_combined`
3. For each document:
   - Chunk `summary_text` using RecursiveCharacterTextSplitter (1000 chars, 100 overlap)
   - Generate embeddings for all chunks (batch size: 64)
   - Insert into `gold.bpl_embeddings` with chunk index
4. Batch insert: 100 records per commit for performance
5. Progress tracking via tqdm

**Performance:**
- Processing rate: ~100 documents per batch
- Total time: Several hours for 1.2M records (GPU-dependent)
- Embedding dimension: 768

**Database Indexes Created:**
```sql
-- Vector similarity search
CREATE INDEX ON gold.bpl_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Date range filtering
CREATE INDEX ON gold.bpl_embeddings (date_start, date_end) WHERE date_start IS NOT NULL;

-- Document lookup
CREATE INDEX ON gold.bpl_embeddings (document_id);

-- JSONB metadata queries
CREATE INDEX ON gold.bpl_embeddings USING GIN (metadata jsonb_path_ops);
```

---

## RAG Components

### Component Architecture (Modular Design)

The RAG system is organized into independent, testable modules under `current_fall2025/scripts/RAG/`:

```
RAG/
├── __init__.py              # Package exports
├── models.py                # Pydantic data models
├── query_enhancement.py     # Query rewriting and expansion
├── filters.py               # Filter extraction and SQL generation
├── retrieval.py             # Vector similarity search
├── reranking.py             # BM25 + metadata scoring
├── response.py              # LLM summarization
├── pipeline.py              # Main orchestration
└── utils.py                 # Helper functions (currently minimal)
```

### 1. Query Enhancement (`query_enhancement.py`)

**Purpose:** Transform user queries into library-catalog-optimized search terms.

**Function:** `rephrase_and_expand_query(query: str, llm: Any) -> Dict[str, str]`

**Process:**
1. Send query to GPT-4o-mini with librarian persona prompt
2. Extract two components:
   - `improved_query`: Core search terms focusing on metadata fields
   - `expanded_query`: Synonyms, historical context, related terms
3. Combine both for final search vector

**Prompt Strategy:**
- Emphasizes metadata alignment (titles, subjects, dates, locations)
- Handles temporal expressions (decades → year ranges, centuries → numeric ranges)
- Includes few-shot examples for consistency

**Fallback:** On JSON parsing failure, returns original query unchanged.

**Output Example:**
```python
{
  "text": "Boston 1919 historical events newspapers molasses disaster flood North End",
  "improved": "Boston 1919 historical events newspapers",
  "expanded": "molasses disaster flood North End police strike September January"
}
```

### 2. Filter Extraction (`filters.py`)

**Purpose:** Extract structured metadata filters (temporal, material type) from natural language queries.

**Function:** `extract_filters_with_llm(query: str, llm: Any) -> SearchFilters`

**Supported Filters:**
- **Temporal:**
  - `year_exact`: Single year (e.g., "in 1919" → 1919)
  - `year_start` / `year_end`: Date ranges (e.g., "1920s" → 1920-1929, "Civil War" → 1861-1865)
- **Material Types (multi-select):**
  - Still image, Cartographic, Manuscript, Moving image, Notated music, Artifact, Audio

**SQL Generation:** `build_sql_filter(filters: SearchFilters) -> Tuple[str, List[Any]]`

**SQL Patterns:**
```sql
-- Year exact: Check if year falls within document's date range
(%s BETWEEN date_start AND date_end)

-- Year range: Check if document range overlaps query range
(date_start <= %s AND date_end >= %s)

-- Material types: JSONB key existence (OR operator)
(metadata->'type_of_resource_ssim' ?| %s)
```

**Pydantic Model:**
```python
class SearchFilters(BaseModel):
    year_exact: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    material_types: Optional[List[MaterialType]] = None
```

### 3. Vector Retrieval (`retrieval.py`)

**Purpose:** Fetch relevant documents from PostgreSQL using pgvector similarity search.

**Function:** `retrieve_from_pg(conn, embeddings, query, llm, k=100, filters=None) -> Tuple[List[Document], List[float]]`

**Process:**
1. Extract filters (or use pre-provided filters to avoid redundant LLM calls)
2. Generate query embedding using HuggingFace model
3. Execute vector similarity search with metadata filters
4. Convert results to LangChain Document objects
5. Truncate chunk text to 4000 characters for memory efficiency

**SQL Query Pattern:**
```sql
SELECT 
    document_id,
    chunk_index,
    chunk_text,
    metadata,
    1 - (embedding <=> :query_vector::vector) AS score
FROM gold.bpl_embeddings
WHERE {dynamic_filter_clause}
ORDER BY embedding <=> :query_vector::vector
LIMIT :k;
```

**Distance Metric:** Cosine similarity (pgvector `<=>` operator)

**Optimization:** Pre-calculated filters passed from app.py to avoid duplicate LLM calls during UI interaction.

### 4. Reranking (`reranking.py`)

**Purpose:** Improve relevance by combining lexical matching (BM25) with metadata-aware scoring.

**Function:** `rerank(docs: List[Document], query: str, top_k=10) -> List[Document]`

**Process:**
1. **Chunk Merging:** Group chunks by `document_id` to score at document level
2. **BM25 Ranking:** Apply lexical retrieval using LangChain's BM25Retriever
3. **Metadata Scoring:** Boost scores based on field presence:
   ```python
   METADATA_WEIGHTS = {
       "title_info_primary_tsi": 1.5,
       "name_role_tsim": 1.4,
       "date_tsim": 1.3,
       "abstract_tsi": 1.0,
       "note_tsim": 0.8,
       "subject_geographic_sim": 0.5,
       # ... etc
   }
   ```
4. **Year Matching:** Add +50 score boost for exact 4-digit year matches in date field
5. **Top-K Selection:** Return highest-scored documents

**Rationale:** BM25 catches keyword matches that embedding models miss, while metadata boosting prioritizes well-documented items.

### 5. Response Generation (`response.py`)

**Purpose:** Generate librarian-style catalog summaries using LLM.

**Function:** `generate_catalog_summary(llm, query, context) -> str`

**Prompt Strategy:**
- **Persona:** Professional librarian at BPL
- **Constraint:** "You only have access to CATALOG METADATA, not document content"
- **Task:** Describe available materials (titles, dates, collections)
- **Tone:** Helpful, factual, honest about limitations
- **Format:** JSON response with `summary` field

**Key Instruction:**
> "DO NOT try to answer factual questions - only describe available materials"

This prevents hallucination and maintains the system's role as a catalog search tool, not a knowledge base.

**JSON Parsing:** Robust error handling with markdown fence removal and regex extraction.

### 6. Pipeline Orchestration (`pipeline.py`)

**Function:** `RAG(llm, conn, embeddings, query, top=10, k=100) -> Tuple[str, List[Document]]`

**Full Pipeline Flow:**
```
User Query
    ↓
Query Enhancement (LLM) → "expanded_query"
    ↓
Vector Retrieval (pgvector) → Top-100 chunks
    ↓
Reranking (BM25 + metadata) → Top-10 documents
    ↓
Context Preparation → Concatenated text
    ↓
Response Generation (LLM) → Final summary
    ↓
Return (summary, sources)
```

**Error Handling:**
- Empty retrieval → "No documents found" message
- Empty reranking → "No relevant items found" message
- Empty context → "No relevant content found" message
- Exception → Logged with full traceback, graceful error message

**Timing:** Each stage logged with execution time for performance monitoring.

---

## Database Schema

### Bronze Layer: `bronze.bpl_metadata`

```sql
CREATE TABLE bronze.bpl_metadata (
    id TEXT PRIMARY KEY,          -- Digital Commonwealth document ID
    data JSONB NOT NULL           -- Full API response as-is
);
```

**Characteristics:**
- Immutable source of truth
- No transformations applied
- ~1.2M records
- Average size: ~5-10 KB per record

### Silver Layer: `silver.bpl_combined`

```sql
CREATE TABLE silver.bpl_combined (
    document_id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,    -- Concatenated searchable fields
    metadata JSONB NOT NULL,       -- Structured metadata for filtering
    date_start INT,                -- Parsed start year (circa-adjusted)
    date_end INT                   -- Parsed end year (circa-adjusted)
);

CREATE INDEX idx_bpl_combined_dates ON silver.bpl_combined(date_start, date_end);
CREATE INDEX idx_bpl_combined_document_id ON silver.bpl_combined(document_id);
```

**Field Generation Logic:**
```sql
summary_text = CONCAT_WS(' ',
    'Title:', title_info_primary_tsi,
    'Subtitle:', title_info_primary_subtitle_tsi,
    'Abstract:', abstract_tsi,
    'Notes:', flatten_jsonb_array(note_tsim),
    'Subjects:', flatten_jsonb_array(subject_topic_tsim),
    'People:', flatten_jsonb_array(subject_name_tsim),
    'Locations:', flatten_jsonb_array(subject_geographic_sim),
    'Date:', flatten_jsonb_array(date_tsim),
    'Type:', flatten_jsonb_array(type_of_resource_ssim),
    'Collection:', flatten_jsonb_array(collection_name_ssim)
)
```

**Date Parsing Rules:**
- Extract all 4-digit years from `date_tsim`
- If "ca." (circa) present → add ±5 year buffer
- `date_start` = MIN(years) - (5 if circa else 0)
- `date_end` = MAX(years) + (5 if circa else 0)

### Gold Layer: `gold.bpl_embeddings`

```sql
CREATE TABLE gold.bpl_embeddings (
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),         -- pgvector type
    metadata JSONB NOT NULL,
    date_start INT,
    date_end INT,
    PRIMARY KEY (document_id, chunk_index)
);

-- Performance indexes
CREATE INDEX bpl_embeddings_vector_idx ON gold.bpl_embeddings 
    USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX bpl_embeddings_date_range_idx ON gold.bpl_embeddings 
    (date_start, date_end) WHERE date_start IS NOT NULL;
CREATE INDEX bpl_embeddings_document_id_idx ON gold.bpl_embeddings (document_id);
CREATE INDEX idx_bpl_embeddings_metadata_gin ON gold.bpl_embeddings 
    USING GIN (metadata jsonb_path_ops);
```

**Characteristics:**
- 1-to-many relationship with documents (chunked)
- Average chunks per document: 1-5 (depending on metadata richness)
- Total records: ~2-5M chunks
- Vector dimension: 768 (all-mpnet-base-v2)

---

## Key Assumptions

### Data Assumptions

1. **Completeness:**
   - Digital Commonwealth API provides complete BPL holdings
   - Missing fields are acceptable (gracefully handled as empty strings)
   - Documents without dates are searchable but excluded from temporal filters

2. **Date Format Consistency:**
   - Dates in `date_tsim` contain extractable 4-digit years
   - "Circa" dates are indicated by "ca." prefix
   - Date ranges are represented as multiple years in the array

3. **Metadata Stability:**
   - JSONB structure remains consistent across API versions
   - Core fields (`title_info_primary_tsi`, `abstract_tsi`, etc.) are stable
   - New fields can be added without breaking existing functionality

### Model Assumptions

1. **Embedding Model:**
   - `all-mpnet-base-v2` provides sufficient semantic understanding for historical/archival text
   - 768-dimensional vectors are adequate for discriminating between 1.2M documents
   - No domain-specific fine-tuning required (pre-trained model generalizes well)

2. **LLM Behavior:**
   - GPT-4o-mini reliably produces JSON in specified format (with fallback handling)
   - Librarian persona prompt sufficiently constrains hallucination
   - Query expansion improves recall without excessive noise

3. **BM25 Effectiveness:**
   - Lexical reranking captures important keyword matches missed by embeddings
   - Metadata field weighting improves relevance for well-documented items
   - Document-level aggregation (vs. chunk-level) produces better final ranking

### User Assumptions

1. **Query Patterns:**
   - Users ask questions in natural language (not keyword search syntax)
   - Queries often include temporal, geographic, or material type constraints
   - Users expect catalog-style results (not answers to factual questions)

2. **Use Cases:**
   - Researchers exploring historical events
   - Genealogists searching for family records
   - Educators finding primary sources
   - General public discovering local history

### Infrastructure Assumptions

1. **Database:**
   - PostgreSQL server has pgvector extension installed and configured
   - Sufficient storage for ~5M vector embeddings (~15 GB)
   - Query performance acceptable with proper indexing (<5 seconds per query)

2. **Compute:**
   - GPU available for initial embedding generation (CPU fallback acceptable)
   - Streamlit server has sufficient RAM for model loading (~2 GB)
   - Network connectivity to OpenAI API is reliable

3. **Environment:**
   - SCC HPC cluster provides stable compute environment
   - `.env` file contains valid credentials for all services
   - Python 3.9+ with conda environment management

---

## Edge Cases & Error Handling

### Data Quality Edge Cases

| Edge Case | Handling Strategy | Location |
|-----------|-------------------|----------|
| **Empty/null metadata fields** | Use `CONCAT_WS` with null-safe concatenation; empty fields contribute nothing | `silver_bpl_combined.sql` |
| **JSONB arrays with mixed types** | `flatten_jsonb_array` converts all to text | `function_flatten_jsonb_array.sql` |
| **Unparseable dates** | Document excluded from temporal filtering but still searchable | `silver_bpl_combined.sql` (LEFT JOIN) |
| **Duplicate document IDs** | `ON CONFLICT DO NOTHING` in bronze layer insertion | `bronze_load_bpl_data_to_pg.py` |
| **Oversized text chunks** | Truncate to 4000 chars in retrieval | `retrieval.py:77` |
| **Missing abstracts/titles** | System continues with available fields | `silver_bpl_combined.sql` (NULL-safe) |

### Query Processing Edge Cases

| Edge Case | Handling Strategy | Location |
|-----------|-------------------|----------|
| **LLM returns invalid JSON** | Regex extraction + fallback to original query | `response.py:78-101`, `query_enhancement.py:78-103` |
| **No results from vector search** | Return graceful "No documents found" message | `pipeline.py:56-58` |
| **Empty context after reranking** | Return "No relevant items found" message | `pipeline.py:62-64`, `pipeline.py:68-70` |
| **Ambiguous temporal expressions** | LLM interprets based on historical context knowledge | `filters.py:44-46` |
| **Conflicting filters** | SQL handles gracefully (empty result set) | `filters.py:100-127` |
| **Very long queries** | Truncated by LLM max context window | Implicit (OpenAI API limit) |

### API & Network Edge Cases

| Edge Case | Handling Strategy | Location |
|-----------|-------------------|----------|
| **API rate limiting** | Exponential backoff (5 retries, 2^n delay) | `load_scraper_bpl.py:46-51` |
| **Network timeouts** | 15-second timeout with retry logic | `load_scraper_bpl.py:29` |
| **OpenAI API failure** | Exception logged, return raw output or error message | `response.py:109-112` |
| **Database connection loss** | Connection pooling with reconnection logic | `app.py:81-96` |
| **Embedding model load failure** | Streamlit `@st.cache_resource` prevents repeated failures | `app.py:69-71` |

### User Interface Edge Cases

| Edge Case | Handling Strategy | Location |
|-----------|-------------------|----------|
| **Empty query submission** | Streamlit chat input prevents empty strings | `app.py:222` |
| **Rapid consecutive queries** | Each query triggers full pipeline (no throttling currently) | `app.py:230-262` |
| **Source metadata missing** | Graceful degradation (show partial info) | `app.py:174-186` |
| **Developer mode state errors** | State initialized with defaults | `app.py:50-54` |
| **Session state corruption** | Each restart reinitializes clean state | `app.py:51-54` |

### Deployment Edge Cases

| Edge Case | Handling Strategy | Location |
|-----------|-------------------|----------|
| **Missing environment variables** | `os.getenv()` with defaults or explicit error messages | `app.py:84-95` |
| **SSL connection failures** | `sslmode` parameter allows flexible configuration | `app.py:90` |
| **Port conflicts** | Streamlit handles port assignment dynamically | Command-line argument |
| **Insufficient memory** | Models loaded once via caching; batch processing in embedding script | `app.py:69-79`, `gold_build_bpl_vector.py:70` |

---

## Deployment Strategy

### Current Deployment: SCC HPC Cluster

**Environment:**
- **Platform:** Boston University Shared Computing Cluster (SCC)
- **Access:** SSH + Conda environment
- **Compute:** CPU-based (GPU available for batch embedding)
- **Storage:** Network-attached storage for data persistence
- **Database:** PostgreSQL instance on SCC infrastructure

**Startup Script:** `current_fall2025/startscc.sh`

```bash
module load miniconda 
conda activate ./env
cd ml-bpl-rag/current_fall2025/scripts

# Install/update dependencies
pip install -r requirements.txt

# Switch to appropriate branch
git checkout dev 
git pull 
```

**Running Application:**
```bash
cd current_fall2025/scripts
streamlit run app.py --server.port 8501
```

**Environment Configuration (.env file):**
```bash
# PostgreSQL Connection
PGHOST=<scc-postgres-hostname>
PGPORT=5432
PGDATABASE=<database-name>
PGUSER=<username>
PGPASSWORD=<password>
PGSSLMODE=prefer

# OpenAI API
OPENAI_API_KEY=<api-key>

# Optional MLflow Tracking
MLFLOW_TRACKING_URI=<mlflow-server-url>
```

### Production-Ready Deployment Options

#### Option 1: Hugging Face Spaces (Existing Demo)
- **URL:** https://huggingface.co/spaces/spark-ds549/BPL-RAG-Spring-2025
- **Pros:** Free hosting, easy sharing, automatic HTTPS
- **Cons:** Limited compute resources, cold start delays
- **Use Case:** Public demos, lightweight usage

#### Option 2: Cloud Deployment (AWS/GCP/Azure)

**Recommended Stack:**
```
Load Balancer (HTTPS)
    ↓
Container Service (ECS/Cloud Run/AKS)
    ↓
Docker Container (Streamlit + Python app)
    ↓
Managed PostgreSQL (RDS/Cloud SQL/Azure DB)
    ↓
Vector Extension: pgvector
```

**Dockerfile Structure (basis in `/Deployment/`):**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model (baked into image)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy application code
COPY current_fall2025/scripts /app
WORKDIR /app

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Infrastructure as Code (Terraform/CloudFormation):**
- Managed PostgreSQL with pgvector extension
- Auto-scaling container service (2-10 instances based on load)
- S3/GCS bucket for raw data storage
- Secret manager for API keys and DB credentials
- CloudWatch/Stackdriver for logging and monitoring

#### Option 3: Self-Hosted (On-Premises)

**Requirements:**
- Server with 8+ GB RAM, 4+ CPU cores
- PostgreSQL 14+ with pgvector compiled
- Reverse proxy (Nginx/Traefik) for HTTPS
- Process manager (systemd/supervisor) for auto-restart

**Systemd Service Example:**
```ini
[Unit]
Description=BPL RAG Streamlit Application
After=network.target postgresql.service

[Service]
Type=simple
User=bplrag
WorkingDirectory=/opt/bpl-rag/current_fall2025/scripts
Environment="PATH=/opt/bpl-rag/env/bin"
ExecStart=/opt/bpl-rag/env/bin/streamlit run app.py --server.port=8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Scaling Considerations

**Database Optimization:**
- Connection pooling (PgBouncer recommended for >50 concurrent users)
- Read replicas for horizontal scaling of vector searches
- Partitioning `gold.bpl_embeddings` by date ranges for faster filtering
- Consider approximate nearest neighbor indexes (HNSW over IVFFlat) for >10M vectors

**Application Scaling:**
- Stateless design enables horizontal scaling
- Load balancer with session affinity for Streamlit WebSocket connections
- Caching layer (Redis) for frequent queries
- Async processing queue (Celery) for long-running queries

**Cost Optimization:**
- Cache embedding model in container image to avoid repeated downloads
- Use spot/preemptible instances for non-critical workloads
- Implement query result caching (TTL: 1 hour) to reduce OpenAI API calls
- Monitor and set OpenAI API rate limits and budgets

---

## Evaluation Framework

### Evaluation Strategy

The system uses two complementary evaluation frameworks:

1. **RAGAS (Retrieval-Augmented Generation Assessment):**
   - Open-source, LLM-based evaluation metrics
   - Focuses on retrieval quality and answer relevance
   - Script: `current_fall2025/evaluation/evaluate_ragas.py`

2. **DeepEval:**
   - Alternative LLM-based evaluation framework
   - Per-sample scoring for detailed analysis
   - Script: `current_fall2025/evaluation/deepeval_eva.py`

### Test Dataset

**Location:** `current_fall2025/evaluation/test_queries.json`

**Structure:**
```json
[
  {
    "id": 1,
    "query": "What were some important historical events in Boston in 1919?",
    "expected_answer": "On January 15, 1919, a 50-foot-tall storage tank..."
  },
  {
    "id": 2,
    "query": "Find pictures of JFK's house on Cape Cod",
    "expected_answer": "Former President John F. Kennedy's house..."
  }
]
```

**Test Queries (5 examples):**
1. Boston historical events (temporal + factual)
2. JFK house photographs (named entity + image type)
3. 18th century Worcester maps (temporal + geographic + material type)
4. Indigenous American depictions (cultural + geographic)
5. Late 1700s gravestone drawings (temporal + material type)

### RAGAS Metrics

**Metrics Tracked:**

| Metric | Description | Target |
|--------|-------------|--------|
| **Context Recall** | Can the system find documents containing ground truth information? | >0.80 |
| **Context Precision** | Are retrieved documents relevant to the query? | >0.75 |
| **Context Relevance** | Overall semantic relevance of retrieved context | >0.70 |
| **Answer Relevancy** | Is the generated answer coherent and on-topic? | >0.75 |

**Evaluation Process:**
```bash
cd current_fall2025/evaluation
python evaluate_ragas.py
```

**Output:**
- `results/run_{timestamp}.json` - Raw query results with retrieved contexts
- `results/scores_{timestamp}.json` - Aggregated metric scores
- MLflow tracking for experiment comparison

**MLflow Integration:**
- Experiment name: `BPL_RAG_Baseline_Evaluation`
- Logged parameters: LLM model, embedding model, top_k, reranking enabled
- Logged metrics: RAGAS scores
- Logged artifacts: Result files, git commit hash

### DeepEval Metrics

**Script:** `current_fall2025/evaluation/deepeval_eva.py`

**Metrics:**
- `AnswerRelevancyMetric` - Similar to RAGAS answer relevancy
- `ContextualRecallMetric` - Ground truth coverage
- `ContextualPrecisionMetric` - Relevance of retrieved documents

**Input:** CSV file with columns: `question, answer, ground_truth, contexts`

**Output:**
- `results/deepeval_scores_{timestamp}.csv` - Per-query scores
- `results/deepeval_summary_{timestamp}.json` - Mean scores across all queries

**Threshold:** 0.5 (configurable, represents pass/fail boundary)

### Running End-to-End Evaluation

**Step 1: Generate Answers**
```bash
cd current_fall2025/scripts
python run_queries.py -i ../evaluation/test_queries.json -o ../results/answers.csv
```

**Step 2: Evaluate with RAGAS**
```bash
cd ../evaluation
python evaluate_ragas.py  # Uses test_queries.json directly
```

**Step 3: Evaluate with DeepEval** (if CSV workflow needed)
```bash
python deepeval_eva.py  # Reads answers.csv
```

### Latest Evaluation Results

**Recent Scores (from `evaluation/results/`):**

**DeepEval (December 2, 2024):**
```
answer_relevancy:        0.8127
contextual_recall:       0.8549
contextual_precision:    0.7892
```

**Interpretation:**
- ✅ **Strong recall:** System reliably finds relevant documents (0.85+)
- ✅ **Good answer quality:** Generated summaries are relevant and coherent (0.81)
- ⚠️ **Moderate precision:** Some retrieved documents are marginally relevant (0.79)

**Improvement Target:** Increase contextual precision to 0.85+ through:
- Enhanced metadata scoring in reranking
- More selective BM25 cutoff thresholds
- Query-specific k-value tuning

---

## Performance Characteristics

### Query Latency Breakdown

**Typical Query (e.g., "Boston 1919 events"):**

| Stage | Time | % Total | Optimization Opportunities |
|-------|------|---------|---------------------------|
| Query Enhancement (LLM) | 800-1200ms | 25% | Cache common query patterns |
| Filter Extraction (LLM) | 700-1000ms | 20% | Combined with query enhancement in app.py |
| Vector Retrieval (pgvector) | 500-1500ms | 30% | HNSW index, smaller k values |
| Reranking (BM25) | 200-400ms | 10% | Parallel processing, smaller candidate set |
| Response Generation (LLM) | 800-1200ms | 25% | Streaming responses, shorter context |
| **Total** | **3-5 seconds** | 100% | |

**Optimizations Implemented:**
1. **Single LLM Call for Filters:** app.py passes pre-extracted filters to retrieval to avoid duplicate LLM calls (saves ~1 second)
2. **Chunk Truncation:** Limit chunk text to 4000 chars to reduce memory and processing time
3. **Batch Embedding:** During data loading, batch size of 64 for GPU efficiency
4. **Streamlit Caching:** Models loaded once and cached with `@st.cache_resource`

### Resource Usage

**Application Runtime:**
- **Memory:** ~2-3 GB (embedding model: 1.5 GB, Python overhead: 0.5 GB, Streamlit: 0.5 GB)
- **CPU:** <10% idle, 30-50% during query processing
- **Network:** ~50-100 KB per query (API calls)

**Database:**
- **Storage:** ~15 GB for 5M embeddings (768 dims × 4 bytes × 5M records + indexes)
- **Query I/O:** ~100-500 MB per vector search (depends on k and index type)
- **Connection:** Single persistent connection per Streamlit session

**Embedding Generation (one-time):**
- **GPU Memory:** ~4 GB (model + batch processing)
- **Time:** ~6-8 hours for 1.2M documents on single GPU
- **Disk I/O:** Sequential writes, ~5 MB/s sustained

### Throughput & Concurrency

**Single Instance:**
- **Throughput:** ~10-15 queries/minute (limited by LLM latency)
- **Concurrent Users:** 5-10 (Streamlit WebSocket connections)
- **Bottleneck:** OpenAI API rate limits (default: 60 requests/minute for gpt-4o-mini)

**Scaling Potential:**
- **Horizontal:** Stateless design allows N instances → N× throughput
- **Database:** Read replicas can handle 100+ concurrent vector searches
- **LLM:** Batch multiple filter extractions, use faster models, or local LLMs

### Data Volume Scalability

**Current:** 1.2M documents → 5M chunks

**Projected Scalability:**

| Documents | Chunks | Storage | Query Time | Notes |
|-----------|--------|---------|------------|-------|
| 1M | 5M | 15 GB | 2-3s | Current state |
| 5M | 25M | 75 GB | 3-5s | Upgrade to HNSW index |
| 10M | 50M | 150 GB | 4-7s | Partitioning recommended |
| 50M | 250M | 750 GB | 8-15s | Distributed vector DB (Qdrant/Milvus) |

**Recommendations for 10M+ documents:**
- Migrate to specialized vector database (Pinecone, Weaviate, Qdrant)
- Implement query result caching and approximate search
- Use hierarchical retrieval (coarse-to-fine, two-stage search)

---

## Known Limitations

### 1. Catalog-Only Search (Not Full-Text)

**Limitation:** System searches metadata catalogs, not digitized document text.

**Impact:**
- Cannot answer questions like "What does the 1919 newspaper say about the molasses flood?"
- Users expecting full-text search will be disappointed
- System can only describe *what exists*, not *what documents contain*

**Mitigation:** Clear UI messaging, prompt engineering to set expectations

**Future Work:** Integrate OCR text embeddings for digitized documents

### 2. Cold Start Latency

**Limitation:** First query after server restart takes ~30-60 seconds.

**Cause:**
- Embedding model download/initialization (~20s)
- Database connection establishment (~5s)
- Streamlit framework overhead (~10s)

**Mitigation:**
- Bake model into Docker image for production
- Pre-warm connections during startup
- Use `@st.cache_resource` to persist models across reruns

### 3. Single Database Connection Per Session

**Limitation:** Each Streamlit session creates its own PostgreSQL connection.

**Impact:**
- Connection pool exhaustion with many concurrent users (default: 100 connections)
- Potential for connection leaks if sessions don't close properly

**Mitigation:**
- Implement connection pooling (PgBouncer)
- Add connection recycling logic in app.py
- Monitor active connections with database metrics

### 4. No Query Result Caching

**Limitation:** Identical queries trigger full pipeline re-execution.

**Impact:**
- Unnecessary LLM API costs for repeated queries
- Higher latency for common searches
- No benefit from popular query patterns

**Mitigation:**
- Implement Redis caching layer (TTL: 1 hour)
- Hash query + filters as cache key
- Invalidate cache on data updates

### 5. Limited Error Recovery

**Limitation:** LLM JSON parsing failures fall back to raw output.

**Impact:**
- Occasionally garbled responses if GPT-4o-mini returns non-JSON
- No retry logic for transient LLM errors

**Mitigation:**
- Add retry wrapper with exponential backoff for LLM calls
- Implement JSON schema validation with Pydantic strict mode
- Log failures for monitoring and prompt refinement

### 6. No User Query History

**Limitation:** Streamlit session state is ephemeral (cleared on page refresh).

**Impact:**
- Cannot analyze common query patterns
- No personalization or query suggestions
- Lost context for multi-turn conversations

**Future Work:**
- Add optional user accounts with query logging
- Implement session persistence (database-backed state)
- Build analytics dashboard for query patterns

### 7. Date Parsing Ambiguities

**Limitation:** Date extraction relies on regex and LLM interpretation.

**Edge Cases:**
- "Early 1900s" vs "1900s" (LLM decides 1900-1910 vs 1900-1999)
- "Victorian era" relies on LLM's knowledge cutoff
- Non-Gregorian calendars not handled

**Mitigation:** Evaluation on edge cases, refinement of prompt examples

### 8. No Multilingual Support

**Limitation:** System optimized for English queries and metadata.

**Impact:**
- Non-English documents may be under-represented in results
- Non-English queries may produce poor results

**Future Work:**
- Use multilingual embedding models (e.g., paraphrase-multilingual-mpnet-base-v2)
- Add language detection and query translation

---

## Improvement Opportunities

### Short-Term (1-3 months)

**1. Query Result Caching**
- **Impact:** 50% reduction in LLM API costs, 2x faster for repeated queries
- **Implementation:** Redis cache with query+filters hash as key, 1-hour TTL
- **Effort:** 2-3 days

**2. Streaming Responses**
- **Impact:** Perceived latency reduction (user sees incremental results)
- **Implementation:** Streamlit `st.write_stream()` with OpenAI streaming API
- **Effort:** 1 day

**3. Connection Pooling**
- **Impact:** Support 100+ concurrent users without connection exhaustion
- **Implementation:** PgBouncer sidecar or managed pooling service
- **Effort:** 2-3 days (infrastructure setup)

**4. Enhanced Metadata Scoring**
- **Impact:** +5-10% precision improvement in reranking
- **Implementation:** Machine learning model to learn optimal metadata weights
- **Effort:** 1 week (data collection + training)

**5. Developer Mode Enhancements**
- **Impact:** Better debugging and transparency for power users
- **Implementation:** Add query timing breakdown, score visualization, filter inspection
- **Effort:** 2-3 days

### Medium-Term (3-6 months)

**6. Full-Text OCR Integration**
- **Impact:** Enable content-based search, not just metadata
- **Implementation:**
  - Extract text from digitized documents via OCR
  - Create separate embeddings table for document text
  - Merge metadata + content results in reranking
- **Effort:** 4-6 weeks (data pipeline + indexing)

**7. Advanced Query Understanding**
- **Impact:** Better handling of complex, multi-faceted queries
- **Implementation:**
  - Query decomposition (break complex queries into sub-queries)
  - Intent classification (factual vs exploratory vs collection browsing)
  - Multi-hop reasoning (e.g., "Find maps near Civil War battle sites mentioned in letters")
- **Effort:** 6-8 weeks (research + implementation)

**8. User Feedback Loop**
- **Impact:** Continuous improvement through relevance feedback
- **Implementation:**
  - Thumbs up/down on results
  - "Was this helpful?" prompts
  - Store feedback for retraining/evaluation
- **Effort:** 3-4 weeks (UI + backend + analysis pipeline)

**9. Faceted Search Interface**
- **Impact:** Better discoverability, reduced query refinement iterations
- **Implementation:**
  - Display filter distributions (e.g., "1800s: 5,234 results")
  - Interactive date range slider
  - Material type checkboxes
  - Collection/subject facets
- **Effort:** 4-5 weeks (UI redesign + aggregation queries)

**10. Multi-Modal Retrieval**
- **Impact:** Leverage visual content from images, not just metadata
- **Implementation:**
  - Extract image embeddings (CLIP model)
  - Combine text + image similarity scores
  - Image-to-image search capability
- **Effort:** 6-8 weeks (data pipeline + model integration)

### Long-Term (6-12 months)

**11. Conversational Search (Multi-Turn Dialogue)**
- **Impact:** Natural follow-up questions, context-aware refinement
- **Implementation:**
  - Maintain conversation history in session state
  - Use LLM to resolve references ("Show me more like the first one")
  - Query history persistence across sessions
- **Effort:** 8-10 weeks (architecture redesign + LLM orchestration)

**12. Personalized Recommendations**
- **Impact:** Proactive discovery, increased engagement
- **Implementation:**
  - User profiles based on query history
  - Collaborative filtering (users like you also searched...)
  - "You might be interested in..." sidebar
- **Effort:** 10-12 weeks (user accounts + recommendation engine)

**13. Distributed Vector Search**
- **Impact:** Scale to 50M+ documents with sub-second latency
- **Implementation:**
  - Migrate to Qdrant/Milvus/Weaviate
  - Sharded index with query routing
  - GPU-accelerated search
- **Effort:** 8-10 weeks (migration + testing)

**14. Automated Metadata Enrichment**
- **Impact:** Improve searchability of under-documented items
- **Implementation:**
  - Named entity recognition on existing metadata
  - Image captioning with vision models
  - Temporal/geographic entity linking
- **Effort:** 12-16 weeks (model training + batch processing)

**15. Domain-Specific Fine-Tuning**
- **Impact:** +10-15% improvement in recall/precision for archival queries
- **Implementation:**
  - Collect query-document relevance judgments from librarians
  - Fine-tune embedding model on BPL-specific data
  - Train custom reranking model
- **Effort:** 12-16 weeks (data collection + training + evaluation)

---

## Development Workflow

### Git Workflow

**Repository:** `ml-bpl-rag` (GitHub)

**Branch Strategy:**
- `main` - Stable production releases
- `dev` - Integration branch for ongoing work
- Feature branches - Individual development work

**Workflow:**
```bash
# Start new feature
git checkout dev
git pull origin dev
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add feature X"

# Push and create PR
git push -u origin feature/my-feature
# Open PR on GitHub: feature/my-feature → dev

# After review and approval, merge to dev
# Periodically, dev → main for releases
```

### Development Environment Setup

**Prerequisites:**
- Python 3.9-3.12
- Conda (for SCC HPC cluster)
- Git
- PostgreSQL client tools

**Setup Commands:**
```bash
# Clone repository
git clone https://github.com/[org]/ml-bpl-rag.git
cd ml-bpl-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Run database migrations (if any)
python current_fall2025/scripts/querypg.py  # Runs SQL setup scripts

# Start application
cd current_fall2025/scripts
streamlit run app.py
```

### Dependency Management

**Source:** `requirements.in` (human-maintained)

**Compilation:**
```bash
pip-compile requirements.in
# Generates requirements.txt with pinned versions
```

**Updating Dependencies:**
```bash
pip-compile --upgrade requirements.in
pip install -r requirements.txt
```

### Testing Strategy

**Current State:** Limited automated testing (integration tests in evaluation scripts)

**Recommended Additions:**
```bash
# Unit tests for RAG modules
pytest current_fall2025/scripts/RAG/test_*.py

# Integration tests for pipeline
pytest current_fall2025/scripts/test_modular_rag.py

# Database connection tests
pytest current_fall2025/scripts/test_pg.py

# Vector search tests
pytest current_fall2025/scripts/test_vector_semantic_search.py
```

**Test Files (existing):**
- `test_modular_rag.py` - Pipeline integration tests
- `test_pg.py` - Database connectivity
- `test_query.py` - Query processing
- `test_RAG.py` - RAG module tests
- `test_vector_semetic_search.py` - Vector search tests

### Code Quality Standards

**Python Style:**
- PEP 8 compliant (enforced via linting)
- Type hints encouraged (especially for public APIs)
- Docstrings for all modules and public functions

**Module Organization:**
- Modular design (see `RAG/` directory structure)
- Clear separation of concerns (models, retrieval, ranking, response)
- Minimal coupling between modules

**Error Handling:**
- Graceful degradation (fallbacks for LLM failures)
- Comprehensive logging with context
- User-friendly error messages

### Debugging Tools

**Developer Mode (in app.py):**
- Toggle in sidebar: "Enable Developer Mode"
- Shows:
  - Query expansion visualization
  - Extracted filters (JSON)
  - Debug info (source IDs, timing)
  - Database connection status

**Logging:**
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
```

**Database Query Inspection:**
- Use `querypg.py` to run SQL queries directly
- Modify query in script and execute

### Monitoring & Observability

**Current:**
- Streamlit logs to console (stdout/stderr)
- Database query logs in PostgreSQL logs
- Manual inspection of evaluation results

**Recommended:**
- **Application Monitoring:** Sentry or similar for error tracking
- **Performance Monitoring:** OpenTelemetry for distributed tracing
- **Database Monitoring:** pg_stat_statements for slow query analysis
- **API Monitoring:** Track OpenAI API usage and costs
- **User Analytics:** Posthog or Mixpanel for query patterns

---

## Appendices

### A. Key File Reference

**Application Entry Points:**
- `current_fall2025/scripts/app.py` - Main Streamlit application

**RAG Pipeline:**
- `current_fall2025/scripts/RAG/pipeline.py` - Main orchestrator
- `current_fall2025/scripts/RAG/query_enhancement.py` - Query rewriting
- `current_fall2025/scripts/RAG/filters.py` - Filter extraction
- `current_fall2025/scripts/RAG/retrieval.py` - Vector search
- `current_fall2025/scripts/RAG/reranking.py` - BM25 reranking
- `current_fall2025/scripts/RAG/response.py` - LLM summarization
- `current_fall2025/scripts/RAG/models.py` - Pydantic models

**Data Pipeline:**
- `current_fall2025/scripts/load_scraper_bpl.py` - API scraping
- `current_fall2025/scripts/bronze_load_bpl_data_to_pg.py` - Bronze loading
- `current_fall2025/scripts/silver_build_bpl_combined_table.py` - Silver transformation
- `current_fall2025/scripts/gold_build_bpl_vector.py` - Embedding generation

**Database Schemas:**
- `current_fall2025/db/bronze_raw_metadata.sql`
- `current_fall2025/db/silver_bpl_combined.sql`
- `current_fall2025/db/gold_bpl_embeddings.sql`
- `current_fall2025/db/function_flatten_jsonb_array.sql`

**Evaluation:**
- `current_fall2025/evaluation/evaluate_ragas.py` - RAGAS metrics
- `current_fall2025/evaluation/deepeval_eva.py` - DeepEval metrics
- `current_fall2025/evaluation/test_queries.json` - Test dataset

**Utilities:**
- `current_fall2025/scripts/run_queries.py` - Batch query runner
- `current_fall2025/scripts/querypg.py` - Database query executor
- `current_fall2025/startscc.sh` - SCC environment setup

### B. Environment Variables

Required variables for `.env` file:

```bash
# PostgreSQL Database
PGHOST=localhost                    # Database host
PGPORT=5432                         # Database port
PGDATABASE=bpl_rag                  # Database name
PGUSER=postgres                     # Database user
PGPASSWORD=your_password            # Database password
PGSSLMODE=prefer                    # SSL mode (disable/allow/prefer/require)

# OpenAI API
OPENAI_API_KEY=sk-...              # OpenAI API key

# MLflow (Optional, for evaluation tracking)
MLFLOW_TRACKING_URI=http://localhost:5000

# Query Configuration (Optional, defaults shown)
TOP=10                              # Number of documents to use for context
K=100                               # Number of documents to retrieve from vector search
```

### C. Common Commands

**Data Pipeline:**
```bash
# Scrape metadata
python load_scraper_bpl.py 1 12000

# Load to bronze
python bronze_load_bpl_data_to_pg.py

# Build silver table
python silver_build_bpl_combined_table.py

# Generate embeddings
python gold_build_bpl_vector.py
```

**Application:**
```bash
# Run locally
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8502

# Run with specific config
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

**Evaluation:**
```bash
# Generate answers for test queries
python run_queries.py -i ../evaluation/test_queries.json -o ../results/answers.csv

# Run RAGAS evaluation
python evaluate_ragas.py

# Run DeepEval evaluation
python deepeval_eva.py
```

**Database:**
```bash
# Connect to PostgreSQL
psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE

# Run SQL file
python querypg.py

# Check vector index
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'bpl_embeddings';
```

### D. Troubleshooting

**Common Issues:**

1. **"No module named 'RAG'"**
   - Ensure `__init__.py` exists in `RAG/` directory
   - Run from `current_fall2025/scripts/` directory

2. **"Database connection failed"**
   - Check `.env` file has correct credentials
   - Verify PostgreSQL is running: `pg_isready -h $PGHOST`
   - Test connection: `psql -h $PGHOST -U $PGUSER -d $PGDATABASE`

3. **"pgvector extension not found"**
   - Install pgvector: `CREATE EXTENSION vector;`
   - Verify: `SELECT * FROM pg_extension WHERE extname = 'vector';`

4. **Slow vector search**
   - Check indexes exist: `\d gold.bpl_embeddings`
   - Rebuild index if needed: `REINDEX INDEX bpl_embeddings_vector_idx;`
   - Consider HNSW index for large datasets

5. **Out of memory during embedding generation**
   - Reduce batch size in `gold_build_bpl_vector.py` (line 81: `batch_size=64` → `batch_size=16`)
   - Process in chunks: Modify script to process date ranges

6. **OpenAI API rate limit errors**
   - Add retry logic with exponential backoff
   - Use slower rate: `time.sleep(1)` between LLM calls
   - Upgrade OpenAI API tier

### E. Contact & Contribution

**Project Maintainers:** [Add team names/emails]

**Repository:** [Add GitHub URL]

**Contribution Guidelines:**
1. Fork repository and create feature branch
2. Follow Python PEP 8 style guidelines
3. Add tests for new functionality
4. Update documentation (this manifest!) for major changes
5. Submit PR to `dev` branch with clear description

**Questions & Support:**
- Open GitHub issue for bugs/feature requests
- [Add Slack/Discord channel if available]
- [Add mailing list if available]

---

## Conclusion

This technical manifest provides a comprehensive overview of the LIBRAG system as implemented in Fall 2025. The modular RAG architecture, Bronze-Silver-Gold data pipeline, and evaluation-driven development approach create a robust, scalable foundation for semantic search over the Boston Public Library's Digital Commonwealth archive.

**Key Takeaways:**
- **Proven at Scale:** Successfully handles 1.2M+ documents with 2-5 second query latency
- **Modular Design:** Clean separation of concerns enables independent testing and optimization
- **Evaluation-Driven:** RAGAS/DeepEval metrics guide continuous improvement
- **Production-Ready:** Deployed on SCC HPC cluster with clear path to cloud hosting

**Next Steps:**
- Implement query result caching for cost reduction
- Add full-text OCR search for digitized documents
- Enhance user feedback loop for continuous improvement

For questions or contributions, please refer to Section E (Contact & Contribution) above.

---

**Document Version:** 1.0  
**Last Updated:** December 7, 2025  
**Authors:** Saksham Goel, Nathan Chang, Penny Lin, Elinor Holt  
**License:** MIT (see LICENSE file)
