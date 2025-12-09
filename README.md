# LIBRAG: Retrieval-Augmented Search for Boston Public Library

**LIBRAG** is a Retrieval-Augmented Generation (RAG) system designed to improve access to the Boston Public Library's (BPL) Digital Commonwealth archive. It enables users to ask natural language questions and retrieve highly relevant historical documents through semantic search.

**Current Implementation (Fall 2025):** PostgreSQL with pgvector extension, modular RAG pipeline, and evaluation-driven development.

---

## Project Context

The Boston Public Library maintains an extensive digital archive of over 1.2 million historical items through the Digital Commonwealth platform, including photographs, manuscripts, maps, newspapers, and audio recordings spanning from the 17th to 23rd centuries. However, traditional keyword-based search makes it difficult for researchers, educators, genealogists, and the general public to discover relevant materials.

**LIBRAG solves this problem by:**
- Transforming natural language questions into catalog-aware searches
- Providing semantic understanding of historical queries (dates, locations, events, people)
- Maintaining transparency through source citations and metadata provenance
- Preserving librarian expertise through catalog-focused prompt engineering



The system acts as a **catalog search tool**, not a knowledge base—it describes what materials are available rather than answering factual questions from document content.

---

## Key Features

### Production-Ready Infrastructure
- **Database:** PostgreSQL 14+ with pgvector extension for vector similarity search
- **Scale:** 1.2M+ metadata records → 5M+ embedded text chunks (768 dimensions)
- **Performance:** 2-5 second query latency with intelligent reranking
- **Architecture:** Bronze-Silver-Gold medallion data pipeline for data quality and traceability

### Modular RAG Pipeline
- **Query Enhancement:** LLM-based query expansion with historical terminology and synonyms
- **Smart Filtering:** Automatic extraction of temporal ranges and material type filters
- **Vector Retrieval:** Fast cosine similarity search with metadata-based filtering
- **BM25 Reranking:** Lexical matching combined with metadata-aware scoring
- **Catalog Responses:** Librarian-style summaries focused on describing available materials

### Evaluation & Quality Assurance
- **RAGAS Metrics:** Context Recall (0.85+), Answer Relevancy (0.78+)
- **DeepEval Framework:** Automated evaluation on curated test queries
- **MLflow Tracking:** Experiment logging and metric comparison for continuous improvement

### Models & Technology
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768-dim vectors)
- **LLM:** OpenAI GPT-4o-mini for query understanding and response generation
- **Reranking:** BM25 lexical scoring with metadata field weights
- **UI Framework:** Streamlit with developer mode for debugging

---

## Live Demo

Try the hosted demo:  
👉 [Hugging Face Spaces - BPL-RAG-Spring-2025](https://huggingface.co/spaces/spark-ds549/BPL-RAG-Spring-2025)

---

## System Architecture

### Bronze-Silver-Gold Data Pipeline

The system implements a medallion architecture pattern for data quality and transformation:

**Bronze Layer (Raw Data):**
```
Digital Commonwealth API → bronze.bpl_metadata
- Purpose: Immutable source of truth
- Size: ~1.2M records as-is from API
- Schema: id (TEXT), data (JSONB)
```

**Silver Layer (Processed Data):**
```
bronze → silver.bpl_combined
- Date extraction and normalization (handles "circa" dates with ±5 year buffer)
- Multi-field text concatenation (title, abstract, notes, subjects, locations, dates)
- Schema: document_id, summary_text, metadata (JSONB), date_start, date_end
```

**Gold Layer (Vector Embeddings):**
```
silver → gold.bpl_embeddings
- Text chunking: RecursiveCharacterTextSplitter (1000 chars, 100 overlap)
- Embedding generation: sentence-transformers/all-mpnet-base-v2
- Schema: document_id, chunk_index, chunk_text, embedding VECTOR(768), metadata
- Indexes: Vector similarity (cosine distance), date ranges, metadata (GIN)
```

### Modular RAG Pipeline

Each stage is independently testable and optimizable:

```
User Query
    ↓
[1] Query Enhancement (LLM)
    → Expand with historical terminology, synonyms, related concepts
    ↓
[2] Filter Extraction (LLM)
    → Extract temporal constraints (years, date ranges, eras)
    → Extract material types (Still image, Cartographic, Manuscript, Audio, etc.)
    ↓
[3] Vector Retrieval (pgvector)
    → Generate query embedding
    → Cosine similarity search with filter constraints
    → Retrieve top-k chunks
    ↓
[4] BM25 Reranking
    → Merge chunks by document
    → Apply lexical matching (BM25)
    → Boost scores based on metadata field presence and exact year matches
    → Select top documents
    ↓
[5] Response Generation (LLM)
    → Generate catalog-style summary
    → Describe available materials (titles, dates, collections)
    ↓
Display Results + Source Metadata
```

**Pipeline Modules:**
- `query_enhancement.py` - Query rewriting and expansion
- `filters.py` - Filter extraction and SQL generation
- `retrieval.py` - Vector similarity search
- `reranking.py` - BM25 + metadata scoring
- `response.py` - LLM-based summary generation
- `pipeline.py` - End-to-end orchestration

---

## Folder Structure

The project is organized into semester-specific directories with shared resources at the root level:

```
ml-bpl-rag/
│
├── current_fall2025/              # Current semester implementation (ACTIVE)
│   ├── db/                        # SQL schema definitions
│   │   ├── bronze_raw_metadata.sql
│   │   ├── silver_bpl_combined.sql
│   │   ├── gold_bpl_embeddings.sql
│   │   └── function_flatten_jsonb_array.sql
│   │
│   ├── scripts/                   # Main application code
│   │   ├── app.py                 # Streamlit web application (ENTRY POINT)
│   │   ├── RAG/                   # Modular RAG pipeline components
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py        # Main orchestrator
│   │   │   ├── query_enhancement.py
│   │   │   ├── filters.py
│   │   │   ├── retrieval.py
│   │   │   ├── reranking.py
│   │   │   ├── response.py
│   │   │   ├── models.py          # Pydantic data models
│   │   │   └── utils.py
│   │   │
│   │   ├── bronze_load_bpl_data_to_pg.py      # Load raw JSON to database
│   │   ├── silver_build_bpl_combined_table.py # Transform to Silver layer
│   │   ├── gold_build_bpl_vector.py           # Generate embeddings
│   │   ├── load_scraper_bpl.py                # Scrape Digital Commonwealth API
│   │   ├── run_queries.py                     # Batch query runner
│   │   └── querypg.py                         # Database query executor
│   │
│   ├── evaluation/                # Evaluation framework
│   │   ├── evaluate_ragas.py      # RAGAS metrics
│   │   ├── deepeval_eva.py        # DeepEval metrics
│   │   ├── test_queries.json      # Curated test queries
│   │   └── results/               # Evaluation results
│   │
│   ├── notebooks/                 # Jupyter notebooks for analysis
│   │   └── ragas_evaluation.ipynb
│   │
│   ├── results/                   # Query results and outputs
│   └── startscc.sh                # SCC HPC cluster setup script
│
├── archive_spring2025/            # Previous semester work (reference)
├── archive_fall2024/              # Earlier semester work (reference)
│
├── Deployment/                    # Deployment configurations
├── docs/                          # Additional documentation
│
├── requirements.txt               # Python dependencies (pip-compile generated)
├── requirements.in                # Human-maintained dependency list
├── .env                           # Environment variables (not in git)
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
├── README.md                      # This file
├── TECHNICAL_MANIFEST.md          # Comprehensive technical documentation
└── COLLABORATORS                  # Project team members
```

**Key Directories Explained:**

- **`current_fall2025/`** - Active development directory for Fall 2025 semester
  - All production code and evaluation scripts are here
  - Organized by function: db schemas, application scripts, evaluation, notebooks

- **`scripts/RAG/`** - Modular RAG pipeline components
  - Each stage (query enhancement, retrieval, reranking, etc.) is a separate module
  - Enables independent testing and optimization
  - `pipeline.py` orchestrates the full flow

- **`db/`** - SQL schema definitions for Bronze-Silver-Gold layers
  - Create these tables in order: bronze → silver → gold
  - Includes custom SQL functions (e.g., `flatten_jsonb_array`)

- **`evaluation/`** - Automated evaluation framework
  - Test queries with ground truth answers
  - RAGAS and DeepEval metric implementations
  - Results tracked in `results/` subdirectory

- **Archive directories** - Historical reference
  - Previous semesters' implementations (Pinecone-based, different architectures)
  - Kept for reference but not actively maintained

---

## Setup Instructions

### Prerequisites

- **Python:** 3.12.4 or later
- **PostgreSQL:** 14+ with pgvector extension installed
- **OpenAI API Key:** For GPT-4o-mini access

### Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/[your-org]/ml-bpl-rag.git
cd ml-bpl-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows


# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```bash
# PostgreSQL Database
PGHOST=localhost
PGPORT=5432
PGDATABASE=bpl_rag
PGUSER=your_username
PGPASSWORD=your_password
PGSSLMODE=prefer

# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here
```

### Set Up Database

```bash
# Connect to PostgreSQL and create database
psql -h $PGHOST -U $PGUSER -d postgres
CREATE DATABASE bpl_rag;
\c bpl_rag
CREATE EXTENSION vector;

# Create schemas
CREATE SCHEMA bronze;
CREATE SCHEMA silver;
CREATE SCHEMA gold;
\q

# Run SQL schema files
cd current_fall2025/scripts
python querypg.py
```

---

## Data Pipeline

To populate the database with BPL metadata:

```bash
cd current_fall2025/scripts

# 1. Scrape metadata from Digital Commonwealth API
python load_scraper_bpl.py 1 12000

# 2. Load raw data to Bronze layer
python bronze_load_bpl_data_to_pg.py

# 3. Transform to Silver layer
python silver_build_bpl_combined_table.py

# 4. Generate embeddings (Gold layer - takes several hours)
python gold_build_bpl_vector.py
```

---

## Running the Application

```bash
cd current_fall2025/scripts
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501/`

### Using the Interface

1. **Enter Query:** Type a natural language question in the chat input
2. **View Results:** System displays catalog summary and referenced archives
3. **Developer Mode:** Toggle in sidebar to see query expansion, filters, and debug info

**Example Queries:**
- "Show me photographs of Boston streets in the 1920s"
- "Find maps of Worcester, MA from the 18th century"
- "What were some important historical events in Boston in 1919?"

---

## Deployment

### Current: BU SCC HPC Cluster

```bash
module load miniconda
conda activate ./env
cd ml-bpl-rag/current_fall2025/scripts
streamlit run app.py --server.port 8501
```

See `TECHNICAL_MANIFEST.md` Section 9 for cloud deployment strategies (Docker, AWS, GCP, Azure).

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Database** | PostgreSQL + pgvector | Vector similarity search |
| **Embeddings** | all-mpnet-base-v2 | 768-dim text embeddings |
| **LLM** | OpenAI GPT-4o-mini | Query understanding & summarization |
| **Framework** | LangChain 0.3.21 | RAG orchestration |
| **UI** | Streamlit 1.45.0 | Web application |
| **Reranking** | rank-bm25 | Lexical reranking |
| **Evaluation** | RAGAS + DeepEval | Quality metrics |

---

## Project Team

**Fall 2025 Contributors:**
- Saksham Goel
- Nathan Chang
- Penny Lin
- Elinor Holt

**Course:** DS/CS 549 - Machine Learning (Boston University)

---

## Documentation

- **README.md** (this file) - Quick start and setup guide
- **TECHNICAL_MANIFEST.md** - Comprehensive technical documentation
  - Architecture deep-dive
  - Edge cases and error handling
  - Performance characteristics
  - Improvement roadmap

---

## Known Limitations

- Searches metadata catalogs only, not full document text
- First query after restart has ~30-60 second cold start
- No query result caching (duplicate queries re-execute)
- Temporal expressions rely on LLM interpretation

*See `TECHNICAL_MANIFEST.md` for detailed limitations and mitigations.*

---

## Future Improvements

- Query result caching and streaming responses
- Full-text OCR integration for document content search
- Conversational search with multi-turn dialogue
- User feedback loop and personalized recommendations
- Distributed vector search for improved scalability

*See `TECHNICAL_MANIFEST.md` for detailed improvement roadmap with effort estimates.*

---

## Contributing

We welcome contributions! Please:
1. Fork the repository and create a feature branch
2. Follow Python PEP 8 style guidelines
3. Add tests for new functionality
4. Submit pull request to `dev` branch

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Last Updated:** December 7, 2025  
**Version:** Fall 2025 Release
