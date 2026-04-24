"""
Central configuration for the BPL RAG pipeline.
All tuneable constants live here — change here, affects everywhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL     = "gpt-4o"
# OPENAI_EMBED_MODEL    = "text-embedding-3-small"   # fallback if not using BGE

# ── BGE Embedding ─────────────────────────────────────────────────────────────
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_DEVICE            = "cuda"                       # V100 CC 7.0 incompatible with installed PyTorch (CC >=7.5)
BGE_BATCH_SIZE        = 128                          # lower if OOM

# ── Neo4j / GraphRAG ──────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# GraphRAG is triggered only for content_driven queries
GRAPH_RAG_ENABLED        = True
GRAPH_TOP_K              = 5000    # max additional docs from graph
GRAPH_MIN_ENTITY_MATCHES = 1    # min query entities a doc must match

# ── PostgreSQL / pgVector ─────────────────────────────────────────────────────
PG_HOST               = os.getenv("PG_HOST", "localhost")
PG_PORT               = int(os.getenv("PG_PORT", 5432))
PG_DB                 = os.getenv("PG_DB", "bpl_rag")
PG_USER               = os.getenv("PG_USER", "postgres")
PG_PASSWORD           = os.getenv("PG_PASSWORD", "")
PG_DSN                = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1024   # was 512 — BGE-M3 handles longer context well
CHUNK_OVERLAP = 150    # was 100 — proportionally larger overlap
CHUNK_TOKENIZER       = "cl100k_base"   # tiktoken encoding

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE           = 5000     # candidates from vector search before rerank
TOP_K_BM25            = 5000    # candidates from BM25
TOP_K_FINAL           = 50      # results returned to the user
RRF_K                 = 60      # RRF constant (standard is 60)

# ── Metadata score blend weight ───────────────────────────────────────────────
# final_score = CONTENT_WEIGHT * content_rrf + METADATA_WEIGHT * metadata_sim
CONTENT_WEIGHT        = 0.80
METADATA_WEIGHT       = 0.20

# ── Ingestion ─────────────────────────────────────────────────────────────────
MIN_CHAR_COUNT        = 100     # skip records with fewer chars of raw_text
JSON_DUMP_DIR         = "data/raw"      # folder containing local JSON dumps

# ── Generation ───────────────────────────────────────────────────────────────
MAX_CONTEXT_CHUNKS    = 5       # how many chunks to pass to GPT-4o
GENERATION_MAX_TOKENS = 600

MIN_RELEVANCE_SCORE = 0.1   # documents below this are considered irrelevant

