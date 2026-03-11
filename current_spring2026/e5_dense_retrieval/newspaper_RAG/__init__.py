"""
Newspaper RAG package.

Pipeline (per ACM 2024 paper):
  User Query
    → E5 embed query
    → Search Title/Summary DB  (threshold gate)
    → Retrieve top-k full articles (E5 dense)
    → Hybrid rerank: Cohere cross-encoder + NER TF-IDF
    → LLM answer generation
"""

from .pipeline import newspaper_rag
from .ingest import ingest_json, load_indexes
from .retrieval import retrieve
from .reranking import rerank_hybrid
from .response import generate_newspaper_answer

__all__ = [
    "newspaper_rag",
    "ingest_json",
    "load_indexes",
    "retrieve",
    "rerank_hybrid",
    "generate_newspaper_answer",
]
