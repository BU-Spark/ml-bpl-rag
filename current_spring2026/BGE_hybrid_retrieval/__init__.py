"""
BGE-M3 Hybrid RAG for Boston Public Library Digital Commonwealth.

Hybrid retrieval combines:
  - Dense vector search (BGE-M3 dense embeddings, pgvector cosine similarity)
  - Sparse lexical search (BGE-M3 SPLADE-style sparse embeddings, PostgreSQL JSONB + GIN index)
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - BGE-M3 cross-encoder reranking on fused candidates
"""

from .pipeline import RAG
from .embedder import BGEM3Embedder

__all__ = ["RAG", "BGEM3Embedder"]
