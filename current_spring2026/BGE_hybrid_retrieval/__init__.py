"""
BGE-M3 Hybrid RAG + GraphRAG for Boston Public Library Digital Commonwealth.

Hybrid retrieval combines:
  - Dense vector search (BGE-M3 dense embeddings, FAISS cosine similarity)
  - Sparse lexical search (BGE-M3 SPLADE-style sparse embeddings)
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - GraphRAG community-cluster retrieval (Microsoft GraphRAG)
  - Query classification to route between semantic and thematic paths
  - BGE-M3 cross-encoder reranking on fused candidates
"""

from pipeline import RAG
from embedder import BGEM3Embedder
from graph_retrieval import GraphRAGStore, load_graph_store, load_all_graph_stores
from query_classifier import classify_query

__all__ = [
    "RAG",
    "BGEM3Embedder",
    "GraphRAGStore",
    "load_graph_store",
    "load_all_graph_stores",
    "classify_query",
]
