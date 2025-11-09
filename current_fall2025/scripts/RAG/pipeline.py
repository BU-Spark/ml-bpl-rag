#!/usr/bin/env python3
"""
Main RAG pipeline orchestration.
Coordinates query enhancement, retrieval, reranking, and response generation.
"""

import time
import logging
from typing import Any, List, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .query_enhancement import rephrase_and_expand_query
from .retrieval import retrieve_from_pg
from .reranking import rerank
from .response import generate_catalog_summary


def RAG(
    llm: Any,
    conn,
    embeddings: HuggingFaceEmbeddings,
    query: str,
    top: int = 10,
    k: int = 100
) -> Tuple[str, List[Document]]:
    """
    Main RAG pipeline for catalog search.
    
    Pipeline stages:
    1. Query rewriting and expansion
    2. Vector similarity search with metadata filtering
    3. BM25 reranking with metadata scoring
    4. LLM-based catalog summary generation
    
    Args:
        llm: Language model instance
        conn: PostgreSQL database connection
        embeddings: HuggingFace embeddings model
        query: User query string
        top: Number of top documents to use for context
        k: Number of documents to retrieve from vector search
        
    Returns:
        Tuple of (summary_string, list_of_reranked_documents)
    """
    total_start = time.time()
    logging.info("🚀 Starting RAG pipeline...")
    
    try:
        # Stage 1: Query enhancement
        expanded_query = rephrase_and_expand_query(query, llm)

        # Stage 2: Vector retrieval with filters
        retrieved, _ = retrieve_from_pg(conn, embeddings, expanded_query, llm, k)
        if not retrieved:
            logging.warning("⚠️ No results retrieved from pgvector.")
            return "No documents found for your query. Try using different search terms or broader keywords.", []

        # Stage 3: Reranking
        reranked = rerank(retrieved, expanded_query, top_k=10)
        if not reranked:
            logging.warning("⚠️ No documents passed reranking.")
            return "No relevant items found in the catalog. Try broadening your search or using different keywords.", []

        # Stage 4: Context preparation
        context = "\n\n".join(d.page_content for d in reranked[:top] if d.page_content)
        if not context.strip():
            logging.warning("⚠️ Context is empty after reranking.")
            return "No relevant content found in catalog entries.", []

        # Stage 5: Response generation
        summary = generate_catalog_summary(llm, expanded_query, context)

        logging.info(f"🏁 RAG completed in {time.time() - total_start:.2f}s total.")
        return summary, reranked

    except Exception as e:
        logging.error(f"❌ Error in RAG: {e}")
        return f"An error occurred while processing your query: {e}", []

