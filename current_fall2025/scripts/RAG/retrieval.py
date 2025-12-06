#!/usr/bin/env python3
"""
Vector retrieval module for RAG system.
Handles pgvector similarity search with metadata filtering.
"""

import json
import time
import logging
from typing import Any, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .filters import extract_filters_with_llm, build_sql_filter
from .models import SearchFilters


def retrieve_from_pg(
    conn,
    embeddings: HuggingFaceEmbeddings,
    query: str,
    llm: Any,
    k: int = 100,
    filters: Optional[SearchFilters] = None
) -> Tuple[List[Document], List[float]]:
    """
    Retrieve relevant documents from PostgreSQL using pgvector similarity search,
    with optional metadata filters (year range, material type) extracted by LLM.
    
    Args:
        conn: PostgreSQL database connection
        embeddings: HuggingFace embeddings model
        query: User query string
        llm: Language model for filter extraction
        k: Number of documents to retrieve
        filters: Optional pre-calculated filters (to avoid re-running LLM)
        
    Returns:
        Tuple of (list of Document objects, list of similarity scores)
    """
    start = time.time()
    logging.info("🔍 Starting similarity search in PostgreSQL (pgvector)...")

    # 1. OPTIMIZATION: Use provided filters if available, else extract them
    if filters is None:
        filters = extract_filters_with_llm(query, llm)
        
    where_clause, params = build_sql_filter(filters)
    logging.info(f"🧩 Applied filters: {filters.model_dump()} → WHERE {where_clause}")

    # Generate query embedding
    qvec = embeddings.embed_query(query)
    qvec_str = "ARRAY[" + ",".join(f"{v:.8f}" for v in qvec) + "]"

    # Execute similarity search with filters
    sql = f"""
        SELECT 
            document_id,
            chunk_index,
            chunk_text,
            metadata,
            1 - (embedding <=> {qvec_str}::vector) AS score
        FROM gold.bpl_embeddings
        WHERE {where_clause}
        ORDER BY embedding <=> {qvec_str}::vector
        LIMIT %s;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (*params, k))
            rows = cur.fetchall()
        
        # Convert results to Document objects
        docs, scores = [], []
        for document_id, chunk_index, chunk_text, metadata, score in rows:
            if len(chunk_text) > 4000:
                chunk_text = chunk_text[:4000]
            
            # Handle metadata being dict or string
            meta_dict = metadata if isinstance(metadata, dict) else json.loads(metadata) if metadata else {}
            
            # Inject source ID and score for downstream usage
            docs.append(Document(
                page_content=chunk_text, 
                metadata={"source": document_id, "vector_score": float(score), **meta_dict}
            ))
            scores.append(float(score))

        logging.info(f"✅ Retrieved {len(docs)} chunks (filters applied) in {time.time() - start:.2f}s.")
        return docs, scores

    except Exception as e:
        logging.error(f"❌ Database retrieval error: {e}")
        return [], []