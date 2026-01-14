#!/usr/bin/env python3
"""
Reranking module for RAG system.
Handles BM25 reranking with metadata-based scoring.
"""

import re
import time
import logging
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# Metadata field weights for scoring
METADATA_WEIGHTS = {
    "title_info_primary_tsi": 1.5,
    "name_role_tsim": 1.4,
    "date_tsim": 1.3,
    "abstract_tsi": 1.0,
    "note_tsim": 0.8,
    "subject_geographic_sim": 0.5,
    "genre_basic_ssim": 0.5,
    "genre_specific_ssim": 0.5,
}


def extract_years_from_query(query: str) -> List[str]:
    """
    Extract 4-digit years from query string.
    
    Args:
        query: User query string
        
    Returns:
        List of year strings found in query
    """
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2}|22\d{2}|23\d{2})\b", query)


def rerank(docs: List[Document], query: str, top_k: int = 10) -> List[Document]:
    """
    Rerank documents using BM25 and metadata-based scoring.
    
    Process:
    1. Merge chunks by document_id
    2. Apply BM25 lexical reranking
    3. Boost scores based on metadata field presence
    4. Add large boost for exact year matches
    5. Return top-k documents
    
    Args:
        docs: List of Document objects to rerank
        query: User query string
        top_k: Number of top documents to return
        
    Returns:
        List of top-k reranked Document objects
    """
    if not docs:
        logging.warning("⚠️ No documents provided for reranking.")
        return []

    logging.info("⚖️ Starting BM25 reranking...")
    start = time.time()

    # Extract years from query for date matching
    query_years = extract_years_from_query(query)
    
    # Group chunks by document_id and merge
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    merged_docs = []
    for src, chunks in grouped.items():
        text = " ".join(c.page_content for c in chunks if c.page_content)
        merged_docs.append(Document(page_content=text, metadata=chunks[0].metadata))

    # Apply BM25 ranking
    bm25 = BM25Retriever.from_documents(merged_docs, k=len(merged_docs))
    ranked = bm25.invoke(query)

    # Apply metadata-based scoring
    final_ranked = []
    for d in ranked:
        score = 1.0
        
        # Add weight for each present metadata field
        for field, weight in METADATA_WEIGHTS.items():
            if field in d.metadata and d.metadata[field]:
                score += weight

        # Large boost for exact year matches in date field
        date_field = str(d.metadata.get("date_tsim", ""))
        for y in query_years:
            if re.search(rf"\b{y}\b", date_field):
                score += 50
                break

        final_ranked.append((d, score))

    # Sort by score and return top-k
    final_ranked.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"✅ Reranked {len(final_ranked)} documents in {time.time() - start:.2f}s.")
    return [doc for doc, _ in final_ranked[:top_k]]

