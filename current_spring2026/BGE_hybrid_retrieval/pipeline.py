#!/usr/bin/env python3
"""
Main RAG pipeline for the BGE-M3 hybrid retrieval system.

Pipeline stages
---------------
1. Query expansion    — LLM rewrites + expands the user query
2. Filter extraction  — LLM extracts year constraints (shared call)
3. Hybrid retrieval   — Chunk-level dense + title-level dense → RRF fusion (FAISS-backed)
4. Reranking          — Metadata-boosted Stage-1 rerank; optional BGE cross-encoder Stage-2
5. Response generation — LLM summarises top catalog entries for the patron
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .embedder import BGEM3Embedder
from .filters import extract_filters_with_llm
from .query_enhancement import rephrase_and_expand_query
from .reranking import BGECrossEncoder, rerank
from .response import generate_catalog_summary
from .retrieval import hybrid_retrieve, load_indexes

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).parent / "indexes" / "1940"


def RAG(
    llm: Any,
    embedder: BGEM3Embedder,
    query: str,
    chunk_index=None,
    title_index=None,
    chunk_meta: Optional[List[dict]] = None,
    title_meta: Optional[List[dict]] = None,
    doc_chunks_map: Optional[Dict] = None,
    index_dir: Optional[Path] = None,
    top_k: int = 10,
    k_dense: int = 150,
    k_title: int = 50,
    use_cross_encoder: bool = False,
    cross_encoder: Optional[BGECrossEncoder] = None,
) -> Tuple[str, List[Document]]:
    """
    Run the full BGE-M3 hybrid RAG pipeline.

    Indexes can be supplied either as pre-loaded objects (chunk_index, title_index,
    chunk_meta, title_meta, doc_chunks_map) or as a directory path (index_dir) from
    which they will be loaded on each call.  Pre-loaded objects are preferred.

    Args:
        llm:               LangChain-compatible language model (e.g. ChatOpenAI).
        embedder:          Loaded BGEM3Embedder instance.
        query:             Raw user query.
        chunk_index:       Loaded FAISS index for full_article chunks.
        title_index:       Loaded FAISS index for title summaries.
        chunk_meta:        List of chunk metadata dicts.
        title_meta:        List of title metadata dicts.
        doc_chunks_map:    Dict mapping doc_id → list of chunk dicts.
        index_dir:         Path to indexes/{year}/ directory (used if objects not provided).
        top_k:             Number of documents to surface to the LLM and return.
        k_dense:           Chunk-level retrieval candidate pool size.
        k_title:           Title-level retrieval candidate pool size.
        use_cross_encoder: Apply BGE cross-encoder reranking (more accurate, slower).
        cross_encoder:     BGECrossEncoder instance; required when use_cross_encoder=True.

    Returns:
        (summary_string, top_k_reranked_documents)
    """
    total_start = time.time()
    logger.info("=== BGE-M3 Hybrid RAG pipeline started ===")

    try:
        # Load indexes from disk if not pre-supplied
        if chunk_index is None or title_index is None:
            load_dir = index_dir or DEFAULT_INDEX_DIR
            chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map = load_indexes(load_dir)

        # Stage 1: Query expansion
        expanded_query = rephrase_and_expand_query(query, llm)

        # Stage 2: Filter extraction (once, shared between both retrieval paths)
        filters = extract_filters_with_llm(expanded_query, llm)

        # Stage 3: Hybrid retrieval
        retrieved, _ = hybrid_retrieve(
            chunk_index=chunk_index,
            title_index=title_index,
            chunk_meta=chunk_meta,
            title_meta=title_meta,
            doc_chunks_map=doc_chunks_map,
            embedder=embedder,
            query=expanded_query,
            llm=llm,
            k_dense=k_dense,
            k_title=k_title,
            filters=filters,
        )

        if not retrieved:
            logger.warning("No results retrieved.")
            return (
                "No documents found for your query. "
                "Try using different search terms or broader keywords.",
                [],
            )

        # Stage 4: Reranking
        reranked = rerank(
            docs=retrieved,
            query=expanded_query,
            top_k=top_k,
            use_cross_encoder=use_cross_encoder,
            cross_encoder=cross_encoder,
        )

        if not reranked:
            logger.warning("No documents after reranking.")
            return (
                "No relevant items found. Try broadening your search or using different keywords.",
                [],
            )

        # Stage 5: Build context and generate response
        context = "\n\n".join(
            d.page_content for d in reranked[:top_k] if d.page_content
        )
        if not context.strip():
            return "No relevant content found in catalog entries.", []

        summary = generate_catalog_summary(llm, expanded_query, context)

        logger.info(
            f"=== RAG complete in {time.time() - total_start:.2f}s "
            f"({len(reranked)} docs returned) ==="
        )
        return summary, reranked

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        return f"An error occurred while processing your query: {e}", []
