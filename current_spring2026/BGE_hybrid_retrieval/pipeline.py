#!/usr/bin/env python3
"""
Main RAG pipeline for the BGE-M3 hybrid retrieval system.

Pipeline stages (see flowchart)
-------------------------------
1. Query expansion + Classification + Filter extraction  (3 LLM calls IN PARALLEL)
2. Parallel retrieval:  Standard RAG (Dense+Sparse+Title → RRF) ∥ GraphRAG
3. Merge both contexts
4. Reranking
5. Response generation

Latency optimisations
---------------------
- All 3 pre-retrieval LLM calls run concurrently via ThreadPoolExecutor
- Standard RAG and GraphRAG retrieval paths run in parallel
- IVF FAISS indexes for sub-linear dense search on large datasets
- Sparse inverted index for fast lexical retrieval (no FAISS overhead)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from embedder import BGEM3Embedder
from filters import extract_filters_with_llm
from graph_retrieval import GraphRAGStore, graphrag_retrieve
from query_classifier import QueryType, classify_query
from query_enhancement import rephrase_and_expand_query
from reranking import BGECrossEncoder, rerank
from response import generate_catalog_summary
from retrieval import SparseInvertedIndex, hybrid_retrieve, load_indexes

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).parent / "indexes" / "1940"


# ---------------------------------------------------------------------------
# Internal: Standard RAG path
# ---------------------------------------------------------------------------

def _standard_rag_path(
    chunk_index,
    title_index,
    chunk_meta: List[dict],
    title_meta: List[dict],
    doc_chunks_map: Dict,
    embedder: BGEM3Embedder,
    expanded_query: str,
    llm: Any,
    k_dense: int,
    k_title: int,
    filters,
    sparse_index,
) -> List[Document]:
    start = time.time()
    try:
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
            sparse_index=sparse_index,
        )
        logger.info(f"Standard RAG path: {len(retrieved)} docs in {time.time() - start:.2f}s")
        return retrieved
    except Exception as e:
        logger.error(f"Standard RAG path failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Internal: GraphRAG path
# ---------------------------------------------------------------------------

def _graphrag_path(
    graph_store: Optional[GraphRAGStore],
    expanded_query: str,
    query_type: QueryType,
    embedder: Optional[BGEM3Embedder] = None,
    use_full_search: bool = True,
    top_k: int = 5,
) -> Tuple[str, List[Document]]:
    start = time.time()
    if graph_store is None or not graph_store.is_valid:
        return "", []

    try:
        graphrag_type = "global" if query_type == "thematic" else "local"
        context_text, docs = graphrag_retrieve(
            store=graph_store,
            query=expanded_query,
            query_type=graphrag_type,
            embedder=embedder,
            top_k=top_k,
            use_full_search=use_full_search,
        )
        logger.info(
            f"GraphRAG path ({graphrag_type}): {len(docs)} docs in "
            f"{time.time() - start:.2f}s"
        )
        return context_text, docs
    except Exception as e:
        logger.error(f"GraphRAG path failed: {e}", exc_info=True)
        return "", []


# ---------------------------------------------------------------------------
# Context merging
# ---------------------------------------------------------------------------

def _merge_contexts(
    standard_docs: List[Document],
    graphrag_context: str,
    graphrag_docs: List[Document],
    query_type: QueryType,
) -> Tuple[List[Document], str]:
    merged_docs: List[Document] = []
    seen_sources = set()

    for doc in standard_docs:
        key = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_index", 0))
        if key not in seen_sources:
            seen_sources.add(key)
            merged_docs.append(doc)

    for doc in graphrag_docs:
        key = doc.metadata.get("community_id", doc.page_content[:100])
        if key not in seen_sources:
            seen_sources.add(key)
            merged_docs.append(doc)

    standard_context = "\n\n".join(
        d.page_content for d in standard_docs[:10] if d.page_content
    )

    if query_type == "thematic":
        combined = ""
        if graphrag_context:
            combined += f"=== Thematic Overview (from knowledge graph) ===\n{graphrag_context}\n\n"
        if standard_context:
            combined += f"=== Relevant Documents ===\n{standard_context}"
    elif query_type == "semantic":
        combined = ""
        if standard_context:
            combined += f"=== Relevant Documents ===\n{standard_context}\n\n"
        if graphrag_context:
            combined += f"=== Broader Context (from knowledge graph) ===\n{graphrag_context}"
    else:
        combined = ""
        if standard_context:
            combined += f"=== Relevant Documents ===\n{standard_context}\n\n"
        if graphrag_context:
            combined += f"=== Thematic Context (from knowledge graph) ===\n{graphrag_context}"

    return merged_docs, combined


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def RAG(
    llm: Any,
    embedder: BGEM3Embedder,
    query: str,
    chunk_index=None,
    title_index=None,
    chunk_meta: Optional[List[dict]] = None,
    title_meta: Optional[List[dict]] = None,
    doc_chunks_map: Optional[Dict] = None,
    sparse_index: Optional[SparseInvertedIndex] = None,
    index_dir: Optional[Path] = None,
    graph_store: Optional[GraphRAGStore] = None,
    use_graphrag: bool = True,
    graphrag_full_search: bool = True,
    top_k: int = 10,
    k_dense: int = 150,
    k_title: int = 50,
    use_cross_encoder: bool = False,
    cross_encoder: Optional[BGECrossEncoder] = None,
) -> Tuple[str, List[Document]]:
    """
    Run the full BGE-M3 hybrid RAG pipeline with parallel GraphRAG.

    Flowchart:
        User Query
        → [Query Expansion ∥ Query Classification ∥ Filter Extraction]  (parallel)
        → [Standard RAG (dense+sparse+title → RRF) ∥ GraphRAG]         (parallel)
        → Merge Both Contexts → Reranking → LLM Response
    """
    total_start = time.time()
    logger.info("=== BGE-M3 Hybrid RAG + GraphRAG pipeline started ===")

    try:
        # Load indexes if not pre-supplied
        if chunk_index is None or title_index is None:
            load_dir = index_dir or DEFAULT_INDEX_DIR
            chunk_index, title_index, chunk_meta, title_meta, doc_chunks_map, sparse_index = load_indexes(load_dir)

        # ── Stage 1: Run all 3 LLM pre-processing calls IN PARALLEL ─────
        # This saves ~2-3s vs sequential (each call is ~0.5-1s)
        expanded_query = query  # fallback
        query_type: QueryType = "both"
        filters = None

        need_classification = use_graphrag and graph_store is not None

        with ThreadPoolExecutor(max_workers=3) as pre_exec:
            fut_expand = pre_exec.submit(rephrase_and_expand_query, query, llm)
            fut_filters = pre_exec.submit(extract_filters_with_llm, query, llm)
            fut_classify = None
            if need_classification:
                fut_classify = pre_exec.submit(classify_query, query)

            try:
                expanded_query = fut_expand.result(timeout=15)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

            try:
                filters = fut_filters.result(timeout=15)
            except Exception as e:
                logger.warning(f"Filter extraction failed: {e}")

            if fut_classify is not None:
                try:
                    query_type = fut_classify.result(timeout=15)
                except Exception as e:
                    logger.warning(f"Query classification failed: {e}")

        logger.info(
            f"Pre-processing done in {time.time() - total_start:.2f}s "
            f"(type={query_type})"
        )

        # ── Stage 2: Parallel retrieval ──────────────────────────────────
        standard_docs: List[Document] = []
        graphrag_context: str = ""
        graphrag_docs: List[Document] = []

        run_standard = query_type in ("semantic", "both")
        run_graphrag = (
            use_graphrag
            and graph_store is not None
            and graph_store.is_valid
            and query_type in ("thematic", "both")
        )

        with ThreadPoolExecutor(max_workers=2) as exec:
            futures = {}

            if run_standard:
                futures["standard"] = exec.submit(
                    _standard_rag_path,
                    chunk_index, title_index, chunk_meta, title_meta,
                    doc_chunks_map, embedder, expanded_query, llm,
                    k_dense, k_title, filters, sparse_index,
                )

            if run_graphrag:
                futures["graphrag"] = exec.submit(
                    _graphrag_path,
                    graph_store, expanded_query, query_type, embedder,
                    graphrag_full_search,
                )

            # For thematic-only, still run standard as fallback
            if query_type == "thematic" and "standard" not in futures:
                futures["standard"] = exec.submit(
                    _standard_rag_path,
                    chunk_index, title_index, chunk_meta, title_meta,
                    doc_chunks_map, embedder, expanded_query, llm,
                    k_dense, k_title, filters, sparse_index,
                )

            for key, future in futures.items():
                try:
                    result = future.result(timeout=120)
                    if key == "standard":
                        standard_docs = result
                    elif key == "graphrag":
                        graphrag_context, graphrag_docs = result
                except Exception as e:
                    logger.error(f"{key} retrieval failed: {e}")

        if not standard_docs and not graphrag_docs:
            return (
                "No documents found for your query. "
                "Try using different search terms or broader keywords.",
                [],
            )

        # ── Stage 3: Merge contexts ──────────────────────────────────────
        merged_docs, combined_context = _merge_contexts(
            standard_docs, graphrag_context, graphrag_docs, query_type,
        )

        # ── Stage 4: Reranking ───────────────────────────────────────────
        reranked = rerank(
            docs=merged_docs,
            query=expanded_query,
            top_k=top_k,
            use_cross_encoder=use_cross_encoder,
            cross_encoder=cross_encoder,
        )

        if not reranked:
            return "No relevant items found. Try broadening your search.", []

        # ── Stage 5: Generate response ───────────────────────────────────
        context_for_llm = combined_context[:8000] if combined_context.strip() else "\n\n".join(
            d.page_content for d in reranked[:top_k] if d.page_content
        )

        if not context_for_llm.strip():
            return "No relevant content found in catalog entries.", []

        summary = generate_catalog_summary(llm, expanded_query, context_for_llm)

        elapsed = time.time() - total_start
        logger.info(
            f"=== Pipeline complete in {elapsed:.2f}s "
            f"({len(reranked)} docs, type={query_type}) ==="
        )
        return summary, reranked

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        return f"An error occurred while processing your query: {e}", []
