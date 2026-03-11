#!/usr/bin/env python3
"""
Main orchestration for the historical newspaper RAG pipeline.

Based on: "Retrieval Augmented Generation for Historical Newspapers"
          ACM Digital Library, 2024. https://dl.acm.org/doi/10.1145/3677389.3702542

Full pipeline:
  1. User Query
  2. Embed query with E5 (intfloat/e5-base-v2)
  3. Search Title/Summary DB
       └─ max similarity < threshold  →  return "no results found"
  4. Retrieve top-k full articles (E5 dense retrieval on Full Article DB)
  5. Hybrid rerank:
       ├─ Cohere cross-encoder score
       ├─ NER-based TF-IDF similarity
       └─ Weighted combination
  6. Pass top reranked articles to LLM for answer generation
"""

import logging
import time
from typing import Any, List, Tuple

import cohere
import spacy
from sentence_transformers import SentenceTransformer

from .models import ArticleResult
from .retrieval import retrieve
from .reranking import rerank_hybrid
from .response import generate_newspaper_answer

logger = logging.getLogger(__name__)


def newspaper_rag(
    query: str,
    # Indexes + metadata (loaded once by the app)
    title_index,
    full_index,
    title_meta: List[dict],
    chunk_meta: List[dict],
    doc_chunks_map: dict,
    # Models / clients
    e5_model: SentenceTransformer,
    llm: Any,
    cohere_client: cohere.ClientV2,
    nlp,
    # Hyperparameters
    threshold: float = 0.30,
    top_k: int = 10,
    rerank_top: int = 5,
    cohere_weight: float = 0.70,
    ner_weight: float = 0.30,
) -> Tuple[str, List[ArticleResult]]:
    """
    Run the full newspaper RAG pipeline end-to-end.

    Args:
        query:           User query string.
        title_index:     FAISS index for Title/Summary DB.
        full_index:      FAISS index for Full Article DB.
        title_meta:      Metadata list aligned with title_index.
        chunk_meta:      Metadata list aligned with full_index.
        doc_chunks_map:  doc_id → list[chunk metadata dicts].
        e5_model:        Loaded intfloat/e5-base-v2 SentenceTransformer.
        llm:             LLM instance for answer generation.
        cohere_client:   Authenticated Cohere V2 client.
        nlp:             Loaded spaCy model for NER.
        threshold:       Min cosine similarity to pass Stage 1 gate (default 0.30).
        top_k:           Articles retrieved in Stage 2 before reranking.
        rerank_top:      Articles returned after reranking (LLM context size).
        cohere_weight:   Weight for Cohere score in hybrid combination (0–1).
        ner_weight:      Weight for NER TF-IDF score in hybrid combination (0–1).

    Returns:
        (answer, reranked_articles)
        answer is a "no results" message when Stage 1 gate rejects the query.
    """
    t0 = time.time()
    logger.info(f"Newspaper RAG started for query: {query!r}")

    # ── Stage 1 + 2: Two-stage retrieval ─────────────────────────────────────
    articles, max_score = retrieve(
        query=query,
        model=e5_model,
        title_index=title_index,
        full_index=full_index,
        title_meta=title_meta,
        chunk_meta=chunk_meta,
        doc_chunks_map=doc_chunks_map,
        threshold=threshold,
        top_k=top_k,
    )

    if not articles:
        msg = (
            f"No relevant articles found "
            f"(best similarity {max_score:.3f} is below the threshold {threshold}). "
            "Try rephrasing your query or using more specific terms."
        )
        logger.info(msg)
        return msg, []

    # ── Stage 3: Hybrid reranking ─────────────────────────────────────────────
    reranked = rerank_hybrid(
        query=query,
        articles=articles,
        cohere_client=cohere_client,
        nlp=nlp,
        top_k=rerank_top,
        cohere_weight=cohere_weight,
        ner_weight=ner_weight,
    )

    if not reranked:
        return "No relevant content found after reranking. Try a different query.", []

    # ── Stage 4: LLM answer generation ───────────────────────────────────────
    answer = generate_newspaper_answer(llm, query, reranked)

    logger.info(f"Newspaper RAG completed in {time.time() - t0:.2f}s.")
    return answer, reranked
