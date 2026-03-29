#!/usr/bin/env python3
"""
GraphRAG retrieval module for the BGE-M3 hybrid RAG system.

Provides query-time retrieval over pre-built GraphRAG community summaries
using Microsoft's GraphRAG library.

Two retrieval modes
-------------------
- **Global search** (thematic queries): Map-reduce over community reports.
  Best for broad/abstract questions ("What themes appear in 1920s Boston?").
- **Local search** (entity-focused): Combines entity descriptions,
  relationships, community context, and source text units.
  Best for specific entity queries ("What happened at the Boston molasses disaster?").

The query classifier (see query_classifier.py) decides which mode to use.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_GRAPHRAG_DIR = Path(__file__).parent / "graphrag_index"


# ---------------------------------------------------------------------------
# Loading GraphRAG artefacts
# ---------------------------------------------------------------------------

class GraphRAGStore:
    """
    Holds pre-loaded GraphRAG parquet artefacts for fast query-time access.
    """

    def __init__(
        self,
        entities: pd.DataFrame,
        relationships: pd.DataFrame,
        communities: pd.DataFrame,
        community_reports: pd.DataFrame,
        text_units: pd.DataFrame,
        project_dir: Path,
    ):
        self.entities = entities
        self.relationships = relationships
        self.communities = communities
        self.community_reports = community_reports
        self.text_units = text_units
        self.project_dir = project_dir

    @property
    def is_valid(self) -> bool:
        return (
            len(self.community_reports) > 0
            and len(self.entities) > 0
        )


def load_graph_store(project_dir: Path) -> Optional[GraphRAGStore]:
    """
    Load GraphRAG parquet artefacts from a project directory.

    Args:
        project_dir: Path to graphrag_index/{year}/ directory.

    Returns:
        GraphRAGStore or None if artefacts are missing.
    """
    output_dir = project_dir / "output"
    required_files = [
        "entities.parquet",
        "communities.parquet",
        "community_reports.parquet",
    ]

    for f in required_files:
        if not (output_dir / f).exists():
            logger.warning(f"GraphRAG artefact missing: {output_dir / f}")
            return None

    try:
        entities = pd.read_parquet(output_dir / "entities.parquet")
        communities = pd.read_parquet(output_dir / "communities.parquet")
        community_reports = pd.read_parquet(output_dir / "community_reports.parquet")

        # These may not exist in all configs
        rel_path = output_dir / "relationships.parquet"
        relationships = pd.read_parquet(rel_path) if rel_path.exists() else pd.DataFrame()

        tu_path = output_dir / "text_units.parquet"
        text_units = pd.read_parquet(tu_path) if tu_path.exists() else pd.DataFrame()

        store = GraphRAGStore(
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            project_dir=project_dir,
        )
        logger.info(
            f"Loaded GraphRAG store from {project_dir}: "
            f"{len(entities)} entities, {len(community_reports)} community reports"
        )
        return store

    except Exception as e:
        logger.error(f"Failed to load GraphRAG store from {project_dir}: {e}")
        return None


def load_all_graph_stores(graphrag_dir: Path = DEFAULT_GRAPHRAG_DIR) -> Dict[str, GraphRAGStore]:
    """
    Load GraphRAG stores for all indexed years.

    Returns:
        Dict mapping year string to GraphRAGStore.
    """
    stores: Dict[str, GraphRAGStore] = {}
    if not graphrag_dir.exists():
        logger.warning(f"GraphRAG directory not found: {graphrag_dir}")
        return stores

    for year_dir in sorted(graphrag_dir.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            store = load_graph_store(year_dir)
            if store is not None and store.is_valid:
                stores[year_dir.name] = store

    logger.info(f"Loaded GraphRAG stores for {len(stores)} years: {list(stores.keys())}")
    return stores


# ---------------------------------------------------------------------------
# GraphRAG Global Search (thematic / community-level)
# ---------------------------------------------------------------------------

async def _async_global_search(
    config,
    store: GraphRAGStore,
    query: str,
    community_level: int = 2,
) -> Tuple[str, dict]:
    """Run GraphRAG global search (async)."""
    import graphrag.api as api

    response, context = await api.global_search(
        config=config,
        entities=store.entities,
        communities=store.communities,
        community_reports=store.community_reports,
        community_level=community_level,
        dynamic_community_selection=True,
        response_type="Multiple Paragraphs",
        query=query,
    )
    return response, context


def global_search(
    store: GraphRAGStore,
    query: str,
    community_level: int = 2,
) -> Tuple[str, List[Document]]:
    """
    Run GraphRAG global search over community summaries.

    Best for thematic / broad questions about patterns, themes, and overviews.

    Args:
        store:           Pre-loaded GraphRAGStore.
        query:           User query string.
        community_level: Hierarchy level (higher = more abstract).

    Returns:
        (response_text, list_of_community_documents)
    """
    start = time.time()
    logger.info(f"GraphRAG global search: '{query[:80]}...'")

    try:
        from graphrag.config.load_config import load_config
        config = load_config(store.project_dir)

        response, context = asyncio.run(
            _async_global_search(config, store, query, community_level)
        )

        # Convert community reports used in the response to Documents
        docs = _community_reports_to_documents(store.community_reports)

        logger.info(f"GraphRAG global search completed in {time.time() - start:.2f}s")
        return response, docs

    except Exception as e:
        logger.error(f"GraphRAG global search failed: {e}", exc_info=True)
        return "", []


# ---------------------------------------------------------------------------
# GraphRAG Local Search (entity-focused)
# ---------------------------------------------------------------------------

async def _async_local_search(
    config,
    store: GraphRAGStore,
    query: str,
    community_level: int = 2,
) -> Tuple[str, dict]:
    """Run GraphRAG local search (async)."""
    import graphrag.api as api

    response, context = await api.local_search(
        config=config,
        entities=store.entities,
        relationships=store.relationships,
        text_units=store.text_units,
        community_reports=store.community_reports,
        communities=store.communities,
        community_level=community_level,
        response_type="Multiple Paragraphs",
        query=query,
    )
    return response, context


def local_search(
    store: GraphRAGStore,
    query: str,
    community_level: int = 2,
) -> Tuple[str, List[Document]]:
    """
    Run GraphRAG local search (entity-focused).

    Best for specific entity-oriented questions.

    Args:
        store:           Pre-loaded GraphRAGStore.
        query:           User query string.
        community_level: Hierarchy level for community context.

    Returns:
        (response_text, list_of_relevant_documents)
    """
    start = time.time()
    logger.info(f"GraphRAG local search: '{query[:80]}...'")

    try:
        from graphrag.config.load_config import load_config
        config = load_config(store.project_dir)

        response, context = asyncio.run(
            _async_local_search(config, store, query, community_level)
        )

        docs = _community_reports_to_documents(store.community_reports)

        logger.info(f"GraphRAG local search completed in {time.time() - start:.2f}s")
        return response, docs

    except Exception as e:
        logger.error(f"GraphRAG local search failed: {e}", exc_info=True)
        return "", []


# ---------------------------------------------------------------------------
# Lightweight fallback: community-summary similarity search
# ---------------------------------------------------------------------------

def community_summary_search(
    store: GraphRAGStore,
    query: str,
    embedder: Any,
    top_k: int = 5,
) -> List[Document]:
    """
    Fast fallback: embed the query and find the most relevant community
    summaries by cosine similarity.  No LLM call required at query time.

    This is used when GraphRAG's full global/local search is too slow or
    the graphrag package is unavailable.

    Args:
        store:    Pre-loaded GraphRAGStore.
        query:    User query string.
        embedder: BGEM3Embedder instance.
        top_k:    Number of community summaries to return.

    Returns:
        List of Documents with community summary content.
    """
    import numpy as np

    start = time.time()
    reports = store.community_reports

    if reports.empty:
        return []

    # Get the summary text column (may be 'full_content' or 'summary')
    text_col = "full_content" if "full_content" in reports.columns else "summary"
    summaries = reports[text_col].dropna().tolist()

    if not summaries:
        return []

    # Embed query
    query_dense, _ = embedder.embed_query(query)
    query_vec = np.array(query_dense, dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-9

    # Embed community summaries in batches
    all_vecs = []
    batch_size = 8
    for i in range(0, len(summaries), batch_size):
        batch = summaries[i:i + batch_size]
        dense_list, _ = embedder.embed_passages(batch)
        all_vecs.extend(dense_list)

    summary_matrix = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(summary_matrix, axis=1, keepdims=True) + 1e-9
    summary_matrix /= norms

    # Cosine similarity
    scores = summary_matrix @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    docs = []
    for idx in top_indices:
        if scores[idx] < 0.1:  # skip very low relevance
            continue
        doc = Document(
            page_content=summaries[idx][:4000],
            metadata={
                "source": "graphrag_community",
                "community_score": float(scores[idx]),
                "retrieval_type": "graphrag_community_summary",
            },
        )
        docs.append(doc)

    logger.info(
        f"Community summary search: {len(docs)} results in {time.time() - start:.2f}s"
    )
    return docs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _community_reports_to_documents(
    reports: pd.DataFrame,
    max_reports: int = 10,
) -> List[Document]:
    """Convert community report rows to LangChain Documents."""
    docs = []
    text_col = "full_content" if "full_content" in reports.columns else "summary"

    # Sort by rank if available
    if "rank" in reports.columns:
        reports = reports.sort_values("rank", ascending=False)

    for _, row in reports.head(max_reports).iterrows():
        content = row.get(text_col, "")
        if not content:
            continue
        doc = Document(
            page_content=str(content)[:4000],
            metadata={
                "source": "graphrag_community",
                "community_id": str(row.get("community_id", row.get("community", ""))),
                "retrieval_type": "graphrag_community_summary",
                "rank": float(row.get("rank", 0)) if "rank" in row.index else 0.0,
            },
        )
        docs.append(doc)

    return docs


def graphrag_retrieve(
    store: GraphRAGStore,
    query: str,
    query_type: str = "global",
    embedder: Any = None,
    top_k: int = 5,
    use_full_search: bool = True,
) -> Tuple[str, List[Document]]:
    """
    Unified GraphRAG retrieval entry point.

    Args:
        store:           Pre-loaded GraphRAGStore.
        query:           User query string.
        query_type:      "global" (thematic) or "local" (entity-focused).
        embedder:        BGEM3Embedder (required for fallback similarity search).
        top_k:           Number of results for fallback search.
        use_full_search: If True, use full GraphRAG search (LLM-based).
                         If False, use fast community-summary similarity.

    Returns:
        (graphrag_context_text, list_of_documents)
    """
    if not store or not store.is_valid:
        logger.warning("No valid GraphRAG store available.")
        return "", []

    if use_full_search:
        try:
            if query_type == "global":
                return global_search(store, query)
            else:
                return local_search(store, query)
        except Exception as e:
            logger.warning(f"Full GraphRAG search failed, falling back to similarity: {e}")
            # Fall through to similarity search

    # Fallback: fast community-summary similarity search
    if embedder is not None:
        docs = community_summary_search(store, query, embedder, top_k=top_k)
        context = "\n\n".join(d.page_content for d in docs)
        return context, docs

    return "", []
