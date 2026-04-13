"""
pipeline.py

Top-level entry point wiring all components together.
GraphRAG is triggered for content_driven queries with use_graph=True.
Uses semantic entity matching + two-hop co-occurrence traversal.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List

from retrieval.query_understanding import classify_query, QueryIntent
from retrieval.retriever import retrieve, RetrievedDocument
from generation.generator import generate, GenerationResult
from evaluation.logger import log_query
from config import (
    TOP_K_FINAL,
    GRAPH_RAG_ENABLED,
    GRAPH_TOP_K,
    GRAPH_MIN_ENTITY_MATCHES,
)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    intent:           QueryIntent
    documents:        List[RetrievedDocument]
    generation:       GenerationResult
    latency_ms:       int
    graph_used:       bool = False
    graph_docs_added: int  = 0


# ── GraphRAG expansion ────────────────────────────────────────────────────────

def _expand_with_graph(
    intent: QueryIntent,
    dense_docs: List[RetrievedDocument],
    top_k: int = GRAPH_TOP_K,
) -> List[RetrievedDocument]:
    """
    Semantic GraphRAG expansion:
      1. Embed raw query text
      2. Find similar Entity nodes via vector index
      3. Two-hop traversal via CO_OCCURS_WITH
      4. Fetch full document details from PostgreSQL
      5. Merge into result list
    """
    from graph.graph_retriever import retrieve_by_query
    from database.schema import get_conn, get_cursor

    existing_ark_ids = {doc.ark_id for doc in dense_docs}

    graph_results = retrieve_by_query(
        query_text      = intent.rewritten_query,
        exclude_ark_ids = existing_ark_ids,
        top_k           = top_k,
    )

    if not graph_results:
        return dense_docs

    # Fetch full document details from PostgreSQL
    graph_ark_ids = [r.ark_id for r in graph_results]

    sql = """
        SELECT
            d.id, d.ark_id, d.title, d.source_url, d.institution,
            d.issue_date, d.year, d.topics, d.geography,
            d.metadata_embedding,
            c.chunk_text, c.chunk_index
        FROM documents d
        LEFT JOIN LATERAL (
            SELECT chunk_text, chunk_index
            FROM chunks
            WHERE document_id = d.id
            ORDER BY chunk_index
            LIMIT 1
        ) c ON true
        WHERE d.ark_id = ANY(%s)
    """

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql, (graph_ark_ids,))
            rows = {row["ark_id"]: row for row in cur.fetchall()}

    graph_score_map = {r.ark_id: r.graph_score for r in graph_results}
    max_graph_score = max(graph_score_map.values()) if graph_score_map else 1.0

    new_docs = []
    for graph_result in graph_results:
        row = rows.get(graph_result.ark_id)
        if not row:
            continue

        # Normalise graph score to 0-0.3 so graph docs appear
        # after dense results but are still included
        normalised_score = (graph_result.graph_score / max_graph_score) * 0.3

        new_docs.append(RetrievedDocument(
            ark_id           = graph_result.ark_id,
            document_id      = row["id"],
            title            = row["title"] or "",
            source_url       = row["source_url"] or "",
            institution      = row["institution"] or "",
            issue_date       = row["issue_date"] or "",
            year             = row["year"] or [],
            topics           = row["topics"] or [],
            geography        = row["geography"] or [],
            best_chunk_text  = row["chunk_text"] or "",
            best_chunk_index = row["chunk_index"] or 0,
            rrf_score        = 0.0,
            metadata_sim     = 0.0,
            final_score      = normalised_score,
        ))

    if new_docs:
        print(f"[pipeline] Graph expanded results by {len(new_docs)} documents")

    return dense_docs + new_docs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_query(raw_query: str, top_k: int = TOP_K_FINAL) -> PipelineResult:
    """
    End-to-end query pipeline:
      1. Classify and rewrite query (GPT-4o)
      2. Retrieve via dense + sparse search (pgVector)
      3. If content_driven + use_graph: expand via GraphRAG
      4. Generate cited response (GPT-4o)
      5. Log event
    """
    start = time.monotonic()

    # ── Step 1: Classify ───────────────────────────────────────────────────
    intent = classify_query(raw_query)
    print(f"[pipeline] Query type   : {intent.query_type}")
    print(f"[pipeline] Rewritten    : {intent.rewritten_query}")
    print(f"[pipeline] Date filter  : {intent.date_filter}")
    print(f"[pipeline] Use graph    : {intent.use_graph}")
    print(f"[pipeline] Weights      : content={intent.content_weight:.2f} metadata={intent.metadata_weight:.2f}")

    # ── Step 2: Dense + sparse retrieval ───────────────────────────────────
    documents = retrieve(intent, top_k=top_k)
    print(f"[pipeline] Retrieved    : {len(documents)} documents (dense/sparse)")

    # ── Step 3: GraphRAG expansion ─────────────────────────────────────────
    graph_used       = False
    graph_docs_added = 0

    if (
        GRAPH_RAG_ENABLED
        and intent.query_type == "content_driven"
        and intent.use_graph
    ):
        try:
            expanded         = _expand_with_graph(intent, documents, top_k=GRAPH_TOP_K)
            graph_docs_added = len(expanded) - len(documents)
            documents        = expanded
            graph_used       = graph_docs_added > 0
        except Exception as e:
            print(f"[pipeline] GraphRAG failed (non-fatal): {e}")

    print(f"[pipeline] Final docs   : {len(documents)} (graph added: {graph_docs_added})")

    # ── Step 4: Generate ───────────────────────────────────────────────────
    if not documents:
        generation = GenerationResult(
            response      = "No relevant materials were found for your query in the Digital Commonwealth collection. Try rephrasing or using a more specific historical topic.",
            source_titles = [],
            source_urls   = [],
        )
    else:
        generation = generate(raw_query, documents)

    latency_ms = int((time.monotonic() - start) * 1000)
    print(f"[pipeline] Latency      : {latency_ms}ms")

    # ── Step 5: Log ────────────────────────────────────────────────────────
    log_query(
        intent            = intent,
        retrieved_docs    = documents,
        generation_result = generation,
        latency_ms        = latency_ms,
    )

    return PipelineResult(
        intent           = intent,
        documents        = documents,
        generation       = generation,
        latency_ms       = latency_ms,
        graph_used       = graph_used,
        graph_docs_added = graph_docs_added,
    )


def print_result(result: PipelineResult):
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result.generation.response)

    print("\n" + "-" * 60)
    print(f"RESULTS  ({len(result.documents)} docs | graph_used={result.graph_used})")
    print("-" * 60)
    for i, doc in enumerate(result.documents, 1):
        date_str = doc.issue_date or (str(doc.year[0]) if doc.year else "unknown")
        print(f"  {i}. {doc.title} ({date_str})")
        print(f"     {doc.source_url}")
        print(f"     score={doc.final_score:.4f}")

    print(f"\n[latency: {result.latency_ms}ms]")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"<query>\"")
        sys.exit(1)

    query  = " ".join(sys.argv[1:])
    result = run_query(query)
    print_result(result)