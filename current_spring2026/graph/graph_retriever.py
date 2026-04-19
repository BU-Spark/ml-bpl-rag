"""
graph/graph_retriever.py

GraphRAG retrieval using semantic entity matching + two-hop traversal.

Flow:
  1. Embed the raw query text with BGE-M3
  2. Find semantically similar Entity nodes via vector index
  3. Two-hop traversal: direct entities → CO_OCCURS_WITH → related entities
  4. Find documents mentioning any of those entities
  5. Return documents not already in dense results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from graph.neo4j_client import get_session
from embedding.embedder import embedder


@dataclass
class GraphResult:
    ark_id:           str
    title:            str
    source_url:       str
    institution:      str
    issue_date:       str
    year:             int
    matched_entities: List[str]
    graph_score:      float
    hop:              int   # 1 = direct entity match, 2 = co-occurrence hop


def retrieve_by_query(
    query_text: str,
    exclude_ark_ids: Set[str] = None,
    top_k: int = 5,
    entity_top_k: int = 10,      # how many similar entities to find via vector search
    co_occur_threshold: int = 10,  # min co-occurrence weight for second hop
) -> List[GraphResult]:
    """
    Semantic GraphRAG retrieval:

    1. Embed raw query text
    2. Find top-K semantically similar Entity nodes (vector index)
    3. Two-hop: also find entities that CO_OCCUR with those entities
    4. Find documents mentioning any entity from either hop
    5. Return ranked by graph_score, excluding already-retrieved docs
    """
    exclude_ark_ids = exclude_ark_ids or set()

    # ── Step 1: Embed query ────────────────────────────────────────────────
    query_emb = embedder.embed_one(query_text, is_query=True)
    query_vec = query_emb.tolist()

    print(f"[graph] Searching for entities similar to: '{query_text[:60]}'")

    # ── Step 2 + 3: Vector search on entities + two-hop traversal ─────────
    cypher = """
        // Step 1: Find semantically similar entities via vector index
        CALL db.index.vector.queryNodes(
            'entity_embeddings',
            $entity_top_k,
            $query_embedding
        ) YIELD node AS e1, score AS similarity
        WHERE similarity >= 0.5

        // Step 2: Two-hop — find co-occurring entities
        OPTIONAL MATCH (e1)-[co:CO_OCCURS_WITH]-(e2:Entity)
        WHERE co.weight >= $co_occur_threshold

        // Collect all entities from both hops
        WITH
            COLLECT(DISTINCT {entity: e1, hop: 1, sim: similarity}) +
            COLLECT(DISTINCT {entity: e2, hop: 2, sim: similarity * 0.7}) AS all_entity_data

        UNWIND all_entity_data AS ed
        WITH ed.entity AS e, ed.hop AS hop, ed.sim AS sim
        WHERE e IS NOT NULL

        // Find documents mentioning these entities
        MATCH (d:Document)-[r:MENTIONS]->(e)
        WHERE NOT d.ark_id IN $exclude_ark_ids

        WITH
            d,
            COUNT(DISTINCT e)       AS matched_count,
            SUM(r.count)            AS total_mentions,
            COLLECT(DISTINCT e.name) AS matched_entities,
            MIN(hop)                AS min_hop,
            MAX(sim)                AS max_similarity

        RETURN
            d.ark_id          AS ark_id,
            d.title           AS title,
            d.source_url      AS source_url,
            d.institution     AS institution,
            d.issue_date      AS issue_date,
            d.year            AS year,
            matched_entities,
            min_hop           AS hop,
            (matched_count * 2.0 + total_mentions * 0.1 + max_similarity) AS graph_score
        ORDER BY graph_score DESC
        LIMIT $top_k
    """

    with get_session() as session:
        result = session.run(
            cypher,
            query_embedding      = query_vec,
            entity_top_k         = entity_top_k,
            co_occur_threshold   = co_occur_threshold,
            exclude_ark_ids      = list(exclude_ark_ids),
            top_k                = top_k,
        )
        rows = result.data()

    if not rows:
        print("[graph] No additional documents found via graph traversal.")
        return []

    hop1 = sum(1 for r in rows if r["hop"] == 1)
    hop2 = sum(1 for r in rows if r["hop"] == 2)
    print(f"[graph] Found {len(rows)} docs — {hop1} direct, {hop2} via co-occurrence")

    return [
        GraphResult(
            ark_id           = row["ark_id"],
            title            = row["title"],
            source_url       = row["source_url"] or "",
            institution      = row["institution"] or "",
            issue_date       = row["issue_date"] or "",
            year             = row["year"],
            matched_entities = row["matched_entities"],
            graph_score      = float(row["graph_score"]),
            hop              = row["hop"],
        )
        for row in rows
    ]