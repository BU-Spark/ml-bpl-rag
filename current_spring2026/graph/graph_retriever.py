"""
graph/graph_retriever.py

GraphRAG retrieval using semantic entity matching + two-hop traversal.

Flow:
  1. Accept pre-computed query embedding (no re-embedding)
  2. Find semantically similar Entity nodes via vector index
  3. Two-hop traversal: direct entities → CO_OCCURS_WITH → related entities
  4. Find documents mentioning any of those entities
  5. Return ranked graph results (ark_id + score)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from graph.neo4j_client import get_session


@dataclass
class GraphResult:
    ark_id:           str
    matched_entities: List[str]
    graph_score:      float
    hop:              int   # 1 = direct entity match, 2 = co-occurrence hop


def retrieve_by_query(
    query_embedding: np.ndarray,
    exclude_ark_ids: Set[str]   = None,
    top_k: int                  = 5,
    entity_top_k: int           = 10,
    co_occur_threshold: int     = 2,
) -> List[GraphResult]:
    """
    Semantic GraphRAG retrieval using a pre-computed query embedding.

    1. Find top-K semantically similar Entity nodes (vector index)
    2. Two-hop: also find entities that CO_OCCUR with those entities
    3. Find documents mentioning any entity from either hop
    4. Return ranked by graph_score
    """
    exclude_ark_ids = exclude_ark_ids or set()
    query_vec       = query_embedding.tolist()

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
            COUNT(DISTINCT e)        AS matched_count,
            SUM(r.count)             AS total_mentions,
            COLLECT(DISTINCT e.name) AS matched_entities,
            MIN(hop)                 AS min_hop,
            MAX(sim)                 AS max_similarity

        RETURN
            d.ark_id         AS ark_id,
            matched_entities,
            min_hop          AS hop,
            (matched_count * 2.0 + total_mentions * 0.1 + max_similarity) AS graph_score
        ORDER BY graph_score DESC
        LIMIT $top_k
    """

    with get_session() as session:
        result = session.run(
            cypher,
            query_embedding    = query_vec,
            entity_top_k       = entity_top_k,
            co_occur_threshold = co_occur_threshold,
            exclude_ark_ids    = list(exclude_ark_ids),
            top_k              = top_k,
        )
        rows = result.data()

    if not rows:
        return []

    hop1 = sum(1 for r in rows if r["hop"] == 1)
    hop2 = sum(1 for r in rows if r["hop"] == 2)
    print(f"[graph] Found {len(rows)} docs — {hop1} direct, {hop2} via co-occurrence")

    return [
        GraphResult(
            ark_id           = row["ark_id"],
            matched_entities = row["matched_entities"],
            graph_score      = float(row["graph_score"]),
            hop              = row["hop"],
        )
        for row in rows
    ]