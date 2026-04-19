"""
graph/graph_builder.py

Builds the knowledge graph in Neo4j from ingested documents.
Embeds Entity nodes using BGE-M3 with "TYPE: name" format
for semantic entity matching at query time.

Run:
    python -m graph.graph_builder --year 1900
    python -m graph.graph_builder --all
"""

from __future__ import annotations

import argparse
import time
from typing import List
from itertools import combinations

import numpy as np

from database.schema import get_conn, get_cursor
from graph.neo4j_client import get_session, create_schema
from graph.entity_extractor import extractor, Entity
from embedding.embedder import embedder


# ── Entity text builder ───────────────────────────────────────────────────────

def build_entity_text(entity: Entity) -> str:
    """
    Build the text string to embed for an entity.
    Format: "TYPE: name" e.g. "PERSON: mayor fitzgerald"
    This gives BGE-M3 enough context to distinguish entity types.
    """
    return f"{entity.type}: {entity.text}"


# ── Fetch documents ───────────────────────────────────────────────────────────

def fetch_documents(year: int = None) -> List[dict]:
    sql = """
        SELECT
            d.id,
            d.ark_id,
            d.title,
            d.year,
            d.institution,
            d.source_url,
            d.issue_date,
            ARRAY_AGG(c.chunk_text ORDER BY c.chunk_index) AS chunks
        FROM documents d
        JOIN chunks c ON c.document_id = d.id
    """
    params = []
    if year:
        sql += " WHERE EXTRACT(YEAR FROM d.date_start) = %s"
        params.append(year)

    sql += " GROUP BY d.id, d.ark_id, d.title, d.year, d.institution, d.source_url, d.issue_date"

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# ── Batch embed entities ──────────────────────────────────────────────────────

def embed_entities(entities: List[Entity]) -> np.ndarray:
    """
    Embed a list of entities using BGE-M3.
    Uses "TYPE: name" format for each entity.
    Returns array of shape (N, 1024).
    """
    texts = [build_entity_text(e) for e in entities]
    return embedder.embed(texts)


# ── Batch Neo4j write ─────────────────────────────────────────────────────────

def write_document_batch(
    session,
    doc: dict,
    entities: List[Entity],
    entity_embeddings: np.ndarray,
    co_occur_pairs: list,
):
    """
    Write document + all entities + relationships in minimal round trips.
    Uses UNWIND for batch efficiency.
    """

    # 1. Upsert document node
    session.run(
        """
        MERGE (d:Document {ark_id: $ark_id})
        SET d.title       = $title,
            d.year        = $year,
            d.institution = $institution,
            d.source_url  = $source_url,
            d.issue_date  = $issue_date
        """,
        ark_id      = doc["ark_id"],
        title       = doc["title"] or "",
        year        = doc["year"][0] if doc["year"] else None,
        institution = doc["institution"] or "",
        source_url  = doc["source_url"] or "",
        issue_date  = doc["issue_date"] or "",
    )

    # 2. Batch upsert entity nodes with embeddings + MENTIONS relationships
    if entities:
        entity_data = [
            {
                "name":      e.text,
                "type":      e.type,
                "count":     e.count,
                "embedding": entity_embeddings[i].tolist(),
            }
            for i, e in enumerate(entities)
        ]

        session.run(
            """
            UNWIND $entities AS ent
            MERGE (e:Entity {name: ent.name, type: ent.type})
            ON CREATE SET e.embedding = ent.embedding
            WITH e, ent
            MATCH (d:Document {ark_id: $ark_id})
            MERGE (d)-[r:MENTIONS]->(e)
            ON CREATE SET r.count = ent.count
            ON MATCH  SET r.count = r.count + ent.count
            """,
            ark_id   = doc["ark_id"],
            entities = entity_data,
        )

    # 3. Batch upsert CO_OCCURS_WITH relationships
    if co_occur_pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (e1:Entity {name: pair.name1, type: pair.type1})
            MATCH (e2:Entity {name: pair.name2, type: pair.type2})
            MERGE (e1)-[r:CO_OCCURS_WITH]->(e2)
            ON CREATE SET r.weight = 1, r.documents = [$ark_id]
            ON MATCH  SET r.weight = r.weight + 1,
                          r.documents = r.documents + [$ark_id]
            """,
            ark_id = doc["ark_id"],
            pairs  = [
                {
                    "name1": e1.text, "type1": e1.type,
                    "name2": e2.text, "type2": e2.type,
                }
                for e1, e2 in co_occur_pairs
            ],
        )


# ── Main build ────────────────────────────────────────────────────────────────

def build_graph(year: int = None):
    print(f"\n{'='*60}")
    print("BPL RAG Graph Builder")
    print(f"  Year filter : {year or 'all'}")
    print(f"{'='*60}\n")

    create_schema()

    print("Fetching documents from PostgreSQL...")
    docs = fetch_documents(year=year)
    print(f"  Found {len(docs)} documents\n")

    total_docs     = 0
    total_entities = 0
    start_time     = time.monotonic()
    CHUNK_SIZE     = 200

    for chunk_start in range(0, len(docs), CHUNK_SIZE):
        chunk     = docs[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_end = min(chunk_start + CHUNK_SIZE, len(docs))

        print(f"\n── Chunk [{chunk_start+1}-{chunk_end}/{len(docs)}] ──")

        # ── Phase 1: Extract entities (CPU/spaCy) ──────────────────────────
        chunk_data = []
        for doc in chunk:
            full_text = " ".join(doc["chunks"] or [])
            entities  = extractor.extract_top(full_text, n=40)
            if entities:
                chunk_data.append((doc, entities))

        print(f"  Extracted entities from {len(chunk_data)} docs")

        if not chunk_data:
            continue

        # ── Phase 2: Embed all entities in one GPU batch ───────────────────
        all_texts = [build_entity_text(e) for _, ents in chunk_data for e in ents]
        print(f"  Embedding {len(all_texts)} entities on GPU...")
        all_embs  = embedder.embed(all_texts)
        print(f"  Embedding complete")

        # Split embeddings back per document
        idx            = 0
        doc_embeddings = []
        for _, entities in chunk_data:
            n = len(entities)
            doc_embeddings.append(all_embs[idx:idx+n])
            idx += n

        # ── Phase 3: Write to Neo4j ────────────────────────────────────────
        for (doc, entities), embs in zip(chunk_data, doc_embeddings):
            top_entities   = entities[:10]
            co_occur_pairs = list(combinations(top_entities, 2))
            with get_session() as session:
                write_document_batch(session, doc, entities, embs, co_occur_pairs)

            total_entities += len(entities)
            total_docs     += 1

        elapsed   = time.monotonic() - start_time
        remaining = (elapsed / total_docs) * (len(docs) - total_docs) if total_docs else 0
        print(
            f"  Written {total_docs}/{len(docs)} docs | "
            f"ETA: {remaining/60:.1f}min"
        )

    print(f"\n✓ Graph build complete.")
    print(f"  Documents processed : {total_docs}")
    print(f"  Total entities      : {total_entities}")
    print(f"  Total time          : {(time.monotonic()-start_time)/60:.1f} min")
    
# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPL RAG Graph Builder")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    build_graph(year=None if args.all else (args.year or 1900))