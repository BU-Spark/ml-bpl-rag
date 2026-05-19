"""
graph/write_graph.py

Phase 3: Write entities and relationships to Neo4j.
Reads from data/graph/embeddings_<year>.npy
       and data/graph/embedding_index_<year>.jsonl
No GPU needed — pure network writes to Neo4j.

IDEMPOTENT: Safe to re-run — relationships only update when ark_id not already present.

Run:
    python -m graph.write_graph --year 1900
    python -m graph.write_graph --all
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# Flush all print output immediately so logs update in real time
print = functools.partial(print, flush=True)

# Prevent numpy from spawning extra threads (avoids memory bloat on cluster)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from graph.neo4j_client import get_session, create_schema

OUTPUT_DIR = Path("data/graph")
BATCH_SIZE = 100  # documents per Neo4j transaction


def write_all(year: int = None, suffix: str = None):
    print("write_all() started")

    if suffix:
        file_suffix = suffix
    else:
        file_suffix = str(year) if year else "all"

    emb_file   = OUTPUT_DIR / f"embeddings_{file_suffix}.npy"
    index_file = OUTPUT_DIR / f"embedding_index_{file_suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Phase 3: Write to Neo4j")
    print(f"  Embeddings : {emb_file}")
    print(f"  Index      : {index_file}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"{'='*60}\n")

    if not emb_file.exists() or not index_file.exists():
        raise FileNotFoundError(
            f"Missing files. Run Phase 1 and Phase 2 first."
        )

    create_schema()

    # Load embeddings
    print("Loading embeddings...")
    embeddings = np.load(emb_file)
    print(f"  Shape: {embeddings.shape}")

    # Load index
    print("Loading index...")
    records = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Found {len(records)} documents\n")

    start_time    = time.monotonic()
    total_written = 0

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch     = records[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(records))

        # Build all data for the batch upfront
        docs_data = []
        for record in batch:
            entities    = record["entities"]
            emb_indices = record["emb_indices"]

            if not entities:
                continue

            doc_embs     = embeddings[emb_indices]
            top_entities = entities[:10]

            docs_data.append({
                "ark_id":      record["ark_id"],
                "title":       record["title"],
                "year":        record["year"][0] if record["year"] else None,
                "institution": record["institution"],
                "source_url":  record["source_url"],
                "issue_date":  record["issue_date"],
                "entities": [
                    {
                        "name":      ent["text"],
                        "type":      ent["type"],
                        "count":     ent["count"],
                        "embedding": doc_embs[i].tolist(),
                    }
                    for i, ent in enumerate(entities)
                ],
                "pairs": [
                    {
                        "name1": e1["text"], "type1": e1["type"],
                        "name2": e2["text"], "type2": e2["type"],
                    }
                    for e1, e2 in combinations(top_entities, 2)
                ],
            })

        with get_session() as session:
            # Single round trip for all document + entity upserts in the batch
            session.run(
                """
                UNWIND $docs AS doc
                MERGE (d:Document {ark_id: doc.ark_id})
                SET d.title       = doc.title,
                    d.year        = doc.year,
                    d.institution = doc.institution,
                    d.source_url  = doc.source_url,
                    d.issue_date  = doc.issue_date
                WITH d, doc
                UNWIND doc.entities AS ent
                MERGE (e:Entity {name: ent.name, type: ent.type})
                ON CREATE SET e.embedding = ent.embedding
                MERGE (d)-[r:MENTIONS]->(e)
                ON CREATE SET r.count = ent.count,
                              r.documents = [doc.ark_id]
                ON MATCH  SET r.count = CASE 
                                          WHEN NOT doc.ark_id IN coalesce(r.documents, [])
                                          THEN coalesce(r.count, 0) + ent.count
                                          ELSE r.count
                                        END,
                              r.documents = CASE
                                              WHEN NOT doc.ark_id IN coalesce(r.documents, [])
                                              THEN coalesce(r.documents, []) + [doc.ark_id]
                                              ELSE r.documents
                                            END
                """,
                docs=docs_data,
            )

            # Co-occurrence — canonicalize pair ordering and make idempotent
            all_pairs_with_ark = []
            for doc in docs_data:
                for p in doc["pairs"]:
                    all_pairs_with_ark.append({
                        "ark_id": doc["ark_id"],
                        "name1": p["name1"],
                        "type1": p["type1"],
                        "name2": p["name2"],
                        "type2": p["type2"],
                    })
            
            if all_pairs_with_ark:
                session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (e1:Entity {name: pair.name1, type: pair.type1})
                    MATCH (e2:Entity {name: pair.name2, type: pair.type2})
                    
                    // Canonicalize: ensure consistent ordering (smaller name/type first)
                    WITH e1, e2, pair,
                         CASE 
                           WHEN pair.name1 < pair.name2 THEN e1
                           WHEN pair.name1 > pair.name2 THEN e2
                           WHEN pair.type1 <= pair.type2 THEN e1
                           ELSE e2
                         END AS a,
                         CASE
                           WHEN pair.name1 < pair.name2 THEN e2
                           WHEN pair.name1 > pair.name2 THEN e1
                           WHEN pair.type1 <= pair.type2 THEN e2
                           ELSE e1
                         END AS b
                    
                    MERGE (a)-[r:CO_OCCURS_WITH]-(b)
                    ON CREATE SET r.weight = 1,
                                  r.documents = [pair.ark_id]
                    ON MATCH  SET r.weight = CASE
                                               WHEN NOT pair.ark_id IN coalesce(r.documents, [])
                                               THEN coalesce(r.weight, 0) + 1
                                               ELSE r.weight
                                             END,
                                  r.documents = CASE
                                                  WHEN NOT pair.ark_id IN coalesce(r.documents, [])
                                                  THEN coalesce(r.documents, []) + [pair.ark_id]
                                                  ELSE r.documents
                                                END
                    """,
                    pairs=all_pairs_with_ark,
                )

        total_written += len(docs_data)
        elapsed   = time.monotonic() - start_time
        remaining = (elapsed / total_written) * (len(records) - total_written) if total_written else 0
        print(
            f"  [{batch_end}/{len(records)}] "
            f"Written {total_written} docs | "
            f"ETA: {remaining/60:.1f}min"
        )

    print(f"\n✓ Graph write complete.")
    print(f"  Documents written : {total_written}")
    print(f"  Total time        : {(time.monotonic()-start_time)/60:.1f} min")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Write graph to Neo4j")
    parser.add_argument("--year",   type=int, default=None)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--suffix", type=str, default=None, help="Explicit file suffix e.g. 'all_gpt' or 'metadata'")
    args = parser.parse_args()

    write_all(
        year   = None if (args.all or args.suffix) else (args.year or 1900),
        suffix = args.suffix,
    )