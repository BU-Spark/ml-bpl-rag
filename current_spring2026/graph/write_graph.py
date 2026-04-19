"""
graph/write_graph.py

Phase 3: Write entities and relationships to Neo4j.
Reads from data/graph/embeddings_<year>.npy
       and data/graph/embedding_index_<year>.jsonl
No GPU needed — pure network writes to Neo4j.

Run:
    python -m graph.write_graph --year 1900
    python -m graph.write_graph --all
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

from graph.neo4j_client import get_session, create_schema

OUTPUT_DIR  = Path("data/graph")
BATCH_SIZE  = 100   # documents per Neo4j transaction


def write_all(year: int = None):
    # suffix     = str(year) if year else "all"
    # emb_file   = OUTPUT_DIR / f"embeddings_{suffix}.npy"
    # index_file = OUTPUT_DIR / f"embedding_index_{suffix}.jsonl"
        # Use explicit suffix if provided, otherwise derive from year
    if suffix:
        file_suffix = suffix
    else:
        file_suffix = str(year) if year else "all"
    
    input_file  = OUTPUT_DIR / f"entities_{file_suffix}.jsonl"
    emb_file    = OUTPUT_DIR / f"embeddings_{file_suffix}.npy"
    index_file  = OUTPUT_DIR / f"embedding_index_{file_suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Phase 3: Write to Neo4j")
    print(f"  Embeddings : {emb_file}")
    print(f"  Index      : {index_file}")
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

    # Process in batches for efficient Neo4j writes
    for batch_start in range(0, len(records), BATCH_SIZE):
        batch     = records[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(records))

        with get_session() as session:
            for record in batch:
                entities    = record["entities"]
                emb_indices = record["emb_indices"]

                if not entities:
                    continue

                # Get embeddings for this document's entities
                doc_embs = embeddings[emb_indices]

                # ── Upsert document node ───────────────────────────────────
                session.run(
                    """
                    MERGE (d:Document {ark_id: $ark_id})
                    SET d.title       = $title,
                        d.year        = $year,
                        d.institution = $institution,
                        d.source_url  = $source_url,
                        d.issue_date  = $issue_date
                    """,
                    ark_id      = record["ark_id"],
                    title       = record["title"],
                    year        = record["year"][0] if record["year"] else None,
                    institution = record["institution"],
                    source_url  = record["source_url"],
                    issue_date  = record["issue_date"],
                )

                # ── Batch upsert entities + MENTIONS ──────────────────────
                entity_data = [
                    {
                        "name":      ent["text"],
                        "type":      ent["type"],
                        "count":     ent["count"],
                        "embedding": doc_embs[i].tolist(),
                    }
                    for i, ent in enumerate(entities)
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
                    ark_id   = record["ark_id"],
                    entities = entity_data,
                )

                # ── CO_OCCURS_WITH relationships ───────────────────────────
                top_entities   = entities[:10]
                co_occur_pairs = list(combinations(top_entities, 2))

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
                        ark_id = record["ark_id"],
                        pairs  = [
                            {
                                "name1": e1["text"], "type1": e1["type"],
                                "name2": e2["text"], "type2": e2["type"],
                            }
                            for e1, e2 in co_occur_pairs
                        ],
                    )

                total_written += 1

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
    # parser = argparse.ArgumentParser(description="Phase 3: Write graph to Neo4j")
    # parser.add_argument("--year", type=int, default=None)
    # parser.add_argument("--all",  action="store_true")
    # args = parser.parse_args()

    # write_all(year=None if args.all else (args.year or 1900))
    parser = argparse.ArgumentParser(description="Phase 3: Write graph to Neo4j")
    parser.add_argument("--year",   type=int, default=None)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--suffix", type=str, default=None, help="Explicit file suffix e.g. 'all_gpt' or 'metadata'")
    args = parser.parse_args()

    write_all(
        year   = None if (args.all or args.suffix) else (args.year or 1900),
        suffix = args.suffix,
    )
