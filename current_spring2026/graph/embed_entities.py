"""
graph/embed_entities.py

Phase 2: Embed all extracted entities using BGE-M3 on GPU.
Reads entities from data/graph/entities_<year>.jsonl
Saves embeddings to data/graph/embeddings_<year>.npy
and a matching index file data/graph/embedding_index_<year>.jsonl

Run:
    python -m graph.embed_entities --year 1900
    python -m graph.embed_entities --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from embedding.embedder import embedder

OUTPUT_DIR = Path("data/graph")


def build_entity_text(entity: dict) -> str:
    return f"{entity['type']}: {entity['text']}"


def embed_all(year: int = None, suffix: str = None):
    # Use explicit suffix if provided, otherwise derive from year
    if suffix:
        file_suffix = suffix
    else:
        file_suffix = str(year) if year else "all"
    
    input_file  = OUTPUT_DIR / f"entities_{file_suffix}.jsonl"
    emb_file    = OUTPUT_DIR / f"embeddings_{file_suffix}.npy"
    index_file  = OUTPUT_DIR / f"embedding_index_{file_suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"BPL Graph — Phase 2: Entity Embedding")
    print(f"  Input  : {input_file}")
    print(f"  Output : {emb_file}")
    print(f"{'='*60}\n")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Entity file not found: {input_file}\n"
            f"Run Phase 1 first: python -m graph.extract_entities"
        )

    # Read all entity records
    print("Reading entity records...")
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Found {len(records)} documents with entities")

    # Flatten all unique entities across all documents
    # We only embed each unique (name, type) pair once
    seen_entities:  dict[tuple, int] = {}   # (name, type) -> index in unique list
    unique_entities: List[dict]      = []

    # Build index: for each doc, map entity -> embedding index
    doc_entity_indices = []   # list of lists of indices

    for record in records:
        doc_indices = []
        for ent in record["entities"]:
            key = (ent["text"], ent["type"])
            if key not in seen_entities:
                seen_entities[key] = len(unique_entities)
                unique_entities.append(ent)
            doc_indices.append(seen_entities[key])
        doc_entity_indices.append(doc_indices)

    print(f"  Unique entities to embed: {len(unique_entities)}")

    # Embed all unique entities in batches
    print(f"\nEmbedding on GPU...")
    start_time = time.monotonic()

    texts      = [build_entity_text(e) for e in unique_entities]
    embeddings = embedder.embed(texts)   # shape (N, 1024)

    elapsed = time.monotonic() - start_time
    print(f"  Embedded {len(unique_entities)} entities in {elapsed:.1f}s")

    # Save embeddings
    np.save(emb_file, embeddings)
    print(f"  Saved embeddings to {emb_file}")

    # Save index file mapping each doc to its entity embedding indices
    with open(index_file, "w", encoding="utf-8") as f:
        for record, indices in zip(records, doc_entity_indices):
            index_record = {
                "ark_id":      record["ark_id"],
                "title":       record["title"],
                "year":        record["year"],
                "institution": record["institution"],
                "source_url":  record["source_url"],
                "issue_date":  record["issue_date"],
                "entities":    record["entities"],
                "emb_indices": indices,   # index into embeddings array
            }
            f.write(json.dumps(index_record) + "\n")

    print(f"  Saved index to {index_file}")
    print(f"\n✓ Embedding complete.")
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Total time       : {elapsed:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Embed entities")
    parser.add_argument("--year",   type=int, default=None)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--suffix", type=str, default=None, help="Explicit file suffix e.g. 'all_gpt' or 'metadata'")
    args = parser.parse_args()

    embed_all(
        year   = None if (args.all or args.suffix) else (args.year or 1900),
        suffix = args.suffix,
    )
