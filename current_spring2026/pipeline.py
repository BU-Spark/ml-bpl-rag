"""
pipeline.py

Top-level entry point wiring all components together.
Graph retrieval is now fully integrated into retriever.py as a third RRF path.
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
from config import TOP_K_FINAL


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    intent:     QueryIntent
    documents:  List[RetrievedDocument]
    generation: GenerationResult
    latency_ms: int


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_query(raw_query: str, top_k: int = TOP_K_FINAL) -> PipelineResult:
    """
    End-to-end query pipeline:
      1. Classify and rewrite query (GPT-4o)
      2. Retrieve via dense + sparse + graph + metadata (all paths, RRF fused)
      3. Generate cited response (GPT-4o)
      4. Log event
    """
    start = time.monotonic()

    # ── Step 1: Classify ───────────────────────────────────────────────────
    intent = classify_query(raw_query)
    # print(f"[pipeline] Query type   : {intent.query_type}")
    print(f"[pipeline] Rewritten    : {intent.rewritten_query}")
    print(f"[pipeline] Date filter  : {intent.date_filter}")

    # ── Step 2: Retrieve (all paths fused via RRF) ─────────────────────────
    documents, _ = retrieve(intent, top_k=top_k)
    print(f"[pipeline] Retrieved    : {len(documents)} documents")

    # ── Step 3: Generate ───────────────────────────────────────────────────
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

    # ── Step 4: Log ────────────────────────────────────────────────────────
    log_query(
        intent            = intent,
        retrieved_docs    = documents,
        generation_result = generation,
        latency_ms        = latency_ms,
    )

    return PipelineResult(
        intent     = intent,
        documents  = documents,
        generation = generation,
        latency_ms = latency_ms,
    )


def print_result(result: PipelineResult):
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result.generation.response)

    print("\n" + "-" * 60)
    print(f"RESULTS  ({len(result.documents)} docs)")
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