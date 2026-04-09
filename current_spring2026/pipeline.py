"""
pipeline.py

Top-level entry point that wires all components together.

Usage (programmatic):
    from pipeline import run_query
    result = run_query("What happened in Boston in 1919?")

Usage (CLI):
    python pipeline.py "What happened in Boston in 1919?"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List

from retrieval.query_understanding import classify_query, QueryIntent
from retrieval.retriever import retrieve, RetrievedDocument
from generation.generator import generate, GenerationResult
from evaluation.logger import log_query
from config import TOP_K_FINAL


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    intent:      QueryIntent
    documents:   List[RetrievedDocument]
    generation:  GenerationResult
    latency_ms:  int


# ── Main function ─────────────────────────────────────────────────────────────

def run_query(raw_query: str, top_k: int = TOP_K_FINAL) -> PipelineResult:
    """
    End-to-end query pipeline:
      1. Classify and rewrite the query (GPT-4o)
      2. Retrieve relevant documents (hybrid search)
      3. Generate a grounded response (GPT-4o)
      4. Log the full event

    Returns a PipelineResult with all intermediate outputs.
    """
    start = time.monotonic()

    # ── Step 1: Understand the query ───────────────────────────────────────
    intent = classify_query(raw_query)
    print(f"[pipeline] Query type   : {intent.query_type}")
    print(f"[pipeline] Rewritten    : {intent.rewritten_query}")
    print(f"[pipeline] Date filter  : {intent.date_filter}")
    print(f"[pipeline] Weights      : content={intent.content_weight:.2f} metadata={intent.metadata_weight:.2f}")

    # ── Step 2: Retrieve ───────────────────────────────────────────────────
    documents = retrieve(intent, top_k=top_k)
    print(f"[pipeline] Retrieved    : {len(documents)} documents")

    if not documents:
        from generation.generator import GenerationResult
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
    """Pretty-print a PipelineResult to stdout."""
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result.generation.response)

    print("\n" + "-" * 60)
    print(f"RESULTS  ({len(result.documents)} documents)")
    print("-" * 60)
    for i, doc in enumerate(result.documents, 1):
        date_str = doc.issue_date or (str(doc.year[0]) if doc.year else "unknown date")
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
