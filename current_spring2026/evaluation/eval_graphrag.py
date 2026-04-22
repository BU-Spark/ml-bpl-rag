"""
evaluation/eval_graphrag.py

Evaluates two isolated retrieval systems on ALL queries:
  System 1 — Dense + Sparse only
  System 2 — GraphRAG only

No filtering — every query is evaluated against both systems.
Summary broken down by use_graph=True vs use_graph=False.

Metrics: Hit@K, Recall@K, Precision@K for K = 5, 10, 20, 30, 50

Run:
    python -m evaluation.eval_graphrag
    python -m evaluation.eval_graphrag --queries test_queries.jsonl --out evaluation/graphrag_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.query_understanding import classify_query
from retrieval.retriever import retrieve
from graph.graph_retriever import retrieve_by_query
from embedding.embedder import embedder
from config import GRAPH_TOP_K



K_VALUES = [5, 10, 20, 30, 50]
MAX_K    = max(K_VALUES)


# ── Metric helpers ────────────────────────────────────────────────────────────

def hit_at_k(retrieved: List[str], ground_truths: List[str], k: int) -> int:
    return int(any(gt in retrieved[:k] for gt in ground_truths))


def recall_at_k(retrieved: List[str], ground_truths: List[str], k: int) -> float:
    if not ground_truths:
        return 0.0
    hits = sum(1 for gt in ground_truths if gt in retrieved[:k])
    return hits / len(ground_truths)


def precision_at_k(retrieved: List[str], ground_truths: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for ark in retrieved[:k] if ark in ground_truths)
    return hits / k


def compute_metrics(retrieved: List[str], ground_truths: List[str]) -> dict:
    metrics = {}
    for k in K_VALUES:
        metrics[f"hit_{k}"]       = hit_at_k(retrieved, ground_truths, k)
        metrics[f"recall_{k}"]    = round(recall_at_k(retrieved, ground_truths, k), 4)
        metrics[f"precision_{k}"] = round(precision_at_k(retrieved, ground_truths, k), 4)
    return metrics


# ── GraphRAG-only retrieval ───────────────────────────────────────────────────

def retrieve_graph_only(query_emb: np.ndarray, top_k: int = MAX_K) -> List[str]:
    """Run GraphRAG retrieval only — no dense or sparse search."""
    graph_results = retrieve_by_query(
        query_embedding = query_emb,
        exclude_ark_ids = set(),
        top_k           = top_k,
    )
    return [r.ark_id for r in graph_results] if graph_results else []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GraphRAG vs Dense+Sparse evaluation")
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).parent.parent / "test_queries.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "graphrag_results.json"),
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: test queries file not found at {queries_path}")
        sys.exit(1)

    with open(queries_path) as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_entries)} queries\n")

    # ── Classify all queries ──────────────────────────────────────────────────
    print("Classifying all queries...")
    classified = []
    for entry in all_entries:
        intent = classify_query(entry["question"])
        classified.append((entry, intent))
    print(f"  Classified {len(classified)} queries")
    print(f"  use_graph=True  : {sum(1 for _, i in classified if i.use_graph)}")
    print(f"  use_graph=False : {sum(1 for _, i in classified if not i.use_graph)}")
    print(f"\nEvaluating at K = {K_VALUES}\n")

    rows = []

    for i, (entry, intent) in enumerate(classified):
        question      = entry["question"]
        qtype         = entry["question_type"]
        ground_truths = [
            g["ark_id"].removeprefix("commonwealth:")
            for g in entry.get("ground_truths", [])
        ]

        print(f"[{i+1:02d}/{len(classified)}] {question[:70]}...")
        print(f"  use_graph={intent.use_graph} | rewritten='{intent.rewritten_query[:60]}'")

        try:
            # ── System 1: Dense + Sparse ───────────────────────────────────
            # retrieve() returns (documents, query_emb) — unpack both
            dense_docs, query_emb = retrieve(intent, top_k=MAX_K)
            dense_ids = [d.ark_id for d in dense_docs]

            # ── System 2: GraphRAG only ────────────────────────────────────
            # Reuse query_emb from dense retrieval — no re-embedding
            graph_ids = retrieve_graph_only(query_emb, top_k=MAX_K)

            # ── Compute metrics ────────────────────────────────────────────
            dense_metrics = compute_metrics(dense_ids, ground_truths)
            graph_metrics = compute_metrics(graph_ids, ground_truths)

            # ── Build row ──────────────────────────────────────────────────
            row = {
                "question":          question,
                "question_type":     qtype,
                "rewritten_query":   intent.rewritten_query,
                "use_graph":         intent.use_graph,
                "num_ground_truths": len(ground_truths),
                "dense_retrieved":   len(dense_ids),
                "graph_retrieved":   len(graph_ids),
                "ground_truth_ids":  ground_truths,
                "dense_ids":         dense_ids,
                "graph_ids":         graph_ids,
                "error":             "",
            }

            for k in K_VALUES:
                row[f"dense_hit_{k}"]       = dense_metrics[f"hit_{k}"]
                row[f"dense_recall_{k}"]    = dense_metrics[f"recall_{k}"]
                row[f"dense_precision_{k}"] = dense_metrics[f"precision_{k}"]
                row[f"graph_hit_{k}"]       = graph_metrics[f"hit_{k}"]
                row[f"graph_recall_{k}"]    = graph_metrics[f"recall_{k}"]
                row[f"graph_precision_{k}"] = graph_metrics[f"precision_{k}"]
                row[f"hit_delta_{k}"]       = graph_metrics[f"hit_{k}"]    - dense_metrics[f"hit_{k}"]
                row[f"recall_delta_{k}"]    = round(graph_metrics[f"recall_{k}"]    - dense_metrics[f"recall_{k}"], 4)
                row[f"precision_delta_{k}"] = round(graph_metrics[f"precision_{k}"] - dense_metrics[f"precision_{k}"], 4)

            for k in [10, 30]:
                d = dense_metrics[f"hit_{k}"]
                g = graph_metrics[f"hit_{k}"]
                print(f"  hit@{k}: dense={d}  graph={g}  Δ={g-d:+d}")

        except Exception as e:
            print(f"  ERROR: {e}")
            row = {
                "question":          question,
                "question_type":     qtype,
                "rewritten_query":   intent.rewritten_query,
                "use_graph":         intent.use_graph,
                "num_ground_truths": len(ground_truths),
                "dense_retrieved":   0,
                "graph_retrieved":   0,
                "ground_truth_ids":  ground_truths,
                "dense_ids":         [],
                "graph_ids":         [],
                "error":             str(e),
            }
            for k in K_VALUES:
                for prefix in ["dense", "graph"]:
                    row[f"{prefix}_hit_{k}"]       = ""
                    row[f"{prefix}_recall_{k}"]    = ""
                    row[f"{prefix}_precision_{k}"] = ""
                row[f"hit_delta_{k}"]       = ""
                row[f"recall_delta_{k}"]    = ""
                row[f"precision_delta_{k}"] = ""

        rows.append(row)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid  = [r for r in rows if r["error"] == ""]
    errors = [r for r in rows if r["error"] != ""]

    def avg(key, subset):
        vals = [r[key] for r in subset if r[key] != ""]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def summary_for(subset):
        if not subset:
            return {}
        return {
            f"hit@{k}":       {"dense": avg(f"dense_hit_{k}", subset),       "graph": avg(f"graph_hit_{k}", subset),       "delta": round(avg(f"graph_hit_{k}", subset)       - avg(f"dense_hit_{k}", subset), 4)}
            for k in K_VALUES
        } | {
            f"recall@{k}":    {"dense": avg(f"dense_recall_{k}", subset),    "graph": avg(f"graph_recall_{k}", subset),    "delta": round(avg(f"graph_recall_{k}", subset)    - avg(f"dense_recall_{k}", subset), 4)}
            for k in K_VALUES
        } | {
            f"precision@{k}": {"dense": avg(f"dense_precision_{k}", subset), "graph": avg(f"graph_precision_{k}", subset), "delta": round(avg(f"graph_precision_{k}", subset) - avg(f"dense_precision_{k}", subset), 4)}
            for k in K_VALUES
        }

    graph_queries    = [r for r in valid if r["use_graph"]]
    no_graph_queries = [r for r in valid if not r["use_graph"]]

    output = {
        "metadata": {
            "total_queries":   len(rows),
            "valid_queries":   len(valid),
            "failed_queries":  len(errors),
            "use_graph_true":  len(graph_queries),
            "use_graph_false": len(no_graph_queries),
            "k_values":        K_VALUES,
        },
        "summary": {
            "all_queries":     summary_for(valid),
            "use_graph_true":  summary_for(graph_queries),
            "use_graph_false": summary_for(no_graph_queries),
        },
        "results": rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    for label, subset in [
        ("ALL QUERIES",     valid),
        ("use_graph=True",  graph_queries),
        ("use_graph=False", no_graph_queries),
    ]:
        n_sub = len(subset)
        if n_sub == 0:
            continue
        print(f"\n{'='*70}")
        print(f"{label}  (n={n_sub})")
        print(f"{'='*70}")
        print(f"\n{'K':<6} {'Dense Hit':>10} {'Graph Hit':>10} {'Δ Hit':>8} {'Dense Rec':>10} {'Graph Rec':>10} {'Δ Rec':>8}")
        print("-" * 70)
        for k in K_VALUES:
            dh = avg(f"dense_hit_{k}", subset)
            gh = avg(f"graph_hit_{k}", subset)
            dr = avg(f"dense_recall_{k}", subset)
            gr = avg(f"graph_recall_{k}", subset)
            print(
                f"{k:<6} {dh:>10.3f} {gh:>10.3f} {gh-dh:>+8.3f} "
                f"{dr:>10.3f} {gr:>10.3f} {gr-dr:>+8.3f}"
            )

    if errors:
        print(f"\nFailed queries: {len(errors)}")
        for r in errors:
            print(f"  - {r['question'][:60]}: {r['error']}")
    print()


if __name__ == "__main__":
    main()