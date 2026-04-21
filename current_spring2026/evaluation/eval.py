"""
evaluation/eval.py

Runs the full pipeline against test_queries.jsonl and computes retrieval metrics.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import List

# ── Run from current_spring2026/ ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import run_query, PipelineResult


# ── Metric helpers ────────────────────────────────────────────────────────────

def hit_at_k(retrieved: List[str], ground_truths: List[str], k: int) -> int:
    return int(any(gt in retrieved[:k] for gt in ground_truths))


def reciprocal_rank(retrieved: List[str], ground_truths: List[str]) -> float:
    for i, ark_id in enumerate(retrieved, start=1):
        if ark_id in ground_truths:
            return 1.0 / i
    return 0.0


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).parent.parent / "test_queries.jsonl"),
        help="Path to test_queries.jsonl",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "eval_results.csv"),
        help="Path to save per-query CSV results",
    )
    args = parser.parse_args()

    K_VALUES = [10, 30, 50]
    MAX_K = max(K_VALUES)

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: test queries file not found at {queries_path}")
        sys.exit(1)

    with open(queries_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} queries from {queries_path}")
    print(f"Evaluating top-{K_VALUES} retrieved results\n")

    rows = []

    for i, entry in enumerate(entries):
        question    = entry["question"]
        ground_truths = [
            g["ark_id"].removeprefix("commonwealth:")
            for g in entry.get("ground_truths", [])
        ]
        reference_answer = entry.get("answer", "")

        print(f"[{i+1:02d}/{len(entries)}] {question[:70]}...")

        try:
            result: PipelineResult = run_query(question, top_k=MAX_K)
            retrieved_ids    = [doc.ark_id for doc in result.documents]
            retrieved_titles = [doc.title  for doc in result.documents]

            mrr = reciprocal_rank(retrieved_ids, ground_truths)

            row = {
                "question":          question,
                "rewritten_query":   result.intent.rewritten_query,
                "num_ground_truths": len(ground_truths),
                "num_retrieved":     len(retrieved_ids),
                "mrr":               round(mrr, 4),
                "response_preview":  result.generation.response[:150].replace("\n", " "),
                "retrieved_ids":     "|".join(retrieved_ids),
                "retrieved_titles":  "|".join(retrieved_titles),
                "ground_truth_ids":  "|".join(ground_truths),
                "latency_ms":        result.latency_ms,
                "error":             "",
            }
            for k in K_VALUES:
                row[f"hit_at_{k}"]       = hit_at_k(retrieved_ids, ground_truths, k)
                row[f"recall_at_{k}"]    = round(recall_at_k(retrieved_ids, ground_truths, k), 4)
                row[f"precision_at_{k}"] = round(precision_at_k(retrieved_ids, ground_truths, k), 4)

            status = "  " + "  ".join(f"hit@{k}={row[f'hit_at_{k}']}" for k in K_VALUES)
            status += f"  mrr={mrr:.3f}"
            print(status)

        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR: {e}")
            row = {
                "question":          question,
                "rewritten_query":   "",
                "num_ground_truths": len(ground_truths),
                "num_retrieved":     0,
                "mrr":               "",
                "response_preview":  "",
                "retrieved_ids":     "",
                "retrieved_titles":  "",
                "ground_truth_ids":  "|".join(ground_truths),
                "latency_ms":        "",
                "error":             str(e),
            }
            for k in K_VALUES:
                row[f"hit_at_{k}"]       = ""
                row[f"recall_at_{k}"]    = ""
                row[f"precision_at_{k}"] = ""

        rows.append(row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-query results saved to {out_path}")

    # ── Summary (all queries) ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    subset = [r for r in rows if r["mrr"] != ""]
    n = len(subset)

    if n:
        avg = lambda key: sum(r[key] for r in subset) / n  # noqa: E731

        print(f"\nAll queries  (n={n})")
        print(f"  MRR                  : {avg('mrr'):.3f}")

        summary_row = {"n": n, "mrr": round(avg("mrr"), 4)}
        for k in K_VALUES:
            print(f"  Hit@{k:<2}               : {avg(f'hit_at_{k}'):.3f}")
            print(f"  Recall@{k:<2}            : {avg(f'recall_at_{k}'):.3f}")
            print(f"  Precision@{k:<2}         : {avg(f'precision_at_{k}'):.3f}")
            summary_row[f"hit_at_{k}"]       = round(avg(f"hit_at_{k}"), 4)
            summary_row[f"recall_at_{k}"]    = round(avg(f"recall_at_{k}"), 4)
            summary_row[f"precision_at_{k}"] = round(avg(f"precision_at_{k}"), 4)

        errors = [r for r in rows if r["error"]]
        if errors:
            print(f"\nFailed queries: {len(errors)}")
            for r in errors:
                print(f"  - {r['question'][:60]}: {r['error']}")

        print()

        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)
        print(f"Summary results saved to {summary_path}")


if __name__ == "__main__":
    main()
