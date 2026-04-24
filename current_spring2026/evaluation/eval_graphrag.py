"""
evaluation/eval.py

Runs the full pipeline against test_queries.jsonl and computes retrieval metrics.
Saves results as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import List

# ── Run from current_spring2026/ ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import run_query, PipelineResult
from retrieval.query_understanding import QueryIntent, DateFilter


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
        default=str(Path(__file__).parent / "eval_results.json"),
        help="Path to save per-query JSON results",
    )
    parser.add_argument(
        "--exp",
        default="",
        help="Experiment suffix",
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

    # ── Load rewrite cache ────────────────────────────────────────────────
    cache_path = Path(__file__).parent / "query_rewrite_cache.json"
    query_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache_hits, cache_misses = 0, 0
    # ─────────────────────────────────────────────────────────────────────

    print(f"Loaded {len(entries)} queries from {queries_path}")
    print(f"Evaluating top-{K_VALUES} retrieved results\n")

    rows = []

    for i, entry in enumerate(entries):
        question      = entry["question"]
        question_type = entry.get("question_type", "")
        ground_truths = [
            g["ark_id"].removeprefix("commonwealth:")
            for g in entry.get("ground_truths", [])
        ]

        print(f"[{i+1:02d}/{len(entries)}] {question[:70]}...")

        try:
            # ── Build or load intent from cache ───────────────────────────
            cache_key = hashlib.md5(question.encode()).hexdigest()

            if cache_key in query_cache:
                cached = query_cache[cache_key]
                intent = QueryIntent(
                    raw_query       = question,
                    rewritten_query = cached["rewritten_query"],
                    is_relevant     = cached.get("is_relevant", True),
                    date_filter     = DateFilter(
                        year_min = cached.get("year_min"),
                        year_max = cached.get("year_max"),
                    ),
                )
                cache_hits += 1
            else:
                intent = None  # will be built inside run_query
                cache_misses += 1
            # ─────────────────────────────────────────────────────────────

            result: PipelineResult = run_query(
                question,
                top_k           = MAX_K,
                skip_generation = True,
                prebuilt_intent = intent,
            )

            # ── Save to cache if this was a miss ──────────────────────────
            if cache_key not in query_cache:
                query_cache[cache_key] = {
                    "rewritten_query": result.intent.rewritten_query,
                    "is_relevant":     result.intent.is_relevant,
                    "year_min":        result.intent.date_filter.year_min,
                    "year_max":        result.intent.date_filter.year_max,
                }
                cache_path.write_text(json.dumps(query_cache, indent=2))
            # ─────────────────────────────────────────────────────────────

            retrieved_ids    = [doc.ark_id for doc in result.documents]
            retrieved_titles = [doc.title  for doc in result.documents]

            mrr = reciprocal_rank(retrieved_ids, ground_truths)

            row = {
                "question":          question,
                "question_type":     question_type,
                "rewritten_query":   result.intent.rewritten_query,
                "num_ground_truths": len(ground_truths),
                "num_retrieved":     len(retrieved_ids),
                "mrr":               round(mrr, 4),
                "response_preview":  result.generation.response[:150].replace("\n", " "),
                "retrieved_ids":     retrieved_ids,
                "retrieved_titles":  retrieved_titles,
                "ground_truth_ids":  ground_truths,
                "latency_ms":        result.latency_ms,
                "error":             "",
            }

            for k in K_VALUES:
                row[f"hit_at_{k}"]       = hit_at_k(retrieved_ids, ground_truths, k)
                row[f"recall_at_{k}"]    = round(recall_at_k(retrieved_ids, ground_truths, k), 4)
                row[f"precision_at_{k}"] = round(precision_at_k(retrieved_ids, ground_truths, k), 4)

            print("  " + "  ".join(f"hit@{k}={row[f'hit_at_{k}']}" for k in K_VALUES) + f"  mrr={mrr:.3f}")

        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR: {e}")
            row = {
                "question":          question,
                "question_type":     question_type,
                "rewritten_query":   "",
                "num_ground_truths": len(ground_truths),
                "num_retrieved":     0,
                "mrr":               "",
                "response_preview":  "",
                "retrieved_ids":     [],
                "retrieved_titles":  [],
                "ground_truth_ids":  ground_truths,
                "latency_ms":        "",
                "error":             str(e),
            }

            for k in K_VALUES:
                row[f"hit_at_{k}"]       = ""
                row[f"recall_at_{k}"]    = ""
                row[f"precision_at_{k}"] = ""

        rows.append(row)

    print(f"\nCache hits: {cache_hits} | Cache misses (GPT-4o calls): {cache_misses}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    if args.exp:
        out_path = out_path.with_name(out_path.stem + "_" + args.exp + out_path.suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nPer-query results saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    def compute_and_save_summary(subset_rows, label):
        subset = [r for r in subset_rows if r["mrr"] != ""]
        n = len(subset)
        if not n:
            return

        avg = lambda key: sum(r[key] for r in subset) / n  # noqa: E731

        summary_row = {"n": n, "mrr": round(avg("mrr"), 4)}

        for k in K_VALUES:
            summary_row[f"hit_at_{k}"]       = round(avg(f"hit_at_{k}"), 4)
            summary_row[f"recall_at_{k}"]    = round(avg(f"recall_at_{k}"), 4)
            summary_row[f"precision_at_{k}"] = round(avg(f"precision_at_{k}"), 4)

        summary_path = out_path.with_name(out_path.stem + f"_summary_{label}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_row, f, indent=2, ensure_ascii=False)

        print(f"Summary ({label}) saved to {summary_path}")

    compute_and_save_summary(rows, "overall")
    compute_and_save_summary(
        [r for r in rows if r.get("question_type") == "metadata"],  "metadata"
    )
    compute_and_save_summary(
        [r for r in rows if r.get("question_type") == "full_text"], "full_text"
    )


if __name__ == "__main__":
    main()