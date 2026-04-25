"""
evaluation/eval_new.py  (fall2025 pipeline)

Runs the fall2025 RAG pipeline (scripts.RAG.RAG) against test_queries.jsonl
and computes the same retrieval metrics used in spring2026:
  Hit@K, Recall@K, Precision@K for K in {10, 30, 50}, MRR,
  and hallucination pass-rate.

Run:
    cd current_fall2025
    python -m evaluation.eval_new
    python -m evaluation.eval_new \
        --queries /path/to/test_queries.jsonl \
        --out evaluation/eval_results_fall25.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List

# ── Make scripts/ importable so `from RAG import RAG` works ──────────────────
ROOT = Path(__file__).parent.parent  # current_fall2025/
sys.path.insert(0, str(ROOT / "scripts"))

import psycopg2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from RAG import RAG  # noqa: E402


# ── Metric helpers ────────────────────────────────────────────────────────────

def hit_at_k(retrieved: List[str], ground_truths: List[str], k: int) -> int:
    return int(any(gt in retrieved[:k] for gt in ground_truths))


def reciprocal_rank(retrieved: List[str], ground_truths: List[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in ground_truths:
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
    hits = sum(1 for doc in retrieved[:k] if doc in ground_truths)
    return hits / k


# ── Query loader (supports .jsonl and .json) ──────────────────────────────────

def load_queries(path: Path):
    text = path.read_text().strip()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    raise ValueError(f"Unrecognised JSON structure in {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fall2025 RAG eval vs spring2026 metrics")
    parser.add_argument(
        "--queries",
        default="/Users/manaswiyadamreddy/Documents/Evaluation/test_queries.jsonl",
        help="Path to test_queries.jsonl (or .json)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "eval_results_fall25.csv"),
        help="Path to save per-query CSV results",
    )
    parser.add_argument("--top", type=int, default=50, help="top reranked docs (>= max K)")
    parser.add_argument("-k", "--k", type=int, default=200, help="pgvector candidate pool")
    args = parser.parse_args()

    K_VALUES = [10, 30, 50]
    MAX_K = max(K_VALUES)
    assert args.top >= MAX_K, f"--top ({args.top}) must be >= max K ({MAX_K})"

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: test queries file not found at {queries_path}")
        sys.exit(1)

    # ── Env & clients ─────────────────────────────────────────────────────────
    load_dotenv()  # picks up .env in cwd if present

    required = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD", "OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"ERROR: missing env vars: {missing}")
        sys.exit(1)

    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatOpenAI(model="gpt-4o-mini")

    entries = load_queries(queries_path)
    print(f"Loaded {len(entries)} queries from {queries_path}")
    print(f"Evaluating top-{K_VALUES} retrieved results (top={args.top}, k={args.k})\n")

    rows = []

    for i, entry in enumerate(entries):
        question = entry["question"]
        qtype = entry["question_type"]
        ground_truths = [
            g["ark_id"].removeprefix("commonwealth:")
            for g in entry.get("ground_truths", [])
        ]

        print(f"[{i+1:02d}/{len(entries)}] ({qtype}) {question[:70]}...")

        t0 = time.time()
        try:
            answer, docs = RAG(llm, conn, embeddings, question, top=args.top, k=args.k)
            # Normalise to bare ark_id (e.g. "70796j511"), matching ground_truths
            retrieved_ids = [
                str(d.metadata["source"]).removeprefix("commonwealth:")
                for d in (docs or [])
                if d.metadata and d.metadata.get("source") is not None
            ]
            latency_ms = int((time.time() - t0) * 1000)

            mrr = reciprocal_rank(retrieved_ids, ground_truths)

            if qtype == "hallucination_test":
                hallucination_pass = int(
                    len(retrieved_ids) == 0
                    or "no relevant" in (answer or "").lower()
                    or "not found" in (answer or "").lower()
                    or "no documents" in (answer or "").lower()
                )
            else:
                hallucination_pass = ""

            row = {
                "question":           question,
                "question_type":      qtype,
                "num_ground_truths":  len(ground_truths),
                "num_retrieved":      len(retrieved_ids),
                "mrr":                round(mrr, 4),
                "hallucination_pass": hallucination_pass,
                "response_preview":   (answer or "")[:150].replace("\n", " "),
                "retrieved_ids":      "|".join(retrieved_ids),
                "ground_truth_ids":   "|".join(ground_truths),
                "latency_ms":         latency_ms,
                "error":              "",
            }
            for k in K_VALUES:
                row[f"hit_at_{k}"]       = hit_at_k(retrieved_ids, ground_truths, k)
                row[f"recall_at_{k}"]    = round(recall_at_k(retrieved_ids, ground_truths, k), 4)
                row[f"precision_at_{k}"] = round(precision_at_k(retrieved_ids, ground_truths, k), 4)

            status = "  " + "  ".join(f"hit@{k}={row[f'hit_at_{k}']}" for k in K_VALUES)
            status += f"  mrr={mrr:.3f}"
            if qtype == "hallucination_test":
                status += f"  hallucination_pass={hallucination_pass}"
            print(status)

        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR: {e}")
            row = {
                "question":           question,
                "question_type":      qtype,
                "num_ground_truths":  len(ground_truths),
                "num_retrieved":      0,
                "mrr":                "",
                "hallucination_pass": "",
                "response_preview":   "",
                "retrieved_ids":      "",
                "ground_truth_ids":   "|".join(ground_truths),
                "latency_ms":         int((time.time() - t0) * 1000),
                "error":              str(e),
            }
            for k in K_VALUES:
                row[f"hit_at_{k}"]       = ""
                row[f"recall_at_{k}"]    = ""
                row[f"precision_at_{k}"] = ""

        rows.append(row)

    conn.close()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-query results saved to {out_path}")

    # ── Summary by query type ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    summary_rows = []
    qtypes_present = []
    for qt in ["metadata", "full_text", "content_driven", "hallucination_test"]:
        if any(r["question_type"] == qt for r in rows):
            qtypes_present.append(qt)

    for qtype in qtypes_present:
        subset = [r for r in rows if r["question_type"] == qtype and r["mrr"] != ""]
        if not subset:
            continue

        n = len(subset)
        avg = lambda key: sum(r[key] for r in subset) / n  # noqa: E731

        print(f"\n{qtype}  (n={n})")
        print(f"  MRR                  : {avg('mrr'):.3f}")
        summary_row = {
            "question_type": qtype,
            "n": n,
            "mrr": round(avg("mrr"), 4),
            "hallucination_pass_rate": "",
        }
        for k in K_VALUES:
            print(f"  Hit@{k:<2}               : {avg(f'hit_at_{k}'):.3f}")
            print(f"  Recall@{k:<2}            : {avg(f'recall_at_{k}'):.3f}")
            print(f"  Precision@{k:<2}         : {avg(f'precision_at_{k}'):.3f}")
            summary_row[f"hit_at_{k}"]       = round(avg(f"hit_at_{k}"), 4)
            summary_row[f"recall_at_{k}"]    = round(avg(f"recall_at_{k}"), 4)
            summary_row[f"precision_at_{k}"] = round(avg(f"precision_at_{k}"), 4)

        if qtype == "hallucination_test":
            hall_subset = [r for r in rows if r["question_type"] == qtype and r["hallucination_pass"] != ""]
            if hall_subset:
                pass_rate = sum(r["hallucination_pass"] for r in hall_subset) / len(hall_subset)
                print(f"  Hallucination pass   : {pass_rate:.3f}")
                summary_row["hallucination_pass_rate"] = round(pass_rate, 4)

        summary_rows.append(summary_row)

    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"\nFailed queries: {len(errors)}")
        for r in errors:
            print(f"  - {r['question'][:60]}: {r['error']}")

    print()

    if summary_rows:
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        summary_fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary results saved to {summary_path}")


if __name__ == "__main__":
    main()
