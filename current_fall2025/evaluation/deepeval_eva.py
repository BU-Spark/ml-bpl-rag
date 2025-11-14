#!/usr/bin/env python3
import os, json, datetime, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

# --- 1. Setup ---
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OpenAI key."

CSV_PATH = "answers.csv"             # input CSV
MODEL = "gpt-4o-mini"                # grading model
THRESHOLD = 0.5                      # DeepEval threshold
OUTDIR = Path("results")             # output directory
OUTDIR.mkdir(exist_ok=True)

# --- 2. Load CSV ---
df = pd.read_csv(CSV_PATH)
required_cols = ["question", "answer", "ground_truth", "contexts"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# --- 3. Instantiate metrics ---
metrics = [
    AnswerRelevancyMetric(model=MODEL, threshold=THRESHOLD),
    ContextualRecallMetric(model=MODEL, threshold=THRESHOLD),
    ContextualPrecisionMetric(model=MODEL, threshold=THRESHOLD),
]

# --- 4. Evaluate each row ---
rows = []
for _, r in df.iterrows():
    tc = LLMTestCase(
        input=r["question"],
        actual_output=r["answer"],
        expected_output=str(r["ground_truth"]),
        retrieval_context=list(r["contexts"]),
    )
    row = {"question": tc.input, "answer": tc.actual_output}

    for m in metrics:
        m.measure(tc)
        # Convert metric class name to snake_case
        name = type(m).__name__.replace("Metric", "")
        snake = "".join(["_"+c.lower() if c.isupper() else c for c in name]).lstrip("_")
        row[snake] = float(getattr(m, "score", 0.0) or 0.0)

    rows.append(row)

# --- 5. Save results ---
out_df = pd.DataFrame(rows)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
per_sample_csv = OUTDIR / f"deepeval_scores_{ts}.csv"
out_df.to_csv(per_sample_csv, index=False)

summary = out_df.select_dtypes("number").mean().to_dict()
summary_json = OUTDIR / f"deepeval_summary_{ts}.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# --- 6. Print summary ---
print("✅ Saved per-sample:", per_sample_csv)
print("✅ Saved summary:", summary_json)
print("\n=== Metric means ===")
for k, v in summary.items():
    print(f"{k:25s}: {v:.4f}")
