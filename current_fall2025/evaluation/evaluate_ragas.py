import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from RAG import RAG

import json
import datetime
import psycopg2
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def run_rag_on_all_tests():
    """Runs the RAG system for all test queries and saves outputs."""
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

    with open("test_queries.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        qid = case["id"]
        query = case["query"]
        ground_truth = case["expected_answer"]

        print(f"\n🚀 Running query {qid}: {query}\n")

        try:
            answer, docs = RAG(llm, conn, embeddings, query)
        except Exception as e:
            print(f"❌ Error while running RAG for ID {qid}: {e}")
            answer, docs = f"Error: {e}", []

        # Extract only the text contents of retrieved documents
        contexts = [d.page_content for d in docs if getattr(d, "page_content", None)]

        results.append({
            "id": qid,
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    # Save results with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    output_path = f"results/run_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved raw results to: {output_path}\n")
    return results


def evaluate_with_ragas(results, run_id):
    """Runs RAGAS evaluation on collected results."""
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"]
        }
        for r in results
    ])

    print("📊 Running RAGAS metrics evaluation...")

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        ContextRelevance(),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)

    print("\n===== 📈 RAGAS Evaluation Summary =====")

    # Convert to pandas and get mean scores (this aggregates across all rows)
    df = result.to_pandas()
    
    # Get only the metric columns (they're in result.scores)
    metric_columns = list(result.scores[0].keys()) if result.scores else []
    
    scores_dict = {}
    for metric in metric_columns:
        if metric in df.columns:
            scores_dict[metric] = df[metric].mean()
            print(f"{metric:20s}: {scores_dict[metric]:.4f}")

    print("=======================================")

    # Save scores
    os.makedirs("results", exist_ok=True)
    summary_path = f"results/scores_{run_id}.json"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved summary scores to: {summary_path}")


if __name__ == "__main__":
    # ❌ Skip rerunning the RAG pipeline if results already exist
    # results = run_rag_on_all_tests()

    # Load previously saved results
    run_id = "20251026_182225"  # Extract the timestamp/ID
    saved_path = f"results/run_{run_id}.json"
    
    with open(saved_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    evaluate_with_ragas(results, run_id)