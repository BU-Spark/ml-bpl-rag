#!/usr/bin/env python3
import os, sys, json, datetime, psycopg2, mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    ContextRelevance,
    AnswerRelevancy,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Allow import of RAG pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from RAG import RAG

load_dotenv()


# --------------------------------------------------------------------
# 🔹 Run RAG pipeline on all test queries
# --------------------------------------------------------------------
def run_rag_on_all_tests(config):
    """Runs the RAG system for all test queries and saves outputs."""
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )

    embeddings = HuggingFaceEmbeddings(model_name=config["embed_model"])
    llm = ChatOpenAI(model=config["llm_model"])

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

        contexts = [d.page_content for d in docs if getattr(d, "page_content", None)]
        results.append({
            "id": qid,
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    output_path = f"results/run_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved raw results to: {output_path}\n")
    return results, timestamp


# --------------------------------------------------------------------
# 🔹 Evaluate RAG results with RAGAS
# --------------------------------------------------------------------
def evaluate_with_ragas(results, run_id):
    """Runs RAGAS evaluation and saves results."""
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ])

    print("📊 Running RAGAS metrics evaluation...")

    # Only metrics relevant to retrieval + output quality
    metrics = [
        ContextRecall(),      # 1️⃣ Can it find the right docs?
        ContextPrecision(),   # 2️⃣ Are retrieved docs relevant?
        ContextRelevance(),   # 3️⃣ Semantic relevance overall
        AnswerRelevancy(),    # 4️⃣ Is the final answer coherent?
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()

    print("\n===== 📈 RAGAS Evaluation Summary =====")
    scores_dict = {}
    ordered_metrics = ["context_recall", "context_precision", "context_relevance", "answer_relevancy"]

    for metric in ordered_metrics:
        if metric in df.columns:
            score = df[metric].mean()
            scores_dict[metric] = score
            print(f"{metric:20s}: {score:.4f}")
    print("=======================================")

    summary_path = f"results/scores_{run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved summary scores to: {summary_path}")

    return scores_dict


# --------------------------------------------------------------------
# 🔹 MLflow Orchestrator
# --------------------------------------------------------------------
if __name__ == "__main__":
    import git

    mlflow.set_experiment("BPL_RAG_Baseline_Evaluation")

    config = {
        "llm_model": "gpt-4o-mini",
        "embed_model": "sentence-transformers/all-mpnet-base-v2",
        "top_k": 100,
        "rerank": True,
    }

    with mlflow.start_run(run_name="prompt_v1_free_summary_mode"):
        mlflow.set_tag("stage", "retrieval")
        mlflow.set_tag("dataset", "test_queries_v1.json")
        mlflow.set_tag("llm_model", config["llm_model"])

        try:
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.object.hexsha)
        except Exception:
            print("⚠️ Could not log git commit hash.")

        mlflow.log_params(config)

        results, run_id = run_rag_on_all_tests(config)
        scores = evaluate_with_ragas(results, run_id)

        mlflow.log_metrics(scores)
        mlflow.log_artifact(f"results/run_{run_id}.json")
        mlflow.log_artifact(f"results/scores_{run_id}.json")

        print("\n✅ MLflow logging complete!")
