#!/usr/bin/env python3
import os, json, csv, time, argparse, psycopg2
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from RAG import RAG
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%H:%M:%S"
)
# ---------- helpers ----------
def load_queries(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept any of these shapes:
    # 1) [{"id":"m1","question":"..."}, ...]
    # 2) [{"question":"..."}...]              -> auto id m1,m2,...
    # 3) {"m1":"...","m2":"..."}              -> expand to list
    # 4) ["...","..."]                        -> auto id m1,m2,...
    if isinstance(data, dict):
        return [{"id": str(k), "question": str(v)} for k, v in data.items()]

    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return [{"id": f"m{i}", "question": q} for i, q in enumerate(data, 1)]

    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        out = []
        for i, item in enumerate(data, 1):
            q = item.get("question") or item.get("query") or item.get("q")
            if not q:
                raise ValueError(f"Item {i} missing 'question' field: {item}")
            _id = str(item.get("id") or f"m{i}")
            out.append({"id": _id, "question": str(q)})
        return out

    raise ValueError("Unsupported JSON shape for test queries.")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)



# ---------- main ----------
def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Run BPL RAG on a queries JSON.")
    ap.add_argument("-i", "--input", required=True, help="Path to test_queries.json")
    ap.add_argument("-o", "--out", default="results/answers.csv", help="Output CSV path")
    ap.add_argument("--top", type=int, default=int(os.getenv("TOP", "10")), help="# reranked chunks to LLM")
    ap.add_argument("-k", "--k", type=int, default=int(os.getenv("K", "100")), help="retriever limit from pgvector")
    args = ap.parse_args()

    conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    database=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "prefer")
)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatOpenAI(model="gpt-4o-mini")

    queries = load_queries(args.input)
    results: List[Dict[str, Any]] = []

    for row in queries:
        qid, question = row["id"], row["question"]
        t0 = time.time()
        try:
            answer, docs = RAG(llm, conn, embeddings, question, top=args.top, k=args.k)
            logging.warning(f"[DEBUG] RAW ANSWER for {qid}: {repr(answer)}")
            
            src_ids = []
            for d in docs or []:
                sid = (d.metadata or {}).get("source")
                if sid is not None:
                    src_ids.append(str(sid))
            latency = int((time.time() - t0) * 1000)
            results.append({
                "id": qid,
                "question": question,
                "answer": answer or "",
                "source_ids": json.dumps(src_ids, ensure_ascii=False),
                "num_sources": len(src_ids),
                "latency_ms": latency,
            })
            print(f"[OK] {qid} {latency}ms  sources={len(src_ids)}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            print(f"[ERR] {qid} {latency}ms  {e}")
            results.append({
                "id": qid,
                "question": question,
                "answer": f"Error: {e}",
                "source_ids": "[]",
                "num_sources": 0,
                "latency_ms": latency,
            })

    write_csv(args.out, results)
    print(f"\nFinished. Wrote {args.out}")

if __name__ == "__main__":
    main()
