#!/usr/bin/env python3
import os
import re
import time
import json
import psycopg2
import logging
import requests

from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from enum import Enum
from pydantic import BaseModel, ValidationError  
from typing import Union, Any

class QueryRewrite(BaseModel):
    improved_query: str
    expanded_query: Optional[str] = ""

class Validity(str, Enum):
    YES = "YES"
    NO = "NO"

class RagResponse(BaseModel):
    reasoning: str
    valid: Validity
    response: Union[str, List[Dict[str, Any]]]
# ==============================================================================
# 🔍 RETRIEVE: pgvector query (no Pinecone)
# ==============================================================================

def retrieve_from_pg(conn, embeddings: HuggingFaceEmbeddings, query: str, k: int = 100) -> Tuple[List[Document], List[float]]:
    start = time.time()
    logging.info("🔍 Starting similarity search in PostgreSQL (pgvector)...")

    qvec = embeddings.embed_query(query)
    # pgvector expects ARRAY[...], not [...]
    qvec_str = "ARRAY[" + ",".join(f"{v:.8f}" for v in qvec) + "]"

    sql = f"""
        SELECT 
            document_id,
            chunk_index,
            chunk_text,
            metadata,
            1 - (embedding <=> {qvec_str}::vector) AS score
        FROM gold.bpl_embeddings
        ORDER BY embedding <=> {qvec_str}::vector
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (k,))
        rows = cur.fetchall()

    docs, scores = [], []
    for document_id, chunk_index, chunk_text, metadata, score in rows:
        if len(chunk_text) > 4000:
            chunk_text = chunk_text[:4000]
        meta_dict = metadata if isinstance(metadata, dict) else json.loads(metadata) if metadata else {}
        docs.append(Document(page_content=chunk_text, metadata={"source": document_id, **meta_dict}))
        scores.append(float(score))

    logging.info(f"✅ Retrieved {len(docs)} chunks from Postgres in {time.time() - start:.2f}s.")
    return docs, scores


# ==============================================================================
# ⚙️ UTILITY HELPERS
# ==============================================================================

def safe_get_json(url: str) -> Optional[Dict]:
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


def extract_text_from_json(json_data: Dict) -> str:
    if not json_data:
        return ""
    fields = [
        "title_info_primary_tsi",
        "abstract_tsi",
        "subject_geographic_sim",
        "genre_basic_ssim",
        "genre_specific_ssim",
        "date_tsim",
    ]
    parts = []
    for field in fields:
        if field in json_data.get("data", {}).get("attributes", {}):
            val = json_data["data"]["attributes"][field]
            if val:
                parts.append(str(val))
    return " ".join(parts) if parts else "No content available"


# ==============================================================================
# 🧠 QUERY REWRITE
# ==============================================================================

def rephrase_and_expand_query(query: str, llm: Any) -> str:
    """
    Rephrase and expand query using LLM, returning a validated JSON object.
    Falls back to original query if validation fails.
    """
    logging.info("🧠 Rephrasing and expanding query using LLM...")
    start = time.time()

    prompt = f"""
    You are a professional librarian skilled at historical research.
    Rewrite and expand the user's query to match metadata tags.
    Include related terms (synonyms, historical names, places, events).

    Respond ONLY in **valid JSON**, no commentary.
    Example:
    {{
        "improved_query": "Boston 1919 historical events",
        "expanded_query": "Boston 1919 molasses flood, North End disaster, early 20th century"
    }}

    Original Query: "{query}"
    """
    logging.info("📝 LLM prompt for rephrase:\n%s", prompt[:1500])  

    response = llm.invoke(prompt)
    logging.info("📩 LLM raw response (rephrase): %s", response.content[:2000])

    try:
        data = json.loads(response.content)
        parsed = QueryRewrite(**data)
        final_q = f"{parsed.improved_query.strip()} {parsed.expanded_query.strip()}".strip()
        logging.info(f"✅ Query rephrased in {time.time() - start:.2f}s: '{final_q}'")
        return final_q
    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"⚠️ JSON parsing or validation failed: {e}")
        return query  # graceful fallback


# ==============================================================================
# ⚖️ RERANK
# ==============================================================================

def extract_years_from_query(query: str) -> List[str]:
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2}|22\d{2}|23\d{2})\b", query)

weights = {
    "title_info_primary_tsi": 1.5,
    "name_role_tsim": 1.4,
    "date_tsim": 1.3,
    "abstract_tsi": 1.0,
    "note_tsim": 0.8,
    "subject_geographic_sim": 0.5,
    "genre_basic_ssim": 0.5,
    "genre_specific_ssim": 0.5,
}

def rerank(docs: List[Document], query: str) -> List[Document]:
    if not docs:
        logging.warning("⚠️ No documents provided for reranking.")
        return []

    logging.info("⚖️ Starting BM25 reranking...")
    start = time.time()

    query_years = extract_years_from_query(query)
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source")].append(doc)

    merged_docs = []
    for src, chunks in grouped.items():
        text = " ".join(c.page_content for c in chunks if c.page_content)
        merged_docs.append(Document(page_content=text, metadata=chunks[0].metadata))

    bm25 = BM25Retriever.from_documents(merged_docs, k=len(merged_docs))
    ranked = bm25.invoke(query)

    final_ranked = []
    for d in ranked:
        score = 1.0
        for field, weight in weights.items():
            if field in d.metadata and d.metadata[field]:
                score += weight

        date_field = str(d.metadata.get("date_tsim", ""))
        for y in query_years:
            if re.search(rf"\b{y}\b", date_field):
                score += 50
                break

        final_ranked.append((d, score))

    final_ranked.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"✅ Reranked {len(final_ranked)} documents in {time.time() - start:.2f}s.")
    return [doc for doc, _ in final_ranked[:10]]


# ==============================================================================
# 🧩 RESPONSE PARSER
# ==============================================================================

def parse_json_and_check(output: str) -> str:
    """
    Parse JSON response safely, validate via schema, and return final message.
    Handles both string and list-based responses.
    """
    try:
        data = json.loads(output)
        parsed = RagResponse(**data)

        # Handle "NO" case
        if parsed.valid.strip().upper() == "NO":
            return "Sorry, I couldn’t find direct answers, but here are related documents."

        # Handle list-based structured responses (like image or multi-result queries)
        if isinstance(parsed.response, list):
            formatted = "\n".join(
                f"- {item.get('title', 'Untitled')} ({item.get('date', 'Unknown')}): {item.get('description', '')}"
                for item in parsed.response
            )
            return formatted or "No structured results provided."

        # Handle string-based simple responses
        if isinstance(parsed.response, str):
            return parsed.response.strip()

        # Unexpected type fallback
        logging.warning("⚠️ Unexpected response type: %s", type(parsed.response))
        return "Unexpected response format from model."

    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"❌ JSON parsing/validation failed: {e}")
        return "No valid response found in model output."


# ==============================================================================
# 🧠 MAIN RAG
# ==============================================================================

def RAG(llm: Any, conn, embeddings: HuggingFaceEmbeddings, query: str, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    total_start = time.time()
    logging.info("🚀 Starting RAG pipeline...")
    try:
        query = rephrase_and_expand_query(query, llm)

        retrieved, _ = retrieve_from_pg(conn, embeddings, query, k)
        if not retrieved:
            logging.warning("⚠️ No results retrieved from pgvector.")
            return "No documents found for your query.", []

        reranked = rerank(retrieved, query)
        if not reranked:
            logging.warning("⚠️ No documents passed reranking.")
            return "Unable to process retrieved documents.", []

        context = "\n\n".join(d.page_content for d in reranked[:top] if d.page_content)
        if not context.strip():
            logging.warning("⚠️ Context is empty after reranking.")
            return "No relevant content found.", []

        logging.info("🗒️ Generating final LLM summary...")
        start = time.time()

        logging.info("🗒️ Generating final LLM summary...")
        start = time.time()

        # Prompt LLM to output strictly JSON
        prompt = f"""
        Pretend you are a professional librarian.
        Summarize the following context as though you had retrieved it for a patron.
        Some results may include image descriptions, captions, or mentions of places/people—treat these as valid and relevant.

        Respond ONLY in **valid JSON**, no commentary.
        Example:
        {{
          "reasoning": "Context mentions Boston 1919 Molasses Flood; relevant to the query.",
          "valid": "YES",
          "response": "The Great Molasses Flood occurred in Boston in 1919, killing 21 people."
        }}

        Context:
        {context}

        Query:
        {query}
        """

        logging.info("📝 LLM prompt for summary:\n%s", prompt[:1500])
        response = llm.invoke(prompt)
        logging.info("📩 LLM raw response (summary): %s", response.content[:2000])

        parsed = parse_json_and_check(response.content)

        logging.info(f"✅ Summary generated in {time.time() - start:.2f}s.")
        logging.info(f"🏁 RAG completed in {time.time() - total_start:.2f}s total.")
        return parsed, reranked


    except Exception as e:
        logging.error(f"❌ Error in RAG: {e}")
        return f"An error occurred while processing your query: {e}", []
