#!/usr/bin/env python3
import os
import re
import time
import json
import psycopg2
import logging
import requests

from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, ValidationError  


# ==============================================================================
# 📦 DATA MODELS
# ==============================================================================

class QueryRewrite(BaseModel):
    improved_query: str
    expanded_query: Optional[str] = ""


class CatalogResponse(BaseModel):
    """Simplified response for catalog search - no YES/NO validation needed"""
    summary: str


class MaterialType(str, Enum):
    STILL_IMAGE = "Still image"
    CARTOGRAPHIC = "Cartographic"
    MANUSCRIPT = "Manuscript"
    MOVING_IMAGE = "Moving image"
    NOTATED_MUSIC = "Notated music"
    ARTIFACT = "Artifact"
    AUDIO = "Audio"


class SearchFilters(BaseModel):
    year_exact: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    material_types: Optional[List[MaterialType]] = None


# ==============================================================================
# 🔍 RETRIEVE (pgvector query)
# ==============================================================================

def retrieve_from_pg(conn, embeddings: HuggingFaceEmbeddings, query: str, llm: Any, k: int = 100) -> Tuple[List[Document], List[float]]:
    """
    Retrieve relevant documents from PostgreSQL using pgvector similarity search,
    with optional metadata filters (year range, material type) extracted by LLM.
    """
    start = time.time()
    logging.info("🔍 Starting similarity search in PostgreSQL (pgvector)...")


    filters = extract_filters_with_llm(query, llm)
    where_clause, params = build_sql_filter(filters)
    logging.info(f"🧩 Applied filters: {filters.dict()} → WHERE {where_clause}")


    qvec = embeddings.embed_query(query)
    qvec_str = "ARRAY[" + ",".join(f"{v:.8f}" for v in qvec) + "]"

    sql = f"""
        SELECT 
            document_id,
            chunk_index,
            chunk_text,
            metadata,
            1 - (embedding <=> {qvec_str}::vector) AS score
        FROM gold.bpl_embeddings
        WHERE {where_clause}
        ORDER BY embedding <=> {qvec_str}::vector
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (*params, k))
        rows = cur.fetchall()
    docs, scores = [], []
    for document_id, chunk_index, chunk_text, metadata, score in rows:
        if len(chunk_text) > 4000:
            chunk_text = chunk_text[:4000]
        meta_dict = metadata if isinstance(metadata, dict) else json.loads(metadata) if metadata else {}
        docs.append(Document(page_content=chunk_text, metadata={"source": document_id, **meta_dict}))
        scores.append(float(score))

    logging.info(f"✅ Retrieved {len(docs)} chunks (filters applied) in {time.time() - start:.2f}s.")
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
    Rephrase and expand query using LLM for better catalog metadata matching.
    Falls back to original query if validation fails.
    """
    logging.info("🧠 Rephrasing and expanding query using LLM...")
    start = time.time()

    prompt = f"""You are a librarian at the Boston Public Library specializing in historical collections and archives.

Your task: Expand the patron's query to better match library catalog metadata (titles, subjects, dates, locations, people, collections).

Include in your expansion:
- Historical synonyms and alternate terminology
- Specific time periods (decades, years, date ranges)
- Related geographic locations (neighborhoods, cities, regions)
- Related historical events, people, or movements
- Relevant collection types (newspapers, photographs, maps, documents)

Respond ONLY in valid JSON format:
{{
    "improved_query": "main search terms focusing on key metadata fields",
    "expanded_query": "additional related terms, synonyms, historical context"
}}

Examples:
Query: "Boston 1919 events"
{{
    "improved_query": "Boston 1919 historical events newspapers",
    "expanded_query": "molasses disaster flood North End police strike September January Dorchester Beacon"
}}

Query: "old photos of Harvard Square"
{{
    "improved_query": "Harvard Square photographs images Cambridge",
    "expanded_query": "historic pictures 1900s 1920s Massachusetts vintage streetscape architecture"
}}

Patron's Query: "{query}"
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()

    try:
        data = json.loads(content)
        parsed = QueryRewrite(**data)
        final_q = f"{parsed.improved_query.strip()} {parsed.expanded_query.strip()}".strip()
        logging.info(f"✅ Query rephrased in {time.time() - start:.2f}s: '{final_q}'")
        return final_q
    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"⚠️ JSON parsing or validation failed: {e}")
        return query


# ==============================================================================
# 🧭 FILTER EXTRACTION (no location for now)
# ==============================================================================

def extract_filters_with_llm(query: str, llm: Any) -> SearchFilters:
    """
    Extract temporal and material filters from a natural-language query.
    Supports multi-select material types.
    """
    prompt = f"""
You are a metadata extraction assistant for the Boston Public Library's catalog.

Extract structured search filters from this query:
"{query}"

Return a JSON object with these fields (use null if not applicable):
- year_exact: Single year (integer)
- year_start: Start year if a range (integer)
- year_end: End year if a range (integer)
- material_types: List of one or more of the following EXACT values, or null if not specified:
  ["Still image", "Cartographic", "Manuscript", "Moving image", "Notated music", "Artifact", "Audio"]

Rules:
- Use "year_exact" if the query refers to a specific year (e.g., "in 1919").
- Use "year_start" and "year_end" if it refers to a century or decade (e.g., "18th century" → 1700–1799, "1920s" → 1920–1929).
- For "material_types", return a list even if only one applies (e.g., ["Still image"]).
- Set missing or irrelevant fields explicitly to null.
- Respond ONLY in valid JSON using exactly these keys.
- Do NOT include markdown or explanations.

Examples:

Query: "photographs of Boston in 1919"
{{
  "year_exact": 1919,
  "year_start": null,
  "year_end": null,
  "material_types": ["Still image"]
}}

Query: "18th century maps and manuscripts of New England"
{{
  "year_exact": null,
  "year_start": 1700,
  "year_end": 1799,
  "material_types": ["Cartographic", "Manuscript"]
}}

Query: "audio recordings from the 1960s"
{{
  "year_exact": null,
  "year_start": 1960,
  "year_end": 1969,
  "material_types": ["Audio"]
}}
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content)
            content = content.strip()

        data = json.loads(content)
        parsed = SearchFilters(**data)
        logging.info(f"🎯 Extracted filters: {parsed.dict()}")
        return parsed

    except ValidationError as ve:
        logging.warning(f"⚠️ LLM response validation failed: {ve}")
        return SearchFilters()
    except Exception as e:
        logging.warning(f"⚠️ Filter extraction error: {e}")
        return SearchFilters()



def build_sql_filter(filters: SearchFilters) -> Tuple[str, List[Any]]:
    """Convert SearchFilters → SQL WHERE clause + parameters for PostgreSQL."""
    conditions, params = [], []

    # --- Year filters ---
    if filters.year_exact:
        conditions.append("(%s BETWEEN date_start AND date_end)")
        params.append(filters.year_exact)
    elif filters.year_start and filters.year_end:
        conditions.append("(date_start <= %s AND date_end >= %s)")
        params.extend([filters.year_end, filters.year_start])

    # --- Material type filters ---
    if filters.material_types:
        conditions.append("(metadata->'type_of_resource_ssim' ?| %s)")
        params.append(filters.material_types)

    # --- Combine all ---
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return where_clause, params



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
    Parse JSON response safely and return the summary.
    Automatically strips markdown code fences if present.
    """
    try:
        # Clean and normalize the model output
        output = output.strip()
        if output.startswith("```"):
            output = re.sub(r"^```[a-zA-Z0-9]*\n?", "", output)
            output = re.sub(r"```$", "", output)
            output = output.strip()

        # If still not valid JSON, attempt to extract the first JSON object
        if not output.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", output)
            if match:
                output = match.group(0).strip()

        # Try to parse the cleaned JSON
        data = json.loads(output)
        parsed = CatalogResponse(**data)
        
        return parsed.summary.strip()

    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"❌ JSON parsing/validation failed: {e}")
        # Fallback: return the raw output if JSON parsing fails
        return output if output else "Unable to generate response."


# ==============================================================================
# 🧠 MAIN RAG
# ==============================================================================

def RAG(llm: Any, conn, embeddings: HuggingFaceEmbeddings, query: str, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    total_start = time.time()
    logging.info("🚀 Starting RAG pipeline...")
    try:
        query = rephrase_and_expand_query(query, llm)

        retrieved, _ = retrieve_from_pg(conn, embeddings, query, llm, k)
        if not retrieved:
            logging.warning("⚠️ No results retrieved from pgvector.")
            return "No documents found for your query. Try using different search terms or broader keywords.", []

        reranked = rerank(retrieved, query)
        if not reranked:
            logging.warning("⚠️ No documents passed reranking.")
            return "No relevant items found in the catalog. Try broadening your search or using different keywords.", []

        context = "\n\n".join(d.page_content for d in reranked[:top] if d.page_content)
        if not context.strip():
            logging.warning("⚠️ Context is empty after reranking.")
            return "No relevant content found in catalog entries.", []

        logging.info("🗒️ Generating final LLM summary...")
        start = time.time()

        # Improved prompt for catalog/discovery system
        prompt = f"""You are a professional librarian at the Boston Public Library helping a patron find relevant materials.

IMPORTANT: You only have access to CATALOG METADATA (titles, dates, locations, subjects, collections) - NOT the actual content of documents.

Your task: Based on the catalog entries below, tell the patron which items, collections, or materials might be relevant to their query.

Guidelines:
- List the most relevant items found (titles, dates, collections)
- Mention key time periods, locations, or subjects that appear
- If results include newspapers, mention specific editions and dates
- If results include images/photographs, describe what collections they're from
- Be helpful even if results aren't perfect - describe what WAS found
- If very few relevant items, suggest broader search terms
- DO NOT make up information not in the catalog entries
- DO NOT try to answer factual questions - only describe available materials

Respond ONLY in valid JSON format:
{{
  "summary": "Your response describing what catalog items are available"
}}

Example response format:
"I found several relevant items in our collection: The Dorchester Beacon newspaper has multiple editions from 1919 that likely covered major Boston events, including editions from January 11, 1919 (around the time of the molasses disaster) and several from September-November 1919 (Boston Police Strike period). These are available in the Boston Public Library Newspapers collection. I also found references to North End historical materials from this era."

Catalog Entries:
{context}

Patron's Query: {query}
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