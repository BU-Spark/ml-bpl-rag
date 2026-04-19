"""
retrieval/query_understanding.py

Uses GPT-4o to:
  1. Rewrite the query for better embedding
  2. Extract hard filters (date range)
  3. Decide if GraphRAG should be triggered

Classification into content_driven/metadata_driven removed.
Retrieval always runs both chunk search and metadata search.
Reranking handles the final ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DateFilter:
    year_min: Optional[int] = None
    year_max: Optional[int] = None


@dataclass
class QueryIntent:
    raw_query:       str        = ""
    rewritten_query: str        = ""
    date_filter:     DateFilter = field(default_factory=DateFilter)
    doc_types:       list[str]  = field(default_factory=list)
    use_graph:       bool       = False

    # Keep for backward compatibility but no longer used for routing
    query_type:      str   = "hybrid"
    content_weight:  float = 0.75
    metadata_weight: float = 0.25


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a search assistant for the Boston Public Library's Digital Commonwealth archive.
The archive contains historical newspapers, photographs, maps, manuscripts, and other
digitised materials from Massachusetts institutions (1900-1946).

Given a user query, return a JSON object with exactly these fields:

{
  "rewritten_query": "<clean semantic version of the query for embedding>",
  "year_min": <integer or null>,
  "year_max": <integer or null>,
  "doc_types": ["newspaper" | "photograph" | "map" | "manuscript" | "book" | "postcard"],
  "use_graph": <true or false>
}

Rules for rewritten_query:
- Expand abbreviations and archaic terms to modern equivalents
- Remove filler words ("can you find me", "I want to see")
- Preserve proper nouns, dates, and geographic names exactly
- Output a concise noun-phrase suitable for semantic embedding

Rules for year_min / year_max:
- Only set if the user explicitly mentions a time period
- "1900s" = year_min 1900, year_max 1909
- "early 20th century" = year_min 1900, year_max 1930
- Otherwise null

Rules for use_graph:
- true only when query asks about specific named people, organizations, or events
  where cross-document connections matter
- Examples needing graph: "who was involved in the 1900 labor strike",
  "what organizations covered the molasses disaster", "find everything about Mayor Fitzgerald"
- Examples not needing graph: "what happened in 1900", "Boston news stories",
  "show me newspapers from June 1900", "find photographs of Greece"

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(raw_query: str) -> QueryIntent:
    response = client.chat.completions.create(
        model       = OPENAI_CHAT_MODEL,
        temperature = 0,
        max_tokens  = 300,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw_query},
        ],
    )

    if not response.choices:
        raise ValueError(f"OpenAI returned empty choices (finish_reason may indicate content filter)")
    raw_json = response.choices[0].message.content.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"GPT-4o returned invalid JSON: {e}\nRaw: {raw_json}")

    return QueryIntent(
        raw_query       = raw_query,
        rewritten_query = parsed.get("rewritten_query", raw_query),
        date_filter     = DateFilter(
            year_min = parsed.get("year_min"),
            year_max = parsed.get("year_max"),
        ),
        doc_types       = parsed.get("doc_types", []),
        use_graph       = parsed.get("use_graph", False),
        query_type      = "hybrid",    # always hybrid now
        content_weight  = 0.75,
        metadata_weight = 0.25,
    )


if __name__ == "__main__":
    test_queries = [
        "What were some important historical events that happened in Boston in 1919?",
        "Find pictures of JFK's house on Cape Cod",
        "Are there any maps of Worcester, MA from the 18th century?",
        "Who was Mayor Fitzgerald?",
    ]
    for q in test_queries:
        intent = classify_query(q)
        print(f"\nQuery     : {q}")
        print(f"Rewritten : {intent.rewritten_query}")
        print(f"Date      : {intent.date_filter}")
        print(f"use_graph : {intent.use_graph}")