"""
retrieval/query_understanding.py

Uses GPT-4o to:
  1. Rewrite the query for better embedding
  2. Extract hard filters (date range)

GraphRAG is always enabled. No classification into content_driven/metadata_driven.
Retrieval always runs all paths: chunk search, metadata search, and graph.
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

# @dataclass
# class QueryIntent:
#     raw_query:        str        = ""
#     rewritten_query:  str        = ""
#     keyword_query:    str        = ""      # ← new
#     query_type:       str        = ""      # ← new
#     date_filter:      DateFilter = field(default_factory=DateFilter)

@dataclass
class QueryIntent:
    raw_query:       str        = ""
    rewritten_query: str        = ""
    is_relevant:     bool       = True   # ← add this
    date_filter:     DateFilter = field(default_factory=DateFilter)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a search assistant for the Boston Public Library's Digital Commonwealth archive.
The archive contains historical newspapers, photographs, maps, manuscripts, and other
digitised materials from Massachusetts institutions (1900-1946).

Given a user query, return a JSON object with exactly these fields:

{
  "rewritten_query": "<clean semantic version of the query for embedding>",
  "year_min": <integer or null>,
  "year_max": <integer or null>
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

Rules for is_relevant:
- Look at the Digital Commonwealth archive description above
- true if the archive could plausibly contain materials related to this query
- false if it is impossible for a historical Massachusetts library archive 
  to contain materials that would answer this query
- When in doubt, return true

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(raw_query: str) -> QueryIntent:
    """
    Rewrite query and extract date filters.
    Name kept as classify_query for backward compatibility.
    """
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
    
    raw_json = response.choices[0].message.content
    if raw_json is None:
        raise ValueError("OpenAI returned null content")
    
    raw_json = raw_json.strip()
    
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
        is_relevant     = parsed.get("is_relevant", True),
        date_filter     = DateFilter(
            year_min = parsed.get("year_min"),
            year_max = parsed.get("year_max"),
        ),
    )
    # return QueryIntent(
    #     raw_query       = raw_query,
    #     rewritten_query = parsed.get("rewritten_query", raw_query),
    #     keyword_query   = parsed.get("keyword_query", ""),
    #     query_type      = parsed.get("query_type", "metadata"),
    #     date_filter     = DateFilter(
    #         year_min = parsed.get("year_min"),
    #         year_max = parsed.get("year_max"),
    #     ),
    # )


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