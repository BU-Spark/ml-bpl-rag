"""
retrieval/query_understanding.py

Uses GPT-4o to:
  1. Classify the query as content_driven, metadata_driven, or hybrid
  2. Extract hard filters (date range, document type, geography)
  3. Rewrite the query into a clean semantic form for BGE embedding

Returns a structured QueryIntent object consumed by the retrieval pipeline.
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
    raw_query:       str  = ""
    rewritten_query: str  = ""

    # One of: "content_driven" | "metadata_driven" | "hybrid"
    query_type:      str  = "hybrid"

    # Hard SQL filters
    date_filter:     DateFilter        = field(default_factory=DateFilter)
    doc_types:       list[str]         = field(default_factory=list)
    geography:       list[str]         = field(default_factory=list)
    topics:          list[str]         = field(default_factory=list)

    # Weights for score fusion (overridden by classifier)
    content_weight:  float = 0.75
    metadata_weight: float = 0.25


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a search assistant for the Boston Public Library's Digital Commonwealth archive.
The archive contains historical newspapers, photographs, maps, manuscripts, and other
digitised materials from Massachusetts institutions.

Given a user query, return a JSON object with exactly these fields:

{
  "query_type": "<content_driven | metadata_driven | hybrid>",
  "rewritten_query": "<clean semantic version of the query for embedding>",
  "year_min": <integer or null>,
  "year_max": <integer or null>,
  "doc_types": ["newspaper" | "photograph" | "map" | "manuscript" | "book" | "postcard"],
  "geography": ["<place names mentioned>"],
  "topics": ["<subject topics mentioned>"],
  "content_weight": <float between 0 and 1>,
  "metadata_weight": <float between 0 and 1>
}

Classification rules:
- content_driven: the user wants to find specific textual content, events, or facts
  within documents. Full-text search is primary. Set content_weight >= 0.75.
  Examples: "articles about the molasses disaster", "find stories about strikes in 1919"

- metadata_driven: the user wants a type of object with specific attributes (date, place,
  format). Metadata fields drive retrieval. Set metadata_weight >= 0.75.
  Examples: "maps of Worcester from the 1700s", "photographs of JFK's house"

- hybrid: both content and metadata signals are important.
  Split weights roughly 50/50.
  Examples: "newspaper articles about indigenous Americans in the 1800s"

Rules for rewritten_query:
- Expand abbreviations and archaic terms to modern equivalents
- Remove filler words ("can you find me", "I want to see")
- Preserve proper nouns, dates, and geographic names exactly
- Output a concise noun-phrase or question suitable for semantic embedding

content_weight + metadata_weight must sum to 1.0.
Return ONLY valid JSON. No markdown, no explanation.
""".strip()


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(raw_query: str) -> QueryIntent:
    """
    Call GPT-4o to classify and rewrite the user query.

    Returns a QueryIntent with all extracted signals.
    Raises ValueError if the model returns unparseable output.
    """
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0,          # deterministic classification
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw_query},
        ],
    )

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

    # Validate weights sum to 1
    cw = float(parsed.get("content_weight",  0.75))
    mw = float(parsed.get("metadata_weight", 0.25))
    total = cw + mw
    if abs(total - 1.0) > 0.01:
        # Normalise instead of failing
        cw, mw = cw / total, mw / total

    intent = QueryIntent(
        raw_query       = raw_query,
        rewritten_query = parsed.get("rewritten_query", raw_query),
        query_type      = parsed.get("query_type", "hybrid"),
        date_filter     = DateFilter(
            year_min = parsed.get("year_min"),
            year_max = parsed.get("year_max"),
        ),
        doc_types       = parsed.get("doc_types", []),
        geography       = parsed.get("geography", []),
        topics          = parsed.get("topics", []),
        content_weight  = cw,
        metadata_weight = mw,
    )

    return intent


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What were some important historical events that happened in Boston in 1919?",
        "Find pictures of JFK's house on Cape Cod",
        "Are there any maps of Worcester, MA from the 18th century?",
        "I want to see a variety of depictions of indigenous Americans.",
    ]
    for q in test_queries:
        intent = classify_query(q)
        print(f"\nQuery : {q}")
        print(f"  Type            : {intent.query_type}")
        print(f"  Rewritten       : {intent.rewritten_query}")
        print(f"  Date filter     : {intent.date_filter}")
        print(f"  Content weight  : {intent.content_weight}")
        print(f"  Metadata weight : {intent.metadata_weight}")
