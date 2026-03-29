#!/usr/bin/env python3
"""
Query classifier for the BGE-M3 hybrid RAG system.

Uses the OpenAI API to classify incoming queries as:
  - **semantic**  → best served by Standard RAG (dense + sparse retrieval)
  - **thematic**  → best served by GraphRAG (community-level summaries)
  - **both**      → run both paths and merge (default when uncertain)

The classification drives routing in the pipeline (see pipeline.py).

Examples
--------
  Semantic : "Find the Boston Traveler article from January 3, 1900"
  Thematic : "What were the major themes covered in Boston newspapers during the 1920s?"
  Both     : "Tell me about coverage of the 1919 Boston police strike"
"""

import json
import logging
import os
import re
import time
from typing import Literal

logger = logging.getLogger(__name__)

QueryType = Literal["semantic", "thematic", "both"]

_CLASSIFIER_PROMPT = """\
You are a query routing classifier for a newspaper archive retrieval system.

Classify the user's query into ONE of these categories:

1. "semantic" — The query asks about a SPECIFIC article, date, person, event, or fact.
   These are best answered by direct keyword/vector search over document chunks.
   Examples:
   - "Find articles about the fire on March 5, 1910"
   - "What did the Boston Traveler report on January 3, 1900?"
   - "Articles mentioning John Smith in 1925"

2. "thematic" — The query asks about broad THEMES, TRENDS, PATTERNS, or SUMMARIES
   across multiple documents. These need community-level knowledge graph analysis.
   Examples:
   - "What were the main topics in 1920s Boston newspapers?"
   - "How did coverage of labor issues evolve over time?"
   - "What themes dominated the newspaper during World War I?"

3. "both" — The query has elements of both, or you are uncertain.
   Examples:
   - "Tell me about the 1919 Boston police strike and its broader impact"
   - "What was happening in Boston in 1910?"

Respond with ONLY a JSON object: {"classification": "<semantic|thematic|both>"}

User Query: "{query}"
"""


def classify_query(query: str, api_key: str | None = None) -> QueryType:
    """
    Classify a query as semantic, thematic, or both using the OpenAI API.

    Args:
        query:   The user's search query.
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).

    Returns:
        "semantic", "thematic", or "both"
    """
    start = time.time()

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        logger.warning("No OpenAI API key for query classification, defaulting to 'both'")
        return "both"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a query routing classifier. Respond only with valid JSON.",
                },
                {
                    "role": "user",
                    "content": _CLASSIFIER_PROMPT.format(query=query),
                },
            ],
            temperature=0,
            max_tokens=50,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content).strip()

        data = json.loads(content)
        classification = data.get("classification", "both").lower().strip()

        if classification not in ("semantic", "thematic", "both"):
            logger.warning(f"Unexpected classification '{classification}', defaulting to 'both'")
            classification = "both"

        logger.info(f"Query classified as '{classification}' in {time.time() - start:.2f}s")
        return classification

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classifier response: {e}")
        return "both"
    except Exception as e:
        logger.warning(f"Query classification failed: {e}")
        return "both"


def classify_query_fast(query: str) -> QueryType:
    """
    Rule-based fast classifier (no LLM call).
    Use as a fallback or for latency-sensitive scenarios.

    Returns:
        "semantic", "thematic", or "both"
    """
    query_lower = query.lower()

    # Thematic indicators
    thematic_keywords = [
        "theme", "themes", "trend", "trends", "pattern", "patterns",
        "overview", "summary", "summarize", "summarise",
        "main topics", "major topics", "key topics",
        "how did", "what were the", "evolution of",
        "compare", "across years", "over time", "throughout",
        "broader impact", "general coverage", "dominant",
    ]

    # Semantic indicators
    semantic_keywords = [
        "find", "search", "article about", "articles about",
        "on january", "on february", "on march", "on april",
        "on may", "on june", "on july", "on august",
        "on september", "on october", "on november", "on december",
        "specific", "exactly", "mentioned", "reported",
        "who is", "who was", "what happened on",
    ]

    thematic_score = sum(1 for kw in thematic_keywords if kw in query_lower)
    semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)

    if thematic_score > semantic_score and thematic_score > 0:
        return "thematic"
    elif semantic_score > thematic_score and semantic_score > 0:
        return "semantic"
    else:
        return "both"
