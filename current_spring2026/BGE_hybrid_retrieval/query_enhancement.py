#!/usr/bin/env python3
"""
Query expansion for the BGE-M3 hybrid RAG system.
Rewrites and expands the user's query via LLM to improve catalog metadata matching.
"""

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from .models import QueryRewrite

logger = logging.getLogger(__name__)


def rephrase_and_expand_query(query: str, llm: Any) -> str:
    """
    Use an LLM to rephrase and expand the query for better catalog recall.

    Falls back to the original query on any failure.
    """
    logger.info("Expanding query with LLM...")
    start = time.time()

    prompt = f"""You are a librarian at the Boston Public Library specializing in historical collections.

Your task: Expand the patron's query to better match library catalog metadata
(titles, subjects, dates, locations, people, collections).

Include in your expansion:
- Historical synonyms and alternate terminology
- Specific time periods (decades, years, date ranges)
- Related geographic locations (neighborhoods, cities, regions)
- Related historical events, people, or movements
- Relevant collection types (newspapers, photographs, maps, documents)

Respond ONLY in valid JSON:
{{
    "improved_query": "main search terms focusing on key metadata fields",
    "expanded_query": "additional related terms, synonyms, historical context"
}}

Examples:
Query: "Boston 1919 events"
{{"improved_query": "Boston 1919 historical events newspapers",
  "expanded_query": "molasses disaster flood North End police strike Dorchester Beacon"}}

Query: "old photos of Harvard Square"
{{"improved_query": "Harvard Square photographs images Cambridge",
  "expanded_query": "historic pictures 1900s 1920s Massachusetts vintage streetscape"}}

Patron's Query: "{query}"
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content).strip()
        data = json.loads(content)
        parsed = QueryRewrite(**data)
        expanded = f"{parsed.improved_query.strip()} {parsed.expanded_query.strip()}".strip()
        logger.info(f"Query expanded in {time.time()-start:.2f}s: '{expanded}'")
        return expanded
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.warning(f"Query expansion failed ({e}), using original query.")
        return query
