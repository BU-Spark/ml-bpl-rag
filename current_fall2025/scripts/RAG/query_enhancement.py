#!/usr/bin/env python3
"""
Query enhancement module for RAG system.
Handles query rewriting and expansion using LLM.
"""

import re
import json
import time
import logging
from typing import Any
from pydantic import ValidationError

from .models import QueryRewrite


def rephrase_and_expand_query(query: str, llm: Any) -> str:
    """
    Rephrase and expand query using LLM for better catalog metadata matching.
    Falls back to original query if validation fails.
    
    Args:
        query: Original user query
        llm: Language model instance
        
    Returns:
        Expanded query string combining improved and expanded terms
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
    
    # Remove markdown code fences if present
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

