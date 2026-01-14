#!/usr/bin/env python3
"""
Query enhancement module for RAG system.
Handles query rewriting and expansion using LLM.
"""

import re
import json
import time
import logging
from typing import Any, Dict, Union
from pydantic import ValidationError

from .models import QueryRewrite


def rephrase_and_expand_query(query: str, llm: Any) -> Dict[str, str]:
    """
    Rephrase and expand query using LLM for better catalog metadata matching.
    Falls back to original query if validation fails.
    
    Args:
        query: Original user query
        llm: Language model instance
        
    Returns:
        Dict with keys:
        - 'text': Combined string (improved + expanded) used for search
        - 'improved': The core rewritten query
        - 'expanded': The additional synonyms/context
    """
    logging.info("🧠 Rephrasing and expanding query using LLM...")
    start = time.time()

    prompt = f"""You are a librarian at the Boston Public Library specializing in historical collections and archives.

Your task: Expand the patron's query to better match library catalog metadata (titles, subjects, dates, locations, people, collections).

Include in your expansion:
- Historical synonyms and alternate terminology
- Related geographic locations (neighborhoods, cities, regions)
- Related historical events, people, or movements
- Relevant collection types (newspapers, photographs, maps, documents)
- **Time Periods**: 
  - If a specific year is provided, prioritize it (e.g., "1919").
  - If a decade or century is provided, convert to a numeric range (e.g., "1900s" -> "1900-1999", "18th century" -> "1700-1799"). Do NOT list random decades (e.g., "1910 1920") unless relevant to a specific event.

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
    "expanded_query": "historic pictures Massachusetts vintage streetscape architecture"
}}

Query: "Maps of 18th century New England"
{{
    "improved_query": "New England maps cartography 1700-1799",
    "expanded_query": "colonial Massachusetts Connecticut Rhode Island Maine New Hampshire Vermont 1700s"
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
        final_q = f"{parsed.improved_query.strip()} {(parsed.expanded_query or '').strip()}".strip()
        logging.info(f"✅ Query rephrased in {time.time() - start:.2f}s: ")
        
        return {
            "text": final_q,
            "improved": parsed.improved_query,
            "expanded": parsed.expanded_query
        }

    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"⚠️ JSON parsing or validation failed: {e}")
        # Return fallback structure
        return {
            "text": query,
            "improved": query,
            "expanded": ""
        }