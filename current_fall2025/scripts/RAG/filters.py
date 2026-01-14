#!/usr/bin/env python3
"""
Filter extraction module for RAG system.
Handles extraction of temporal and material type filters from queries.
"""

import re
import json
import logging
from typing import Any, List, Tuple
from pydantic import ValidationError

from .models import SearchFilters


def extract_filters_with_llm(query: str, llm: Any) -> SearchFilters:
    """
    Extract temporal and material filters from a natural-language query.
    Supports multi-select material types.
    
    Args:
        query: User query string
        llm: Language model instance
        
    Returns:
        SearchFilters object with extracted filters
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
- Use "year_start" and "year_end" if the query refers to:
    - A century or decade (e.g., "1920s" → 1920–1929, "18th century" → 1700–1799).
    - A specific historical event or era. Use your general knowledge to estimate the date range (e.g., "Civil War" → 1861–1865, "Victorian era" → 1837–1901).
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

Query: "Civil War maps and manuscripts"
{{
  "year_exact": null,
  "year_start": 1861,
  "year_end": 1865,
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
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content)
            content = content.strip()

        data = json.loads(content)
        parsed = SearchFilters(**data)
        logging.info(f"🎯 Extracted filters: {parsed.model_dump()}")
        return parsed

    except ValidationError as ve:
        logging.warning(f"⚠️ LLM response validation failed: {ve}")
        return SearchFilters()
    except Exception as e:
        logging.warning(f"⚠️ Filter extraction error: {e}")
        return SearchFilters()


def build_sql_filter(filters: SearchFilters) -> Tuple[str, List[Any]]:
    """
    Convert SearchFilters to SQL WHERE clause + parameters for PostgreSQL.
    
    Args:
        filters: SearchFilters object with year and material type filters
        
    Returns:
        Tuple of (where_clause_string, parameters_list)
    """
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