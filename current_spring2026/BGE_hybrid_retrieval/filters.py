#!/usr/bin/env python3
"""
Metadata filter extraction for the BGE-M3 hybrid RAG system.
Extracts temporal filters from natural-language queries via LLM.
"""

import re
import json
import logging
from typing import Any

from pydantic import ValidationError

from .models import SearchFilters


def extract_filters_with_llm(query: str, llm: Any) -> SearchFilters:
    """
    Extract temporal and material-type filters from a query using an LLM.

    Args:
        query: User query string.
        llm: LangChain-compatible language model.

    Returns:
        SearchFilters with extracted year/material constraints.
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
- Use "year_exact" for a specific year (e.g., "in 1919").
- Use "year_start"/"year_end" for a range (e.g., "1920s" → 1920–1929, "18th century" → 1700–1799,
  "Civil War" → 1861–1865).
- Return a list for material_types even if only one applies.
- Set missing fields explicitly to null.
- Respond ONLY in valid JSON — no markdown, no explanations.

Examples:

Query: "photographs of Boston in 1919"
{{"year_exact": 1919, "year_start": null, "year_end": null, "material_types": ["Still image"]}}

Query: "Civil War maps and manuscripts"
{{"year_exact": null, "year_start": 1861, "year_end": 1865, "material_types": ["Cartographic", "Manuscript"]}}

Query: "audio recordings from the 1960s"
{{"year_exact": null, "year_start": 1960, "year_end": 1969, "material_types": ["Audio"]}}

Query: "{query}"
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content).strip()
        data = json.loads(content)
        parsed = SearchFilters(**data)
        logging.info(f"Extracted filters: {parsed.model_dump()}")
        return parsed
    except ValidationError as ve:
        logging.warning(f"Filter validation failed: {ve}")
        return SearchFilters()
    except Exception as e:
        logging.warning(f"Filter extraction error: {e}")
        return SearchFilters()


