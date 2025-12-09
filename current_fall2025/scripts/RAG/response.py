#!/usr/bin/env python3
"""
Response generation module for RAG system.
Handles LLM-based catalog summary generation and JSON parsing.
"""

import re
import json
import time
import logging
from typing import Any, List
from pydantic import ValidationError
from langchain_core.documents import Document

from .models import CatalogResponse


def generate_catalog_summary(
    llm: Any,
    query: str,
    context: str
) -> str:
    """
    Generate a catalog-style summary using LLM.
    
    Args:
        llm: Language model instance
        query: Original (or expanded) user query
        context: Context string from reranked documents
        
    Returns:
        Summary string describing available catalog materials
    """
    logging.info("🗒️ Generating final LLM summary...")
    start = time.time()

    # Catalog-focused prompt
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

    # logging.info("📝 LLM prompt for summary:\n%s", prompt[:1500])
    response = llm.invoke(prompt)
    # logging.info("📩 LLM raw response (summary): %s", response.content[:2000])

    parsed = parse_json_response(response.content)

    logging.info(f"✅ Summary generated in {time.time() - start:.2f}s.")
    return parsed


def parse_json_response(output: str) -> str:
    """
    Parse JSON response safely and return the summary.
    Automatically strips markdown code fences if present.
    
    Args:
        output: Raw LLM output string
        
    Returns:
        Parsed summary string, or raw output if parsing fails
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

