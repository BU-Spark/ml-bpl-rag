#!/usr/bin/env python3
"""
LLM response generation for the BGE-M3 hybrid RAG system.
Generates a librarian-style catalog summary from the reranked documents.
"""

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from .models import CatalogResponse

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a professional librarian at the Boston Public Library helping a patron find relevant materials.

IMPORTANT: You only have access to CATALOG METADATA (titles, dates, locations, subjects, collections)
— NOT the actual content of documents.

Your task: Based on the catalog entries below, tell the patron which items, collections, or materials
might be relevant to their query.

Guidelines:
- List the most relevant items found (titles, dates, collections)
- Mention key time periods, locations, or subjects that appear
- If results include newspapers, mention specific editions and dates
- If results include images/photographs, describe what collections they're from
- Be helpful even if results aren't perfect — describe what WAS found
- If very few relevant items, suggest broader search terms
- DO NOT make up information not in the catalog entries
- DO NOT try to answer factual questions — only describe available materials

Respond ONLY in valid JSON:
{{"summary": "Your response describing what catalog items are available"}}

Catalog Entries:
{context}

Patron's Query: {query}
"""


def generate_catalog_summary(llm: Any, query: str, context: str) -> str:
    """
    Generate a catalog-discovery summary using the LLM.

    Args:
        llm:     LangChain-compatible language model.
        query:   Expanded user query.
        context: Concatenated page content from reranked documents.

    Returns:
        Summary string, or a fallback message on failure.
    """
    logger.info("Generating LLM catalog summary...")
    start = time.time()

    prompt = _PROMPT_TEMPLATE.format(context=context[:6000], query=query)
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
            content = re.sub(r"```$", "", content).strip()

        if not content.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                content = match.group(0).strip()

        data = json.loads(content)
        parsed = CatalogResponse(**data)
        logger.info(f"Summary generated in {time.time()-start:.2f}s")
        return parsed.summary.strip()

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"Response generation failed: {e}")
        return "Unable to generate a summary. Please try rephrasing your query."
