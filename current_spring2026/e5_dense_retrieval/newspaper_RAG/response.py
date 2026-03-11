#!/usr/bin/env python3
"""
LLM-based answer generation for the historical newspaper RAG pipeline.

Unlike the original BPL catalog pipeline (which only had access to metadata),
this pipeline has access to actual OCR article text, so the LLM can answer
factual questions grounded in the newspaper content.
"""

import re
import json
import logging
from typing import Any, List

from pydantic import ValidationError

from .models import ArticleResult, NewsRAGResponse

logger = logging.getLogger(__name__)


def _build_context(articles: List[ArticleResult]) -> str:
    """Format reranked articles into an LLM-readable context block."""
    parts = []
    for i, a in enumerate(articles, 1):
        header = (
            f"[Source {i}: {a.newspaper}, {a.issue_date}]"
            + (f" Topics: {', '.join(a.topics)}" if a.topics else "")
            + (f" Geography: {', '.join(a.geography)}" if a.geography else "")
        )
        parts.append(f"{header}\n{a.full_text}")
    return "\n\n---\n\n".join(parts)


def generate_newspaper_answer(
    llm: Any,
    query: str,
    articles: List[ArticleResult],
) -> str:
    """
    Generate an answer grounded in historical newspaper excerpts.

    Args:
        llm:      LLM instance (must support .invoke(prompt) → response.content).
        query:    User query string.
        articles: Reranked ArticleResult objects (full_text available).

    Returns:
        Answer string extracted from the LLM JSON response,
        or raw output if JSON parsing fails.
    """
    if not articles:
        return "No relevant newspaper articles were found for your query."

    context = _build_context(articles)

    prompt = f"""You are a research assistant specialising in historical Boston newspapers. \
You have access to digitised OCR text from the Boston Evening Transcript archive.

Use the newspaper excerpts below to answer the patron's question accurately. \
The text may contain OCR artefacts (misspellings, broken words) — interpret them \
charitably when the meaning is clear.

Guidelines:
- Answer directly using evidence from the articles.
- Cite specific issues (newspaper name and date) when making claims.
- If the excerpts only partially address the question, say what was found and what is missing.
- Do NOT invent facts not present in the excerpts.
- Keep the answer concise and informative.

Respond ONLY in valid JSON:
{{
  "answer": "Your detailed, source-grounded answer here."
}}

Newspaper Excerpts:
{context}

Patron's Question: {query}
"""

    logger.info("Invoking LLM for answer generation...")
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\n?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()

    # Extract first JSON object if surrounded by prose
    if not content.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0).strip()

    try:
        data = json.loads(content)
        parsed = NewsRAGResponse(**data)
        return parsed.answer.strip()
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error(f"JSON parsing failed: {exc}")
        return content if content else "Unable to generate a response."
