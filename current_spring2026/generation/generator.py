"""
generation/generator.py

Generates a single grounded summary that cites each retrieved document
inline by number, e.g. [1], [2], so users can trace claims to results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, GENERATION_MAX_TOKENS
from retrieval.retriever import RetrievedDocument

client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class GenerationResult:
    response:      str
    source_titles: List[str]
    source_urls:   List[str]
    is_relevant:   bool = True


SYSTEM_PROMPT = """
You are a search assistant for the Boston Public Library's Digital Commonwealth archive,
which contains historical newspapers, photographs, maps, manuscripts, and other materials
from Massachusetts institutions.

You will be given a user query and a numbered list of retrieved documents with their
metadata and text excerpts.

Return a JSON object with exactly these fields:
{
  "is_relevant": <true if the retrieved documents are genuinely relevant to the query, false otherwise>,
  "response": "<your response here>"
}

Rules for response:
- If is_relevant is true: write a 3-5 sentence summary that cites specific documents
  inline using their number e.g. [1], [2], [3], grounded ONLY in the provided documents,
  using clear accessible language suitable for researchers and the general public,
  without mentioning scores rankings or technical retrieval details
- If is_relevant is false: explain that no relevant materials were found for this query
  in the Digital Commonwealth collection and suggest refining the search.
  DO NOT include any citation numbers like [1], [2] in this case.
  DO NOT reference "the retrieved documents" or "the results" — 
  write as if speaking directly to the user about their query, 
  not about the internal search process. suggest how the user could rephrase their query to find 
  something related within the historical archive, or acknowledge that 
  this topic is outside the scope of the collection.


Rules for is_relevant:
- true if at least some of the retrieved documents genuinely relate to what the user is asking
- false if the retrieved documents are clearly unrelated to the user query, e.g. the user
  asked about a modern topic and the results are historical materials with no connection

Example format when relevant:
{
  "is_relevant": true,
  "response": "Several materials related to the 1919 Boston Molasses Disaster are available [1][2]. The Boston Traveler covered the event extensively [1], while photographs document the structural damage [3]."
}

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


def _build_context(docs: List[RetrievedDocument]) -> str:
    """Format retrieved documents as a numbered list for GPT-4o."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        date_str = doc.issue_date or (str(doc.year[0]) if doc.year else "unknown date")
        excerpt  = doc.best_chunk_text[:300] if doc.best_chunk_text else "No text excerpt — collection-level record."
        lines.append(f"[{i}] Title: {doc.title}")
        lines.append(f"    Date: {date_str} | Institution: {doc.institution}")
        if doc.topics:
            lines.append(f"    Topics: {', '.join(doc.topics)}")
        lines.append(f"    Excerpt: {excerpt}")
        lines.append("")
    return "\n".join(lines)


def generate(raw_query: str, docs: List[RetrievedDocument]) -> GenerationResult:
    """
    Generate a single cited summary referencing documents by [number].
    """
    if not docs:
        return GenerationResult(
            response      = "No relevant materials were found for your query. Try rephrasing or using a more specific historical topic.",
            source_titles = [],
            source_urls   = [],
            is_relevant   = False,
        )

    context = _build_context(docs)

    user_message = f"""User query: {raw_query}

Retrieved documents:
{context}

Write a concise summary that cites the relevant documents inline by number."""

    response = client.chat.completions.create(
        model       = OPENAI_CHAT_MODEL,
        temperature = 0.2,
        max_tokens  = GENERATION_MAX_TOKENS,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    if not response.choices:
        raise ValueError("OpenAI returned empty choices (finish_reason may indicate content filter)")

    raw_json = response.choices[0].message.content.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        # Fallback — treat raw text as response, assume relevant
        return GenerationResult(
            response      = raw_json,
            source_titles = [d.title for d in docs],
            source_urls   = [d.source_url for d in docs],
            is_relevant   = True,
        )

    return GenerationResult(
        response      = parsed.get("response", ""),
        source_titles = [d.title for d in docs],
        source_urls   = [d.source_url for d in docs],
        is_relevant   = parsed.get("is_relevant", True),
    )