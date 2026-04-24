"""
generation/generator.py

Generates a single grounded summary that cites each retrieved document
inline by number, e.g. [1], [2], so users can trace claims to results.
"""

from __future__ import annotations
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


SYSTEM_PROMPT = """
You are a search assistant for the Boston Public Library's Digital Commonwealth archive,
which contains historical newspapers, photographs, maps, manuscripts, and other materials
from Massachusetts institutions.

You will be given a user query and a numbered list of retrieved documents with their
metadata and text excerpts.

Write a 3-5 sentence response that:
- Summarizes what was found and why it is relevant to the query
- Cites specific documents inline using their number, e.g. [1], [2], [3]
- Is grounded ONLY in the provided documents — do not invent facts
- Uses clear, accessible language suitable for researchers and the general public
- Does not mention scores, rankings, or technical retrieval details

Example format:
"Several materials related to the 1919 Boston Molasses Disaster are available [1][2].
The Boston Traveler covered the event extensively in its January 1919 issues [1],
while photographs of the aftermath document the structural damage to the North End [3]."

If the documents are not relevant to the query, say so clearly and suggest refining the search.
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
    response_text = response.choices[0].message.content.strip()

    return GenerationResult(
        response      = response_text,
        source_titles = [d.title for d in docs],
        source_urls   = [d.source_url for d in docs],
    )