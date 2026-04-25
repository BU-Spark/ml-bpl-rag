"""
retrieval/refine.py

Phase-2 human-in-the-loop helper. Given the docs the user thumbed up, ask
GPT-4o for two narrower follow-up queries, run each through the full pipeline
(skipping generation), and RRF-merge the new results with the originals.

Public API:
    refine_search(original_query, original_results, thumbed_up_docs, ...)
        → (merged_docs, follow_up_queries, child_query_ids)

The returned `child_query_ids` link back to query_logs rows (so feedback on a
refined-result card can still tie back to the right Postgres row, and the
nightly promotion script can group refinement queries with their parent).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import List, Optional, Tuple

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, RRF_K
from retrieval.retriever import RetrievedDocument


_client = OpenAI(api_key=OPENAI_API_KEY)


_REFINE_PROMPT = """
You are helping refine a search of the Boston Public Library / Digital
Commonwealth archive (Massachusetts historical photos, maps, manuscripts,
newspapers).

The user originally searched:
    "{query}"

These are catalog records the user marked as RELEVANT (the kind of thing
they want more of):
{liked_block}

Write {n} short follow-up search queries that would surface MORE results
similar to the liked items. Each query should:
  - be a clean noun phrase or short question (no filler, no "find me")
  - vary the angle slightly (e.g., a different proper noun, a related event,
    a related place, a related document type)
  - stay within the spirit of the original query

Return ONLY valid JSON, no markdown:
{{"queries": ["...", "..."]}}
""".strip()


def _liked_block(docs: List[RetrievedDocument], max_items: int = 5) -> str:
    lines = []
    for d in docs[:max_items]:
        title = (d.title or "").strip() or "(untitled)"
        date  = d.issue_date or (str(d.year[0]) if d.year else "")
        snippet = (d.best_chunk_text or "").strip().replace("\n", " ")[:160]
        line = f"  - {title}"
        if date:
            line += f" ({date})"
        if snippet:
            line += f" — {snippet}"
        lines.append(line)
    return "\n".join(lines) if lines else "  (none)"


def generate_followups(
    original_query: str,
    thumbed_up_docs: List[RetrievedDocument],
    n: int = 2,
) -> List[str]:
    """One GPT-4o call. Returns up to n follow-up queries. Falls back to []
    on any failure (the caller should treat that as 'no refinement possible').
    """
    if not thumbed_up_docs:
        return []
    prompt = _REFINE_PROMPT.format(
        query=original_query,
        liked_block=_liked_block(thumbed_up_docs),
        n=n,
    )
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=0.2,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        queries = parsed.get("queries", [])
        return [q.strip() for q in queries if q and q.strip()][:n]
    except Exception as e:
        print(f"[refine] follow-up generation failed: {e}")
        return []


def _rrf_merge(
    *doc_lists: List[RetrievedDocument],
    k: int = RRF_K,
) -> List[RetrievedDocument]:
    """RRF-merge multiple ranked RetrievedDocument lists. Each list contributes
    1/(k+rank). Final list is sorted by combined score; final_score is updated
    to reflect the merged score."""
    scores: dict[str, float] = defaultdict(float)
    rows:   dict[str, RetrievedDocument] = {}
    for docs in doc_lists:
        for rank, d in enumerate(docs, start=1):
            scores[d.ark_id] += 1.0 / (k + rank)
            rows.setdefault(d.ark_id, d)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[RetrievedDocument] = []
    for ark_id, s in ranked:
        d = rows[ark_id]
        d.final_score = s
        out.append(d)
    return out


def refine_search(
    original_query:   str,
    original_results: List[RetrievedDocument],
    thumbed_up_docs:  List[RetrievedDocument],
    top_k:            int = 50,
    session_id:       Optional[str] = None,
    parent_query_id:  Optional[int] = None,
    n_followups:      int = 2,
) -> Tuple[List[RetrievedDocument], List[str], List[Optional[int]]]:
    """
    Run two follow-up retrievals based on what the user liked, RRF-merge with
    the original results, return (merged_docs, follow_up_queries, child_ids).

    Each follow-up is logged to query_logs with `parent_query_id` set, so the
    refinement chain is reconstructable.
    """
    # Lazy import to break the pipeline → retrieval → refine cycle
    from pipeline import run_query

    follow_ups = generate_followups(original_query, thumbed_up_docs, n=n_followups)
    if not follow_ups:
        return original_results, [], []

    new_lists:  List[List[RetrievedDocument]] = [original_results]
    child_ids:  List[Optional[int]] = []

    for fq in follow_ups:
        try:
            r = run_query(
                fq,
                top_k           = top_k,
                skip_generation = True,
                session_id      = session_id,
                parent_query_id = parent_query_id,
            )
            new_lists.append(r.documents)
            child_ids.append(r.query_id)
        except Exception as e:
            print(f"[refine] follow-up retrieval failed for {fq!r}: {e}")
            child_ids.append(None)

    merged = _rrf_merge(*new_lists)[:top_k]
    return merged, follow_ups, child_ids
