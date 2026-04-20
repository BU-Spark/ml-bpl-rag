"""
retrieval/fusion_interleave.py

Ported from Documents/rag_test/experiments/strategies/_fusion_core.py and
Documents/rag_test/experiments/strategies/fusion_interleave.py.

Runs TWO retrievals per query and interleaves them 1:1:
  A) unified_tier1 config (title-cap + entity-expand) with GraphRAG ON
  B) unified config (plain) with GraphRAG OFF

This matches the offline-winning combo
`tier1+graphrag-on ⊕ unified+graphrag-off` from the rag_test experiments.

Entry point for the pipeline: `fusion_interleave_retrieve(intent, top_k)`.
"""

from __future__ import annotations

import os
from typing import List

from retrieval.retriever import RetrievedDocument
from retrieval.unified_core import UnifiedConfig, unified_retrieve


# ── Graph-on-anchor toggle ────────────────────────────────────────────────────
def graph_on_anchor_default() -> bool:
    """FUSION_GRAPH_ON_ANCHOR env var; default True to match the offline
    winning combo (tier1+graphrag-on ⊕ unified+graphrag-off)."""
    val = os.environ.get("FUSION_GRAPH_ON_ANCHOR", "1").strip().lower()
    return val not in ("0", "false", "off", "no", "")


# ── Graph expansion (mirrors pipeline._expand_with_graph + RRF merge) ─────────
_GRAPHRAG_MERGE_K = 60


def _graph_expand(intent, base_docs, graph_top_k=None, merge_top_k=100):
    """Apply GraphRAG expansion then RRF-merge so graph-added docs can
    displace base results. Safe on failure: returns base_docs unchanged.
    """
    if not base_docs:
        return base_docs
    try:
        from pipeline import _expand_with_graph
    except Exception:
        return base_docs

    if graph_top_k is None:
        try:
            from config import GRAPH_TOP_K as graph_top_k  # noqa: N806
        except Exception:
            graph_top_k = 10

    try:
        expanded = _expand_with_graph(intent, list(base_docs), top_k=graph_top_k)
    except Exception:
        return base_docs

    graph_only = expanded[len(base_docs):]
    scores: dict[str, float] = {}
    rows:   dict[str, RetrievedDocument] = {}
    for rank, d in enumerate(base_docs, start=1):
        scores[d.ark_id] = 1.0 / (_GRAPHRAG_MERGE_K + rank)
        rows[d.ark_id]   = d
    for rank, d in enumerate(graph_only, start=1):
        scores[d.ark_id] = scores.get(d.ark_id, 0.0) + 1.0 / (_GRAPHRAG_MERGE_K + rank)
        rows.setdefault(d.ark_id, d)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:merge_top_k]
    out = []
    for ark, s in ranked:
        d = rows[ark]
        d.final_score = s
        out.append(d)
    return out


# ── Tier configs ──────────────────────────────────────────────────────────────
def _tier1_cfg(pool_size: int = 300) -> UnifiedConfig:
    return UnifiedConfig(
        pool_size=pool_size,
        rewrite_mode="entity_expand",
        apply_title_cap=True,
        per_title_cap=2,
        per_newspaper_cap=3,
        content_weight=0.6,
        metadata_weight=0.4,
    )


def _unified_cfg(pool_size: int = 200) -> UnifiedConfig:
    return UnifiedConfig(
        pool_size=pool_size,
        rewrite_mode="rewrite",
        apply_title_cap=False,
        content_weight=0.6,
        metadata_weight=0.4,
    )


# ── Dual retrieval ────────────────────────────────────────────────────────────
def dual_retrieve(
    intent,
    anchor_pool: int  = 300,
    partner_pool: int = 200,
    graph_on_anchor: bool = True,
    wide_k: int = 100,
) -> tuple[List[RetrievedDocument], List[RetrievedDocument]]:
    """Returns (tier1_docs, unified_docs), each up to wide_k rows."""
    tier1 = unified_retrieve(intent, top_k=wide_k, cfg=_tier1_cfg(anchor_pool))
    if graph_on_anchor and getattr(intent, "use_graph", False):
        tier1 = _graph_expand(intent, tier1, merge_top_k=wide_k)
    unified = unified_retrieve(intent, top_k=wide_k, cfg=_unified_cfg(partner_pool))
    return tier1, unified


# ── 1:1 interleave with dedupe ────────────────────────────────────────────────
def interleave(
    a_docs: List[RetrievedDocument],
    b_docs: List[RetrievedDocument],
    ratio: tuple[int, int] = (1, 1),
) -> List[RetrievedDocument]:
    ra, rb = ratio
    out, seen = [], set()
    ia = ib = 0
    while ia < len(a_docs) or ib < len(b_docs):
        for _ in range(ra):
            while ia < len(a_docs) and a_docs[ia].ark_id in seen:
                ia += 1
            if ia < len(a_docs):
                d = a_docs[ia]
                out.append(d)
                seen.add(d.ark_id)
                ia += 1
        for _ in range(rb):
            while ib < len(b_docs) and b_docs[ib].ark_id in seen:
                ib += 1
            if ib < len(b_docs):
                d = b_docs[ib]
                out.append(d)
                seen.add(d.ark_id)
                ib += 1
    return out


# ── Public entry point ────────────────────────────────────────────────────────
def fusion_interleave_retrieve(
    intent,
    top_k: int = 10,
) -> List[RetrievedDocument]:
    """1:1 interleave of (unified_tier1 + GraphRAG-on) ⊕ (unified + GraphRAG-off)."""
    tier1_docs, unified_docs = dual_retrieve(
        intent, graph_on_anchor=graph_on_anchor_default(),
    )
    fused = interleave(tier1_docs, unified_docs, ratio=(1, 1))
    return fused[:top_k]
