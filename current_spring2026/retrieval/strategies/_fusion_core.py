"""
_fusion_core
-------------
Shared machinery for the fusion_* strategies. Each fusion strategy runs TWO
retrievals per query — the unified_tier1 configuration (entity_expand rewrite,
diversity caps, graph-on) and the unified configuration (rewrite, no caps,
graph-off) — then combines their ranked outputs via a strategy-specific method.
Mirrors the offline winning combo `tier1+graphrag-on ⊕ unified+graphrag-off`.

Not a registered strategy itself.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List

from retrieval.strategies._unified_core import UnifiedConfig, unified_retrieve


def graph_on_anchor_default() -> bool:
    """Read FUSION_GRAPH_ON_ANCHOR env var. Default True to match the offline
    winning combo (tier1+graphrag-on ⊕ unified+graphrag-off)."""
    val = os.environ.get("FUSION_GRAPH_ON_ANCHOR", "1").strip().lower()
    return val not in ("0", "false", "off", "no", "")


# ── Dual retrieval ────────────────────────────────────────────────────────────

def _tier1_cfg(pool_size: int = 300, use_graph: bool = True) -> UnifiedConfig:
    return UnifiedConfig(
        pool_size=pool_size,
        rewrite_mode="entity_expand",
        apply_title_cap=True,
        per_title_cap=2,
        per_newspaper_cap=3,
        content_weight=0.6,
        metadata_weight=0.4,
        use_graph=use_graph,
    )


def _unified_cfg(pool_size: int = 200) -> UnifiedConfig:
    return UnifiedConfig(
        pool_size=pool_size,
        rewrite_mode="rewrite",
        apply_title_cap=False,
        content_weight=0.6,
        metadata_weight=0.4,
        use_graph=False,
    )


def dual_retrieve(
    intent,
    anchor_pool: int = 300,
    partner_pool: int = 200,
    graph_on_anchor: bool = True,
    wide_k: int = 100,
):
    """Returns (tier1_docs, unified_docs), each up to wide_k rows."""
    tier1   = unified_retrieve(intent, top_k=wide_k, cfg=_tier1_cfg(anchor_pool, use_graph=graph_on_anchor))
    unified = unified_retrieve(intent, top_k=wide_k, cfg=_unified_cfg(partner_pool))
    return tier1, unified


# ── Fusion methods ────────────────────────────────────────────────────────────

def interleave(a_docs, b_docs, ratio=(1, 1)):
    """Pull ratio[0] from a then ratio[1] from b, dedupe by ark_id."""
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


def weighted_rrf(a_docs, b_docs, wa=1.0, wb=1.0, k=60):
    scores = defaultdict(float)
    rows: dict[str, object] = {}
    for rank, d in enumerate(a_docs):
        scores[d.ark_id] += wa / (k + rank + 1)
        rows.setdefault(d.ark_id, d)
    for rank, d in enumerate(b_docs):
        scores[d.ark_id] += wb / (k + rank + 1)
        rows.setdefault(d.ark_id, d)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for ark, s in ranked:
        d = rows[ark]
        d.final_score = s
        out.append(d)
    return out


def title_dedup(docs, keep_first_per_title: bool = True, max_per_title: int = 1):
    """Collapse rows sharing an identical (normalized) title. Post-fusion step
    to kill the 'Boston Traveler x100' monoculture."""
    if not keep_first_per_title:
        return docs
    seen_counts: dict[str, int] = defaultdict(int)
    out = []
    for d in docs:
        title = (getattr(d, "title", "") or "").strip().lower()
        if not title:
            out.append(d)
            continue
        if seen_counts[title] >= max_per_title:
            continue
        seen_counts[title] += 1
        out.append(d)
    return out
