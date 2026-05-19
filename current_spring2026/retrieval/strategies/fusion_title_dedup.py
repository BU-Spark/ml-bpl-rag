"""
fusion_title_dedup
------------------
Interleave then collapse near-duplicate titles.

Many retrieved results share identical titles (multiple "Indians", duplicate
"His hunting ground of yesterday, National Parks", long runs of "The Boston
Traveler"). Those duplicates occupy rank-5/10 slots without adding information.
Collapsing them to one-per-title after fusion promotes the next distinct
document up the ranking, lifting hit@5 and hit@10 without losing anything.
"""

from __future__ import annotations

from retrieval.strategies._fusion_core import (
    dual_retrieve,
    graph_on_anchor_default,
    interleave,
    title_dedup,
)

NAME = "fusion_title_dedup"
DESCRIPTION = (
    "Fusion method 1: 1:1 interleave + strict title dedup (keep first "
    "occurrence of each title) on the fused head."
)


def retrieve(intent, top_k: int = 10, **_):
    tier1_docs, unified_docs = dual_retrieve(
        intent, graph_on_anchor=graph_on_anchor_default()
    )
    fused = interleave(tier1_docs, unified_docs, ratio=(1, 1))
    deduped = title_dedup(fused, max_per_title=1)
    return deduped[:top_k]
