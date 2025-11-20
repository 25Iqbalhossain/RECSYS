# candidate_generation/hybrid.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def fuse_sources(
    covis: List[Dict[str, Any]],
    ann:   List[Dict[str, Any]],
    seq_boost: List[str] | None = None,
    w_covis: float = 1.0,
    w_ann:   float = 1.0,
    w_seq:   float = 0.2,
    topn:    int = 50,
):
    """Merge candidates from co-vis + ann; add small boost for items appearing in seq_boost list (recent-first)."""
    scores: dict[str, float] = {}
    for r in covis:
        iid, s = str(r["item_id"]), float(r.get("score", 0.0))
        scores[iid] = scores.get(iid, 0.0) + w_covis * s
    for r in ann:
        iid, s = str(r["item_id"]), float(r.get("score", 0.0))
        scores[iid] = scores.get(iid, 0.0) + w_ann * s 
    if seq_boost:
        # exponential small boost for items that also appear in the user's short history
        for rank, iid in enumerate(seq_boost):
            scores[iid] = scores.get(iid, 0.0) + w_seq * (0.9 ** rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [{"item_id": k, "score": float(v)} for k, v in ranked]
