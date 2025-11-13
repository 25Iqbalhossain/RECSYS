# ranker/inference.py
from __future__ import annotations
from typing import List, Dict, Any

def rank_simple(candidates: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    """
    Rules-only v0:
      - input: [{"item_id": "...", "score": <count>}, ...]
      - sort by score desc, return top-N
    """
    if not candidates:
        return []
    ranked = sorted(
        ({"item_id": str(c.get("item_id")), "score": float(c.get("score", 0.0))} for c in candidates),
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[: max(1, n)]
