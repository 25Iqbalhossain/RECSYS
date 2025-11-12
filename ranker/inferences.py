def rank_simple(candidates, n: int = 20):
    """Day-1: sort by co-vis count (already in 'score')."""
    return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:n]
