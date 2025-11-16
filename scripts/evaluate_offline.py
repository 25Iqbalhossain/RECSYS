from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.covis.covis_services import CoVis
from retrieval.vector_store import FaissStore
from embedding.models import embed_item_id


SESS = Path("data/sessions/sessions.parquet")


def ndcg_at_k(rank, k=10):
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / np.log2(rank + 2)


def mrr_at_k(rank, k=10):
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / (rank + 1)


def evaluate(n=10, max_sessions=5000):
    if not SESS.exists():
        raise SystemExit(f"sessions not found at {SESS}")
    df = pd.read_parquet(SESS)
    covis = CoVis()
    faiss = FaissStore()
    ndcgs = []
    mrrs = []
    cnt = 0
    for row in df.itertuples(index=False):
        items = getattr(row, "item_seq")
        if not isinstance(items, list) or len(items) < 2:
            continue
        items = [str(x) for x in items if x]
        for i in range(len(items) - 1):
            ctx = items[i]
            nxt = items[i + 1]
            vq = embed_item_id(ctx)
            if vq is None or getattr(faiss, "index", None) is None:
                continue
            ann = faiss.search_vec(vq.astype(np.float32), k=n)
            ranked = [r["item_id"] for r in ann]
            try:
                rank = ranked.index(nxt)
            except ValueError:
                rank = None
            ndcgs.append(ndcg_at_k(rank, k=n))
            mrrs.append(mrr_at_k(rank, k=n))
            cnt += 1
            if max_sessions and cnt >= max_sessions:
                break
        if max_sessions and cnt >= max_sessions:
            break
    print(f"N={len(ndcgs)}  NDCG@{n}: {np.mean(ndcgs):.4f}  MRR@{n}: {np.mean(mrrs):.4f}")


if __name__ == "__main__":
    evaluate()

