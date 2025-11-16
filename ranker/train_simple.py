from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.simple_ltr import train_logistic, save_weights
from retrieval.covis.covis_services import CoVis
from retrieval.vector_store import FaissStore
from embedding.models import embed_item_id


SESS = Path("data/sessions/sessions.parquet")


def build_examples(max_sessions: int | None = 10000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not SESS.exists():
        raise SystemExit(f"sessions not found at {SESS}. Run processing/sessionization/job.py")
    df = pd.read_parquet(SESS)
    if df.empty:
        raise SystemExit("no sessions available for training")

    covis = CoVis()
    faiss = FaissStore()

    feat_names = ["semantic", "covis", "popularity"]
    X = []
    y = []
    cnt = 0
    for row in df.itertuples(index=False):
        items = getattr(row, "item_seq")
        if not isinstance(items, list) or len(items) < 2:
            continue
        items = [str(x) for x in items if x]
        for i in range(len(items) - 1):
            ctx = items[i]
            pos = items[i + 1]
            f = {}
            vq = embed_item_id(ctx)
            vp = embed_item_id(pos)
            if vq is not None and vp is not None:
                vq = vq.astype(np.float32)
                vp = vp.astype(np.float32)
                nvq = np.linalg.norm(vq) or 1.0
                nvp = np.linalg.norm(vp) or 1.0
                f["semantic"] = float(np.dot(vq / nvq, vp / nvp))
            else:
                f["semantic"] = 0.0
            nbrs = {r["item_id"]: r["score"] for r in covis.topk_for_item(ctx, k=200)}
            f["covis"] = float(nbrs.get(pos, 0.0))
            df_c = getattr(covis, "df", None)
            if df_c is not None and not df_c.empty:
                try:
                    pop = float(df_c[df_c["neighbor"].astype(str) == pos]["count"].sum())
                except Exception:
                    pop = 0.0
            else:
                pop = 0.0
            f["popularity"] = pop
            X.append([f[k] for k in feat_names])
            y.append(1.0)

            # negative: choose a different FAISS neighbor
            neg = None
            if vq is not None:
                ann = faiss.search_vec(vq.astype(np.float32), k=50)
                neg = next((r["item_id"] for r in ann if r["item_id"] != pos and r["item_id"] != ctx), None)
            if not neg:
                continue
            vn = embed_item_id(neg)
            if vn is not None:
                vn = vn.astype(np.float32)
                nvn = np.linalg.norm(vn) or 1.0
                f2 = {
                    "semantic": float(np.dot(vq / nvq, vn / nvn)) if vq is not None else 0.0,
                    "covis": float(nbrs.get(neg, 0.0)),
                    "popularity": float(df_c[df_c["neighbor"].astype(str) == neg]["count"].sum()) if df_c is not None and not df_c.empty else 0.0,
                }
                X.append([f2[k] for k in feat_names])
                y.append(0.0)

            cnt += 1
            if max_sessions and cnt >= max_sessions:
                break
        if max_sessions and cnt >= max_sessions:
            break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y, feat_names


def main():
    X, y, feat_names = build_examples()
    if X.size == 0:
        raise SystemExit("no training data built")
    w, b = train_logistic(X, y, lr=0.2, epochs=300, l2=1e-4)
    save_weights(feat_names, w, b)
    print(f"[OK] trained logistic reranker on {len(y)} examples, features={feat_names}")


if __name__ == "__main__":
    main()

