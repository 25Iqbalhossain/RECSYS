# scripts/build_faiss.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

EMB = Path("data/artifacts/item_embeddings.parquet")
IDX = Path("data/artifacts/faiss.index")
IDS = Path("data/artifacts/faiss_ids.parquet")

def main():
    if not EMB.exists():
        raise SystemExit(f"embeddings not found at {EMB}. Run scripts/backfill_embeddings.py first.")
    df = pd.read_parquet(EMB)
    total_rows = len(df)
    if df.empty:
        raise SystemExit("no embeddings in parquet; aborting.")
    ids = []
    vecs_list = []
    dim = None
    skipped_empty = 0
    skipped_non_numeric = 0
    skipped_dim_mismatch = 0
    for row in df.itertuples(index=False):
        iid = str(getattr(row, "item_id"))
        v = getattr(row, "vec")
        try:
            arr = np.asarray(v, dtype=np.float32).ravel()
        except Exception:
            skipped_non_numeric += 1
            continue
        if arr.size == 0:
            skipped_empty += 1
            continue
        if dim is None:
            dim = int(arr.size)
        if arr.size != dim:
            skipped_dim_mismatch += 1
            continue
        ids.append(iid)
        vecs_list.append(arr)
    kept = len(vecs_list)
    if not vecs_list or dim is None or dim <= 0:
        raise SystemExit("no valid vectors found to build FAISS index (dim<=0 or empty)")
    vecs = np.vstack(vecs_list)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    IDX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(IDX))
    pd.DataFrame({"item_id": ids}).to_parquet(IDS, index=False)
    print(
        "[OK] faiss index built: {nt} vectors, dim={dim}\n"
        "source rows: {rows}, kept: {kept}, skipped_empty: {se}, skipped_non_numeric: {sn}, skipped_dim_mismatch: {sd}\n"
        "-> {idx}\n-> {ids}"
        .format(nt=index.ntotal, dim=dim, rows=total_rows, kept=kept,
                se=skipped_empty, sn=skipped_non_numeric, sd=skipped_dim_mismatch,
                idx=IDX, ids=IDS)
    )

if __name__ == "__main__":
    main()
