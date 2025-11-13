# scripts/build_faiss.py
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

EMB = Path("data/artifacts/item_embeddings.parquet")
IDX = Path("data/artifacts/faiss.index")
IDS = Path("data/artifacts/faiss_ids.parquet")

def main():
    df = pd.read_parquet(EMB)
    vecs = np.vstack(df["vec"].to_list()).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)         # cosine via normalized vectors
    # ensure normalized
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, str(IDX))
    df[["item_id"]].to_parquet(IDS, index=False)
    print(f"[OK] faiss index built: {index.ntotal} vectors, dim={dim}\n-> {IDX}\n-> {IDS}")

if __name__ == "__main__":
    main()
