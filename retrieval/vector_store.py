# retrieval/vector_store.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

class FaissStore:
    def __init__(self,
                 index_path: str | Path = "data/artifacts/faiss.index",
                 ids_path: str | Path = "data/artifacts/faiss_ids.parquet"):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.index = None
        self.ids: list[str] = []
        self.reload()

    def reload(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.ids_path.exists():
            df = pd.read_parquet(self.ids_path)
            self.ids = [str(x) for x in df["item_id"].tolist()]

    def search_vec(self, v: np.ndarray, k: int = 50):
        if self.index is None or not self.ids:
            return []
        v = v.astype("float32")[None, :]
        faiss.normalize_L2(v)
        D, I = self.index.search(v, k)
        rows = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            rows.append({"item_id": self.ids[idx], "score": float(score)})
        return rows
