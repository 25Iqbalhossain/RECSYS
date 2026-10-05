# retrieval/vector_store.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd


class FaissStore:
    """
    Thin wrapper around a FAISS index plus an ID mapping.

    The index is expected to be an `IndexFlatIP` (inner product) built on
    L2‑normalized embedding vectors (see `index/faiss_builder.py`). Under
    this convention:

    - Stored vectors are unit‑length.
    - Query vectors are L2‑normalized before search.
    - The raw FAISS score is the cosine similarity in [-1, 1].

    For convenience, `search_vec` returns both:
      - `score`: cosine similarity (higher is better).
      - `distance`: 1 - score (smaller is better), useful for APIs that
        prefer a distance metric.
    """

    def __init__(
        self,
        index_path: str | Path = "data/artifacts/faiss.index",
        ids_path: str | Path = "data/artifacts/faiss_ids.parquet",
    ) -> None:
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.index: faiss.Index | None = None
        self.ids: list[str] = []
        self.reload()

    def reload(self) -> None:
        """Load/reload the FAISS index and the item_id mapping if present."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.ids_path.exists():
            df = pd.read_parquet(self.ids_path)
            self.ids = [str(x) for x in df["item_id"].tolist()]

    def search_vec(self, v: np.ndarray, k: int = 50) -> List[Dict[str, Any]]:
        """
        Search the ANN index with a query vector.

        Parameters
        ----------
        v : np.ndarray
            Query embedding. It will be cast to float32 and L2‑normalized
            before calling FAISS so that inner product == cosine similarity.
        k : int, optional
            Number of neighbors to retrieve (default: 50).

        Returns
        -------
        List[Dict[str, Any]]
            One dict per neighbor with at least:
              - `item_id`: catalog identifier (string).
              - `score`: cosine similarity (float, higher is better).
              - `distance`: 1 - score (float, smaller is better).
        """
        if self.index is None or not self.ids:
            return []

        v = np.asarray(v, dtype="float32")[None, :]
        faiss.normalize_L2(v)
        D, I = self.index.search(v, k)

        rows: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            s_val = float(score)
            rows.append(
                {
                    "item_id": self.ids[idx],
                    "score": s_val,
                    "distance": 1.0 - s_val,
                }
            )
        return rows
