# retrieval/covis_service.py  (যদি তুমি retrieval/covis/covis_services.py নামে রাখো,
# তাহলে import path অনুযায়ী server.py তে ঠিক করো)

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

class CoVis:
    """
    Read/Reload co-vis table and return top-k neighbors for an item.
    Looks for data/artifacts/covis.parquet (fallback: covis.csv).
    """
    def __init__(self, path: str | Path = "data/artifacts/covis.parquet"):
        self.path_parquet = Path(path)
        self.path_csv = self.path_parquet.with_suffix(".csv")
        self.df: pd.DataFrame | None = None
        self._index: dict[str, pd.DataFrame] = {}
        self.reload()

    def _read_any(self) -> pd.DataFrame | None:
        try:
            if self.path_parquet.exists():
                return pd.read_parquet(self.path_parquet)
        except Exception:
            pass
        if self.path_csv.exists():
            return pd.read_csv(self.path_csv)
        return None

    def reload(self) -> None:
        """(Re)load table and build an in-memory index by item_id."""
        self.df = self._read_any()
        if self.df is None or self.df.empty:
            self._index = {}
            return
        self.df["item_id"] = self.df["item_id"].astype(str)
        self.df["neighbor"] = self.df["neighbor"].astype(str)
        self.df["count"] = self.df["count"].astype(float)
        self._index = {
            k: g[["neighbor", "count"]].sort_values("count", ascending=False).reset_index(drop=True)
            for k, g in self.df.groupby("item_id", sort=False)
        }

    def topk_for_item(self, item_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """Return [{'item_id': neighbor, 'score': count}, ...]"""
        if not self._index:
            return []
        key = str(item_id)
        if key not in self._index:
            return []
        df = self._index[key].head(k)
        return [{"item_id": n, "score": float(c)} for n, c in zip(df["neighbor"], df["count"])]
