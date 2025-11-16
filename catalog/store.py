from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd


CAT_PARQUET = Path("data/catalog/items.parquet")
CAT_CSV = Path("data/catalog/items.csv")


@dataclass
class CatalogItem:
    item_id: str
    title: Optional[str] = None
    desc: Optional[str] = None
    created_ts: Optional[float] = None

    def text(self) -> str:
        parts = [self.title or "", self.desc or ""]
        return ". ".join([p for p in parts if p]).strip()


class Catalog:
    def __init__(self, path_parquet: Path = CAT_PARQUET, path_csv: Path = CAT_CSV) -> None:
        self.path_parquet = Path(path_parquet)
        self.path_csv = Path(path_csv)
        self.df: Optional[pd.DataFrame] = None
        self._map: Dict[str, CatalogItem] = {}
        self.reload()

    def _read_any(self) -> Optional[pd.DataFrame]:
        try:
            if self.path_parquet.exists():
                return pd.read_parquet(self.path_parquet)
        except Exception:
            pass
        if self.path_csv.exists():
            return pd.read_csv(self.path_csv)
        return None

    def reload(self) -> None:
        self.df = self._read_any()
        self._map = {}
        if self.df is None or self.df.empty:
            return
        df = self.df.copy()
        for col in ["item_id", "title", "desc"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if "created_ts" in df.columns:
            try:
                df["created_ts"] = pd.to_numeric(df["created_ts"], errors="coerce").astype(float)
            except Exception:
                df["created_ts"] = None
        else:
            df["created_ts"] = None
        for row in df.itertuples(index=False):
            self._map[str(getattr(row, "item_id"))] = CatalogItem(
                item_id=str(getattr(row, "item_id")),
                title=str(getattr(row, "title", "")) if hasattr(row, "title") else None,
                desc=str(getattr(row, "desc", "")) if hasattr(row, "desc") else None,
                created_ts=float(getattr(row, "created_ts", 0.0)) if hasattr(row, "created_ts") and pd.notna(getattr(row, "created_ts")) else None,
            )

    def get(self, item_id: str) -> Optional[CatalogItem]:
        return self._map.get(str(item_id))

    def all_ids(self) -> list[str]:
        return list(self._map.keys())

