"""Co‑visitation (co‑vis) store used as a classical retrieval signal.

The co‑visitation table is precomputed offline (see
``scripts/build_covis.py``) and stored under
``data/artifacts/covis.parquet`` with a ``.csv`` fallback.  Each row
typically has:

* ``item_id``   – the anchor item.
* ``neighbor``  – an item frequently viewed/clicked together with the anchor.
* ``count``     – number of co‑occurrences in sessions.
* ``score``     – optional association‑rule score (e.g. confidence/lift).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class CoVis:
    """
    Lightweight co‑visitation store.

    This class reads the co‑visitation table and exposes
    :meth:`topk_for_item` to fetch the strongest neighbors for a given
    ``item_id``.

    Scoring convention
    ------------------
    * If a ``score`` column is present, it is used as the primary
      association‑rule score.
    * Otherwise, the raw ``count`` column is used for ranking and also
      copied into ``score``.
    """

    def __init__(self, path: str | Path = "data/artifacts/covis.parquet"):
        self.path_parquet = Path(path)
        self.path_csv = self.path_parquet.with_suffix(".csv")
        self.df: pd.DataFrame | None = None
        self._index: dict[str, pd.DataFrame] = {}
        self.reload()

    def _read_any(self) -> pd.DataFrame | None:
        """Read either the parquet or CSV co‑visitation file, if it exists."""
        try:
            if self.path_parquet.exists():
                return pd.read_parquet(self.path_parquet)
        except Exception:
            # If parquet fails to load, silently fall back to CSV.
            pass
        if self.path_csv.exists():
            return pd.read_csv(self.path_csv)
        return None

    def reload(self) -> None:
        """
        (Re)load the co‑visitation table and build an in‑memory index.

        After this call, ``topk_for_item(item_id)`` is O(1) for lookup
        plus O(k) for slicing, since we pre‑group and sort by the chosen
        score column.
        """
        self.df = self._read_any()
        if self.df is None or self.df.empty:
            self._index = {}
            return

        self.df["item_id"] = self.df["item_id"].astype(str)
        self.df["neighbor"] = self.df["neighbor"].astype(str)

        # Normalize numeric columns.
        if "count" in self.df.columns:
            self.df["count"] = self.df["count"].astype(float)
        else:
            self.df["count"] = 0.0

        if "score" in self.df.columns:
            self.df["score"] = self.df["score"].astype(float)
            order_col = "score"
        else:
            # If no separate association‑rule score is provided, reuse
            # the co‑visitation count as the score.
            self.df["score"] = self.df["count"].astype(float)
            order_col = "count"

        self._index = {
            k: g[["neighbor", "score", "count"]]
            .sort_values(order_col, ascending=False)
            .reset_index(drop=True)
            for k, g in self.df.groupby("item_id", sort=False)
        }

    def topk_for_item(self, item_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Return the top‑k co‑visited neighbors for ``item_id``.

        The result format is a list of dictionaries::

            [{"item_id": "<neighbor>", "score": <score>}, ...]

        where ``score`` prefers the explicit association‑rule metric if
        available, and otherwise falls back to ``count``.
        """
        if not self._index:
            return []

        key = str(item_id)
        if key not in self._index:
            return []

        df = self._index[key].head(k)
        return [{"item_id": n, "score": float(s)} for n, s in zip(df["neighbor"], df["score"])]

