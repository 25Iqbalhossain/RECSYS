from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root on sys.path when running via absolute path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding.models import embed_item_id, embed_text, DIM
from catalog.store import Catalog, CatalogItem
OUT = Path("data/artifacts/item_embeddings.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def read_catalog_items() -> list[CatalogItem]:
    cat = Catalog()
    if cat.df is None or cat.df.empty:
        raise SystemExit("catalog not found. Provide data/catalog/items.parquet or items.csv")
    return [cat.get(i) for i in cat.all_ids() if cat.get(i) is not None]


def main() -> None:
    items = read_catalog_items()
    pairs = []
    for it in items:
        text = it.text()
        v = None
        if text:
            v = embed_text(text)
        if v is None:
            v = embed_item_id(it.item_id)
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float32).ravel()
        if a.size == 0:
            continue
        pairs.append((it.item_id, a))
    if not pairs:
        raise SystemExit("no valid item embeddings produced. Is the embedding model loaded?")
    ids, vecs = zip(*pairs)
    embs = np.vstack(vecs)
    out = pd.DataFrame({"item_id": list(ids), "vec": list(embs)})
    out.to_parquet(OUT, index=False)
    print(f"[OK] wrote {len(out)} item embeddings -> {OUT} (dim={DIM})")


if __name__ == "__main__":
    main()

