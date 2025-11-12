from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root on sys.path when running via absolute path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding.models import embed_item_id, DIM

COVIS = Path("data/artifacts/covis.parquet")
COVIS_CSV = COVIS.with_suffix(".csv")
OUT = Path("data/artifacts/item_embeddings.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def read_covis() -> pd.DataFrame:
    if COVIS.exists():
        return pd.read_parquet(COVIS)
    if COVIS_CSV.exists():
        return pd.read_csv(COVIS_CSV)
    raise SystemExit("covis not found. Run scripts/build_covis.py first.")


def main() -> None:
    df = read_covis()
    items = sorted(set(df["item_id"].astype(str)) | set(df["neighbor"].astype(str)))
    embs = np.vstack([embed_item_id(i) for i in items])
    out = pd.DataFrame({"item_id": items, "vec": list(embs)})
    out.to_parquet(OUT, index=False)
    print(f"[OK] wrote {len(out)} item embeddings -> {OUT} (dim={DIM})")


if __name__ == "__main__":
    main()

