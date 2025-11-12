# scripts/build_covis.py
"""
Create co-visitation neighbors from sessions (windowed, optional time-decay).

Input  : data/sessions/sessions.parquet   (fallback: sessions.csv)
Output : data/artifacts/covis.parquet     (fallback: covis.csv)
Schema : item_id, neighbor, count
Run    : python scripts/build_covis.py --window 3 --topk 100 --time-decay --gamma 0.98
"""

from pathlib import Path
from collections import defaultdict
import argparse
import itertools
import pandas as pd
import math

SESS_PARQUET = Path("data/sessions/sessions.parquet")
SESS_CSV     = Path("data/sessions/sessions.csv")
OUT_PARQUET  = Path("data/artifacts/covis.parquet")
OUT_CSV      = Path("data/artifacts/covis.csv")
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def read_sessions() -> pd.DataFrame:
    if SESS_PARQUET.exists():
        df = pd.read_parquet(SESS_PARQUET)
    elif SESS_CSV.exists():
        # item_seq in CSV is a stringified list -> eval back to list
        df = pd.read_csv(SESS_CSV, converters={"item_seq": eval})
    else:
        raise SystemExit("[ERR] sessions file not found. Run processing/sessionization/job.py first.")

    # ensure columns
    need = {"user_id", "session_id", "start", "end", "item_seq"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[ERR] sessions missing columns: {miss}")

    # coerce dtypes
    df["user_id"] = df["user_id"].astype(str)
    # start/end to datetime
    try:
        df["start"] = pd.to_datetime(df["start"])
        df["end"]   = pd.to_datetime(df["end"])
    except Exception:
        # if already datetime, it's fine
        pass

    return df


def write_covis(df: pd.DataFrame):
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"[OK] wrote {len(df):,} rows -> {OUT_PARQUET}")
    except Exception as e:
        df.to_csv(OUT_CSV, index=False)
        print(f"[WARN] {e}\n[WARN] Parquet engine missing. Wrote CSV instead -> {OUT_CSV}")


def build(topk: int = 100, window: int = 3, symmetric: bool = True,
          time_decay: bool = False, gamma: float = 0.98):
    """
    Build co-vis counts.

    window>0 : sliding window over ordered item_seq, counts close-by pairs (more robust than set-permutations)
    window=0 : original behavior using unique set-permutations per session (counts each pair once per session)

    symmetric : if True, also count B->A when seeing A->B
    time_decay: if True, weight each session by gamma**days_since_session_end
    """
    sess = read_sessions()

    now = pd.Timestamp.utcnow()
    counts = defaultdict(lambda: defaultdict(float))

    # iterate rows
    for row in sess.itertuples(index=False):
        items = row.item_seq
        # if item_seq accidentally string, try to eval
        if isinstance(items, str):
            try:
                items = eval(items)
            except Exception:
                items = []
        # clean empties
        items = [str(x) for x in items if x]
        if not items:
            continue

        # session weight (time decay)
        w = 1.0
        if time_decay:
            try:
                age_days = max(0.0, (now - pd.to_datetime(row.end)).total_seconds() / 86400.0)
                w = gamma ** age_days
            except Exception:
                w = 1.0

        if window and window > 0:
            # WINDOWED, ORDER-PRESERVING (counts can be >1 within a session)
            L = len(items)
            for i in range(L):
                jmax = min(i + 1 + window, L)
                for j in range(i + 1, jmax):
                    a, b = items[i], items[j]
                    counts[a][b] += w
                    if symmetric:
                        counts[b][a] += w
        else:
            # ORIGINAL: unique pairs once per session (sparser)
            uniq = list(set(items))
            for a, b in itertools.permutations(uniq, 2):
                counts[a][b] += w

    # flatten and take topk per item
    rows = []
    for a, nbrs in counts.items():
        # sort by weighted count desc
        top = sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[:topk]
        for b, c in top:
            rows.append({"item_id": a, "neighbor": b, "count": float(c)})

    covis = pd.DataFrame(rows)
    if covis.empty:
        print("[WARN] No co-vis pairs built (maybe too few sessions?).")
    write_covis(covis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100, help="keep top-K neighbors per item")
    ap.add_argument("--window", type=int, default=3,
                    help="sliding window size over item_seq; 0 = use unique set-permutations")
    ap.add_argument("--symmetric", action="store_true", default=True,
                    help="count both A->B and B->A (default True)")
    ap.add_argument("--no-symmetric", action="store_false", dest="symmetric",
                    help="disable symmetric counting (directed only)")
    ap.add_argument("--time-decay", action="store_true", help="apply time decay (gamma**days)")
    ap.add_argument("--gamma", type=float, default=0.98, help="decay base per day (0<gamma<1)")
    args = ap.parse_args()

    if not (0.0 < args.gamma <= 1.0):
        raise SystemExit("--gamma must be in (0,1]. e.g., 0.98")

    build(topk=args.topk,
          window=args.window,
          symmetric=args.symmetric,
          time_decay=args.time_decay,
          gamma=args.gamma)


if __name__ == "__main__":
    main()
