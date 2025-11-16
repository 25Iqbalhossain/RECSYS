"""
Build co‑vis neighbors using pairwise association rules (support / confidence / lift).

Input  : data/sessions/sessions.parquet   (fallback: sessions.csv)
Output : data/artifacts/covis.parquet     (fallback: covis.csv)
Schema : item_id, neighbor, count, support, confidence, lift, score
Rule   : score = confidence * lift; keep top‑K neighbors per item.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import itertools
import pandas as pd

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
    return df


def write_covis(df: pd.DataFrame):
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"[OK] wrote {len(df):,} rows -> {OUT_PARQUET}")
    except Exception as e:
        df.to_csv(OUT_CSV, index=False)
        print(f"[WARN] {e}\n[WARN] Parquet engine missing. Wrote CSV instead -> {OUT_CSV}")


def build_assoc(topk: int = 100):
    sess = read_sessions()

    # Count singletons and unordered pair co-occurrences per session (unique itemset per session)
    count_item: dict[str, float] = defaultdict(float)
    count_pair: dict[frozenset[str], float] = defaultdict(float)
    n_sessions = 0

    for row in sess.itertuples(index=False):
        items = getattr(row, "item_seq", [])
        if isinstance(items, str):
            try:
                items = eval(items)
            except Exception:
                items = []
        uniq = [str(x) for x in set(items) if x]
        if not uniq:
            continue
        for a in uniq:
            count_item[a] += 1.0
        if len(uniq) >= 2:
            for a, b in itertools.combinations(sorted(uniq), 2):
                count_pair[frozenset((a, b))] += 1.0
        n_sessions += 1

    if not count_pair:
        print("[WARN] No pairs found (not enough multi-item sessions).")
        write_covis(pd.DataFrame(columns=["item_id","neighbor","count","support","confidence","lift","score"]))
        return

    rows = []
    for pair, cab in count_pair.items():
        a, b = sorted(list(pair))
        ca = count_item.get(a, 0.0) or 0.0
        cb = count_item.get(b, 0.0) or 0.0
        # support over total sessions
        support_ab = cab / max(1.0, float(n_sessions))
        # A->B
        if ca > 0:
            conf_a_b = cab / ca
            supp_b = cb / max(1.0, float(n_sessions))
            lift_a_b = conf_a_b / supp_b if supp_b > 0 else 0.0
            score_a_b = conf_a_b * lift_a_b
            rows.append({
                "item_id": a, "neighbor": b, "count": float(cab),
                "support": float(support_ab), "confidence": float(conf_a_b),
                "lift": float(lift_a_b), "score": float(score_a_b),
            })
        # B->A
        if cb > 0:
            conf_b_a = cab / cb
            supp_a = ca / max(1.0, float(n_sessions))
            lift_b_a = conf_b_a / supp_a if supp_a > 0 else 0.0
            score_b_a = conf_b_a * lift_b_a
            rows.append({
                "item_id": b, "neighbor": a, "count": float(cab),
                "support": float(support_ab), "confidence": float(conf_b_a),
                "lift": float(lift_b_a), "score": float(score_b_a),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] Association-rule table empty.")
        write_covis(df)
        return

    # keep top-K per item by score, then by count
    df = df.sort_values(["item_id", "score", "count"], ascending=[True, False, False])
    df = df.groupby("item_id", as_index=False, sort=False).head(topk)
    write_covis(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100, help="keep top-K neighbors per item")
    args = ap.parse_args()
    build_assoc(topk=args.topk)


if __name__ == "__main__":
    main()
