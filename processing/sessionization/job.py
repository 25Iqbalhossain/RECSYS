"""
Build sessions from raw events with a 30-minute inactivity gap.
Input : data/raw_events/events.jsonl
Output: data/sessions/sessions.parquet
"""
from pathlib import Path
import json
import pandas as pd

RAW = Path("data/raw_events/events.jsonl")
OUT = Path("data/sessions/sessions.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

def run():
    if not RAW.exists():
        raise SystemExit(f"no events at {RAW}. Post some /track events first.")

    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("events file is empty.")
    df = pd.DataFrame(rows)

    # types & sorting
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(["user_id", "ts"], kind="mergesort")

    # new session after 30 minutes of inactivity per user
    gap = pd.Timedelta(minutes=30)
    df["new_session"] = (df.groupby("user_id")["ts"].diff() > gap).fillna(True)
    df["session_id"] = df.groupby("user_id")["new_session"].cumsum().astype(int)

    # group to per-session sequences
    sessions = (
        df.groupby(["user_id", "session_id"])
          .agg(start=("ts","min"),
               end=("ts","max"),
               item_seq=("item_id", lambda s: [x for x in s if pd.notna(x)]))
          .reset_index()
    )

    sessions.to_parquet(OUT, index=False)
    print(f"wrote {len(sessions)} sessions -> {OUT}")

if __name__ == "__main__":
    run()
