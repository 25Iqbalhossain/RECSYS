# scripts/generate_events.py
"""
Generate a large dummy events.jsonl for Day-1 pipeline.
Creates users, sessions (>=30m gap), and item views/clicks.

Edit the CONFIG block to change dataset size.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import json, random, math, os

# ---------------- CONFIG ----------------
SEED = 42
NUM_USERS = 5_000          
NUM_ITEMS = 20_000         
AVG_SESSIONS_PER_USER = 6  
AVG_EVENTS_PER_SESSION = 8 
CLICK_RATE = 0.15          
START_TIME = datetime(2024, 1, 1)  
DAYS_SPAN = 30            
OUT_PATH = Path("data/raw_events/events.jsonl")
# ----------------------------------------

random.seed(SEED)

def ensure_dirs():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def zipf_item(pop_alpha=1.2):
    """Heavier head popularity: small IDs more popular."""
    # Zipf-like sampling using 1/rank^alpha over [1..NUM_ITEMS]
    r = random.random()
    # inverse CDF approx for Zipf—simple rejection for stability
    while True:
        k = random.randint(1, NUM_ITEMS)
        p = (1.0 / (k ** pop_alpha)) / 1.2020569031595942  # normalize ~ zeta(1.2)
        if random.random() < p * 1.5:
            return f"item_{k}"

def gen_user_id(i: int) -> str:
    return f"user_{i:05d}"

def write_events():
    ensure_dirs()
    n_users = NUM_USERS
    n_sessions = int(n_users * AVG_SESSIONS_PER_USER)
    # Poisson-ish counts around AVG_EVENTS_PER_SESSION
    def session_len():
        # 2..(2*avg+2) range with heavier center
        lam = AVG_EVENTS_PER_SESSION
        k = max(2, min(2*lam+2, int(random.expovariate(1/lam)) + 2))
        return k

    total_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for u in range(n_users):
            user_id = gen_user_id(u)
            # random starting anchor within DAYS_SPAN
            t = START_TIME + timedelta(days=random.random() * DAYS_SPAN)
            # number of sessions for this user
            s_count = max(1, int(random.expovariate(1/AVG_SESSIONS_PER_USER)) + 1)

            for _ in range(s_count):
                # keep each session within ~5–20 minutes
                ev_in_session = session_len()
                for _ in range(ev_in_session):
                    item_id = zipf_item()
                    # view event
                    ev = {
                        "user_id": user_id,
                        "item_id": item_id,
                        "event_type": "view",
                        "ts": t.timestamp()
                    }
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    total_written += 1

                    # small chance of click after view
                    if random.random() < CLICK_RATE:
                        t_click = t + timedelta(seconds=random.randint(1, 30))
                        evc = {
                            "user_id": user_id,
                            "item_id": item_id,
                            "event_type": "click",
                            "ts": t_click.timestamp()
                        }
                        f.write(json.dumps(evc, ensure_ascii=False) + "\n")
                        total_written += 1

                    # next event inside session (2–15s later)
                    t += timedelta(seconds=random.randint(2, 15))

                # next session after >=31 minutes to guarantee a new session
                t += timedelta(minutes=31 + random.randint(0, 120))

    return total_written

if __name__ == "__main__":
    n = write_events()
    print(f"Wrote {n:,} events -> {OUT_PATH}")
