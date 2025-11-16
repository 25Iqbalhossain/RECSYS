"""
Run daily refresh jobs: build_covis -> backfill_embeddings -> build_faiss.
Use APScheduler to schedule at a fixed hour (UTC).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import subprocess

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cmd(cmd: list[str]) -> int:
    print(f"[{datetime.utcnow().isoformat()}] RUN: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except Exception as e:
        print(f"[ERR] {e}")
        return 1


def refresh_job():
    ok = run_cmd([sys.executable, "scripts/build_covis.py"]) == 0
    ok = ok and run_cmd([sys.executable, "scripts/backfill_embeddings.py"]) == 0
    ok = ok and run_cmd([sys.executable, "scripts/build_faiss.py"]) == 0
    print(f"refresh complete: ok={ok}")


def main(hour_utc: int = 3):
    sched = BlockingScheduler()
    trigger = CronTrigger(hour=hour_utc, minute=0)
    sched.add_job(refresh_job, trigger=trigger, id="daily_refresh", replace_existing=True)
    # run once now
    refresh_job()
    print(f"scheduler started: daily refresh at {hour_utc}:00 UTC")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("scheduler stopping...")
        sched.shutdown(wait=False)


if __name__ == "__main__":
    main()

