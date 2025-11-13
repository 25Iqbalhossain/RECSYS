from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import subprocess, sys, requests

def refresh():
    print(f"[{datetime.utcnow().isoformat()}] refresh start")
    subprocess.check_call([sys.executable, "processing/sessionization/job.py"])
    subprocess.check_call([sys.executable, "scripts/build_covis.py"])
    try:
        requests.post("http://127.0.0.1:8000/reload_covis", timeout=2)
    except Exception:
        pass
    print(f"[{datetime.utcnow().isoformat()}] refresh done")

if __name__ == "__main__":
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(refresh, "interval", minutes=30, next_run_time=None)
    print("Scheduler running every 30 minutes…")
    sched.start()
