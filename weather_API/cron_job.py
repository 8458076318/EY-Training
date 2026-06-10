"""
cron_job.py — Runs the weather ETL pipeline on a schedule.

Two ways to use this:
  1. Run this script directly (uses the `schedule` library — pure Python, cross-platform)
  2. Register with the OS cron daemon (see instructions at the bottom)
"""

import sys
import time
import logging
import schedule
from datetime import datetime
from etl_pipeline import run_etl

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cron.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
    ],
)


# ─── Job Definition ───────────────────────────────────────────────────────────
def etl_job():
    log.info("[TRIGGER] Scheduled ETL job triggered at %s", datetime.utcnow().isoformat())
    try:
        csv_path = run_etl(historical=True)
        log.info("[OK] Job finished successfully. Output: %s", csv_path)
    except Exception as e:
        log.error("[FAIL] Job failed: %s", e, exc_info=True)


# ─── Schedule Configuration ───────────────────────────────────────────────────
# Collect new readings every hour
schedule.every(1).hours.do(etl_job)

# Also run a full 6-month backfill once a day at midnight UTC
schedule.every().day.at("00:00").do(lambda: run_etl(historical=True))

# Optional: run immediately on start
etl_job()

log.info("Scheduler running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)   # check every minute


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: OS-Level Cron (Linux/Mac)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Open crontab editor:
#   $ crontab -e
#
# Add one of these lines (adjust paths):
#
#   Run every hour:
#   0 * * * * /usr/bin/python3 /path/to/etl_pipeline.py >> /path/to/cron.log 2>&1
#
#   Run every day at midnight:
#   0 0 * * * /usr/bin/python3 /path/to/etl_pipeline.py >> /path/to/cron.log 2>&1
#
#   Run every 6 hours:
#   0 */6 * * * /usr/bin/python3 /path/to/etl_pipeline.py >> /path/to/cron.log 2>&1
#
# Cron syntax reminder:
#   ┌─ minute  (0-59)
#   │ ┌─ hour    (0-23)
#   │ │ ┌─ day of month (1-31)
#   │ │ │ ┌─ month  (1-12)
#   │ │ │ │ ┌─ day of week (0-7, Sun=0 or 7)
#   * * * * *  command
# ═══════════════════════════════════════════════════════════════════════════════