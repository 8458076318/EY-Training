import os, shutil, pathlib, time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Install watchdog in Colab
# !pip install watchdog -q

EXT_MAP = {
    ".pdf": "PDFs", ".jpg": "Images",
    ".png": "Images", ".csv": "Data",
    ".xlsx": "Data", ".mp4": "Videos",
}

def organise(folder: str):
    src = Path(folder)
    src.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            dest = src / EXT_MAP.get(f.suffix.lower(), "Other")
            dest.mkdir(exist_ok=True)
            shutil.move(str(f), dest / f.name)
            print(f"Moved {f.name} → {dest.name}/")

organise(Path(__file__).resolve().parent / "sample_data")

import os, shutil, pathlib, time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Install watchdog in Colab
# pip install watchdog -q
EXT_MAP = {
    ".pdf": "PDFs", ".jpg": "Images",
    ".png": "Images", ".csv": "Data",
    ".xlsx": "Data", ".mp4": "Videos",
}

def organise(folder: str):
    src = Path(folder)
    src.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            dest = src / EXT_MAP.get(f.suffix.lower(), "Other")
            dest.mkdir(exist_ok=True)
            shutil.move(str(f), dest / f.name)
            print(f"Moved {f.name} → {dest.name}/")

organise(Path(__file__).resolve().parent / "sample_data")

#Section 2 — Web scraping automation
#What we'll do
#Scrape a webpage on a schedule
#Parse content with BeautifulSoup
#Save results to CSV automatically
# !pip install requests beautifulsoup4 -q
import requests, csv
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_quotes() -> list:
    url = "https://quotes.toscrape.com"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    quotes = []
    for q in soup.select(".quote")[:5]:
        quotes.append({
            "text": q.find("span", class_="text").text,
            "author": q.find("small").text,
            "scraped_at": datetime.now().isoformat(),
        })
    return quotes

def save_to_csv(rows, path="quotes.csv"):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text","author","scraped_at"])
        w.writerows(rows)

data = scrape_quotes()
save_to_csv(data)
print(f"Saved {len(data)} quotes")

# SMTP email sender
import smtplib
from email.mime.text import MIMEText

def send_alert(subject: str, body: str,
               to: str, gmail_user: str, app_pw: str):
    print("Email alert preview")
    print(f"To      : {to}")
    print(f"From    : {gmail_user}")
    print(f"Subject : {subject}")
    print(f"Body    : {body}")

# Usage example: print only, do not send a real email
send_alert("Script done", "Scraping finished.", "you@gmail.com",
           "bot@gmail.com", "not-used")

# !pip install schedule -q
import schedule, time

def job_scrape():
    data = scrape_quotes()
    save_to_csv(data)
    print(f"[{datetime.now():%H:%M:%S}] Scraped {len(data)} rows")

def job_report():
    print(f"[{datetime.now():%H:%M:%S}] Daily report sent")

# Every 10 seconds (for demo; change to .minutes / .hours in prod)
schedule.every(10).seconds.do(job_scrape)
# Every day at 08:00
schedule.every().day.at("08:00").do(job_report)

# Run for 35 seconds (3 ticks) then stop
deadline = time.time() + 35
while time.time() < deadline:
    schedule.run_pending()
    time.sleep(1)

# !pip install apscheduler -q
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

scheduler = BackgroundScheduler()

# Interval job — every 15 seconds
scheduler.add_job(job_scrape, "interval", seconds=15,
                   id="scraper")

# Cron job — weekdays at 07:30
scheduler.add_job(job_report,
                   CronTrigger(day_of_week="mon-fri", hour=7, minute=30),
                   id="daily_report")

scheduler.start()
print("Scheduler running. Jobs:")
for job in scheduler.get_jobs():
    print(f" • {job.id} — next: {job.next_run_time}")

time.sleep(40)   # keep alive for demo
scheduler.shutdown()

# argparse is built into Python — no pip install needed
import argparse

# Build the parser
parser = argparse.ArgumentParser(
    prog="scraper",
    description="Scrape quotes and optionally email a report.",
)

# Positional argument (required, no -- prefix)
parser.add_argument("output",
                    help="Path to save the output CSV")

# Optional: value argument
parser.add_argument("--limit", type=int, default=5,
                    help="Max quotes to scrape (default: 5)")

# Optional: boolean flag (True when present, False when absent)
parser.add_argument("--email", action="store_true",
                    help="Send email alert when done")

# Optional: restricted choices
parser.add_argument("--log-level",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    default="INFO")

# Simulate: python scraper.py results.csv --limit 3 --email
args = parser.parse_args(["results.csv", "--limit", "3", "--email"])

print(f"Output    : {args.output}")
print(f"Limit     : {args.limit}")
print(f"Email?    : {args.email}")
print(f"Log level : {args.log_level}")

# Now use the parsed args to drive the scraper
if args.limit:
    print(f"\nRunning scraper with limit={args.limit} -> {args.output}")
    data = scrape_quotes()[:args.limit]
    save_to_csv(data, args.output)
    print(f"Saved {len(data)} rows to {args.output}")

# subprocess is built into Python — no pip install needed
import subprocess
import sys

# 1. Run a simple command and capture output
result = subprocess.run(
    [sys.executable, "-c", "print('Hello from subprocess!')"],
    capture_output=True,   # capture stdout + stderr
    text=True,             # decode bytes to str automatically
    check=True,            # raise an error if exit code != 0
)
print("stdout      :", result.stdout.strip())
print("return code :", result.returncode)

# 2. Run a Python snippet as a sub-process
result = subprocess.run(
    [sys.executable, "-c", "import sys; print('Python:', sys.version.split()[0])"],
    capture_output=True, text=True, check=True
)
print(result.stdout.strip())

# 3. Error handling — catch failures without crashing
try:
    subprocess.run(
        [sys.executable, "-c", "import sys; print('Demo error', file=sys.stderr); sys.exit(2)"],
        capture_output=True, text=True, check=True
    )
except subprocess.CalledProcessError as e:
    print(f"Command failed (exit {e.returncode}): {e.stderr.strip()}")

# 4. Pipe two commands together
producer = subprocess.Popen(
    [sys.executable, "-c", "print('python.exe'); print('notepad.exe')"],
    stdout=subprocess.PIPE,
    text=True
)
filterer = subprocess.Popen(
    [sys.executable, "-c", "import sys; print(''.join(line for line in sys.stdin if 'python' in line).strip())"],
    stdin=producer.stdout,
    stdout=subprocess.PIPE,
    text=True
)
producer.stdout.close()
output, _ = filterer.communicate()
print("Filtered process names:")
print(output.strip() or "  (none found)")

# 5. Reusable helper + practical examples
def run_cmd(cmd: list, timeout: int = 30) -> str:
    """Run a shell command and return stdout, raising on error."""
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        check=True, timeout=timeout
    )
    return result.stdout.strip()

# Run a small cross-platform command
print(run_cmd([sys.executable, "-c", "from pathlib import Path; print('Current folder:', Path.cwd())"]))

# Package install demo skipped to avoid changing your environment during practice
print("Package install skipped in this demo")

# Count lines in the CSV we created earlier
try:
    print("CSV rows:", run_cmd([
        sys.executable, "-c",
        "from pathlib import Path; p=Path('quotes.csv'); print(sum(1 for _ in p.open()) if p.exists() else 'quotes.csv not found')"
    ]))
except subprocess.CalledProcessError:
    print("quotes.csv not found — run Section 2 first")


if os.name == "nt":
    print("CronTab demo skipped: crontab is a Unix/Linux scheduler, not a Windows feature.")
else:
    from crontab import CronTab

    cron = CronTab(user=True)            # edit the current user's crontab
    cron.remove_all(comment="demo_job")  # clean up any leftover demo jobs

    # Add a job: run scraper every day at 06:00
    job = cron.new(
        command=f"{sys.executable} /home/user/scraper.py >> /tmp/scraper.log 2>&1",
        comment="demo_job"
    )
    job.setall("0 6 * * *")   # min  hour  day  month  weekday

    cron.write()
    print("Job written to crontab:")
    for j in cron:
        print(" ", j)
