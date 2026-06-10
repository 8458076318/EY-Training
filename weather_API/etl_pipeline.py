"""
Weather ETL Pipeline - OpenWeatherMap API
Extracts last 6 months of historical weather data and stores it locally (CSV/SQLite).
"""

import os
import time
import sqlite3
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ─── Configuration ────────────────────────────────────────────────────────────
API_KEY   = os.getenv("OWM_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL  = "https://api.openweathermap.org/data/2.5"
HIST_URL  = "https://history.openweathermap.org/data/2.5/history/city"

# Target cities (name, lat, lon)
CITIES = [
    {"name": "London",    "lat": 51.5074, "lon": -0.1278},
    {"name": "New York",  "lat": 40.7128, "lon": -74.0060},
    {"name": "Tokyo",     "lat": 35.6895, "lon": 139.6917},
]

DB_PATH  = "weather_data.db"
CSV_DIR  = "csv_exports"
os.makedirs(CSV_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_raw (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city        TEXT    NOT NULL,
            ts          INTEGER NOT NULL,   -- Unix timestamp
            dt_utc      TEXT    NOT NULL,   -- Human-readable UTC
            temp_c      REAL,
            feels_like  REAL,
            temp_min    REAL,
            temp_max    REAL,
            humidity    INTEGER,
            pressure    INTEGER,
            wind_speed  REAL,
            wind_deg    INTEGER,
            weather_main TEXT,
            weather_desc TEXT,
            clouds      INTEGER,
            visibility  INTEGER,
            UNIQUE(city, ts)               -- Prevent duplicates
        )
    """)
    con.commit()
    con.close()
    log.info("Database initialised at %s", DB_PATH)


# ─── Extract ──────────────────────────────────────────────────────────────────
def fetch_current_weather(city: dict) -> dict | None:
    """
    Fetch current weather for a city.
    Free-tier fallback when historical API is unavailable.
    """
    url = f"{BASE_URL}/weather"
    params = {
        "lat":   city["lat"],
        "lon":   city["lon"],
        "appid": API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.error("Failed to fetch current weather for %s: %s", city["name"], e)
        return None


def fetch_historical_weather(city: dict, start_ts: int, end_ts: int) -> list[dict]:
    """
    Fetch historical hourly weather (requires paid OWM History API).
    Falls back to current reading on free tier.
    """
    url = HIST_URL
    params = {
        "lat":   city["lat"],
        "lon":   city["lon"],
        "type":  "hour",
        "start": start_ts,
        "end":   end_ts,
        "appid": API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("list", [])
    except requests.HTTPError as e:
        if resp.status_code == 401:
            log.warning(
                "Historical API requires a paid plan. "
                "Falling back to current weather for %s.", city["name"]
            )
        else:
            log.error("HTTP error for %s: %s", city["name"], e)
        return []
    except requests.RequestException as e:
        log.error("Request error for %s: %s", city["name"], e)
        return []


def parse_record(city_name: str, item: dict) -> dict:
    """Normalize a raw OWM JSON record into a flat dict."""
    main    = item.get("main", {})
    wind    = item.get("wind", {})
    weather = item.get("weather", [{}])[0]
    ts      = item.get("dt", int(time.time()))
    return {
        "city":         city_name,
        "ts":           ts,
        "dt_utc":       datetime.utcfromtimestamp(ts).isoformat(),
        "temp_c":       main.get("temp"),
        "feels_like":   main.get("feels_like"),
        "temp_min":     main.get("temp_min"),
        "temp_max":     main.get("temp_max"),
        "humidity":     main.get("humidity"),
        "pressure":     main.get("pressure"),
        "wind_speed":   wind.get("speed"),
        "wind_deg":     wind.get("deg"),
        "weather_main": weather.get("main"),
        "weather_desc": weather.get("description"),
        "clouds":       item.get("clouds", {}).get("all"),
        "visibility":   item.get("visibility"),
    }


# ─── Load ─────────────────────────────────────────────────────────────────────
def load_records(records: list[dict]):
    """Insert records into SQLite, ignoring duplicates."""
    if not records:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    inserted = 0
    for r in records:
        try:
            cur.execute("""
                INSERT OR IGNORE INTO weather_raw
                (city, ts, dt_utc, temp_c, feels_like, temp_min, temp_max,
                 humidity, pressure, wind_speed, wind_deg,
                 weather_main, weather_desc, clouds, visibility)
                VALUES
                (:city,:ts,:dt_utc,:temp_c,:feels_like,:temp_min,:temp_max,
                 :humidity,:pressure,:wind_speed,:wind_deg,
                 :weather_main,:weather_desc,:clouds,:visibility)
            """, r)
            inserted += cur.rowcount
        except sqlite3.Error as e:
            log.error("DB insert error: %s | record: %s", e, r)
    con.commit()
    con.close()
    log.info("Inserted %d new records (skipped duplicates).", inserted)


def export_csv():
    """Export the full table to a timestamped CSV."""
    con   = sqlite3.connect(DB_PATH)
    df    = pd.read_sql("SELECT * FROM weather_raw ORDER BY city, ts", con)
    con.close()
    fname = os.path.join(CSV_DIR, f"weather_{datetime.utcnow():%Y%m%d_%H%M%S}.csv")
    df.to_csv(fname, index=False)
    log.info("Exported %d rows → %s", len(df), fname)
    return fname


# ─── ETL Orchestrator ─────────────────────────────────────────────────────────
def run_etl(historical: bool = True):
    """
    Main ETL run.
    - historical=True  → tries to pull 6 months of hourly data (paid API)
    - historical=False → pulls current snapshot only (free API)
    """
    log.info("=== ETL run started at %s ===", datetime.utcnow().isoformat())
    init_db()

    now       = datetime.utcnow()
    start_dt  = now - timedelta(days=180)   # 6 months back
    start_ts  = int(start_dt.timestamp())
    end_ts    = int(now.timestamp())

    all_records: list[dict] = []

    for city in CITIES:
        log.info("Processing city: %s", city["name"])

        if historical:
            items = fetch_historical_weather(city, start_ts, end_ts)
        else:
            items = []

        # Fallback to current weather if historical returned nothing
        if not items:
            current = fetch_current_weather(city)
            if current:
                items = [current]

        records = [parse_record(city["name"], item) for item in items]
        log.info("  → %d records fetched", len(records))
        all_records.extend(records)

    load_records(all_records)
    csv_path = export_csv()
    log.info("=== ETL run complete. CSV: %s ===", csv_path)
    return csv_path


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_etl(historical=True)
