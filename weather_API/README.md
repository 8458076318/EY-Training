# Weather ETL Pipeline — OpenWeatherMap

## Project Structure
```
weather_etl/
├── etl_pipeline.py   # Core ETL logic (Extract → Transform → Load)
├── cron_job.py       # Scheduler (Python schedule library + OS cron notes)
├── requirements.txt
├── weather_data.db   # SQLite database (auto-created on first run)
└── csv_exports/      # Timestamped CSV snapshots (auto-created)
```

## Setup

### 1. Get an API Key
- Free tier: https://openweathermap.org/api (current weather only)
- Paid tier: needed for `/history` endpoint (6-month backfill)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API Key
```bash
export OWM_API_KEY="your_key_here"
```
Or replace `"YOUR_API_KEY_HERE"` in `etl_pipeline.py` directly.

### 4. Configure Cities
Edit the `CITIES` list in `etl_pipeline.py` to add/remove locations.

---

## Running

### One-off run (backfill + current snapshot):
```bash
python etl_pipeline.py
```

### Continuous scheduler (every hour + daily midnight backfill):
```bash
python cron_job.py
```

### OS Cron (Linux/Mac) — run every hour:
```
0 * * * * /usr/bin/python3 /path/to/etl_pipeline.py >> /path/to/cron.log 2>&1
```

---

## Output

| Location       | Description                              |
|----------------|------------------------------------------|
| `weather_data.db` | SQLite DB with `weather_raw` table    |
| `csv_exports/` | Timestamped CSV export after each run    |
| `etl.log`      | ETL run logs                             |
| `cron.log`     | Scheduler logs                           |

### `weather_raw` Table Schema
| Column         | Type    | Description                   |
|----------------|---------|-------------------------------|
| city           | TEXT    | City name                     |
| ts             | INTEGER | Unix timestamp                |
| dt_utc         | TEXT    | ISO datetime (UTC)            |
| temp_c         | REAL    | Temperature (°C)              |
| feels_like     | REAL    | Feels-like temperature (°C)   |
| humidity       | INTEGER | Humidity (%)                  |
| pressure       | INTEGER | Pressure (hPa)                |
| wind_speed     | REAL    | Wind speed (m/s)              |
| weather_main   | TEXT    | e.g. "Rain", "Clear"          |
| weather_desc   | TEXT    | e.g. "light rain"             |

---

## Next Steps
- **EDA**: Load `weather_data.db` into pandas for time series analysis
- **ML**: Use ARIMA, Prophet, or LSTM for forecasting
- **Azure**: Deploy on Azure Functions with a Timer Trigger for serverless cron
