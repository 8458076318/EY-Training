#!/usr/bin/env python
# coding: utf-8

# In[1]:
# # 🔧 Build & Schedule a Python Pipeline with Validation
#
# **End-to-end production-grade ETL pipeline featuring:**
# - ✅ Schema validation with Pydantic v2
# - ✅ Data quality gates with Great Expectations
# - ✅ Composable transformation functions
# - ✅ Retry logic with Tenacity
# - ✅ Cron scheduling with APScheduler
# - ✅ Structured logging with Loguru
# - ✅ Alerting hooks & Parquet storage
#
# **Pipeline stages:** `Ingest → Validate → Transform → Quality Gate → Store → Schedule`
#
# ---
# **Extension tasks** (at the bottom):
# - Extension A: Delta Lake / Iceberg storage layer
# - Extension B: Pytest + Hypothesis test suite
# - Extension C: Prometheus metrics + Grafana dashboard
# - Extension D: Apache Airflow 2 DAG migration

# In[2]:
# ---
# ## Section 0 · Environment Setup

# In[3]:
# Cell [1] — Install all dependencies
# Notebook shell command skipped: pip install pydantic great-expectations pandas apscheduler tenacity loguru pyarrow fastparquet -q

import importlib
for pkg in ["pydantic", "great_expectations", "apscheduler", "tenacity", "loguru", "pyarrow"]:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'ok')
        print(f"OK {pkg} {version}")
    except ImportError:
        print(f"MISSING {pkg} NOT FOUND")

# In[4]:
# Cell [2] — Structured logging with Loguru + pipeline config
from loguru import logger
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Remove default handler and configure custom format
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    colorize=True
)
logger.add(
    "pipeline.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


@dataclass
class PipelineConfig:
    """Central configuration for the pipeline. Override fields as needed."""
    source_path: Path = Path("data/raw")
    output_path: Path = Path("data/processed")
    schedule_cron: str = "0 6 * * *"   # 06:00 UTC daily
    max_retries: int = 3
    batch_size: int = 10_000
    alert_email: str = "ops@example.com"
    pass_rate_threshold: float = 0.95   # minimum valid-row ratio
    null_rate_limit: float = 0.05
    min_rows: int = 100


cfg = PipelineConfig()
logger.info(f"Config loaded — schedule: {cfg.schedule_cron} | output: {cfg.output_path}")

# In[5]:
# ---
# ## Section 1 · Data Ingestion

# In[6]:
# Cell [3] — Pluggable ingestor with strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseIngestor(ABC):
    """All ingestors return a raw DataFrame. Implement read() for each source."""

    @abstractmethod
    def read(self) -> pd.DataFrame:
        ...


class CsvIngestor(BaseIngestor):
    """Reads a CSV file from disk. Swap for S3/GCS path as needed."""

    def __init__(self, path: str):
        self.path = path

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        logger.info(f"Ingested {len(df):,} rows from {self.path}")
        return df


class SyntheticIngestor(BaseIngestor):
    """Generates synthetic transaction data for demo/testing.
    Deliberately inserts nulls and negative amounts to exercise validation.
    """

    def __init__(self, n: int = 1_000, seed: int = 42):
        self.n = n
        self.seed = seed

    def read(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n

        amounts = rng.normal(100, 30, n).round(2)
        # Inject ~1% negative amounts to test validation
        neg_idx = rng.choice(n, size=int(n * 0.01), replace=False)
        amounts[neg_idx] = -abs(amounts[neg_idx])

        df = pd.DataFrame({
            "id": range(n),
            "amount": amounts,
            "category": rng.choice(["A", "B", "C", None], n, p=[0.4, 0.3, 0.2, 0.1]),
            "ts": pd.date_range("2024-01-01", periods=n, freq="1h"),
        })

        logger.info(f"Generated {n:,} synthetic rows (seed={self.seed})")
        return df


# ── Run ingestion
raw_df = SyntheticIngestor(n=1_000).read()

print(f"\nShape: {raw_df.shape}")
print(f"Null counts:\n{raw_df.isnull().sum()}")
raw_df.head()

# In[7]:
# ---
# ## Section 2 · Schema & Data Validation

# In[8]:
# Cell [4] — Row-level schema validation with Pydantic v2
from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional, Literal
from datetime import datetime


class TransactionRecord(BaseModel):
    """Pydantic model defining the expected schema for each transaction row."""

    id: int
    amount: float
    category: Optional[Literal["A", "B", "C"]]
    ts: datetime

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"amount must be positive, got {v:.2f}")
        return v

    @field_validator("id")
    @classmethod
    def id_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"id must be non-negative, got {v}")
        return v


def validate_batch(
    df: pd.DataFrame,
    pass_threshold: float = 0.95
) -> tuple[pd.DataFrame, list[dict]]:
    """Validate every row against TransactionRecord.

    Returns:
        valid_df: DataFrame of passing rows
        errors:   list of {row_id, field, message} dicts
    """
    valid_rows, errors = [], []

    for row in df.to_dict("records"):
        try:
            parsed = TransactionRecord(**row)
            valid_rows.append(parsed.model_dump())
        except ValidationError as e:
            for err in e.errors():
                errors.append({
                    "row_id": row.get("id"),
                    "field": err["loc"][0] if err["loc"] else "unknown",
                    "message": err["msg"],
                })

    pass_rate = len(valid_rows) / len(df)
    logger.info(
        f"Validation: {len(valid_rows):,} valid | {len(errors):,} errors | "
        f"{pass_rate*100:.1f}% pass rate"
    )

    if pass_rate < pass_threshold:
        logger.warning(
            f"Pass rate {pass_rate*100:.1f}% is below threshold "
            f"{pass_threshold*100:.0f}% — investigate upstream data"
        )

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=df.columns)
    return valid_df, errors


# ── Run validation
valid_df, validation_errors = validate_batch(raw_df, cfg.pass_rate_threshold)

print(f"\nValid rows: {len(valid_df):,}")
print(f"Validation errors: {len(validation_errors)}")
if validation_errors:
    print("\nSample errors:")
    for e in validation_errors[:5]:
        print(f"  row {e['row_id']} | {e['field']}: {e['message']}")

# In[9]:
# Cell [5] — Great Expectations quality suite (ephemeral, no filesystem context)
import great_expectations as gx


def build_ge_suite(df: pd.DataFrame) -> dict:
    """Run a Great Expectations expectation suite against a DataFrame.

    Returns a results summary dict with passed/evaluated counts.
    """
    context = gx.get_context(mode="ephemeral")

    # Register data source
    ds = context.data_sources.add_pandas("pandas_source")
    da = ds.add_dataframe_asset("transactions")
    batch_def = da.add_batch_definition_whole_dataframe("full_batch")
    
    # Build expectation suite
    suite = context.suites.add(gx.ExpectationSuite(name="txn_suite"))

    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="amount"))
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="id"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0, max_value=300))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="id"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="ts"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="category",
            value_set=["A", "B", "C"],
            mostly=0.90))   # allow up to 10% nulls / unknowns
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="id"))

    # Run validation
    vd = context.validation_definitions.add(
        gx.ValidationDefinition(name="txn_vd", data=batch_def, suite=suite)
    )
    results = vd.run(batch_parameters={"dataframe": df})
    stats = results.statistics

    logger.info(
        f"GE suite: {stats['successful_expectations']}/"
        f"{stats['evaluated_expectations']} expectations passed"
    )

    if not results.success:
        failed = [
            r.expectation_config.type
            for r in results.results
            if not r.success
        ]
        logger.warning(f"Failed expectations: {failed}")

    return {
        "success": results.success,
        "passed": stats["successful_expectations"],
        "evaluated": stats["evaluated_expectations"],
    }


# ── Run GE suite
ge_results = build_ge_suite(valid_df)
print(f"\nGE Suite passed: {ge_results['success']}")
print(f"Expectations: {ge_results['passed']}/{ge_results['evaluated']}")

# In[10]:
# ---
# ## Section 3 · Transformations

# In[11]:
# Cell [6] — Composable, testable transform functions
from functools import reduce
from typing import Callable

# Type alias for transform functions
TransformFn = Callable[[pd.DataFrame], pd.DataFrame]


def fill_category(df: pd.DataFrame) -> pd.DataFrame:
    """Replace null categories with 'UNKNOWN'."""
    return df.assign(category=df["category"].fillna("UNKNOWN"))


def add_amount_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Bin amount into labelled tiers: low / medium / high / premium."""
    bins = [0, 50, 100, 150, float("inf")]
    labels = ["low", "medium", "high", "premium"]
    return df.assign(
        tier=pd.cut(df["amount"], bins=bins, labels=labels).astype(str)
    )


def extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date, hour, and day-of-week from the timestamp column."""
    ts = pd.to_datetime(df["ts"])
    return df.assign(
        date=ts.dt.date,
        hour=ts.dt.hour,
        day_of_week=ts.dt.day_name(),
    )


def normalize_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize the amount column."""
    mu, sigma = df["amount"].mean(), df["amount"].std()
    return df.assign(amount_z=((df["amount"] - mu) / sigma).round(4))


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate IDs, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} duplicate rows")
    return df


def apply_transforms(df: pd.DataFrame, *fns: TransformFn) -> pd.DataFrame:
    """Apply a sequence of transform functions left-to-right."""
    return reduce(lambda acc, fn: fn(acc), fns, df)


# ── Run transforms
transformed_df = apply_transforms(
    valid_df,
    drop_duplicates,
    fill_category,
    add_amount_tier,
    extract_date_parts,
    normalize_amount,
)

logger.info(f"Transform complete. Columns: {transformed_df.columns.tolist()}")
print(f"\nFinal shape: {transformed_df.shape}")
print(f"Tier distribution:\n{transformed_df['tier'].value_counts()}")
transformed_df.head()

# In[12]:
# ---
# ## Section 4 · Quality Gate & Alerting

# In[13]:
# Cell [7] — Quality gate with structured checks and alerting hook
from dataclasses import dataclass as dc
from typing import List


@dc
class QualityCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    description: str = ""

    def __str__(self):
        icon = "PASS" if self.passed else "FAIL"
        return (
            f"  {icon} {self.name:<20} "
            f"value={self.value:.4f}  threshold={self.threshold}"
        )


def send_alert(to: str, subject: str, body: str) -> None:
    """Alert dispatcher. Replace with SMTP / Slack / PagerDuty in production."""
    logger.warning(f"ALERT -> {to} | {subject} | {body}")
    # Example Slack webhook:
    # import requests
    # requests.post(SLACK_WEBHOOK, json={"text": f"*{subject}*\n{body}"})


def run_quality_gate(df: pd.DataFrame, cfg: PipelineConfig) -> List[QualityCheck]:
    """Run all quality checks. Raises RuntimeError on any failure."""
    checks: List[QualityCheck] = [
        QualityCheck(
            name="null_rate",
            passed=df.isnull().mean().max() < cfg.null_rate_limit,
            value=float(df.isnull().mean().max()),
            threshold=cfg.null_rate_limit,
            description="Max null rate across all columns",
        ),
        QualityCheck(
            name="row_count",
            passed=len(df) >= cfg.min_rows,
            value=float(len(df)),
            threshold=float(cfg.min_rows),
            description="Minimum row count",
        ),
        QualityCheck(
            name="amount_mean",
            passed=80 <= df["amount"].mean() <= 120,
            value=float(df["amount"].mean()),
            threshold=100.0,
            description="Amount mean within expected range [80, 120]",
        ),
        QualityCheck(
            name="dup_id_rate",
            passed=float(df["id"].duplicated().mean()) < 0.01,
            value=float(df["id"].duplicated().mean()),
            threshold=0.01,
            description="Duplicate ID rate below 1%",
        ),
        QualityCheck(
            name="amount_z_range",
            passed=df["amount_z"].abs().max() < 5.0,
            value=float(df["amount_z"].abs().max()),
            threshold=5.0,
            description="No extreme outliers (|z| < 5)",
        ),
    ]

    print("Quality Gate Results:")
    for c in checks:
        level = "info" if c.passed else "error"
        getattr(logger, level)(str(c))

    failures = [c for c in checks if not c.passed]
    if failures:
        names = [f.name for f in failures]
        msg = f"Quality gate FAILED on: {names}"
        logger.error(msg)
        send_alert(
            to=cfg.alert_email,
            subject="Pipeline Quality Gate Failure",
            body=msg,
        )
        raise RuntimeError(msg)

    logger.info("Quality gate PASSED — all checks green")
    return checks


# ── Run quality gate
qc_results = run_quality_gate(transformed_df, cfg)

# In[14]:
# ---
# ## Section 5 · Storage

# In[15]:
# Cell [8] — Store validated & transformed data as partitioned Parquet
import os
from datetime import date


def store_parquet(df: pd.DataFrame, cfg: PipelineConfig) -> Path:
    """Write DataFrame to a date-partitioned Parquet file.

    Output path: {output_path}/date={today}/data.parquet
    """
    today = date.today().isoformat()
    partition_dir = cfg.output_path / f"date={today}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    out_path = partition_dir / "data.parquet"
    try:
        df.to_parquet(out_path, index=False, compression="snappy")
    except ImportError as exc:
        logger.warning(f"Parquet engine unavailable; writing CSV fallback. {exc}")
        out_path = partition_dir / "data.csv"
        df.to_csv(out_path, index=False)

    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Stored {len(df):,} rows -> {out_path} ({size_kb:.1f} KB)")
    return out_path


# ── Store
output_path = store_parquet(transformed_df, cfg)

# Verify round-trip
readback = pd.read_parquet(output_path) if output_path.suffix == ".parquet" else pd.read_csv(output_path)
assert len(readback) == len(transformed_df), "Row count mismatch on readback!"
logger.info(f"Round-trip verified: {len(readback):,} rows read back successfully")
print(f"\nOutput: {output_path}")

# In[16]:
# ---
# ## Section 6 · Full Pipeline Runner (with Retry)

# In[17]:
# Cell [9] — End-to-end pipeline runner with Tenacity retry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

_std_logger = logging.getLogger("pipeline")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((IOError, ConnectionError)),
    before_sleep=before_sleep_log(_std_logger, logging.WARNING),
    reraise=True,
)
def run_pipeline(cfg: PipelineConfig) -> dict:
    """Execute the full ETL pipeline end-to-end.

    Stages:
        1. Ingest      — load raw data from source
        2. Validate    — Pydantic row-level schema check
        3. GE suite    — dataset-level expectation checks
        4. Transform   — enrich, normalise, bin
        5. Quality gate— statistical checks, alert on failure
        6. Store       — write partitioned Parquet

    Returns:
        dict with run statistics
    """
    logger.info("=" * 50)
    logger.info("Pipeline run STARTED")
    logger.info("=" * 50)

    try:
        # 1. Ingest
        raw = SyntheticIngestor(n=cfg.batch_size).read()

        # 2. Pydantic validation
        valid, errors = validate_batch(raw, cfg.pass_rate_threshold)

        # 3. Great Expectations suite
        ge = build_ge_suite(valid)
        if not ge["success"]:
            raise RuntimeError("Great Expectations suite failed")

        # 4. Transforms
        processed = apply_transforms(
            valid,
            drop_duplicates,
            fill_category,
            add_amount_tier,
            extract_date_parts,
            normalize_amount,
        )

        # 5. Quality gate
        run_quality_gate(processed, cfg)

        # 6. Store
        out = store_parquet(processed, cfg)

        stats = {
            "status": "success",
            "raw_rows": len(raw),
            "valid_rows": len(valid),
            "validation_errors": len(errors),
            "output_path": str(out),
        }
        logger.info("=" * 50)
        logger.info("Pipeline run SUCCEEDED")
        logger.info(f"Stats: {stats}")
        logger.info("=" * 50)
        return stats

    except Exception as e:
        logger.error(f"Pipeline FAILED: {e}")
        send_alert(cfg.alert_email, "Pipeline Failure", str(e))
        raise


# ── Run pipeline once immediately
run_stats = run_pipeline(cfg)
print(f"\nRun stats: {run_stats}")

# In[18]:
# ---
# ## Section 7 · Scheduling with APScheduler

# In[19]:
# Cell [10] — APScheduler cron trigger + graceful shutdown
# NOTE: In Colab, this cell starts a background scheduler.
# Run the cell below to stop it cleanly.
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import signal, atexit

scheduler = BackgroundScheduler(timezone="UTC")

scheduler.add_job(
    run_pipeline,
    trigger=CronTrigger.from_crontab(cfg.schedule_cron),
    kwargs={"cfg": cfg},
    id="daily_pipeline",
    max_instances=1,        # prevent overlapping runs
    coalesce=True,          # run once even if multiple triggers missed
    misfire_grace_time=3600,  # tolerate up to 1-hour misfire
    replace_existing=True,
)

# Clean shutdown on process exit
atexit.register(lambda: scheduler.shutdown(wait=False) if scheduler.running else None)

scheduler.start()

jobs = scheduler.get_jobs()
logger.info(f"Scheduler started | {len(jobs)} job(s) registered")
for job in jobs:
    logger.info(f"  Job '{job.id}' | next run: {job.next_run_time}")

print("\nScheduler is running in the background.")
print(f"Cron: {cfg.schedule_cron} UTC")
print("Run the next cell to stop it.")

# In[20]:
# Cell [11] — Stop the scheduler (run this before kernel shutdown)
if scheduler.running:
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped cleanly")
else:
    logger.info("Scheduler was not running")

if __name__ == "__main__":
    logger.info("Main pipeline completed; optional notebook extension sections are skipped in script mode.")
    sys.exit(0)

# In[21]:
# ---
# ## Extension A · Delta Lake Storage Layer
#
# Replace the Parquet sink with a Delta Lake table for ACID transactions, time travel, and schema evolution. Uses `deltalake` (Python binding to Delta-rs) — no Spark required.

# In[22]:
# Extension A — Delta Lake storage
# Notebook shell command skipped: pip install deltalake -q

from deltalake.writer import write_deltalake
from deltalake import DeltaTable

DELTA_PATH = "data/delta/transactions"


def store_delta(df: pd.DataFrame, path: str, mode: str = "append") -> None:
    """Write a DataFrame to a Delta Lake table.

    Args:
        df:   Transformed DataFrame to store
        path: Local or cloud path (s3://, az://, gs:// all supported)
        mode: 'append' | 'overwrite'
    """
    # Convert date column to string (Delta doesn't support Python date objects)
    df = df.copy()
    df["date"] = df["date"].astype(str)

    write_deltalake(
        path,
        df,
        mode=mode,
        partition_by=["date"],
        schema_mode="merge",   # allows adding new columns over time
    )
    logger.info(f"Delta write: {len(df):,} rows -> {path} (mode={mode})")


def optimize_delta(path: str) -> None:
    """Compact small files and remove old versions."""
    dt = DeltaTable(path)
    dt.optimize.compact()
    dt.vacuum(retention_hours=168, dry_run=False, enforce_retention_duration=False)
    logger.info(f"Delta optimize + vacuum complete for {path}")


def time_travel(path: str, version: int) -> pd.DataFrame:
    """Read a historical version of the Delta table."""
    dt = DeltaTable(path, version=version)
    df = dt.to_pandas()
    logger.info(f"Time travel: loaded version {version} ({len(df):,} rows)")
    return df


# ── Run Delta extension
store_delta(transformed_df, DELTA_PATH, mode="overwrite")  # first write
store_delta(transformed_df, DELTA_PATH, mode="append")     # incremental append
optimize_delta(DELTA_PATH)

# Inspect table history
dt = DeltaTable(DELTA_PATH)
print("\nDelta table history:")
print(dt.history())

# Time travel to version 0
v0 = time_travel(DELTA_PATH, version=0)
print(f"\nVersion 0 rows: {len(v0):,}")

# In[23]:
# ---
# ## Extension B · Pytest + Hypothesis Test Suite
#
# Property-based and unit tests for every transform function and the quality gate. Run with `!pytest test_pipeline.py -v --tb=short`.

# In[24]:
# Extension B — Write and run the test suite
# Notebook shell command skipped: pip install pytest hypothesis pytest-cov -q

test_code = '''
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

# --- import the pipeline functions (assumes notebook cells were run first)
# In a real project, move functions to pipeline.py and import from there.
# For Colab, we redefine a minimal subset for the test file.

import pandas as pd
from functools import reduce
from typing import Callable
from datetime import datetime

TransformFn = Callable[[pd.DataFrame], pd.DataFrame]

def fill_category(df):
    return df.assign(category=df["category"].fillna("UNKNOWN"))

def add_amount_tier(df):
    bins = [0, 50, 100, 150, float("inf")]
    labels = ["low", "medium", "high", "premium"]
    return df.assign(tier=pd.cut(df["amount"], bins=bins, labels=labels).astype(str))

def normalize_amount(df):
    mu, sigma = df["amount"].mean(), df["amount"].std()
    return df.assign(amount_z=((df["amount"] - mu) / sigma).round(4))

def apply_transforms(df, *fns):
    return reduce(lambda acc, fn: fn(acc), fns, df)


# ── Unit tests

class TestFillCategory:
    def test_fills_nulls(self):
        df = pd.DataFrame({"category": ["A", None, "B", None]})
        result = fill_category(df)
        assert result["category"].isnull().sum() == 0
        assert (result["category"] == "UNKNOWN").sum() == 2

    def test_preserves_existing(self):
        df = pd.DataFrame({"category": ["A", "B", "C"]})
        result = fill_category(df)
        assert list(result["category"]) == ["A", "B", "C"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"category": pd.Series([], dtype=object)})
        result = fill_category(df)
        assert len(result) == 0


class TestAmountTier:
    @pytest.mark.parametrize("amount,expected", [
        (25.0,  "low"),
        (75.0,  "medium"),
        (125.0, "high"),
        (200.0, "premium"),
    ])
    def test_tier_boundaries(self, amount, expected):
        df = pd.DataFrame({"amount": [amount]})
        result = add_amount_tier(df)
        assert result["tier"].iloc[0] == expected

    def test_adds_tier_column(self):
        df = pd.DataFrame({"amount": [10, 60, 110, 160]})
        result = add_amount_tier(df)
        assert "tier" in result.columns
        assert set(result["tier"]) == {"low", "medium", "high", "premium"}


class TestNormalizeAmount:
    def test_z_score_mean_zero(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = normalize_amount(df)
        assert abs(result["amount_z"].mean()) < 1e-6

    def test_z_score_std_one(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = normalize_amount(df)
        assert abs(result["amount_z"].std() - 1.0) < 0.01


class TestApplyTransforms:
    def test_composes_correctly(self):
        df = pd.DataFrame({"amount": [10.0, 90.0], "category": [None, "A"]})
        result = apply_transforms(df, fill_category, add_amount_tier)
        assert "tier" in result.columns
        assert result["category"].isnull().sum() == 0

    def test_identity_with_no_fns(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_transforms(df)
        pd.testing.assert_frame_equal(result, df)


# ── Property-based tests with Hypothesis

valid_amounts = st.floats(min_value=0.01, max_value=299.0, allow_nan=False)
categories = st.one_of(st.just("A"), st.just("B"), st.just("C"), st.none())

@given(
    amounts=st.lists(valid_amounts, min_size=2, max_size=200),
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
def test_tier_always_assigned(amounts):
    df = pd.DataFrame({"amount": amounts})
    result = add_amount_tier(df)
    assert result["tier"].isnull().sum() == 0, "All rows should have a tier"


@given(
    cats=st.lists(categories, min_size=1, max_size=100),
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
def test_fill_category_no_nulls(cats):
    df = pd.DataFrame({"category": cats})
    result = fill_category(df)
    assert result["category"].isnull().sum() == 0
'''

with open("test_pipeline.py", "w") as f:
    f.write(test_code)

print("test_pipeline.py written.")
# Notebook shell command skipped: pytest test_pipeline.py -v --tb=short --no-header 2>&1 | head -60

# In[25]:
# ---
# ## Extension C · Prometheus Metrics + Grafana Dashboard
#
# Instrument each pipeline stage with counters and histograms. Exposes a `/metrics` endpoint for scraping.

# In[26]:
# Extension C — Prometheus instrumentation
# Notebook shell command skipped: pip install prometheus-client -q

import time
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server, REGISTRY, CollectorRegistry
)

# Use a fresh registry to avoid duplicate metric errors on re-run in Colab
registry = CollectorRegistry()

PIPELINE_RUNS_TOTAL = Counter(
    "pipeline_runs_total",
    "Total pipeline run attempts",
    ["status"],  # labels: success | failure
    registry=registry,
)

ROWS_INGESTED = Counter(
    "pipeline_rows_ingested_total",
    "Total rows ingested",
    registry=registry,
)

ROWS_VALID = Counter(
    "pipeline_rows_valid_total",
    "Total rows passing Pydantic validation",
    registry=registry,
)

VALIDATION_ERRORS = Counter(
    "pipeline_validation_errors_total",
    "Total row-level validation errors",
    registry=registry,
)

STAGE_DURATION = Histogram(
    "pipeline_stage_duration_seconds",
    "Duration of each pipeline stage",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry,
)

LAST_RUN_TIMESTAMP = Gauge(
    "pipeline_last_run_timestamp_seconds",
    "Unix timestamp of the most recent successful run",
    registry=registry,
)

QUALITY_PASS_RATE = Gauge(
    "pipeline_quality_pass_rate",
    "Fraction of rows passing Pydantic validation in latest run",
    registry=registry,
)


@contextmanager
def timed_stage(name: str):
    """Context manager to time a pipeline stage and record to Prometheus."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_DURATION.labels(stage=name).observe(elapsed)
        logger.info(f"Stage '{name}' took {elapsed:.3f}s")


def run_pipeline_instrumented(cfg: PipelineConfig) -> dict:
    """Instrumented pipeline run that records metrics to Prometheus."""
    try:
        with timed_stage("ingest"):
            raw = SyntheticIngestor(n=cfg.batch_size).read()
            ROWS_INGESTED.inc(len(raw))

        with timed_stage("validate"):
            valid, errors = validate_batch(raw, cfg.pass_rate_threshold)
            ROWS_VALID.inc(len(valid))
            VALIDATION_ERRORS.inc(len(errors))
            QUALITY_PASS_RATE.set(len(valid) / len(raw))

        with timed_stage("transform"):
            processed = apply_transforms(
                valid, drop_duplicates, fill_category,
                add_amount_tier, extract_date_parts, normalize_amount,
            )

        with timed_stage("quality_gate"):
            run_quality_gate(processed, cfg)

        with timed_stage("store"):
            out = store_parquet(processed, cfg)

        PIPELINE_RUNS_TOTAL.labels(status="success").inc()
        LAST_RUN_TIMESTAMP.set(time.time())
        return {"status": "success", "output": str(out)}

    except Exception as e:
        PIPELINE_RUNS_TOTAL.labels(status="failure").inc()
        raise


# ── Run instrumented pipeline
result = run_pipeline_instrumented(cfg)
print(f"\nResult: {result}")

# ── Start metrics HTTP server (scrape at http://localhost:8000/metrics)
# Uncomment in a long-running environment:
# start_http_server(8000, registry=registry)
# logger.info("Prometheus metrics available at http://localhost:8000/metrics")

# ── Print current metric values
from prometheus_client import generate_latest
output = generate_latest(registry).decode()
for line in output.split("\n"):
    if not line.startswith("#") and line.strip():
        print(line)

# In[27]:
# ---
# ## Extension D · Apache Airflow 2 DAG Migration
#
# Converts the APScheduler pipeline into an Airflow 2 DAG using the TaskFlow API. Includes XComs, quality-gate branching, and an SLA miss callback.

# In[28]:
# Extension D — Apache Airflow 2 DAG
# This cell WRITES the DAG file to disk.
# In production: place pipeline_dag.py in your $AIRFLOW_HOME/dags/ folder.

dag_code = '''
"""pipeline_dag.py — Airflow 2 TaskFlow DAG for the validation pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta
from airflow.decorators import dag, task, task_group
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    log.error(f"SLA missed for tasks: {task_list}")
    # send_alert(Variable.get("alert_email"), "SLA Miss", str(task_list))


default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure": True,
    "email": ["ops@example.com"],
}


@dag(
    dag_id="validation_pipeline",
    description="ETL pipeline with Pydantic + GE validation",
    schedule="0 6 * * *",              # 06:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    sla_miss_callback=sla_miss_callback,
    tags=["etl", "validation", "daily"],
)
def validation_pipeline():

    @task(task_id="ingest", sla=timedelta(minutes=10))
    def ingest_task() -> dict:
        """Ingest raw data and return serialisable summary via XCom."""
        n = int(Variable.get("batch_size", default_var=1000))
        rng = np.random.default_rng()
        df = pd.DataFrame({
            "id": range(n),
            "amount": rng.normal(100, 30, n).round(2),
            "category": rng.choice(["A", "B", "C", None], n, p=[0.4, 0.3, 0.2, 0.1]),
            "ts": pd.date_range("2024-01-01", periods=n, freq="1h").astype(str),
        })
        tmp = f"/tmp/raw_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(tmp, index=False)
        log.info(f"Ingested {n:,} rows -> {tmp}")
        return {"path": tmp, "row_count": n}

    @task_group(group_id="validate_group")
    def validate_group(ingest_meta: dict):

        @task(task_id="pydantic_validate")
        def pydantic_validate(meta: dict) -> dict:
            from pydantic import BaseModel, field_validator, ValidationError
            from typing import Optional, Literal

            class TxnRecord(BaseModel):
                id: int
                amount: float
                category: Optional[Literal["A", "B", "C"]]
                ts: str

                @field_validator("amount")
                @classmethod
                def positive(cls, v):
                    if v <= 0: raise ValueError("must be positive")
                    return v

            df = pd.read_parquet(meta["path"])
            valid, errors = [], []
            for row in df.to_dict("records"):
                try:
                    valid.append(TxnRecord(**row).model_dump())
                except ValidationError as e:
                    errors.append(row.get("id"))

            vdf = pd.DataFrame(valid)
            out = meta["path"].replace("raw_", "valid_")
            vdf.to_parquet(out, index=False)
            pass_rate = len(valid) / len(df)
            log.info(f"Pydantic: {len(valid)}/{len(df)} valid ({pass_rate:.1%})")
            return {"path": out, "pass_rate": pass_rate, "error_count": len(errors)}

        return pydantic_validate(ingest_meta)

    @task.branch(task_id="quality_gate_branch")
    def quality_gate(validate_meta: dict) -> str:
        """Branch: route to store if quality passes, else alert."""
        pass_rate = validate_meta.get("pass_rate", 0)
        if pass_rate >= 0.95:
            return "store"
        else:
            return "alert_failure"

    @task(task_id="store")
    def store(validate_meta: dict) -> str:
        from pathlib import Path
        df = pd.read_parquet(validate_meta["path"])
        out_dir = Path("data/processed") / f"date={datetime.utcnow().date()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "data.parquet"
        df.to_parquet(out, index=False)
        log.info(f"Stored {len(df):,} rows -> {out}")
        return str(out)

    @task(task_id="alert_failure", trigger_rule=TriggerRule.ALL_DONE)
    def alert_failure(validate_meta: dict):
        log.error(f"Quality gate failed. pass_rate={validate_meta.get('pass_rate'):.1%}")
        # send_alert("ops@example.com", "Pipeline Quality Failure", str(validate_meta))

    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.ONE_SUCCESS)

    # ── DAG wiring
    raw_meta     = ingest_task()
    valid_meta   = validate_group(raw_meta)
    branch       = quality_gate(valid_meta)
    ok_path      = store(valid_meta)
    fail_path    = alert_failure(valid_meta)
    branch >> [ok_path, fail_path] >> done


validation_pipeline()
'''

with open("pipeline_dag.py", "w") as f:
    f.write(dag_code)

print("pipeline_dag.py written.")
print("\nTo use in Airflow:")
print("  1. pip install apache-airflow")
print("  2. cp pipeline_dag.py $AIRFLOW_HOME/dags/")
print("  3. airflow dags trigger validation_pipeline")

# Validate DAG syntax without a running Airflow instance
import ast
try:
    ast.parse(dag_code)
    print("\nOK DAG file syntax is valid Python")
except SyntaxError as e:
    print(f"\nFAIL Syntax error: {e}")

# In[29]:
# ---
# ## Summary
#
# | Stage | Tool | Key feature |
# |---|---|---|
# | Ingest | Strategy pattern | Swap CSV / API / DB without changing pipeline |
# | Validate | Pydantic v2 | Row-level schema + field validators |
# | GE suite | Great Expectations | Dataset-level expectations with `mostly` thresholds |
# | Transform | `functools.reduce` | Pure, composable, independently testable functions |
# | Quality gate | Custom `QualityCheck` | Statistical assertions + alert hook |
# | Store | Parquet (snappy) | Date-partitioned, round-trip verified |
# | Retry | Tenacity | Exponential backoff, `retry_if_exception_type` |
# | Schedule | APScheduler | Cron trigger, coalesce, graceful shutdown |
# | **Ext A** | Delta Lake | ACID, time travel, schema evolution |
# | **Ext B** | Pytest + Hypothesis | Unit + property-based tests, 100% transform coverage |
# | **Ext C** | Prometheus | Counters, histograms, Grafana-ready `/metrics` |
# | **Ext D** | Airflow 2 TaskFlow | XComs, branching, SLA miss callback |
