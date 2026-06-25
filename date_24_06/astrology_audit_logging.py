"""Audit logging for the astrology multi-agent pipeline.

A runnable script version of the notebook that adds a tamper-evident audit
trail across every agent in the horoscope service (input, astrology,
retrieval, interpretation, guardrails, chart render, orchestrator).

This is audit logging, not tracing. Traces (OpenTelemetry) are for debugging
latency; an audit log is the durable, append-only, PII-safe record of what
each agent did with someone's birth data.
"""

import contextvars
import hashlib
import hmac
import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps

import pandas as pd


ZODIAC = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]
random.seed(7)  # reproducible demo


try:  # display() is provided by Colab/Jupyter
    display  # noqa: F821
except NameError:  # fall back to print if run as a plain script
    display = print


# 1. The audit log core

AUDIT_SIGNING_KEY = b"demo-audit-key--load-from-a-secret-manager-in-prod"


class AuditLogger:
    def __init__(self, signing_key: bytes = AUDIT_SIGNING_KEY):
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._key = signing_key

    _CORE = (
        "seq",
        "timestamp",
        "request_id",
        "agent",
        "action",
        "status",
        "latency_ms",
        "severity",
        "attributes",
        "error",
        "prev_hash",
    )

    def _canonical(self, rec: dict) -> str:
        core = {k: rec[k] for k in self._CORE}
        return json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)

    def _hash(self, rec: dict) -> str:
        return hashlib.sha256(self._canonical(rec).encode()).hexdigest()

    def _sign(self, record_hash: str) -> str:
        return hmac.new(self._key, record_hash.encode(), hashlib.sha256).hexdigest()

    def append(
        self,
        *,
        agent,
        action,
        status,
        latency_ms,
        attributes,
        severity="info",
        error=None,
    ) -> dict:
        with self._lock:
            seq = len(self._records)
            prev_hash = self._records[-1]["record_hash"] if self._records else "GENESIS"
            rec = {
                "seq": seq,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": REQUEST_ID.get(),
                "agent": agent,
                "action": action,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "severity": severity,
                "attributes": attributes,
                "error": error,
                "prev_hash": prev_hash,
            }
            rec["record_hash"] = self._hash(rec)
            rec["signature"] = self._sign(rec["record_hash"])
            self._records.append(rec)
            return rec

    def verify_chain(self):
        prev = "GENESIS"
        for rec in self._records:
            if rec["prev_hash"] != prev:
                return False, rec["seq"], "broken link (prev_hash mismatch)"
            if self._hash(rec) != rec["record_hash"]:
                return False, rec["seq"], "payload altered (hash mismatch)"
            if self._sign(rec["record_hash"]) != rec["signature"]:
                return False, rec["seq"], "bad signature"
            prev = rec["record_hash"]
        return True, None, "chain intact"

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "seq": r["seq"],
                    "agent": r["agent"],
                    "action": r["action"],
                    "status": r["status"],
                    "sev": r["severity"],
                    "ms": r["latency_ms"],
                    "attributes": json.dumps(r["attributes"], default=str),
                    "hash": r["record_hash"][:10] + "\u2026",
                }
                for r in self._records
            ]
        )

    def export_jsonl(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, default=str) + "\n")
        return path


# 2. Request context + the @audited decorator

AUDIT = AuditLogger()
REQUEST_ID = contextvars.ContextVar("request_id", default="-")
_CURRENT_ATTRS = contextvars.ContextVar("current_attrs", default=None)


def audit_attr(**kw):
    """Attach PII-safe fields to the current audit event from inside an agent.

    Reserved keys: _status, _severity (let an agent mark an event blocked/warn).
    """

    d = _CURRENT_ATTRS.get()
    if d is not None:
        d.update(kw)


def audited(agent: str, action: str | None = None):
    def deco(fn):
        act = action or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            attrs: dict = {}
            tok = _CURRENT_ATTRS.set(attrs)
            t0 = time.perf_counter()
            status, severity, err = "ok", "info", None
            try:
                result = fn(*args, **kwargs)
                status = attrs.pop("_status", "ok")
                severity = attrs.pop("_severity", "info")
                return result
            except Exception as e:
                attrs.pop("_status", None)
                attrs.pop("_severity", None)
                status, severity, err = "error", "error", f"{type(e).__name__}: {e}"
                raise
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                AUDIT.append(
                    agent=agent,
                    action=act,
                    status=status,
                    latency_ms=dt,
                    attributes=dict(attrs),
                    severity=severity,
                    error=err,
                )
                _CURRENT_ATTRS.reset(tok)

        return wrapper

    return deco


# 3. PII redaction


def _h12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def redact_birth(birth: dict) -> dict:
    return {
        "name_hash": _h12(birth["name"]) if birth.get("name") else None,
        "year": int(str(birth["date"])[:4]),
        "date": str(birth["date"])[:4] + "-XX-XX",
        "time_known": bool(birth.get("time")),
        "place_hash": _h12(birth["place"]) if birth.get("place") else None,
        "lat_bucket": round(birth["latitude"]) if birth.get("latitude") is not None else None,
        "lon_bucket": round(birth["longitude"]) if birth.get("longitude") is not None else None,
        "tz": birth.get("timezone"),
    }


# 4. The agents (lightweight mocks of the real ones)

SHARED_PREFIX_TOKENS = 1500  # system + chart JSON, cached across the 12 houses
HOUSE_SPECIFIC_TOKENS = 1000  # per-house retrieved context + instruction


@audited("input_agent", "validate_geocode")
def input_agent(birth: dict) -> dict:
    audit_attr(birth=redact_birth(birth))  # redacted, never raw
    if not birth.get("date"):
        audit_attr(_status="blocked", _severity="warning")
        raise ValueError("missing birth date")
    geocoded = birth.get("latitude") is None
    resolved = {
        **birth,
        "latitude": birth.get("latitude", 18.52),
        "longitude": birth.get("longitude", 73.85),
        "timezone": birth.get("timezone", 5.5),
    }
    audit_attr(geocoded=geocoded, tz=resolved["timezone"])
    return resolved


@audited("astrology_agent", "fetch_chart")
def astrology_agent(resolved: dict) -> dict:
    audit_attr(source="AstrologyAPI.com", endpoint="western_horoscope", cache_hit=False, mock=True)
    time.sleep(0.04)  # simulate the external call
    seed = int(str(resolved["date"])[8:10] or 0) % 12
    houses = [{"house": i + 1, "sign": ZODIAC[(seed + i) % 12]} for i in range(12)]
    planets = [
        {"name": n, "house": (i % 12) + 1, "sign": ZODIAC[(i * 3) % 12]}
        for i, n in enumerate(["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"])
    ]
    audit_attr(planets=len(planets), houses=len(houses))
    return {"houses": houses, "planets": planets, "ascendant_sign": houses[0]["sign"]}


@audited("retrieval_agent", "retrieve")
def retrieve_house(chart: dict, house_no: int) -> list[str]:
    score = round(random.uniform(0.55, 0.92), 3)
    n = random.randint(2, 4)
    audit_attr(
        house=house_no,
        top_k=4,
        n_results=n,
        max_score=score,
        store="qdrant",
        embed_model="bge-small-en-v1.5",
    )
    if score < 0.60:
        audit_attr(_severity="warning", retrieval_low_confidence=True)
    return [f"passage {i} for house {house_no}" for i in range(n)]


@audited("interpretation_agent", "generate")
def interpret_house(chart: dict, house_no: int, passages: list[str], warm: bool) -> str:
    out = random.randint(300, 450)
    if warm:  # cache hit: only house-specific tokens billed
        cache_read, cache_write, billed_in = SHARED_PREFIX_TOKENS, 0, HOUSE_SPECIFIC_TOKENS
    else:  # cold write: full prefix billed once
        cache_read, cache_write, billed_in = 0, SHARED_PREFIX_TOKENS, SHARED_PREFIX_TOKENS + HOUSE_SPECIFIC_TOKENS
    time.sleep(0.02)
    audit_attr(
        model="claude-sonnet-4-6",
        house=house_no,
        input_tokens=billed_in,
        output_tokens=out,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cache_hit=warm,
    )
    cusp = next(h for h in chart["houses"] if h["house"] == house_no)
    return f"House {house_no} ({cusp['sign']}): " + " | ".join(passages)


@audited("guardrails", "house_output_check")
def guard_house(chart: dict, house_no: int, text: str) -> bool:
    cusp = next(h for h in chart["houses"] if h["house"] == house_no)
    real = cusp["sign"]
    passed = real in text  # deterministic chart-consistency check
    audit_attr(house=house_no, real_sign=real, passed=passed, decision="allow" if passed else "flag")
    if not passed:
        audit_attr(_status="blocked", _severity="warning")
    return passed


@audited("guardrails", "aggregation_check")
def guard_aggregate(chart: dict, readings: dict) -> bool:
    missing = [h for h in range(1, 13) if h not in readings]
    audit_attr(houses_present=len(readings), missing=missing, decision="allow" if not missing else "flag")
    if missing:
        audit_attr(_status="blocked", _severity="warning")
    return not missing


@audited("chart_agent", "render_svg")
def chart_agent(chart: dict) -> str:
    audit_attr(format="svg", planets=len(chart["planets"]))
    return "<svg><!-- natal wheel --></svg>"


# 5. Orchestrator - the 12-house parallel fan-out


def _run_in_ctx(fn, *a):
    return contextvars.copy_context().run(fn, *a)


def run_request(birth: dict) -> dict:
    REQUEST_ID.set("req_" + uuid.uuid4().hex[:8])
    resolved = input_agent(birth)
    chart = astrology_agent(resolved)

    # retrieval: one query per house, in parallel
    retrieved: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_run_in_ctx, retrieve_house, chart, h): h for h in range(1, 13)}
        for fut, h in futs.items():
            retrieved[h] = fut.result()

    # interpretation: WARM the cache with house 1, then fan out houses 2..12
    readings: dict[int, str] = {1: interpret_house(chart, 1, retrieved[1], False)}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {
            ex.submit(_run_in_ctx, interpret_house, chart, h, retrieved[h], True): h
            for h in range(2, 13)
        }
        for fut, h in futs.items():
            readings[h] = fut.result()

    # guardrails + render
    for h in range(1, 13):
        guard_house(chart, h, readings[h])
    guard_aggregate(chart, readings)
    chart_agent(chart)
    return readings


def main() -> None:
    # 6. Run it
    birth = {
        "name": "Ada Lovelace",
        "date": "1990-04-23",
        "time": "14:30",
        "place": "Pune, India",
        "timezone": 5.5,
    }
    readings = run_request(birth)
    print(f"Generated {len(readings)} house readings for request {AUDIT._records[-1]['request_id']}")
    print(f"Total audit events: {len(AUDIT._records)}")
    for reading in readings.values():
        print(reading)
    print("\nSample reading (house 10):\n", readings[10])

    # 7. The audit trail
    pd.set_option("display.max_colwidth", 70)
    display(AUDIT.frame())

    # 8. Per-agent metrics from the audit log
    recs = pd.DataFrame(AUDIT._records)
    summary = (
        recs.groupby("agent")
        .agg(
            events=("seq", "count"),
            errors=("status", lambda s: int((s != "ok").sum())),
            warnings=("severity", lambda s: int((s == "warning").sum())),
            mean_ms=("latency_ms", "mean"),
            p95_ms=("latency_ms", lambda s: float(s.quantile(0.95))),
        )
        .round(2)
        .sort_values("events", ascending=False)
    )
    display(summary)

    # Token & cache accounting pulled straight from the interpretation events
    interp = [r["attributes"] for r in AUDIT._records if r["agent"] == "interpretation_agent"]
    tot_in = sum(a["input_tokens"] for a in interp)
    tot_out = sum(a["output_tokens"] for a in interp)
    tot_read = sum(a["cache_read_tokens"] for a in interp)
    print(
        f"\nInterpretation: {len(interp)} houses | billed input {tot_in:,} tok | "
        f"output {tot_out:,} tok | served from cache {tot_read:,} tok"
    )
    print("Cache read tokens are the shared prefix the 11 warm calls did NOT re-bill.")

    # 9. Verify the chain, then tamper with it
    ok, seq, msg = AUDIT.verify_chain()
    print(f"Before tamper -> intact={ok}  ({msg})")

    # Simulate someone editing a past audit record to hide a token overage.
    victim = next(r for r in AUDIT._records if r["agent"] == "interpretation_agent")
    print(f"Tampering with seq {victim['seq']} (was output_tokens={victim['attributes']['output_tokens']})...")
    victim["attributes"]["output_tokens"] = 1

    ok, seq, msg = AUDIT.verify_chain()
    print(f"After tamper  -> intact={ok}  caught at seq={seq}  ({msg})")

    # 10. PII check - what actually got stored
    raw = birth
    stored = next(r for r in AUDIT._records if r["agent"] == "input_agent")["attributes"]["birth"]
    print("RAW input (in memory only):")
    print(json.dumps(raw, indent=2))
    print("\nSTORED in audit log (redacted):")
    print(json.dumps(stored, indent=2))

    # 11. Export the audit log
    path = AUDIT.export_jsonl("astrology_audit_log.jsonl")
    print("Wrote", path, "with", len(AUDIT._records), "records")
    print("\nFirst record:\n", json.dumps(AUDIT._records[0], indent=2)[:600], "...")

    try:
        from google.colab import files  # type: ignore

        files.download(path)
    except Exception:
        print("\n(Not in Colab - file saved locally at", path + ")")


if __name__ == "__main__":
    main()

