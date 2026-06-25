"""Day 18: Colab 2 - Multi-tool Agent Chaining + Redis Event Queue.

Converted from `Day18_Colab2_Chaining_Redis_Event_Queue.ipynb` into a plain
Python script that:

- loads the repo-root `.env` file automatically,
- keeps the SQLite order workflow from the notebook,
- uses `fakeredis` when available and a small in-process fallback otherwise,
- runs the Anthropic tool loop when `ANTHROPIC_API_KEY` is available,
- falls back to a deterministic offline mock when the API key or live call
  is unavailable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


def _load_repo_env() -> Path | None:
    """Load key=value pairs from the repo-root `.env` file.

    The repo-local `.env` is the source of truth for this workspace, so we load
    it before reading any provider settings. Existing process values are
    overwritten intentionally to keep notebook-derived behavior consistent.
    """

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

    return env_path


_ENV_PATH = _load_repo_env()


try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional for offline demo mode
    Anthropic = None


def _build_stream_backend():
    """Prefer fakeredis, otherwise use a tiny local stream-capable fallback."""

    try:
        import fakeredis

        return fakeredis.FakeStrictRedis()
    except Exception:
        return _MiniStreamRedis()


class _MiniStreamRedis:
    """Small Redis-like stand-in for the demo path.

    It only implements the commands used by this notebook: `xgroup_create`,
    `xadd`, `xreadgroup`, `xack`, `set`, and `get`.
    """

    def __init__(self):
        self._kv: dict[str, bytes] = {}
        self._streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self._groups: dict[tuple[str, str], dict[str, int]] = {}

    def xgroup_create(self, stream: str, group: str, id: str = "0", mkstream: bool = True):
        key = (stream, group)
        if key in self._groups:
            raise ValueError("group exists")
        if mkstream:
            self._streams.setdefault(stream, [])
        self._groups[key] = {"cursor": 0}
        return True

    def xadd(self, stream: str, fields: dict[str, str]):
        self._streams.setdefault(stream, [])
        msg_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        payload = {str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in fields.items()}
        self._streams[stream].append((msg_id, payload))
        return msg_id

    def xreadgroup(self, group: str, consumer: str, streams: dict[str, str], count: int = 10):
        out = []
        for stream, requested_id in streams.items():
            if requested_id != ">":
                continue
            state = self._groups.setdefault((stream, group), {"cursor": 0})
            messages = self._streams.get(stream, [])
            cursor = state["cursor"]
            chunk = messages[cursor : cursor + count]
            if chunk:
                state["cursor"] = cursor + len(chunk)
                out.append((stream, [(msg_id, fields) for msg_id, fields in chunk]))
        return out

    def xack(self, stream: str, group: str, msg_id: str):
        return 1

    def set(self, key: str, value: str):
        self._kv[key] = str(value).encode("utf-8")
        return True

    def get(self, key: str):
        return self._kv.get(key)


# Step 1 - Configure the agent runtime.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LIVE = bool(ANTHROPIC_API_KEY)
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


# Step 2 - SQLite orders + inventory database.
db = sqlite3.connect(":memory:", check_same_thread=False)
db.execute("CREATE TABLE inventory (sku TEXT PRIMARY KEY, name TEXT, qty INTEGER, price REAL)")
db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT, qty INTEGER, total REAL, status TEXT)")
db.executemany(
    "INSERT INTO inventory VALUES (?,?,?,?)",
    [
        ("KB-01", "Mechanical keyboard", 12, 129.0),
        ("HUB-2", "USB-C hub", 0, 58.0),
        ("MON-4", "4K monitor", 5, 410.0),
    ],
)
db.commit()


# Step 3 - A Redis Stream as the email job queue.
r = _build_stream_backend()
STREAM = "emails"
GROUP = "mailers"
try:
    r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
except Exception as exc:
    print("group exists:", exc)


def enqueue_email(to: str, subject: str, body: str):
    job_id = uuid.uuid4().hex[:8]
    r.xadd(STREAM, {"job_id": job_id, "to": to, "subject": subject, "body": body})
    r.set(f"jobresult:{job_id}", "queued")
    return job_id


def run_worker(max_msgs: int = 10) -> int:
    """Drain the stream on demand and mark queued jobs as sent."""

    processed = 0
    resp = r.xreadgroup(GROUP, "worker-1", {STREAM: ">"}, count=max_msgs)
    for _stream, msgs in resp or []:
        for msg_id, fields in msgs:
            decoded = {k.decode(): v.decode() for k, v in fields.items()}
            r.set(f"jobresult:{decoded['job_id']}", "sent")
            r.xack(STREAM, GROUP, msg_id)
            processed += 1
    return processed


print("queue ready; sample job id ->", enqueue_email("a@x.com", "hi", "test"))
print("worker processed", run_worker(), "job(s)")


# Step 4 - Tool functions (sync + async).
def check_inventory(sku: str):
    row = db.execute(
        "SELECT sku,name,qty,price FROM inventory WHERE sku=? LIMIT 1",
        (sku,),
    ).fetchone()
    if not row:
        return {"error": f"unknown sku {sku}"}
    return {"sku": row[0], "name": row[1], "qty": row[2], "price": row[3]}


def create_order(sku: str, qty: int):
    cur = db.execute("SELECT qty,price FROM inventory WHERE sku=? LIMIT 1", (sku,)).fetchone()
    if not cur:
        return {"error": f"unknown sku {sku}"}
    have, price = cur
    if qty <= 0:
        return {"error": "qty must be positive"}
    if have < qty:
        return {"error": f"insufficient stock: have {have}, need {qty}"}
    db.execute("UPDATE inventory SET qty=qty-? WHERE sku=?", (qty, sku))
    cur2 = db.execute(
        "INSERT INTO orders (sku,qty,total,status) VALUES (?,?,?,?)",
        (sku, qty, round(price * qty, 2), "created"),
    )
    db.commit()
    return {"order_id": cur2.lastrowid, "sku": sku, "qty": qty, "total": round(price * qty, 2)}


def send_confirmation(to: str, order_id: int):
    job_id = enqueue_email(to, f"Order {order_id} confirmed", f"Your order {order_id} is on its way.")
    return {"job_id": job_id, "status": "queued"}


def check_job(job_id: str):
    v = r.get(f"jobresult:{job_id}")
    return {"job_id": job_id, "status": v.decode() if v else "unknown"}


inv = check_inventory("KB-01")
print(inv)
od = create_order("KB-01", 2)
print(od)
jb = send_confirmation("asha@x.com", od["order_id"])
print(jb)
print("worker processed", run_worker(), "job(s)")
print(check_job(jb["job_id"]))


# Step 5 - Tool schemas + dispatch.
TOOLS = [
    {
        "name": "check_inventory",
        "description": "Check stock and price for a product SKU before ordering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "Product SKU, e.g. KB-01.",
                }
            },
            "required": ["sku"],
        },
    },
    {
        "name": "create_order",
        "description": "Create an order for a SKU and quantity. Fails if stock is insufficient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "qty": {"type": "integer", "description": "Units to order (>0)."},
            },
            "required": ["sku", "qty"],
        },
    },
    {
        "name": "send_confirmation",
        "description": "Queue a confirmation email for a created order. Returns a job_id immediately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Customer email."},
                "order_id": {"type": "integer"},
            },
            "required": ["to", "order_id"],
        },
    },
    {
        "name": "check_job",
        "description": "Check the status of a queued email job by job_id.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
]

DISPATCH = {
    "check_inventory": check_inventory,
    "create_order": create_order,
    "send_confirmation": send_confirmation,
    "check_job": check_job,
}


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}, True
    try:
        out = fn(**args)
        return out, isinstance(out, dict) and "error" in out
    except Exception as exc:
        return {"error": repr(exc)}, True


print("tools ready")


# Step 6 - The orchestrating agent loop.
SYSTEM = (
    "You are an ordering agent. To place an order: first check_inventory, then create_order, "
    "then send_confirmation with the returned order_id. Report the order total and the email job status. "
    "If stock is insufficient, say so and do not create the order."
)


def _offline_agent_turn(user_text: str, verbose: bool = True) -> str:
    if verbose:
        print("... (mock) chaining check_inventory -> create_order -> send_confirmation")
    inv, _ = run_tool("check_inventory", {"sku": "MON-4"})
    od, _ = run_tool("create_order", {"sku": "MON-4", "qty": 1})
    jb, _ = run_tool("send_confirmation", {"to": "asha@x.com", "order_id": od["order_id"]})
    run_worker()
    st, _ = run_tool("check_job", {"job_id": jb["job_id"]})
    return f"(mock) Order {od['order_id']} total ${od['total']}; email {st['status']}."


def agent(user_text: str, max_steps: int = 8, verbose: bool = True):
    if not LIVE or Anthropic is None:
        return _offline_agent_turn(user_text, verbose=verbose)

    try:
        client = Anthropic()
        messages = [{"role": "user", "content": user_text}]
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0

        for _ in range(max_steps):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )

            usage = getattr(resp, "usage", None)
            if usage is not None:
                cumulative_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                cumulative_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        if verbose:
                            print(f"tool: {block.name}({block.input})")
                        out, is_err = run_tool(block.name, block.input)
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(out),
                                "is_error": is_err,
                            }
                        )
                messages.append({"role": "user", "content": results})
                run_worker()
                continue

            text = "".join(block.text for block in resp.content if block.type == "text")
            if verbose:
                total = cumulative_input_tokens + cumulative_output_tokens
                print(
                    f"token usage: input={cumulative_input_tokens} "
                    f"output={cumulative_output_tokens} total={total}"
                )
            return text

        return "(max steps reached)"
    except Exception as exc:
        print(f"Warning: live Anthropic call failed, falling back to offline mock: {exc}")
        return _offline_agent_turn(user_text, verbose=verbose)


print(agent("Order one 4K monitor (SKU MON-4) and email asha@x.com the confirmation."))


# Step 7 - Show the resulting state.
def main() -> None:
    print("Loaded repo env from:", _ENV_PATH if _ENV_PATH is not None else "(none)")
    print("LIVE mode" if LIVE else "OFFLINE mock mode")
    print("Orders table:")
    for row in db.execute("SELECT id,sku,qty,total,status FROM orders").fetchall():
        print(" ", row)
    print("Remaining stock:")
    for row in db.execute("SELECT sku,qty FROM inventory").fetchall():
        print(" ", row)
    out_of_stock, is_err = run_tool("create_order", {"sku": "HUB-2", "qty": 1})
    if is_err:
        print("Out-of-stock attempt:", out_of_stock["error"])
    else:
        print("Out-of-stock attempt:", out_of_stock)


if __name__ == "__main__":
    main()
