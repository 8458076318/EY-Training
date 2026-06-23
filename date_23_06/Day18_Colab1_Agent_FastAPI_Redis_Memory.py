"""Day 18: Claude Agent + FastAPI Backend + Redis Memory.

Converted from the notebook `Day18_Colab1_Agent_FastAPI_Redis_Memory.ipynb`
into a plain Python script that:

- loads the repo-root `.env` file automatically,
- runs the FastAPI order API in-process,
- uses Redis-like memory for short-term history and long-term facts,
- executes a Claude tool loop when `ANTHROPIC_API_KEY` is available,
- falls back to a deterministic offline mock when the key or package is
  unavailable.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _load_repo_env() -> Path | None:
    """Load key=value pairs from the repo-root `.env` without extra deps.

    The repo `.env` intentionally overrides any stale process-level values so
    the notebook behaves the same way every run.
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
    from fastapi import FastAPI, HTTPException
except Exception as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "Missing FastAPI. Install `fastapi` to run this script."
    ) from exc

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional for the demo path
    TestClient = None


def _build_redis_backend():
    """Prefer real Redis when REDIS_URL is set, otherwise use fakeredis."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis

            return redis.Redis.from_url(redis_url)
        except Exception as exc:
            print(f"Warning: REDIS_URL is set, but real Redis is unavailable: {exc}")

    try:
        import fakeredis

        return fakeredis.FakeStrictRedis()
    except Exception:
        return _SimpleRedis()


class _SimpleRedis:
    """Small in-memory Redis stand-in for the demo path."""

    def __init__(self):
        self._lists: dict[str, list[str]] = {}
        self._hashes: dict[str, dict[str, tuple[str, float | None]]] = {}

    def _purge_expired(self, key: str) -> None:
        now = time.time()
        table = self._hashes.get(key)
        if not table:
            return
        expired = [field for field, (_value, ttl_at) in table.items() if ttl_at is not None and ttl_at <= now]
        for field in expired:
            del table[field]

    def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        values = self._lists.get(key, [])
        if not values:
            self._lists[key] = []
            return
        if end == -1:
            end = len(values) - 1
        self._lists[key] = values[start : end + 1]

    def lrange(self, key: str, start: int, end: int):
        values = self._lists.get(key, [])
        if end == -1:
            end = len(values) - 1
        sliced = values[start : end + 1]
        return [item.encode("utf-8") for item in sliced]

    def hset(self, key: str, field: str, value: str) -> None:
        self._hashes.setdefault(key, {})[field] = (value, None)

    def expire(self, key: str, ttl_seconds: int) -> None:
        table = self._hashes.get(key)
        if not table:
            return
        ttl_at = time.time() + ttl_seconds
        for field, (value, _old_ttl) in list(table.items()):
            table[field] = (value, ttl_at)

    def hget(self, key: str, field: str):
        self._purge_expired(key)
        table = self._hashes.get(key, {})
        if field not in table:
            return None
        value, ttl_at = table[field]
        if ttl_at is not None and ttl_at <= time.time():
            del table[field]
            return None
        return value.encode("utf-8")

    def hgetall(self, key: str):
        self._purge_expired(key)
        table = self._hashes.get(key, {})
        out = {}
        for field, (value, ttl_at) in table.items():
            if ttl_at is not None and ttl_at <= time.time():
                continue
            out[field.encode("utf-8")] = value.encode("utf-8")
        return out


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY and _ENV_PATH is not None:
    # If the repo-root .env did not provide the key, stay fully offline.
    pass

LIVE = bool(ANTHROPIC_API_KEY)
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


app = FastAPI()

_ORDERS = {
    "A1001": {
        "id": "A1001",
        "item": "Mechanical keyboard",
        "qty": 1,
        "status": "shipped",
        "total": 129.0,
        "customer_id": "C1002",
    },
    "A1002": {
        "id": "A1002",
        "item": "USB-C hub",
        "qty": 2,
        "status": "processing",
        "total": 58.0,
        "customer_id": "C1001",
    },
    "A1003": {
        "id": "A1003",
        "item": "4K monitor",
        "qty": 1,
        "status": "delivered",
        "total": 410.0,
        "customer_id": "C1001",
    },
}

_CUSTOMERS = {
    "C1001": {"id": "C1001", "name": "Asha", "tier": "gold", "default_order_id": "A1002"},
    "C1002": {"id": "C1002", "name": "Ravi", "tier": "silver", "default_order_id": "A1001"},
}


def lookup_order(order_id: str) -> dict[str, Any] | None:
    return _ORDERS.get(order_id.upper())


def lookup_customer(customer_id: str) -> dict[str, Any] | None:
    return _CUSTOMERS.get(customer_id.upper())


@app.get("/orders/{order_id}")
def get_order(order_id: str):
    order = lookup_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="order not found")
    return order


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    customer = lookup_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="customer not found")
    return customer


if TestClient is not None:
    try:
        client = TestClient(app)
    except Exception as exc:  # pragma: no cover - environment-specific
        client = None
        print(f"Warning: FastAPI TestClient unavailable; using direct lookup demo: {exc}")
else:
    client = None


class RedisMemory:
    def __init__(self, r, session_id: str, history_limit: int = 40):
        self.r = r
        self.sid = session_id
        self.history_limit = history_limit
        self.h_key = f"hist:{session_id}"
        self.f_key = f"facts:{session_id}"

    def append_turn(self, role: str, content):
        self.r.rpush(self.h_key, json.dumps({"role": role, "content": content}))
        self.r.ltrim(self.h_key, -self.history_limit, -1)

    def load_history(self):
        return [json.loads(x) for x in self.r.lrange(self.h_key, 0, -1)]

    def replace_history(self, turns):
        if hasattr(self.r, "delete"):
            try:
                self.r.delete(self.h_key)
            except Exception:
                pass
        elif hasattr(self.r, "_lists"):
            self.r._lists[self.h_key] = []
        for turn in turns:
            self.r.rpush(self.h_key, json.dumps(turn))
        self.r.ltrim(self.h_key, -self.history_limit, -1)

    def set_fact(self, key: str, value: str, ttl_seconds: int | None = None):
        self.r.hset(self.f_key, key, value)
        if ttl_seconds:
            self.r.expire(self.f_key, ttl_seconds)

    def get_fact(self, key: str):
        v = self.r.hget(self.f_key, key)
        return v.decode() if isinstance(v, bytes) else v

    def all_facts(self):
        return {k.decode(): v.decode() for k, v in self.r.hgetall(self.f_key).items()}

    def forget_fact(self, key: str):
        if hasattr(self.r, "hdel"):
            try:
                self.r.hdel(self.f_key, key)
            except Exception:
                pass
        elif hasattr(self.r, "_hashes"):
            self.r._hashes.get(self.f_key, {}).pop(key, None)


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def summarize_history_turns(turns):
    if not turns:
        return "No prior context."
    parts = []
    for turn in turns:
        content = turn.get("content")
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        parts.append(f"{turn.get('role', 'unknown')}: {content}")
    summary = " | ".join(parts)
    return summary[:800] + ("..." if len(summary) > 800 else "")


def compact_history(mem_obj: RedisMemory, max_turns: int = 8, keep_last: int = 4, verbose: bool = True):
    history = mem_obj.load_history()
    if len(history) <= max_turns:
        return False
    prefix = history[:-keep_last]
    suffix = history[-keep_last:]
    summary_text = summarize_history_turns(prefix)
    compacted = [{"role": "assistant", "content": f"Summary of earlier turns: {summary_text}"}] + suffix
    mem_obj.replace_history(compacted)
    if verbose:
        print(f"History compacted from {len(history)} turns to {len(compacted)} turns.")
    return True


redis_backend = _build_redis_backend()
mem = RedisMemory(redis_backend, session_id="demo-user")


TOOLS = [
    {
        "name": "get_order",
        "description": "Look up a customer order by its ID and return item, quantity, status and total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID like 'A1001'.",
                    "pattern": "^[Aa][0-9]{4}$",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Persist a durable fact about the user (e.g. shipping preference) for future turns. Reject obvious PII.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": 'Short fact key, e.g. "shipping_pref".',
                },
                "value": {
                    "type": "string",
                    "description": "The fact value to store.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_fact",
        "description": "Retrieve a previously stored fact about the user by key. Returns empty if unknown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The fact key to look up.",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "forget_fact",
        "description": "Delete a stored fact so it can no longer be recalled.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The fact key to delete.",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "get_customer",
        "description": "Look up a customer by ID and return profile details, including default order context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID like 'C1001'.",
                    "pattern": "^[Cc][0-9]{4}$",
                }
            },
            "required": ["customer_id"],
        },
    },
]


def tool_get_order(order_id: str):
    order = lookup_order(order_id)
    if order is None:
        return {"error": f"No order {order_id} found."}
    return order


def tool_get_customer(customer_id: str):
    customer = lookup_customer(customer_id)
    if customer is None:
        return {"error": f"No customer {customer_id} found."}
    return customer


def tool_remember_fact(key: str, value: str):
    if re.search(r"\b[\w.+-]+@[\w.-]+\.\w+\b", value):
        return {"error": "PII blocked: email addresses are not allowed in remembered facts."}
    if re.search(r"\b(?:\d[ -]*?){13,19}\b", value):
        return {"error": "PII blocked: card-like numbers are not allowed in remembered facts."}
    mem.set_fact(key, value)
    return {"ok": True, "stored": {key: value}}


def tool_recall_fact(key: str):
    value = mem.get_fact(key)
    return {"key": key, "value": value} if value is not None else {"key": key, "value": None}


def tool_forget_fact(key: str):
    mem.forget_fact(key)
    return {"ok": True, "forgotten": key}


DISPATCH = {
    "get_order": tool_get_order,
    "get_customer": tool_get_customer,
    "remember_fact": tool_remember_fact,
    "recall_fact": tool_recall_fact,
    "forget_fact": tool_forget_fact,
}


def run_tool(name, args):
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, True
    try:
        out = fn(**args)
        is_err = isinstance(out, dict) and "error" in out
        return out, is_err
    except Exception as exc:
        return {"error": repr(exc)}, True


SYSTEM = (
    "You are an order-support assistant. Use get_order for any order question. "
    "Use remember_fact / recall_fact to keep durable user preferences across turns. "
    "Be concise."
)


def _normalize_fact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _extract_order_id(text: str) -> str | None:
    match = re.search(r"\bA\d{4}\b", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _offline_agent_turn(user_text: str, verbose: bool = True) -> str:
    lower = user_text.lower().strip()

    remember_match = re.search(r"remember that my\s+(.+?)\s+is\s+(.+?)(?:[.?!]?$)", user_text, re.IGNORECASE)
    if remember_match:
        key_phrase = remember_match.group(1).strip()
        value = remember_match.group(2).strip().rstrip(".?!")
        fact_key = _normalize_fact_key(key_phrase)
        mem.set_fact(fact_key, value)
        reply = f"Saved {key_phrase} = {value}."
        mem.append_turn("assistant", reply)
        return reply

    recall_match = re.search(r"what did i say my\s+(.+?)\s+was\??$", user_text, re.IGNORECASE)
    if recall_match:
        key_phrase = recall_match.group(1).strip()
        fact_key = _normalize_fact_key(key_phrase)
        value = mem.get_fact(fact_key)
        reply = f"You said your {key_phrase} was {value}." if value is not None else f"I do not have a fact for {key_phrase}."
        mem.append_turn("assistant", reply)
        return reply

    if lower.startswith("hi, i am "):
        name = user_text.split("Hi, I am ", 1)[-1].strip().rstrip(".")
        mem.set_fact("name", name)
        reply = f"Nice to meet you, {name}."
        mem.append_turn("assistant", reply)
        return reply

    order_id = _extract_order_id(user_text)
    if "status of order" in lower and order_id:
        out, _ = run_tool("get_order", {"order_id": order_id})
        mem.set_fact("last_order_id", order_id)
        reply = f"Order {order_id} is {out.get('status', '?')}."
        mem.append_turn("assistant", reply)
        return reply

    if ("how much was order" in lower or "what was order" in lower or "how much is order" in lower) and order_id:
        out, _ = run_tool("get_order", {"order_id": order_id})
        mem.set_fact("last_order_id", order_id)
        reply = f"Order {order_id} total is {out.get('total', '?')}."
        mem.append_turn("assistant", reply)
        return reply

    if "given my budget cap" in lower and "within it" in lower:
        budget_text = mem.get_fact("budget_cap") or mem.get_fact("budget cap")
        last_order_id = mem.get_fact("last_order_id")
        if budget_text is None:
            reply = "I do not yet know your budget cap."
            mem.append_turn("assistant", reply)
            return reply
        try:
            budget = float(re.sub(r"[^0-9.]", "", budget_text))
        except Exception:
            budget = None
        order_ref = order_id or last_order_id
        if not order_ref:
            reply = f"Your budget cap is {budget_text}, but I do not know which order to compare."
            mem.append_turn("assistant", reply)
            return reply
        order = lookup_order(order_ref)
        if not order:
            reply = f"I could not find order {order_ref}."
            mem.append_turn("assistant", reply)
            return reply
        within = budget is not None and order.get("total", 0) <= budget
        reply = (
            f"Yes, order {order_ref} at {order['total']} is within your budget cap of {budget_text}."
            if within
            else f"No, order {order_ref} at {order['total']} is above your budget cap of {budget_text}."
        )
        mem.append_turn("assistant", reply)
        return reply

    if "customer" in lower and "order" in lower and "default order" in lower:
        customer = lookup_customer("C1001")
        order = lookup_order(customer["default_order_id"]) if customer else None
        reply = (
            f"Customer {customer['name']} is on tier {customer['tier']} and their default order "
            f"{order['id']} is {order['status']}."
            if customer and order
            else "I could not resolve both customer and order."
        )
        mem.append_turn("assistant", reply)
        return reply

    if verbose:
        print("[offline] model requests get_order A1001")
    out, _ = run_tool("get_order", {"order_id": "A1001"})
    reply = f"(offline) Order A1001 is {out.get('status', '?')}."
    mem.append_turn("assistant", reply)
    return reply


def agent_turn(user_text, max_steps: int = 6, verbose: bool = True):
    mem.append_turn("user", user_text)
    messages = mem.load_history()
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0

    compact_history(mem, verbose=verbose)
    messages = mem.load_history()

    if not LIVE:
        reply = _offline_agent_turn(user_text, verbose=verbose)
        if verbose:
            print("token usage: input=0 output=0 total=0")
        return reply

    try:
        from anthropic import Anthropic
    except Exception as exc:
        if verbose:
            print(f"Warning: anthropic is unavailable, using offline mock mode: {exc}")
        reply = _offline_agent_turn(user_text, verbose=verbose)
        if verbose:
            print("token usage: input=0 output=0 total=0")
        return reply

    client_a = Anthropic()
    for _step in range(max_steps):
        try:
            resp = client_a.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:
            if verbose:
                print(f"Warning: live Anthropic call failed, using offline mock mode: {exc}")
            reply = _offline_agent_turn(user_text, verbose=verbose)
            if verbose:
                print("token usage: input=0 output=0 total=0")
            return reply

        usage = getattr(resp, "usage", None)
        if usage is not None:
            cumulative_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            cumulative_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": [block.model_dump() for block in resp.content]})
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
            continue

        text = "".join(block.text for block in resp.content if block.type == "text")
        mem.append_turn("assistant", text)
        if verbose:
            total = cumulative_input_tokens + cumulative_output_tokens
            print(f"token usage: input={cumulative_input_tokens} output={cumulative_output_tokens} total={total}")
        return text

    return "(stopped: max steps reached)"


def main() -> None:
    print("deps installed")
    if _ENV_PATH is not None:
        print(f"Loaded repo env from: {_ENV_PATH} (repo values override process env)")
    print("LIVE mode" if LIVE else "OFFLINE mock mode (no key) - agent loop will be simulated")
    print()

    print("Step 2 - FastAPI backend")
    if client is not None:
        print(client.get("/orders/A1001").json())
    else:
        print(lookup_order("A1001"))
    print()

    print("Step 3 - Redis memory")
    mem.set_fact("name", "Asha")
    mem.append_turn("user", "hello")
    print("facts:", mem.all_facts())
    print("history:", mem.load_history())
    print()

    print("Step 4 - Tools")
    print(len(TOOLS), "tools declared")
    print()

    print("Step 5 - Tool dispatch")
    print(run_tool("get_order", {"order_id": "A1002"}))
    print(run_tool("get_order", {"order_id": "A9999"}))
    print(run_tool("get_customer", {"customer_id": "C1001"}))
    print()

    print("Step 5b - Extension checks")
    mem.set_fact("temp_note", "expires soon", ttl_seconds=1)
    print("temp_note before expiry:", mem.get_fact("temp_note"))
    time.sleep(1.1)
    print("temp_note after expiry:", mem.get_fact("temp_note"))
    print(run_tool("forget_fact", {"key": "temp_note"}))
    print(run_tool("remember_fact", {"key": "email", "value": "user@example.com"}))
    print(run_tool("remember_fact", {"key": "card", "value": "4111 1111 1111 1111"}))
    print()

    print("Step 6 - Agent loop")
    print(agent_turn("What is the status of order A1002?"))
    print()

    print("Step 7 - Persistence check")
    print(agent_turn("Please remember that my shipping preference is express."))
    print("---")
    print(agent_turn("What did I say my shipping preference was?"))
    print("---")
    print("Raw facts in Redis:", mem.all_facts())
    print("History length:", len(mem.load_history()), "turns")
    print()

    print("Step 8 - Multi-turn chat demo")
    for msg in [
        "Hi, I am Asha.",
        "How much was order A1003?",
        "Remember that my budget cap is 500 dollars.",
        "Given my budget cap, was that order within it?",
    ]:
        print("USER:", msg)
        print("AGENT:", agent_turn(msg, verbose=False))
        print()

    print("Step 9 - Parallel lookup demo")
    print(agent_turn("Tell me about customer C1001 and their default order.", verbose=False))
    print()

    print("Step 10 - Rolling summary demo")
    demo_mem = RedisMemory(_build_redis_backend(), session_id="summary-demo", history_limit=40)
    for idx in range(10):
        demo_mem.append_turn("user", f"message {idx}")
        demo_mem.append_turn("assistant", f"reply {idx}")
    print("before:", len(demo_mem.load_history()))
    compact_history(demo_mem, max_turns=8, keep_last=4)
    print("after:", len(demo_mem.load_history()))


if __name__ == "__main__":
    main()
