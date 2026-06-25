"""Day 18 capstone: small support agent with short-term Redis memory,
long-term vector recall, queue-backed async work, approval gating, and
lightweight tracing / prompt-cache accounting.

This is a notebook-to-script style artifact intended to be runnable on its own.
It loads the repo-root `.env` automatically and uses live Anthropic mode when a
valid `ANTHROPIC_API_KEY` is present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


def _load_repo_env() -> Path | None:
    """Load key=value pairs from the repo-root `.env` file."""

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
except Exception:  # pragma: no cover - optional live path
    Anthropic = None


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ENABLE_LIVE = True if ANTHROPIC_API_KEY else False
LIVE = bool(ANTHROPIC_API_KEY) and ENABLE_LIVE


def _now_ms() -> int:
    return int(time.time() * 1000)


class _MiniRedis:
    """Tiny Redis-like fallback for the notebook demo path."""

    def __init__(self):
        self._kv: dict[str, bytes] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self._groups: dict[tuple[str, str], dict[str, int]] = {}

    def set(self, key: str, value: str):
        self._kv[key] = str(value).encode("utf-8")
        return True

    def get(self, key: str):
        return self._kv.get(key)

    def rpush(self, key: str, value: str):
        self._lists.setdefault(key, []).append(str(value).encode("utf-8"))

    def lrange(self, key: str, start: int, end: int):
        values = self._lists.get(key, [])
        if end == -1:
            end = len(values) - 1
        return values[start : end + 1]

    def ltrim(self, key: str, start: int, end: int):
        values = self._lists.get(key, [])
        if end == -1:
            end = len(values) - 1
        self._lists[key] = values[start : end + 1]

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
        msg_id = f"{_now_ms()}-{uuid.uuid4().hex[:8]}"
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


class _HashEmbedding:
    """Small deterministic embedding model for offline vector recall."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            weight = 1.0 + (int(digest[8:12], 16) % 5) / 10.0
            vec[idx] += weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SimpleVectorStore:
    def __init__(self, embedder: _HashEmbedding):
        self.embedder = embedder
        self._rows: list[dict[str, Any]] = []

    def add_text(self, text: str, metadata: dict[str, Any]):
        self._rows.append(
            {
                "text": text,
                "metadata": metadata,
                "embedding": self.embedder.embed(text),
            }
        )

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        qvec = self.embedder.embed(query)

        def cosine(vec: list[float]) -> float:
            return sum(a * b for a, b in zip(qvec, vec))

        ranked = sorted(
            self._rows,
            key=lambda row: cosine(row["embedding"]),
            reverse=True,
        )
        return ranked[:k]


class SupportMemory:
    """Short-term memory plus long-term semantic recall."""

    def __init__(self, redis_backend, session_id: str = "support-session", history_limit: int = 24):
        self.r = redis_backend
        self.session_id = session_id
        self.history_key = f"support:history:{session_id}"
        self.facts_key = f"support:facts:{session_id}"
        self.history_limit = history_limit
        self.vector = SimpleVectorStore(_HashEmbedding())

    def add_turn(self, role: str, text: str):
        payload = json.dumps({"role": role, "text": text, "ts": _now_ms()})
        self.r.rpush(self.history_key, payload)
        self.r.ltrim(self.history_key, -self.history_limit, -1)

    def get_history(self) -> list[dict[str, Any]]:
        rows = self.r.lrange(self.history_key, 0, -1)
        out = []
        for row in rows:
            raw = row.decode("utf-8") if isinstance(row, (bytes, bytearray)) else row
            try:
                out.append(json.loads(raw))
            except Exception:
                out.append({"role": "unknown", "text": raw})
        return out

    def remember_fact(self, key: str, value: str):
        self.r.set(f"{self.facts_key}:{key}", value)
        self.vector.add_text(f"{key}: {value}", {"kind": "fact", "key": key})

    def get_fact(self, key: str):
        val = self.r.get(f"{self.facts_key}:{key}")
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else val

    def recall(self, query: str, k: int = 3):
        return self.vector.search(query, k=k)


def _build_redis_backend():
    """Prefer fakeredis if available, otherwise use the local fallback."""

    try:
        import fakeredis

        return fakeredis.FakeStrictRedis()
    except Exception:
        return _MiniRedis()


redis_backend = _build_redis_backend()
mem = SupportMemory(redis_backend, session_id="capstone-demo")


# Knowledge base for long-term recall.
KB_DOCS = [
    {
        "id": "reset_password",
        "title": "Password reset",
        "content": "If a customer cannot log in, verify email ownership, then trigger password reset.",
    },
    {
        "id": "refund_policy",
        "title": "Refund policy",
        "content": "Refunds above 100 require human approval. Refunds under 100 can be auto-approved.",
    },
    {
        "id": "delivery_delay",
        "title": "Delivery delay",
        "content": "For shipping delays, create a ticket and enqueue a callback job for the customer.",
    },
    {
        "id": "plan_change",
        "title": "Plan change",
        "content": "Plan downgrades need a confirmation note, but immediate upgrades can be processed in one step.",
    },
]

for doc in KB_DOCS:
    mem.vector.add_text(doc["content"], {"kind": "kb", **doc})


db = sqlite3.connect(":memory:", check_same_thread=False)
db.execute(
    "CREATE TABLE tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, issue TEXT, severity TEXT, status TEXT, amount REAL DEFAULT 0)"
)
db.execute(
    "CREATE TABLE approvals (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, approver TEXT, approved_at INTEGER)"
)
db.commit()


r = redis_backend
STREAM = "support:callbacks"
GROUP = "support-workers"
try:
    r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
except Exception as exc:
    print("queue already initialized:", exc)


def enqueue_callback(customer_email: str, ticket_id: int, message: str):
    job_id = uuid.uuid4().hex[:8]
    r.xadd(
        STREAM,
        {
            "job_id": job_id,
            "customer_email": customer_email,
            "ticket_id": str(ticket_id),
            "message": message,
        },
    )
    r.set(f"jobresult:{job_id}", "queued")
    return {"job_id": job_id, "status": "queued"}


def run_worker(max_msgs: int = 10) -> int:
    processed = 0
    resp = r.xreadgroup(GROUP, "worker-1", {STREAM: ">"}, count=max_msgs)
    for _stream, msgs in resp or []:
        for msg_id, fields in msgs:
            decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in fields.items()}
            r.set(f"jobresult:{decoded['job_id']}", "sent")
            r.xack(STREAM, GROUP, msg_id)
            processed += 1
    return processed


TRACE_LOG: list[dict[str, Any]] = []
PROMPT_CACHE: dict[str, dict[str, Any]] = {}


def _prompt_cache_key(system_prompt: str, tool_block: list[dict[str, Any]]) -> str:
    blob = json.dumps({"system": system_prompt, "tools": tool_block}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compile_prompt_bundle(system_prompt: str, tool_block: list[dict[str, Any]]):
    key = _prompt_cache_key(system_prompt, tool_block)
    hit = key in PROMPT_CACHE
    if not hit:
        PROMPT_CACHE[key] = {
            "compiled_at": _now_ms(),
            "system_prompt": system_prompt,
            "tool_count": len(tool_block),
        }
    return PROMPT_CACHE[key], hit


TOOLS = [
    {
        "name": "recall_memory",
        "description": "Search short-term history and long-term vector memory for relevant support context.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket for a customer issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "issue": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["customer_id", "issue", "severity"],
        },
    },
    {
        "name": "queue_callback",
        "description": "Queue an asynchronous customer callback job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_email": {"type": "string"},
                "ticket_id": {"type": "integer"},
                "message": {"type": "string"},
            },
            "required": ["customer_email", "ticket_id", "message"],
        },
    },
    {
        "name": "request_refund",
        "description": "Request a refund, but return needs_approval when the amount is above the approval threshold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer"},
                "amount": {"type": "number"},
                "approver": {"type": "string"},
            },
            "required": ["ticket_id", "amount", "approver"],
        },
    },
]


APPROVAL_THRESHOLD = float(os.environ.get("DAY18_REFUND_APPROVAL_THRESHOLD", "100"))


def _record_trace(tool: str, args: dict[str, Any], started_ms: int, ok: bool, retries: int):
    TRACE_LOG.append(
        {
            "tool": tool,
            "args": args,
            "ms": _now_ms() - started_ms,
            "ok": ok,
            "retries": retries,
        }
    )


def run_tool(name: str, args: dict[str, Any], retries: int = 1):
    fn = DISPATCH.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}, True

    started = _now_ms()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            out = fn(**args)
            ok = isinstance(out, dict) and "error" not in out
            _record_trace(name, args, started, ok, attempt)
            return out, not ok
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.05 * (attempt + 1))
                continue
            _record_trace(name, args, started, False, attempt)
            return {"error": repr(exc)}, True

    _record_trace(name, args, started, False, retries)
    return {"error": repr(last_error) if last_error else "unknown error"}, True


def recall_memory(query: str):
    recent = [row for row in mem.get_history() if query.lower() in row.get("text", "").lower()]
    semantic = mem.recall(query, k=3)
    return {
        "recent_hits": recent[-3:],
        "semantic_hits": [
            {"text": row["text"], "metadata": row["metadata"]} for row in semantic
        ],
    }


def create_ticket(customer_id: str, issue: str, severity: str):
    cur = db.execute(
        "INSERT INTO tickets (customer_id, issue, severity, status) VALUES (?,?,?,?)",
        (customer_id, issue, severity, "open"),
    )
    db.commit()
    ticket_id = cur.lastrowid
    mem.remember_fact(f"ticket:{ticket_id}", issue)
    mem.add_turn("assistant", f"Created ticket {ticket_id} for {customer_id}")
    return {"ticket_id": ticket_id, "status": "open", "severity": severity}


def queue_callback(customer_email: str, ticket_id: int, message: str):
    return enqueue_callback(customer_email, ticket_id, message)


def request_refund(ticket_id: int, amount: float, approver: str):
    if amount > APPROVAL_THRESHOLD:
        return {
            "status": "needs_approval",
            "ticket_id": ticket_id,
            "amount": amount,
            "approver": approver,
        }
    return refund_customer(ticket_id=ticket_id, amount=amount)


def approve_refund(ticket_id: int, approver: str):
    db.execute(
        "INSERT INTO approvals (ticket_id, approver, approved_at) VALUES (?,?,?)",
        (ticket_id, approver, _now_ms()),
    )
    db.execute("UPDATE tickets SET status='approved' WHERE id=?", (ticket_id,))
    db.commit()
    return {"status": "approved", "ticket_id": ticket_id, "approver": approver}


def refund_customer(ticket_id: int, amount: float):
    db.execute("UPDATE tickets SET status=?, amount=? WHERE id=?", ("refunded", amount, ticket_id))
    db.commit()
    return {"status": "refunded", "ticket_id": ticket_id, "amount": amount}


DISPATCH = {
    "recall_memory": recall_memory,
    "create_ticket": create_ticket,
    "queue_callback": queue_callback,
    "request_refund": request_refund,
    "approve_refund": approve_refund,
    "refund_customer": refund_customer,
}


SYSTEM_PROMPT = (
    "You are a customer support agent. "
    "Use recall_memory before responding when context is missing. "
    "For customer issues, chain recall_memory -> create_ticket -> queue_callback. "
    "Refunds above the approval threshold must return needs_approval and require approve_refund before any final refund. "
    "Keep the conversation concise and explain the next step."
)


def _offline_orchestrate(user_text: str, verbose: bool = True) -> str:
    text = user_text.strip()
    lowered = text.lower()
    mem.add_turn("user", text)

    if "refund" in lowered:
        amount_match = re.search(r"(\d+(?:\.\d+)?)", lowered)
        amount = float(amount_match.group(1)) if amount_match else 150.0
        ticket, _ = run_tool("create_ticket", {"customer_id": "C1001", "issue": f"Refund request: {text}", "severity": "high"})
        refund, _ = run_tool("request_refund", {"ticket_id": ticket["ticket_id"], "amount": amount, "approver": "human"})
        if refund.get("status") == "needs_approval":
            approval, _ = run_tool("approve_refund", {"ticket_id": ticket["ticket_id"], "approver": "manager-on-duty"})
            final, _ = run_tool("refund_customer", {"ticket_id": ticket["ticket_id"], "amount": amount})
            return (
                f"Refund {amount} was escalated, approved by {approval['approver']}, "
                f"and finished with status {final['status']} for ticket {ticket['ticket_id']}."
            )
        return f"Refund completed for ticket {ticket['ticket_id']}."

    if "ticket" in lowered or "issue" in lowered or "callback" in lowered or "support" in lowered:
        recall, _ = run_tool("recall_memory", {"query": text})
        ticket, _ = run_tool("create_ticket", {"customer_id": "C1001", "issue": text, "severity": "medium"})
        callback, _ = run_tool(
            "queue_callback",
            {
                "customer_email": "customer@example.com",
                "ticket_id": ticket["ticket_id"],
                "message": f"We created ticket {ticket['ticket_id']} and will call you back soon.",
            },
        )
        run_worker()
        job_status = r.get(f"jobresult:{callback['job_id']}")
        job_status = job_status.decode("utf-8") if isinstance(job_status, (bytes, bytearray)) else job_status
        return (
            f"Created ticket {ticket['ticket_id']} and queued callback job {callback['job_id']} "
            f"(status: {job_status}). Recall summary: {len(recall['semantic_hits'])} semantic hit(s)."
        )

    recall, _ = run_tool("recall_memory", {"query": text})
    if not recall["recent_hits"] and not recall["semantic_hits"]:
        return "I do not have enough context yet. Tell me the issue, ticket number, or account detail."
    return f"Relevant memory: {recall['semantic_hits'][:2]}"


def agent(user_text: str, max_steps: int = 6, verbose: bool = True):
    compiled, hit = compile_prompt_bundle(SYSTEM_PROMPT, TOOLS)
    if verbose:
        print(f"prompt cache: {'hit' if hit else 'miss'}; compiled_at={compiled['compiled_at']}")

    if not LIVE or Anthropic is None:
        return _offline_orchestrate(user_text, verbose=verbose)

    try:
        client = Anthropic()
        messages = [{"role": "user", "content": user_text}]
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0

        for _ in range(max_steps):
            resp = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            usage = getattr(resp, "usage", None)
            if usage is not None:
                cumulative_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                cumulative_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        if verbose:
                            print(f"tool: {block.name}({block.input})")
                        out, is_err = run_tool(block.name, block.input, retries=1)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(out),
                                "is_error": is_err,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
                continue

            text = "".join(block.text for block in resp.content if block.type == "text")
            if verbose:
                total = cumulative_input_tokens + cumulative_output_tokens
                print(
                    f"token usage: input={cumulative_input_tokens} "
                    f"output={cumulative_output_tokens} total={total}"
                )
            mem.add_turn("assistant", text)
            return text

        return "(max steps reached)"
    except Exception as exc:
        print(f"Warning: live Anthropic call failed, using offline orchestrator: {exc}")
        return _offline_orchestrate(user_text, verbose=verbose)


def print_trace_table():
    print("tool trace:")
    for row in TRACE_LOG:
        print(f"  {row['tool']:<18} {row['ms']:>4} ms  ok={row['ok']}  retries={row['retries']}  args={row['args']}")


def main() -> None:
    print("Loaded repo env from:", _ENV_PATH if _ENV_PATH is not None else "(none)")
    print("LIVE mode" if LIVE else "OFFLINE demo mode")
    print()

    mem.remember_fact("name", "Asha")
    mem.remember_fact("plan", "premium")
    mem.add_turn("user", "Hi, my delivery is delayed and I need support.")
    mem.add_turn("assistant", "I can help with that.")

    print("Demo 1 - support routing")
    print(agent("My delivery is delayed. Please open a ticket and call me back.", verbose=True))
    print()

    print("Demo 2 - approval gate")
    print(agent("Please refund me 250 dollars for ticket 42.", verbose=True))
    print()

    print("Demo 3 - recall")
    print(agent("What do you remember about my plan?", verbose=False))
    print()

    print("Demo 4 - memory state")
    print("history:", mem.get_history()[-4:])
    print("ticket facts:", mem.get_fact("ticket:1"))
    print()

    print_trace_table()
    print()
    print(f"prompt cache entries: {len(PROMPT_CACHE)}")
    print("Orders/tickets table:")
    for row in db.execute("SELECT id, customer_id, issue, severity, status, amount FROM tickets").fetchall():
        print(" ", row)


if __name__ == "__main__":
    main()
