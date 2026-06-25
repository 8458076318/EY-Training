# Day 18 Capstone Architecture

## Overview

This support agent uses a small offline-friendly architecture:

- repo-root `.env` loader for secrets and runtime settings
- Redis-style short-term memory for recent turns and queued jobs
- vector-store style long-term recall for support knowledge and remembered facts
- tool layer with support ticket creation, callback queuing, refund handling, and approval
- trace logging for each tool call with latency and retry counts
- prompt-bundle cache keyed by the system prompt plus tool schema

## Flow

1. User message enters the agent.
2. The agent first compiles the prompt bundle and reports a cache hit or miss.
3. The router checks whether the request is a recall, ticket, callback, or refund case.
4. Ticket flows chain `recall_memory -> create_ticket -> queue_callback`.
5. Refund flows enforce a human approval gate when the amount exceeds the threshold.
6. The async callback tool writes to a Redis stream and a worker drains it later.

## Storage

- Short-term memory: Redis list keyed by session
- Long-term recall: local vector store seeded with support guidance
- Tickets and approvals: SQLite tables for simple persistence

## Runtime

- Offline mode is the default so the demo runs in restricted environments.
- Live Anthropic tool use is enabled automatically when `ANTHROPIC_API_KEY` is present in `.env`.
- If `fakeredis` is unavailable, a tiny in-process Redis fallback is used.
