# Trace/Spans vs. Audit Log — Decision Cheatsheet

*Day 20 · Observability & Governance (MeridianPay scenario)*

## The Core Distinction

| | **Trace / Spans** | **Audit Log** |
|---|---|---|
| **What it records** | A timed tree of spans for one request | An append-only, hash-chained record of each decision |
| **Completeness** | Sampled (e.g. 10%) | Complete — no gaps |
| **Lifespan** | Days | Years |
| **Tamper-evidence** | None | Yes — hash chain (`prev → sha256:...`) breaks if altered |
| **Read by** | Engineers | Auditors / regulators / compliance |
| **Answers** | *where* it broke, *why* it's slow, *how much* it cost | *who* decided, *prove* it happened, *keep* it unaltered |
| **Linked by** | `trace_id` (same id ties a trace to its audit entries) | `trace_id` |

**Listen for the verbs:**
- `where` / `why` / `how fast` / `how much` → **Trace / Spans**
- `who` / `prove` / `keep` / `all of them` → **Audit Log**
- Needs both a technical explanation *and* durable proof → **Both**

---

## Ticket Answers & Reasoning

### Ticket 02 — Regulator: who approved loan #88213, model version, inputs, proof never altered
**→ Audit Log**
Discrete decision facts (who, version, inputs) plus a tamper-evidence requirement. Only the hash-chained log proves nothing was altered after the fact.

### Ticket 03 — Prove PII-redaction guardrail ran before model call, on all 12,400 approvals last quarter
**→ Audit Log**
Two tells: **"all 12,400"** (traces are sampled, most wouldn't exist) and **"last quarter"** (traces live days, not months). The chain's `prev ←` links also prove ordering, not just timestamps.

### Ticket 04 — Reproduce a single failing request in staging, show full call tree, pinpoint where it errored
**→ Trace / Spans**
Need the request's execution tree (retrieval → underwrite → model.call → guardrail) with durations to find where it broke. The audit log shows decisions, not call structure or timing.

### Ticket 05 — FinOps wants average cost & tokens per request on a live dashboard; 10% sampling is fine
**→ Trace / Spans**
"Sampling is fine" is the giveaway — audit logs can't be sampled (their value is completeness). Per-call cost/token metrics belong on `model.call` spans, feeding a live engineering dashboard.

### Ticket 06 — Decision record must survive a DBA with write access trying to alter it 3 years from now
**→ Audit Log**
Tamper-evidence against someone with direct write access is exactly what the hash chain defends against — any edit breaks verification. Traces live days and have no tamper protection, so they fail on both lifespan and integrity.

### Ticket 08 — Guardrail blocked a transaction; compliance needs permanent proof it was blocked, on-call needs to debug why it fired
**→ Both**
Two audiences, two needs: Audit Log gives compliance the immutable `guardrail.blocked` record; Trace/Spans gives on-call the full call tree (what retrieval returned, what the model scored, what the guardrail evaluated) to explain *why* it fired on a legit-looking request.

---

## Quick Pattern Recap

- **Audit Log** signals: *who*, *prove*, *never altered*, *all of them*, *years from now*, *regulator/compliance*, *sampling not acceptable*
- **Trace/Spans** signals: *why slow*, *where it broke*, *show the call tree*, *debug*, *live dashboard*, *sampling is fine*
- **Both** signals: one stakeholder needs durable proof **and** another needs technical root cause, in the same request
