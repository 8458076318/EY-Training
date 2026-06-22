# Agent Architecture Scenarios — Single vs Multi-agent

> Five industry scenarios designed to surface genuine architectural trade-offs.
> **Decision logic:**
> - **Single agent** when sub-tasks share one context, pipeline is linear, or there's a hard latency SLA
> - **Multi-agent** when work can be parallelised across independent units, stages need different tool surfaces or security scopes, or you need failure isolation and human approval gates between stages

---

## 1. Healthcare / Clinical AI · Apollo Diagnostics

### Automated radiology report + care pathway

**Scenario:** Apollo wants to automate the end-to-end workflow when a chest CT scan arrives:
1. A radiologist-grade model reads the scan and drafts a findings report
2. A clinical-decision-support system cross-checks findings against the patient's medication history for contraindications
3. A scheduling agent books the recommended follow-up (e.g. biopsy, PET scan) in the hospital's EMR
4. A communication agent drafts the GP letter and patient-facing summary

Each step has a distinct tool, knowledge domain, and failure mode.

**Tags:** `4 distinct domains` · `Sequential with gates` · `Different tool access`

### ✅ Answer: Multi-agent

### Justification

| Signal | Reasoning |
|---|---|
| 4 distinct domains | Radiology AI, clinical decision support, EMR scheduling, and GP communication are fundamentally different knowledge domains — one model cannot hold all four |
| Sequential with gates | Cannot book a biopsy before contraindications are cleared; cannot draft a GP letter before a procedure is booked. Human approval gates are mandatory between stages |
| Different tool access | DICOM viewer, drug interaction DB, hospital EMR API, GP portal — four separate tool surfaces. Giving one agent access to all four is a security boundary violation |

### Architecture: Sequential multi-agent with human approval gates

```
CT scan arrives
      │
      ▼
┌─────────────────────────────┐
│  Agent 1 — Radiology AI     │  Tool: DICOM viewer / imaging model
│  Reads scan, drafts report  │
└─────────────────────────────┘
      │
  ⬡ Gate — findings validated (radiologist)
      │
      ▼
┌─────────────────────────────┐
│  Agent 2 — Clinical DSS     │  Tool: Drug interaction DB, medication records
│  Cross-checks contraindic.  │
└─────────────────────────────┘
      │
  ⬡ Gate — clinician approval
      │
      ▼
┌─────────────────────────────┐
│  Agent 3 — Scheduling       │  Tool: Hospital EMR API, calendar
│  Books follow-up procedure  │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Agent 4 — Communication    │  Tool: Messaging platform, GP portal
│  GP letter + patient summary│
└─────────────────────────────┘
      │
      ▼
  Output delivered
```

**Why not single agent?** A single agent would need simultaneous access to all four tool surfaces (security violation), no failure isolation (one tool crash corrupts everything), and no human gate before an AI autonomously books a clinical procedure.

---

## 2. E-commerce / Retail · ShopIQ

### Personalised product recommendation email

**Scenario:** ShopIQ runs a nightly batch job to send personalised recommendation emails to 4 million users. For each user:
1. Pull their 6-month purchase and browse history
2. Run a collaborative-filtering model to get top-10 candidates
3. Apply business rules (exclude out-of-stock, exclude recently-purchased)
4. Write a short personalised intro paragraph
5. Assemble the email HTML

The model call, rules, copywriting, and assembly all use the **same user context object** and must complete in **under 3 seconds per user**.

**Tags:** `Batch: 4M users` · `Shared user context` · `< 3s per user`

### ✅ Answer: Single agent

### Justification

| Signal | Reasoning |
|---|---|
| Shared user context | All four steps consume the same user object — no data handoff problem to solve, no reason to split context across agents |
| < 3s per user | Multi-agent coordination adds 10–50ms per hop. A chain of agents per user would blow the SLA |
| Batch: 4M users | The parallelism needed is *across* users (4M independent single-agent instances), not *within* a single user's job |
| Linear pipeline | Pull → filter → rules → copy → HTML is a linear transformation of one object. No branching, no independent sub-tasks |

### Architecture: Single agent × 4M parallel instances

```
┌─────────────────────────────────────────────────────────┐
│  Single agent (one context window per user)             │
│                                                         │
│  [User context object — shared across all steps]        │
│       │                                                 │
│       ▼                                                 │
│  Step 1 — Retrieve 6-month history                      │
│       │                                                 │
│       ▼                                                 │
│  Step 2 — Collaborative filtering → top-10 candidates   │
│       │                                                 │
│       ▼                                                 │
│  Step 3 — Business rules (OOS, recent-purchase filter)  │
│       │                                                 │
│       ▼                                                 │
│  Step 4 — Write personalised intro paragraph            │
│       │                                                 │
│       ▼                                                 │
│  Step 5 — Assemble email HTML → dispatch                │
└─────────────────────────────────────────────────────────┘
         × 4,000,000 parallel instances (one per user)
```

**The trap:** Multi-agent feels right because there are 5 distinct steps. But the decision rule is: single agent when sub-tasks share one context and the pipeline is linear. Both conditions are explicitly stated.

---

## 3. Legal / LegalTech · ContractIQ

### M&A due diligence on 800 contracts

**Scenario:** A PE firm uploads 800 supplier and employment contracts ahead of an acquisition. ContractIQ must:
1. Extract key obligations and risk clauses from each document *(parallelisable across contracts)*
2. Cross-reference extracted clauses against a jurisdiction-specific regulatory checklist
3. Identify inter-contract dependencies (e.g. change-of-control clauses that cascade)
4. Produce an executive risk summary with a red/amber/green heat map

Total turnaround required: **under 4 hours**. Documents are independent at extraction but **interdependent at the synthesis stage**.

**Tags:** `800 docs, parallel` · `Cross-doc synthesis` · `4-hour SLA`

### ✅ Answer: Multi-agent

### Justification

| Signal | Reasoning |
|---|---|
| 800 docs, parallel | 800 contracts sequentially at even 1 min/contract = 13+ hours. Parallel extraction agents are the only way to meet the 4-hour SLA |
| Cross-doc synthesis | Jurisdiction check, inter-contract dependency mapping, and heat map generation all require seeing *all* outputs together — a synthesis orchestrator must fan back in |
| 4-hour SLA | Forces fan-out at extraction; the SLA is the architectural constraint |

### Architecture: Fan-out / Fan-in (Map-Reduce)

```
800 contracts uploaded
         │
    ┌────┴────┐  (fan-out)
    │         │
    ▼         ▼          ▼  ... × 800
┌────────┐ ┌────────┐ ┌────────┐
│ Agent  │ │ Agent  │ │ Agent  │   Each agent:
│ Doc 1  │ │ Doc 2  │ │ Doc 800│   ① Extract obligations + risk clauses
└────────┘ └────────┘ └────────┘   ② Tag clause types
    │         │          │         ③ Initial R/A/G score
    └────┬────┘ (fan-in) ┘
         │
         ▼
  [Structured clause store]
         │
         ▼
┌──────────────────────────────┐
│  Synthesis orchestrator      │
│                              │
│  Step A: Jurisdiction check  │  Cross-ref all clauses vs regulatory checklist
│  Step B: Dependency map      │  Identify cascading change-of-control clauses
│  Step C: Risk heat map       │  Aggregate R/A/G scores across all 800 docs
└──────────────────────────────┘
         │
         ▼
  Executive risk summary → PE firm
```

**Pattern:** Documents independent at extraction → parallel agents. Documents interdependent at synthesis → single orchestrator fans back in. Classic map-reduce.

**Why not single agent?** Fitting 800 contracts into one context window is impossible. Even if it were possible, sequential processing would miss the 4-hour SLA by 3×.

---

## 4. RegTech / Compliance · FinSecure Bank

### Real-time transaction fraud screening

**Scenario:** FinSecure processes 2 million card transactions per second across retail and corporate accounts. Each transaction must be assessed for fraud risk within **80ms** or the authorization gateway times out. The check involves:
1. Running the transaction against a **rules engine**
2. A **risk-score model** (ML inference)
3. A **velocity lookup** (how many txns has this card done in the last 60 seconds?)

All three checks draw from the **same transaction context**. Hard latency SLA: **p99 < 80ms**.

**Tags:** `Latency SLA: p99 < 80ms` · `Shared context` · `Stateless checks`

### ✅ Answer: Single agent

### Justification

| Signal | Reasoning |
|---|---|
| p99 < 80ms | Multi-agent coordination (even lightweight) adds 10–50ms per hop. With 3 checks, a multi-agent chain would blow the p99 SLA. 80ms is the entire budget |
| Shared context | All three checks consume the same transaction object — no data handoff, no reason to split context |
| Stateless checks | Stateless ≠ multi-agent. Stateless means the three checks can run as **parallel tool calls within a single agent**, not as separate agents communicating with each other |

### Architecture: Single agent with parallel tool calls

```
Transaction arrives
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Single agent (one context window per transaction)       │
│                                                          │
│  Transaction context: card_id · amount · merchant ·      │
│  timestamp · account_type                                │
│         │                                                │
│    ┌────┴────┬────────────┐  (parallel tool calls)       │
│    ▼         ▼            ▼                              │
│  Rules     Risk-score   Velocity                         │
│  engine    model        lookup                           │
│  Blocklist ML inference Txns on card                     │
│  patterns  on features  in last 60s                      │
│    │         │            │                              │
│    └────┬────┴────────────┘  (results fan back in)       │
│         ▼                                                │
│  Agent synthesises → APPROVE / DECLINE / FLAG            │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  Authorization gateway response
```

**The key distinction from ContractIQ:** Both have parallel work. But ContractIQ's parallelism justifies agents (800 docs, minutes each). FinSecure's parallelism is 3 fast lookups within an 80ms budget — handled by parallel tool calls inside one agent, not separate agents.

---

## Decision Framework Summary

| Scenario | Answer | Decisive signal |
|---|---|---|
| Apollo — radiology pathway | **Multi-agent** | Different tools + human approval gates + 4 distinct domains |
| ShopIQ — recommendation email | **Single agent** | Shared user context + hard 3s SLA + linear pipeline |
| ContractIQ — M&A due diligence | **Multi-agent** | 800 docs parallel (fan-out) + cross-doc synthesis (fan-in) |
| FinSecure — fraud screening | **Single agent** | p99 < 80ms + shared context + stateless parallel tool calls |

### Quick decision rules

```
Is there a hard sub-100ms latency SLA?
  → Single agent (coordination overhead kills it)

Do sub-tasks share one context object and form a linear pipeline?
  → Single agent

Are there N independent documents/items to process in parallel
  AND a synthesis step that needs all outputs?
  → Multi-agent (fan-out / fan-in)

Do stages need different tool surfaces, security scopes,
  or human approval gates between them?
  → Multi-agent

Is the parallelism needed across jobs (not within one job)?
  → Single agent × N parallel instances
```
