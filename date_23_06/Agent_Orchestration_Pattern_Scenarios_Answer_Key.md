# Agent Orchestration Pattern Scenarios — Answer Key

**Patterns:** Round-robin · Selector · Swarm/Handoff · GraphFlow · Magentic

---

## Summary Table

| # | Industry | Scenario | Pattern Chosen | Runner-up |
|---|---|---|---|---|
| 1 | Banking | Loan underwriting | **GraphFlow** | — |
| 2 | SaaS Support | Customer support triage | **Swarm / Handoff** | — |
| 3 | Media | Editorial workflow | **Round-robin** | — |
| 4 | Enterprise IT | Security incident investigation | **Magentic** | — |
| 5 | Professional Services | Client help desk | **Selector** | — |
| 6 | Insurance | Claims adjudication | **GraphFlow** | Magentic |
| 7 | Retail | Buyer's research assistant | **Magentic** | GraphFlow |
| 8 | Manufacturing | RFP response builder | **GraphFlow** | Swarm / Handoff |

---

## 1. Banking — Loan Underwriting

**Brief:** Every mortgage application passes the same steps in the same order — income verification → credit assessment → risk scoring → compliance sign-off — and the bank must replay the exact path for audit.

**Pattern: GraphFlow**

**Justification:** Same order every time, plus replay-for-audit, is the textbook case for an explicit, deterministic graph. The steps, their sequence, and the named edges between them are fixed and known in advance — exactly what a graph captures and what makes it replayable.

**Block Diagram:**
```
[Income Verification] ──► [Credit Assessment] ──► [Risk Scoring] ──► [Compliance Sign-off]
                                                                              │
                                                                              ▼
                                                                      [Audit Log / Replay]
```

**Why not the others:**
- *Selector* — never want a model improvising the route inside a compliance pipeline.
- *Round-robin* — order matters, but you also need the named, auditable edges a graph gives you.
- *Swarm/Handoff* — no dynamic peer routing; the path is fixed, not negotiated.
- *Magentic* — nothing here is open-ended; every step is known ahead of time.

---

## 2. SaaS Support — Customer Support Triage

**Brief:** A first-line assistant triages billing/technical/account issues and passes to the matching specialist. A specialist who realizes it's the wrong area hands it back to the front desk to re-route.

**Pattern: Swarm / Handoff**

**Justification:** Specialists pass control directly to each other by name (front desk ↔ specialist) and hand it back when it's the wrong fit. That's peer-to-peer control passing with no central router re-deciding every turn — the defining signature of Swarm/Handoff.

**Block Diagram:**
```
                  ┌──────────────┐
        ┌────────►│   Billing    │────────┐
        │         └──────────────┘        │ (wrong area → hand back)
        │                                  ▼
 ┌──────────────┐                  ┌──────────────┐
 │  Front Desk  │◄─────────────────┤  Technical   │
 │ (first-line) │                  └──────────────┘
        │                                  ▲
        │         ┌──────────────┐         │
        └────────►│Account Access│─────────┘
                  └──────────────┘
        peers hand control directly — no central re-router
```

**Why not the others:**
- *Selector* — a hub re-picking the next speaker each turn from one central point; here control passes directly between named peers.
- *Round-robin* — no fixed turn order; routing depends on issue type.
- *GraphFlow* — no fixed branch/merge structure; the back-and-forth re-routing is dynamic, not predetermined.

---

## 3. Media — Editorial Workflow

**Brief:** A newsroom drafts a short article with a fixed three-stage loop — writer drafts, editor critiques, writer revises — same roles, same order, repeating until the editor signs off.

**Pattern: Round-robin**

**Justification:** Fixed roles, a known order, a draft → review → revise loop, repeating mechanically until a stop condition. No routing brain is needed — just a steady ring between two fixed participants.

**Block Diagram:**
```
        ┌────────────┐   draft    ┌────────────┐
        │   Writer    │──────────►│   Editor    │
        │  (drafts/   │           │ (critiques) │
        │   revises)  │◄──────────│             │
        └────────────┘  feedback  └────────────┘
              ▲                          │
              └──── loop until editor signs off ──┘
```

**Why not the others:**
- *Swarm/Handoff* — no dynamic routing to named peers based on judgment; the order is fixed and mechanical.
- *Selector* — no one is picking "the best next speaker"; it's always the same two roles alternating.
- *GraphFlow* — no branch/merge structure, just a simple repeating cycle.

---

## 4. Enterprise IT — Security Incident Investigation

**Brief:** When an alert fires, an assistant investigates an unknown situation — pulls logs, runs queries, reads config files, looks up the relevant CVE — and decides its own next move as findings emerge. Nobody can script the steps in advance.

**Pattern: Magentic**

**Justification:** "Decide its own next move as findings emerge... nobody can script the steps in advance" is the defining signal for Magentic — an open-ended task where the next action depends on what was just discovered, so a planner adapts dynamically rather than following a predetermined path.

**Block Diagram:**
```
                ┌─────────────────┐
        ┌──────►│  Planner/Driver  │◄──────┐
        │       │ (decides next    │       │
        │       │  move from       │       │
        │       │  findings)       │       │
        │       └─────────────────┘       │
        │                                  │
  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐
  │ Pull Logs │ │   Run     │ │   Read    │ │  Look up │
  │           │ │  Queries  │ │  Configs  │ │   CVE    │
  └───────────┘ └───────────┘ └───────────┘ └──────────┘
        (steps dispatched dynamically, order unknown upfront)
```

**Why not the others:**
- *GraphFlow* — a graph requires branches to be known ahead of time; the investigation path is fundamentally unpredictable.
- *Swarm/Handoff* — no set of named peers passing control; one investigation adapting step by step.
- *Selector* — picks between known fixed speakers each turn; this is about discovering unknown next actions.

---

## 5. Professional Services — Client Help Desk

**Brief:** A consultancy's assistant fields mixed inbound questions — tax, legal, technical. For each new question, the system should pick the single best-suited expert to answer, based on what was asked.

**Pattern: Selector**

**Justification:** "Pick the single best-suited expert" per question is a central hub evaluating each new input and choosing one expert per turn — exactly "a model picks the best next speaker each turn."

**Block Diagram:**
```
                  ┌──────────────┐
   Question ─────►│   Selector    │
                  │ (hub picks 1) │
                  └───────┬──────┘
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌───────────┐    ┌───────────┐     ┌───────────┐
  │ Tax Expert │    │Legal Expert│     │Tech Expert │
  └───────────┘    └───────────┘     └───────────┘
```

**Why not the others:**
- *Swarm/Handoff* — no peer-to-peer handing off; one central decision point picks the responder each time.
- *Round-robin* — no fixed turn order; the expert chosen depends entirely on the question.
- *GraphFlow* — no branch/merge structure; it's a one-shot pick-and-answer per question.

---

## 6. Insurance — Claims Adjudication *(Close call)*

**Brief:** A claim needs three independent checks — fraud screening, policy-coverage check, medical-coding review — that run at the same time. A final decision agent combines all three results into approve/deny.

**Pattern: GraphFlow** | **Runner-up: Magentic**

**Justification:** Three independent checks running in parallel, then merged by a final decision agent, is a textbook fork → join. The checks and the combine step are fixed and known in advance — GraphFlow's explicit, deterministic graph fits exactly.

**Why Magentic is the close runner-up:** A planner could dispatch three workers in parallel and synthesize their outputs — superficially similar. The difference: Magentic is for tasks where the planner is discovering *what* work is needed as it goes. Here the three checks and the merge step are already known and fixed — nothing is being figured out dynamically — so it's a structured graph, not planner improvisation.

**Block Diagram:**
```
                    ┌──────────────────┐
              ┌────►│  Fraud Screening  │────┐
              │     └──────────────────┘    │
 [New Claim]──┼────►┌──────────────────┐    ├────►[Decision Agent]──►Approve/Deny
              │     │ Coverage Check    │    │
              │     └──────────────────┘    │
              └────►┌──────────────────┐    │
                    │ Coding Review     │────┘
                    └──────────────────┘
                 (fork — parallel)      (join — merge)
```

**Why not the others:**
- *Selector/Swarm* — no single "speaker" being picked, no handoff between named peers; this is simultaneous parallel execution.
- *Round-robin* — not a fixed shared-context loop; independent parallel branches.

---

## 7. Retail — Buyer's Research Assistant *(Close call)*

**Brief:** A merchandising team asks: "Find three trending materials for outdoor furniture this season and summarise supplier options." The number and type of sub-tasks isn't known in advance and may need web search and data lookups.

**Pattern: Magentic** | **Runner-up: GraphFlow**

**Justification:** "The number and type of sub-tasks isn't known in advance" is the defining Magentic signal — an open task where a planner figures out what work is needed (how many materials, how many supplier lookups) as it goes, not a task with predetermined steps.

**Why GraphFlow is the close runner-up:** The task looks like it splits cleanly into fork/join branches (find materials → look up suppliers → summarize). But GraphFlow needs branches fixed ahead of time. Here you don't know how many materials you'll find or how many lookups each needs until partway through — that unpredictability rules out a pre-built graph.

**Block Diagram:**
```
                ┌──────────────────┐
        ┌──────►│  Planner/Driver   │◄──────┐
        │       │ (decides sub-tasks│       │
        │       │  as they emerge)  │       │
        │       └──────────────────┘       │
        │                                   │
  ┌───────────┐ ┌────────────┐ ┌───────────┐
  │Web Search │ │  Material   │ │ Supplier   │ ──► [Summary]
  │           │ │  Lookup     │ │   Lookup   │
  └───────────┘ └────────────┘ └───────────┘
     (sub-tasks dispatched dynamically — count/type unknown upfront)
```

**Why not the others:**
- *Selector* — not picking one fixed expert per turn from a known roster; it's decomposing one task into unknown sub-work.
- *Swarm/Handoff* — no named peers passing control back and forth; one planner driving variable work toward a goal.

---

## 8. Manufacturing — RFP Response Builder *(Close call)*

**Brief:** A bid response has four sections (technical, pricing, compliance, timeline), each owned by a specialist and assembled in order. A reviewer then checks the assembled draft and may send specific sections back for rework before final sign-off.

**Pattern: GraphFlow** | **Runner-up: Swarm / Handoff**

**Justification:** Four fixed, known sections, each with a named owner, assembled in a set order, followed by a review/rework step — a deterministic graph with branch, join, and a conditional edge back for rework. The structure is fully known in advance.

**Why Swarm/Handoff is the close runner-up:** "Reviewer sends specific sections back for rework" looks like a handoff — control passing to a named peer based on judgment. Reasonable read. But the rest of the scenario (fixed sections, fixed owners, fixed assembly order) is far more structured than Swarm/Handoff implies, which is meant for dynamic path routing. A graph can model "reviewer may send section X back" as a defined conditional edge, without needing open-ended peer routing.

**Block Diagram:**
```
 [Technical]  ┐
 [Pricing]    ├──► [Assemble Draft] ──► [Reviewer] ──┐
 [Compliance] │                              ▲        │
 [Timeline]   ┘                              │        ▼
                              (rework edge) ──┘   [Final Sign-off]
                       reviewer routes specific sections back
```

**Why not the others:**
- *Selector* — no single hub picking one speaker per turn; multiple specialists own fixed parallel sections.
- *Round-robin* — not a simple repeating turn cycle; parallel ownership assembled into one document.
- *Magentic* — nothing is unknown in advance; sections, owners, and rework path are all specified upfront.

---

## Decision Lens Recap

| Signal in the brief | Pattern |
|---|---|
| Fixed roles, known order, draft → review → revise loop | **Round-robin** |
| Best responder depends on the question; pick one expert per turn | **Selector** |
| Specialists route to each other and hand control back (triage) | **Swarm / Handoff** |
| Same path every time — auditable, reproducible, or parallel checks that merge | **GraphFlow** |
| Steps/sub-tasks unknown in advance; planner must adapt as findings emerge | **Magentic** |
