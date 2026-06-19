# Framework Selection Table

| Scenario | Recommended framework | One-line why | Main decision cue |
|---|---|---|---|
| Scenario 1 — The understaffed marketing team | **Dify** | Best fit because non-technical ops users can build and later change routing logic through a no-code/low-code UI without engineering help. | Non-technical business users, quick prototype, editable logic without code. |
| Scenario 2 — The research-brief assembly line | **CrewAI** | Best fit because the workflow has clear roles like researcher, analyst, writer, and editor, with structured handoffs between agents. | Multiple predefined agents with role → task → handoff flow. |
| Scenario 3 — The self-debugging data analyst | **AutoGen** | Best fit because coder and critic agents can have open-ended back-and-forth conversations and execute/debug Python code in a sandbox. | Unknown number of turns, agent conversation, code execution needed. |
| Scenario 4 — The regulated enterprise platform | **LangGraph** | Best fit because it supports custom stateful workflows, branching, loops, checkpoints, pause/resume, human-in-the-loop gates, and production observability. | Engineering-owned production system with control, compliance, tracing, and long-running workflows. |

## Quick rule of thumb

| Question | Choose |
|---|---|
| Non-technical users need to build or tweak it quickly | **Dify / no-code-low-code tool** |
| Clearly defined agent roles and sequential handoffs | **CrewAI** |
| Agents need to chat, critique, and iterate dynamically | **AutoGen** |
| Production-grade custom orchestration with state, branching, HITL, and observability | **LangGraph** |