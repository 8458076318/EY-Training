# Architecture Overview

## Agent Pipeline

```
User Request
    │
    ▼
FastAPI (async)
    │
    ▼
AgentOrchestrator
    ├─► Agent 1: Analyser (Groq) — enriches context from history
    │
    └─► Agent 2: Day Planner (Gemini → GPT-4 fallback) — generates plan
              │
              ▼
         Background
    └─► Agent 3: Feedback (Ollama local) — improves future plans
```

## Cost Strategy
- Gemini free tier handles ~90% of requests
- GPT-4 fallback only on rate-limit or error
- Groq free tier for fast history analysis
- Ollama runs locally inside K8s — zero API cost
