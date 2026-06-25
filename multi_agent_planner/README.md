# Multi-Agent Day Planner + RAG Knowledge Assistant

Production-grade multi-agent system with cost-aware routing.

## Architecture

| Agent | Model | Cost | Handles |
|-------|-------|------|---------|
| OpenAI | gpt-4o | Paid | Day planning, Indian meal generation |
| Groq | llama-3.1-8b-instant | Free | RAG retrieval, hallucination check |
| Ollama | mistral (local) | Free | Summarisation, evaluation, fallback |

## Quick Start

```bash
git clone <repo>
cd multi-agent-planner
bash scripts/setup.sh
uvicorn api.main:app --reload
```

## Docker

```bash
docker-compose up -d
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /planner/generate | Generate day plan + schedule SMS reminders |
| POST | /rag/query | RAG knowledge assistant query |
| GET  | /health | Service health check |

## SMS Reminders
- **India**: Set `FAST2SMS_API_KEY` (cheapest)
- **International**: Set `TWILIO_*` vars
- Fires 10 minutes before every scheduled activity

## Observability
- Prometheus metrics: `http://localhost:8001`
- Grafana dashboards: `http://localhost:3000`
