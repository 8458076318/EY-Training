# 🗓️ AI Day Planner — 3-Agent Orchestration System

Production-ready multi-agent day planner with Streamlit UI, FastAPI backend,
and intelligent orchestration via CrewAI / LangGraph.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI (async)
- **Agents**: Gemini (free) → GPT-4 (fallback) | Groq | Ollama + CrewAI
- **Database**: PostgreSQL (async) + Pinecone / FAISS
- **Queue**: Celery + Redis
- **Infra**: Docker + Kubernetes (AKS) + Helm
- **Observability**: Prometheus + Grafana + Loki

## Quick Start
```bash
cp .env.example .env          # fill in your keys
make dev                      # docker-compose up --build
make migrate                  # run alembic migrations
```

## Docker Troubleshooting
If `docker-compose` fails with a named-pipe or API-version error on Windows:

1. Make sure Docker Desktop is running and the Docker Desktop service is started.
2. Confirm your Windows user is in the `docker-users` group, then sign out and sign back in.
3. If your profile has a broken `C:\Users\<you>\.docker` directory, move it aside or set `DOCKER_CONFIG` to a writable folder before retrying.
4. Retry the stack with `docker compose -f infra/docker/docker-compose.yml up postgres redis ollama api frontend worker`.

## Project Layout
```
src/         FastAPI app + agents + services
ui/          Streamlit UI
infra/       Docker, K8s, Helm charts
tests/       unit / integration / e2e
scripts/     DB seed, deploy helpers
docs/        Architecture, API reference
```
