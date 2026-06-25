#!/bin/bash
set -e

echo "==> Setting up Multi-Agent Day Planner"

# 1. Python deps
pip install -r requirements.txt

# 2. Pull Ollama model (free, local)
if command -v ollama &>/dev/null; then
  echo "==> Pulling Ollama mistral model (free)"
  ollama pull mistral
else
  echo "==> Ollama not found — install from https://ollama.com"
fi

# 3. Copy env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> .env created — fill in your API keys"
fi

echo "==> Done. Run: uvicorn api.main:app --reload"
