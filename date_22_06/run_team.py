import asyncio
import os
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ── Load .env from C:\Training\AI-ML-Training-Projects\.env ──
env_path = Path(r"C:\Training\AI-ML-Training-Projects\.env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    print(f"✅ Loaded .env from {env_path}")
else:
    print(f"❌ .env not found at {env_path}")

# ── Read keys ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = "https://api.openai.com/v1"   # standard OpenAI

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env file!")

print(f"✅ API Key loaded: {OPENAI_API_KEY[:8]}...")

async def run_team(task: str):

    # ── Models (using OpenAI GPT) ──
    researcher_model = OpenAIChatCompletionClient(
        model="gpt-4o-mini",          # cheap & fast
        api_key=OPENAI_API_KEY,
    )
    editor_model = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

    # ── Agents ──
    researcher = AssistantAgent(
        name="Researcher",
        model_client=researcher_model,
        system_message="You are an expert researcher. Provide detailed summaries in Markdown.",
    )
    editor = AssistantAgent(
        name="Editor",
        model_client=editor_model,
        system_message="You are a strict editor. Critique and optimize the researcher's work.",
    )

    # ── Team ──
    team = RoundRobinGroupChat(
        participants=[researcher, editor],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    # ── Run (non-streaming, most reliable) ──
    print(f"\n🚀 Task: {task}\n{'='*50}")
    
    result = await team.run(task=task)
    
    # result is a TaskResult object
    for msg in result.messages:
        print(f"\n[{msg.source}]:\n{msg.content}")
        print("-" * 40)
    
    print("\n✅ Session complete!")

if __name__ == "__main__":
    task = "Explain why GPUs are important for training large language models."
    asyncio.run(run_team(task))