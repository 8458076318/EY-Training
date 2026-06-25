from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid
import os
from pathlib import Path

# ── Load .env ──
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

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env!")

app = FastAPI(title="AutoGen Multi-Agent API")

# Store results in memory
results = {}

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

# ── Build team function ──
def build_team():
    researcher_model = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )
    editor_model = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

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

    team = RoundRobinGroupChat(
        participants=[researcher, editor],
        termination_condition=MaxMessageTermination(max_messages=4),
    )
    return team

# ── Background task executor ──
async def execute_task(task_id: str, task: str):
    try:
        team = build_team()
        output = []

        async for message in team.run_stream(task=task):
            if isinstance(message, TaskResult):
                # Final result object
                for msg in message.messages:
                    output.append({
                        "agent": msg.source,
                        "content": msg.content
                    })
            elif hasattr(message, "source") and hasattr(message, "content"):
                # Individual streaming message
                output.append({
                    "agent": message.source,
                    "content": message.content
                })

        results[task_id] = {"status": "completed", "output": output}
        print(f"✅ Task {task_id} completed with {len(output)} messages")

    except Exception as e:
        results[task_id] = {"status": "error", "error": str(e)}
        print(f"❌ Task {task_id} failed: {e}")

# ── Routes ──

@app.post("/run", response_model=TaskResponse)
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    results[task_id] = {"status": "running", "output": []}
    background_tasks.add_task(execute_task, task_id, request.task)
    print(f"🚀 Started task {task_id}: {request.task}")
    return {"task_id": task_id, "status": "running"}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    return results.get(task_id, {"status": "not_found"})

@app.get("/health")
def health():
    return {"status": "ok", "agents": ["Researcher", "Editor"]}

@app.get("/tasks")
def list_tasks():
    return {
        "total": len(results),
        "tasks": {tid: r["status"] for tid, r in results.items()}
    }