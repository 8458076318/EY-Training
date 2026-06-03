import os, json
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
 
 
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        f"GROQ_API_KEY was not found. Add it to {env_path} as GROQ_API_KEY=your_key"
    )

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=api_key
)
 
# --- shared state schema ---
class AgentState(TypedDict):
    goal:       str
    tasks:      List[str]
    results:    List[str]
    critique:   str
    approved:   bool
    iterations: int
 
def planner(state: AgentState) -> AgentState:
    system = """You are a planning agent. Break the user's goal into
at most 5 concrete, actionable tasks. Respond ONLY with a
valid JSON array of strings. No preamble, no markdown."""
 
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    response = llm.invoke(messages).content.strip()
 
    try:
        clean = response.replace("```json","").replace("```","").strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [response]   # fallback: treat whole response as one task
 
    print(f"\n[Planner] Generated {len(tasks)} tasks:")
    for i, t in enumerate(tasks): print(f"  {i+1}. {t}")
 
    return {**state, "tasks": tasks}
 
initial_state: AgentState = {
    "goal":       "Research and summarise the top 3 trends in agriculture for 2025",
    "tasks":      [],
    "results":    [],
    "critique":   "",
    "approved":   False,
    "iterations": 0
}
planner(initial_state)
