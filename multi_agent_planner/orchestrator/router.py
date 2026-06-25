"""
Cost-aware router: assigns each task type to the cheapest capable agent.
  openai  → paid, highest quality  (day planning, meal generation)
  groq    → free, very fast        (RAG, hallucination check, book recs)
  ollama  → free, local            (summarisation, eval, fallback)
"""

ROUTING_TABLE: dict[str, str] = {
    # Day Planner
    "generate_schedule": "openai",
    "generate_meals": "openai",
    "weekly_cheat_meal": "openai",
    # RAG
    "rag_query": "groq",
    "hallucination_check": "groq",
    "rerank": "groq",
    # Knowledge / Light tasks
    "book_recommendation": "groq",
    "summarise": "ollama",
    "evaluate": "ollama",
    "fallback": "ollama",
}


class AgentRouter:
    def select(self, task_type: str) -> str:
        agent = ROUTING_TABLE.get(task_type, "ollama")
        return agent
