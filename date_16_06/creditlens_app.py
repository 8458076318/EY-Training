# CreditLens FastAPI App — run with: uvicorn creditlens_app:app --reload
# Requires: pip install fastapi uvicorn


from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
from collections import defaultdict

app = FastAPI(
    title="CreditLens API",
    description="FinanceGuard AI Credit Policy Assistant — with safety guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://financeGuard.internal"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ── In-memory rate limiter ────────────────────────────────────
RATE_LIMIT      = 30   # max requests per window
WINDOW_SECONDS  = 60
request_counts  = defaultdict(list)

def rate_limit_check(request: Request):
    client_ip = request.client.host
    now = time.time()
    window_start = now - WINDOW_SECONDS

    # Prune old timestamps
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] if t > window_start
    ]

    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT} requests per {WINDOW_SECONDS}s"
        )
    request_counts[client_ip].append(now)

# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000,
                       description="Natural language credit policy question")
    user_id: str = Field(..., description="Loan officer employee ID")
    application_context: Optional[str] = Field(None, description="Optional application ID for context")

class QueryResponse(BaseModel):
    query_id: str
    response: str
    action: str          # answered | blocked | redirected
    latency_ms: float
    pii_redacted: bool
    safety_passed: bool

# ── Endpoints ─────────────────────────────────────────────────
@app.post("/v1/query", response_model=QueryResponse)
async def query_creditlens(
    body: QueryRequest,
    _: None = Depends(rate_limit_check)
):
    result = pipeline.run(body.query, user_id=body.user_id)
    m = result["metrics"]
    return QueryResponse(
        query_id=result["query_id"],
        response=result["response"],
        action=result["action"],
        latency_ms=m.latency_ms,
        pii_redacted=m.pii_redacted,
        safety_passed=m.output_safety_safe
    )

@app.get("/health")
def health(): return {"status": "ok", "service": "CreditLens"}

@app.get("/v1/metrics")
def get_metrics():
    df = pipeline.metrics_dataframe()
    if df.empty: return {"total": 0}
    return {
        "total_queries": len(df),
        "block_rate": (df["final_action"] == "block").mean(),
        "avg_latency_ms": df["latency_ms"].mean(),
        "pii_redaction_rate": df["pii_redacted"].mean(),
        "estimated_daily_cost_usd": df["estimated_cost_usd"].sum()
    }
