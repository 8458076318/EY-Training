from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


app = FastAPI(
    title="Order Management API",
    description=(
        "End-to-end FastAPI demo with nested Pydantic models and the "
        "extension response pattern."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health"])
def health() -> dict:
    return {"status": "ok", "message": "Order Management API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapiapp.app.main:app", host="127.0.0.1", port=8000, reload=False)
