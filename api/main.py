"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config.logging import setup_logging
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(title="GraphRAG Engine", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    settings = get_settings()
    if settings.api_key and request.url.path != "/health":
        if request.headers.get("Authorization") != f"Bearer {settings.api_key}":
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


from api.routes import documents, ingest, query, graph, summary

app.include_router(ingest.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(graph.router)
app.include_router(summary.router)
