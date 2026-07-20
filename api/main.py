"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(title="GraphRAG Engine", version="0.1.0", lifespan=lifespan)


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
