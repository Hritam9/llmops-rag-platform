"""FastAPI serving layer for the RAG platform.

Run directly:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Or via Docker Compose (see docker/docker-compose.yml).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.config import load_config
from src.monitoring.metrics import (
    CHUNKS_RETRIEVED,
    QUERY_COUNT,
    QUERY_ERRORS,
    QUERY_LATENCY,
    TOKENS_USED,
)
from src.rag.generation.rag_chain import RAGChain

rag_chain: RAGChain | None = None
config: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, config
    config = load_config()
    rag_chain = RAGChain(config)
    print("[api] RAG chain initialized and ready")
    yield
    print("[api] Shutting down")


app = FastAPI(
    title="LLMOps RAG Platform",
    description="Production-style RAG API with MLflow tracking and Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        vector_store_ready=rag_chain is not None,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    QUERY_COUNT.inc()
    try:
        with QUERY_LATENCY.time():
            result = rag_chain.query(request.question)

        CHUNKS_RETRIEVED.observe(result["num_chunks_retrieved"])
        TOKENS_USED.labels(token_type="prompt").inc(result["usage"]["prompt_tokens"])
        TOKENS_USED.labels(token_type="completion").inc(result["usage"]["completion_tokens"])

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
            total_latency_seconds=result["total_latency_seconds"],
        )
    except Exception as e:
        QUERY_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
