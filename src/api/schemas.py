from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    total_latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
