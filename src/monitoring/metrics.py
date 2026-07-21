"""Prometheus metric definitions, imported by the FastAPI app.

Scraped by Prometheus (see docker/prometheus.yml) and visualized in Grafana.
"""
from prometheus_client import Counter, Histogram

QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total number of RAG queries served",
)

QUERY_ERRORS = Counter(
    "rag_query_errors_total",
    "Total number of RAG queries that raised an error",
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end latency of a RAG query (retrieval + generation)",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
)

CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 2, 3, 4, 5, 8, 10],
)

TOKENS_USED = Counter(
    "rag_tokens_used_total",
    "Total tokens consumed by LLM calls",
    ["token_type"],  # "prompt" | "completion"
)
