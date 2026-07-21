# About This Project

This project is an LLMOps platform demonstrating production-grade RAG
(Retrieval-Augmented Generation) with experiment tracking, CI/CD, and
observability, built entirely on free and open-source tools.

## Vector Database

The project uses ChromaDB as its vector database. Chroma is open-source,
runs embedded (no server required), and persists data to local disk,
making it ideal for free-tier development and small-to-medium production
workloads.

## Experiment Tracking

MLflow is used for experiment tracking. Every ingestion run and every
query logs its parameters (chunk size, embedding model, retrieval top_k)
and metrics (latency, token usage, number of chunks retrieved) to MLflow,
making experiments fully reproducible and comparable over time.

## Embeddings

Embeddings are generated locally using a HuggingFace sentence-transformers
model (all-MiniLM-L6-v2), which requires no API key and runs on CPU.

## LLM Inference

LLM inference is powered by either the Groq API (free tier, extremely
fast) or Ollama (fully local, zero cost), configurable via config.yaml.

## CI/CD

GitHub Actions runs the test suite and the evaluation suite on every pull
request. If the average evaluation score drops below a configured
threshold, the CI pipeline fails, preventing quality regressions from
being merged.
