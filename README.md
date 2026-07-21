# LLMOps + RAG Platform (100% Free Stack)

An end-to-end LLMOps platform demonstrating production-grade RAG (Retrieval-Augmented
Generation) with experiment tracking, CI/CD, and observability — built entirely on
free/open-source tools.

## Why this project

Most RAG tutorials stop at "chatbot that answers questions from a PDF." This project
adds the **Ops** layer that separates a toy demo from a production system:

- **Prompt & experiment versioning** (MLflow) — every retrieval config and prompt template is tracked
- **Automated evaluation pipeline** — retrieval quality (hit rate, MRR) + generation quality (faithfulness) on every change
- **CI/CD** — GitHub Actions runs tests + eval suite on every PR, blocks merge on regression
- **Observability** — Prometheus + Grafana dashboards for latency, token usage, retrieval scores
- **Containerized serving** — FastAPI + Docker Compose, one command to run the whole stack locally

## Architecture

```
                    ┌─────────────────────┐
                    │   FastAPI Service   │  ← /query, /health, /metrics
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌───────▼────────┐
      │  Retriever    │ │  Generator   │ │  MLflow Tracker│
      │  (Chroma +    │ │  (Groq /     │ │  (params,      │
      │  HF Embeds)   │ │   Ollama)    │ │  metrics, runs)│
      └───────┬──────┘ └──────────────┘ └────────────────┘
              │
      ┌───────▼──────┐
      │  Ingestion    │  ← chunk, embed, upsert docs
      │  Pipeline     │
      └──────────────┘

      Prometheus ← scrapes /metrics ← Grafana dashboards
```

## Free stack used

| Component          | Tool                                  | Why free                          |
|---------------------|----------------------------------------|-------------------------------------|
| Embeddings          | `sentence-transformers` (HuggingFace)  | Runs locally, no API key           |
| Vector DB           | ChromaDB                               | Open-source, embedded/local        |
| LLM inference        | Groq API (free tier) or Ollama (local) | No cost for dev-scale usage         |
| Experiment tracking | MLflow (self-hosted)                   | Open-source, run via Docker        |
| Serving             | FastAPI + Uvicorn                      | Open-source                        |
| CI/CD               | GitHub Actions                         | Free for public repos              |
| Monitoring          | Prometheus + Grafana                   | Open-source, via Docker Compose    |
| Container runtime   | Docker + Docker Compose                | Free                                |

## Quickstart

```bash
# 1. Clone and install
git clone <your-repo-url>
cd llmops-rag-platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set your free Groq API key (https://console.groq.com — free tier)
cp .env.example .env
# edit .env and add GROQ_API_KEY=...
# (Alternative: install Ollama locally and skip the API key — see configs/config.yaml)

# 3. Ingest sample documents into the vector store
python -m src.rag.ingestion.ingest --source data/raw

# 4. Run the full stack (API + MLflow + Prometheus + Grafana)
docker compose -f docker/docker-compose.yml up --build

# 5. Query it
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# 6. Run the evaluation suite (also runs automatically in CI)
python -m src.rag.evaluation.evaluate
```

## Repo layout

```
llmops-rag-platform/
├── src/
│   ├── rag/
│   │   ├── ingestion/      # load, chunk, embed, upsert documents
│   │   ├── retrieval/      # vector search + retriever config
│   │   ├── generation/     # prompt templates + LLM call wrapper
│   │   └── evaluation/     # retrieval + generation quality metrics
│   ├── api/                # FastAPI app (serving layer)
│   └── monitoring/         # Prometheus metrics instrumentation
├── pipelines/               # MLflow-tracked pipeline entrypoints
├── configs/                 # YAML configs (model names, chunk sizes, etc.)
├── data/raw, data/processed # sample corpus + processed chunks
├── tests/                   # pytest unit + integration tests
├── docker/                  # Dockerfile + docker-compose (API, MLflow, Prometheus, Grafana)
├── .github/workflows/       # CI: lint, test, eval-gate
└── scripts/                 # helper shell scripts
```

## Resume bullet (example)

> Built an end-to-end LLMOps platform (RAG + MLflow + CI/CD eval gating + Prometheus/
> Grafana observability) using entirely open-source tooling; automated retrieval/
> generation quality checks blocked regressions pre-merge via GitHub Actions.

## Roadmap / stretch goals
- Swap Chroma for a hosted free-tier vector DB (e.g., Qdrant Cloud free tier) to show multi-backend support
- Add a re-ranker (e.g., `bge-reranker` via HuggingFace) as an ablation in MLflow
- A/B test two prompt templates and compare via MLflow metrics
