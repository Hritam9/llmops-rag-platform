#!/usr/bin/env bash
# Run ingestion then start the API locally (no Docker) — useful for quick dev loops.
set -e

echo "Ingesting documents from data/raw ..."
python -m src.rag.ingestion.ingest --source data/raw

echo "Starting API on http://localhost:8000 ..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
