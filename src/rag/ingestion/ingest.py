"""End-to-end ingestion pipeline: load -> chunk -> embed -> upsert into Chroma.

Usage:
    python -m src.rag.ingestion.ingest --source data/raw
"""
import argparse
import time

import chromadb
import mlflow

import os

from src.config import load_config
from src.rag.ingestion.embedder import get_embedding_model
from src.rag.ingestion.loader import chunk_documents, load_documents


def run_ingestion(source_dir: str, config: dict) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="ingestion"):
        start = time.time()

        # 1. Log config params for reproducibility
        mlflow.log_params({
            "chunk_size": config["chunking"]["chunk_size"],
            "chunk_overlap": config["chunking"]["chunk_overlap"],
            "embedding_model": config["embedding"]["model_name"],
            "source_dir": source_dir,
        })

        # 2. Load + chunk
        docs = load_documents(source_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )

        if not chunks:
            print("[ingestion] No documents found — add files to data/raw/ first.")
            mlflow.log_metric("num_chunks", 0)
            return

        # 3. Embed
        embedder = get_embedding_model(
            config["embedding"]["model_name"],
            config["embedding"]["device"],
        )
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        vectors = embedder.embed_documents(texts)

        # 4. Upsert into Chroma
        client = chromadb.PersistentClient(path=config["vector_store"]["persist_dir"])
        collection = client.get_or_create_collection(
            name=config["vector_store"]["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        elapsed = time.time() - start
        mlflow.log_metrics({
            "num_chunks": len(chunks),
            "ingestion_seconds": elapsed,
        })
        print(f"[ingestion] Upserted {len(chunks)} chunks into '{config['vector_store']['collection_name']}' "
              f"in {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/raw", help="Directory of source documents")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_ingestion(args.source, cfg)
