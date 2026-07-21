"""Orchestrates a single RAG query: retrieve -> build prompt -> generate -> log to MLflow."""
import os
import time
from typing import Dict

import mlflow

from src.rag.generation.generator import Generator
from src.rag.generation.prompt_builder import build_rag_prompt
from src.rag.retrieval.retriever import Retriever


class RAGChain:
    def __init__(self, config: dict):
        self.config = config
        self.retriever = Retriever(config)
        self.generator = Generator(config)

    def query(self, question: str, log_to_mlflow: bool = True) -> Dict:
        start = time.time()

        chunks = self.retriever.retrieve(question)
        messages = build_rag_prompt(
            question=question,
            context_chunks=chunks,
            system_prompt=self.config["generation"]["system_prompt"],
        )
        answer, usage = self.generator.generate(messages)

        total_latency = round(time.time() - start, 3)
        result = {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "num_chunks_retrieved": len(chunks),
            "usage": usage,
            "total_latency_seconds": total_latency,
        }

        if log_to_mlflow:
            self._log_run(result)

        return result

    def _log_run(self, result: Dict) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="query", nested=mlflow.active_run() is not None):
            mlflow.log_param("question", result["question"][:250])
            mlflow.log_param("model_name", self.config["generation"]["model_name"])
            mlflow.log_param("top_k", self.config["retrieval"]["top_k"])
            mlflow.log_metrics({
                "num_chunks_retrieved": result["num_chunks_retrieved"],
                "total_latency_seconds": result["total_latency_seconds"],
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
            })
