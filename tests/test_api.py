"""API tests. The RAG chain is mocked so these run without needing a live
vector store, Groq API key, or MLflow server — keeping CI fast and free.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.main.RAGChain") as mock_chain_cls:
        mock_chain = MagicMock()
        mock_chain.query.return_value = {
            "question": "test question",
            "answer": "test answer",
            "retrieved_chunks": [{"text": "chunk", "score": 0.9, "metadata": {}}],
            "num_chunks_retrieved": 1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "total_latency_seconds": 0.5,
        }
        mock_chain_cls.return_value = mock_chain

        from src.api.main import app
        with TestClient(app) as test_client:
            yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint(client):
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "test answer"
    assert len(body["retrieved_chunks"]) == 1


def test_query_endpoint_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
