from src.rag.generation.prompt_builder import build_rag_prompt


def test_build_rag_prompt_structure():
    chunks = [
        {"text": "Chroma is a vector database.", "metadata": {}, "score": 0.9},
        {"text": "MLflow tracks experiments.", "metadata": {}, "score": 0.8},
    ]
    messages = build_rag_prompt(
        question="What is Chroma?",
        context_chunks=chunks,
        system_prompt="You are a helpful assistant.",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Chroma is a vector database." in messages[1]["content"]
    assert "What is Chroma?" in messages[1]["content"]


def test_build_rag_prompt_empty_context():
    messages = build_rag_prompt(
        question="Anything?",
        context_chunks=[],
        system_prompt="System.",
    )
    assert "Question: Anything?" in messages[1]["content"]
