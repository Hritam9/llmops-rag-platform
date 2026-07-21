from langchain_core.documents import Document

from src.rag.ingestion.loader import chunk_documents


def test_chunk_documents_respects_chunk_size():
    long_text = "word " * 1000
    docs = [Document(page_content=long_text, metadata={"source": "test"})]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 250  # allow slight overshoot from splitter


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="short text", metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "test.txt"


def test_chunk_documents_empty_input():
    assert chunk_documents([], chunk_size=500, chunk_overlap=50) == []
