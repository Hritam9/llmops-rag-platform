"""Wraps a free, local HuggingFace sentence-transformers model for embeddings."""
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance.

    Cached so repeated calls (e.g. across ingestion + retrieval) don't reload
    the model weights from disk every time.
    """
    print(f"[embeddings] Loading '{model_name}' on {device} (first call downloads weights, then cached)")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
