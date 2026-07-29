"""Vector similarity retriever backed by ChromaDB."""

import chromadb

from src.rag.ingestion.embedder import get_embedding_model


class Retriever:
    """Thin wrapper around a Chroma collection for top-k semantic search."""

    def __init__(self, config: dict):
        self.config = config
        self.embedder = get_embedding_model(
            config["embedding"]["model_name"],
            config["embedding"]["device"],
        )
        self.client = chromadb.PersistentClient(path=config["vector_store"]["persist_dir"])
        self.collection = self.client.get_or_create_collection(
            name=config["vector_store"]["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k most similar chunks for a query, with scores."""
        k = top_k or self.config["retrieval"]["top_k"]
        query_vector = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )

        threshold = self.config["retrieval"]["score_threshold"]
        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Collection is created with hnsw:space="cosine", so Chroma's
            # "distance" here is true cosine distance (0-2, lower = closer).
            # similarity = 1 - distance is only valid under that metric —
            # Chroma's default (squared L2) would make this conversion wrong
            # and silently over-filter correct matches.
            similarity = 1 - dist
            if similarity >= threshold:
                hits.append({"text": doc, "metadata": meta, "score": round(similarity, 4)})

        return hits
