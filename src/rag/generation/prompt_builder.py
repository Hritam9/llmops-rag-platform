"""Builds the final prompt sent to the LLM from retrieved context + user question."""


def build_rag_prompt(question: str, context_chunks: list[dict], system_prompt: str) -> list[dict[str, str]]:
    """Assemble a chat-format message list for the LLM call.

    Keeping this as its own function (rather than inline in generator.py) means
    prompt changes are isolated and easy to A/B test + log as an MLflow param.
    """
    context_text = "\n\n".join(
        f"[Source {i+1}]\n{chunk['text']}" for i, chunk in enumerate(context_chunks)
    )

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
