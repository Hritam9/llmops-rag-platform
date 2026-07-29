"""Load raw text/markdown/PDF documents and split them into chunks."""
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_documents(source_dir: str) -> list[Document]:
    """Load all supported files (.txt, .md, .pdf) from a directory."""
    source_path = Path(source_dir)
    documents: list[Document] = []

    for loader_cls, glob_pattern in [
        (TextLoader, "**/*.txt"),
        (TextLoader, "**/*.md"),
        (PyPDFLoader, "**/*.pdf"),
    ]:
        try:
            loader = DirectoryLoader(
                str(source_path),
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=False,
            )
            documents.extend(loader.load())
        except Exception as e:  # noqa: BLE001 - one bad/missing file type
            # (corrupt PDF, encoding error, etc.) must not abort ingestion
            # of the other file types in this loop.
            print(f"[ingestion] Warning: failed loading {glob_pattern}: {e}")

    print(f"[ingestion] Loaded {len(documents)} raw documents from {source_dir}")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingestion] Produced {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
