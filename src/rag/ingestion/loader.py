"""Load raw text/markdown/PDF documents and split them into chunks."""
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document


def load_documents(source_dir: str) -> List[Document]:
    """Load all supported files (.txt, .md, .pdf) from a directory."""
    source_path = Path(source_dir)
    documents: List[Document] = []

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
        except Exception as e:
            print(f"[ingestion] Warning: failed loading {glob_pattern}: {e}")

    print(f"[ingestion] Loaded {len(documents)} raw documents from {source_dir}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingestion] Produced {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
