"""Document ingestion and vector store module for Mini AI Knowledge Assistant."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "faiss_index"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return local HuggingFace embedding model (all-MiniLM-L6-v2).

    Disables tqdm progress bars which cause OSError in non‑tty environments like Streamlit.
    """
    # Suppress progress bars and parallel tokenizers warnings in Streamlit
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "show_progress_bar": False},
    )


def load_pdf(file_path: str) -> List:
    """Extract pages and metadata from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    filename = Path(file_path).name

    for doc in raw_docs:
        # Normalize page number to 1-based index and preserve clean filename
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(raw_page) + 1
        doc.metadata["source"] = filename

    return raw_docs


def load_file(file_path: str) -> List:
    """Load a document of various supported types using appropriate loader.
    Supports PDF, DOCX/DOC, PPTX/PPT via UnstructuredFileLoader.
    Returns a list of LangChain Document objects with `source` metadata.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    # Fallback to UnstructuredFileLoader for other formats
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    filename = Path(file_path).name
    for doc in docs:
        # Unstructured loader may not provide page number; default to 1
        doc.metadata.setdefault("page", 1)
        doc.metadata["source"] = filename
    return docs


def chunk_documents(
    docs: List,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List:
    """Split documents into chunks while preserving source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Ensure source and page metadata are present in every chunk
    for chunk in chunks:
        chunk.metadata.setdefault("source", "Unknown Document")
        chunk.metadata.setdefault("page", 1)

    return chunks


def build_vector_store(
    chunks: List,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """Build a FAISS vector store from document chunks and persist to disk."""
    if not chunks:
        raise ValueError("Cannot build index from empty document chunks.")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    return vector_store


def load_vector_store(index_dir: str = DEFAULT_INDEX_DIR) -> Optional[FAISS]:
    """Reload an existing FAISS vector store from disk without re-embedding."""
    index_path = Path(index_dir)
    if not (index_path / "index.faiss").exists():
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )




def ingest_multiple_files(
    file_paths: List[str],
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """Ingest multiple documents of supported types, extract, chunk, and index.
    Supports PDFs, DOCX/DOC, PPTX/PPT via load_file.
    """
    all_chunks = []
    for path in file_paths:
        if not os.path.exists(path):
            continue
        docs = load_file(path)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text content could be extracted from provided documents.")

    return build_vector_store(all_chunks, index_dir=index_dir)
