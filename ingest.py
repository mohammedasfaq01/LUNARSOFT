"""Document ingestion and vector store module for Mini AI Knowledge Assistant."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
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
    Sets an explicit cache_folder to a writable path to prevent Windows OSError 22 (invalid
    argument) when the default HF hub cache resolves to a path with spaces/Unicode characters.
    """
    import pathlib

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Explicit writable cache dir avoids Windows path-resolution errors
    cache_dir = str(pathlib.Path.home() / ".cache" / "huggingface" / "mini_ai_rag")
    os.makedirs(cache_dir, exist_ok=True)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=cache_dir,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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
    """Load a document of supported types using lightweight, purpose-specific loaders.

    Supports:
      - PDF  → PyPDFLoader (preserves page numbers)
      - DOCX/DOC → Docx2txtLoader (requires docx2txt)
      - PPTX/PPT → python-pptx (one Document per slide)
    """
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    if ext == ".pdf":
        return load_pdf(file_path)

    if ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("page", 1)
            doc.metadata["source"] = filename
        return docs

    if ext in (".pptx", ".ppt"):
        from pptx import Presentation
        from langchain_core.documents import Document

        prs = Presentation(file_path)
        docs = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            text_parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = " ".join(run.text for run in para.runs).strip()
                        if line:
                            text_parts.append(line)
            slide_text = "\n".join(text_parts).strip()
            if slide_text:
                docs.append(
                    Document(
                        page_content=slide_text,
                        metadata={"source": filename, "page": slide_num},
                    )
                )
        return docs

    raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, PPTX.")


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
