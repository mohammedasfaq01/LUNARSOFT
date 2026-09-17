"""Streamlit Chat Application for Mini AI Knowledge Assistant (RAG)."""

import os
import shutil
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from ingest import (
    DEFAULT_INDEX_DIR,
    ingest_multiple_files,
    load_vector_store,
)
from rag import query_rag

load_dotenv()

st.set_page_config(
    page_title="Mini AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)


def get_secret(key_name: str) -> str:
    """Retrieve secret from Streamlit secrets or OS environment."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, "")


# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    # Ensure the index directory exists before attempting to load
    os.makedirs(DEFAULT_INDEX_DIR, exist_ok=True)
    # Attempt to load previously persisted FAISS index if available
    st.session_state.vector_store = load_vector_store(DEFAULT_INDEX_DIR)

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings & Ingestion")

    provider_options = ["Offline Grounded Engine (No API Key Required)", "Gemini", "Groq"]
    default_gemini_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    default_groq_key = get_secret("GROQ_API_KEY")
    
    default_provider_idx = 1 if default_gemini_key else (2 if default_groq_key else 0)

    provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=default_provider_idx,
        help="Select between Offline Grounded Engine (free local) or Gemini / Groq free tiers.",
    )

    if provider == "Gemini":
        api_key = st.text_input(
            "Gemini API Key",
            value=default_gemini_key,
            type="password",
            help="Provide your Gemini API key (or set in .env / st.secrets).",
        )
        model_name = st.selectbox(
            "Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
            index=0,
        )
        if not api_key:
            st.info("💡 Don't have an API key? Switch to 'Offline Grounded Engine' to test without a key.")
    elif provider == "Groq":
        api_key = st.text_input(
            "Groq API Key",
            value=default_groq_key,
            type="password",
            help="Provide your Groq API key (or set in .env / st.secrets).",
        )
        model_name = st.selectbox(
            "Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=1,  # default to free-tier model likely available
        )
        if not api_key:
            st.info("💡 Don't have an API key? Switch to 'Offline Grounded Engine' to test without a key.")
    else:
        api_key = None
        model_name = "offline-grounded"
        st.success("🟢 Local CPU Grounded Engine active. Zero external API calls.")

    st.divider()
    st.subheader("📄 Document Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "doc", "pptx", "ppt"],
        accept_multiple_files=True,
        help="Upload PDFs, Word docs, or PowerPoint presentations to index into FAISS.",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_clicked = st.button("🚀 Index Uploads", use_container_width=True)
    with col_btn2:
        load_samples_clicked = st.button("📂 Load Samples", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Extracting, chunking, and indexing documents..."):
                temp_dir = tempfile.mkdtemp(prefix="rag_docs_")
                saved_paths = []
                doc_names = []
                try:
                    for up_file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, up_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(up_file.getbuffer())
                        saved_paths.append(temp_file_path)
                        doc_names.append(up_file.name)

                    # Build FAISS index from multiple documents
                    vs = ingest_multiple_files(saved_paths, index_dir=DEFAULT_INDEX_DIR)
                    st.session_state.vector_store = vs
                    st.session_state.indexed_docs = doc_names
                    st.success(f"Successfully indexed {len(doc_names)} document(s)!")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    if load_samples_clicked:
        with st.spinner("Indexing sample multi-document PDFs (Apollo & Security)..."):
            sample_dir = Path(__file__).parent / "sample_docs"
            if not (sample_dir / "project_apollo.pdf").exists():
                from eval import setup_sample_documents
                setup_sample_documents()
            samples = [
                str(sample_dir / "project_apollo.pdf"),
                str(sample_dir / "security_policy.pdf"),
            ]
            vs = ingest_multiple_files(samples, index_dir=DEFAULT_INDEX_DIR)
            st.session_state.vector_store = vs
            st.session_state.indexed_docs = ["project_apollo.pdf", "security_policy.pdf"]
            st.success("Sample documents loaded and indexed!")
            st.rerun()

    # ── Local Path Indexing ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Index from Local Path")
    st.caption("Enter a file path or a folder path. All supported files inside a folder will be indexed.")

    local_path_input = st.text_input(
        "File or Folder Path",
        placeholder=r"e.g. C:\Users\You\Documents\reports  or  C:\report.pdf",
        help="Supports PDF, DOCX, DOC, PPTX, PPT. For folders, all matching files are indexed recursively.",
    )
    index_path_clicked = st.button("📥 Index from Path", use_container_width=True)

    if index_path_clicked:
        raw_path = local_path_input.strip().strip('"').strip("'")
        if not raw_path:
            st.warning("Please enter a file or folder path.")
        else:
            target = Path(raw_path)
            SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}
            if not target.exists():
                st.error(f"Path not found: `{raw_path}`")
            else:
                with st.spinner(f"Scanning and indexing from `{target.name}`..."):
                    try:
                        if target.is_file():
                            if target.suffix.lower() not in SUPPORTED_EXTS:
                                st.error(f"Unsupported file type: `{target.suffix}`. Use PDF, DOCX, or PPTX.")
                                file_paths = []
                            else:
                                file_paths = [str(target)]
                                doc_names = [target.name]
                        else:
                            # Recursively collect all supported files in folder
                            file_paths = [
                                str(p) for p in sorted(target.rglob("*"))
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                            ]
                            doc_names = [Path(p).name for p in file_paths]

                        if file_paths:
                            if not doc_names:
                                doc_names = [Path(p).name for p in file_paths]
                            vs = ingest_multiple_files(file_paths, index_dir=DEFAULT_INDEX_DIR)
                            st.session_state.vector_store = vs
                            st.session_state.indexed_docs = doc_names
                            st.success(f"✅ Indexed {len(doc_names)} document(s) from `{target.name}`!")
                            st.rerun()
                        elif target.exists():
                            st.warning("No supported documents found at that path.")
                    except Exception as e:
                        st.error(f"Error during ingestion: {str(e)}")

    # Display Index Status
    if st.session_state.vector_store is not None:
        st.success("✅ Knowledge Base Active (FAISS loaded)")
        if st.session_state.indexed_docs:
            st.caption("Indexed files:")
            for name in st.session_state.indexed_docs:
                st.write(f"- 📄 `{name}`")
        else:
            st.caption("Indexed chunks loaded from disk.")
    else:
        st.info("ℹ️ No active index. Upload documents or enter a path to get started.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🤖 Mini AI Knowledge Assistant")
st.caption(
    "Strictly grounded RAG assistant powered by FAISS, "
    "`all-MiniLM-L6-v2`, and verifiable page-level citations."
)

# Render Chat History
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            with st.expander(f"📚 View Sources & Citations ({len(msg['sources'])} chunks)"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**Citation {idx}**: 📄 `{src['source']}` — **Page {src['page']}**"
                    )
                    st.caption(f"> {src['snippet']}")

# User Query Input
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    # 1. Display and save user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate response via RAG
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating grounded answer..."):
            result = query_rag(
                query=user_query,
                vector_store=st.session_state.vector_store,
                provider=provider,
                api_key=api_key,
                model=model_name,
                k=4,
            )
            answer_text = result["answer"]
            citations = result.get("sources", [])

            st.markdown(answer_text)

            if citations:
                with st.expander(f"📚 View Sources & Citations ({len(citations)} chunks)"):
                    for idx, src in enumerate(citations, start=1):
                        st.markdown(
                            f"**Citation {idx}**: 📄 `{src['source']}` — **Page {src['page']}**"
                        )
                        st.caption(f"> {src['snippet']}")

    # 3. Persist assistant message in session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": citations,
        }
    )
