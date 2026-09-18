"""Streamlit Chat Application for Mini AI Knowledge Assistant (RAG)."""

import os

# ── Silence all tqdm / HuggingFace / Transformers progress output ────────────
# Must be set BEFORE importing sentence-transformers, transformers, or langchain
# to prevent OSError 22 (Invalid argument) on Windows non-TTY streams (Streamlit).
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISABLE_TQDM", "true")

# Patch tqdm to write to a null sink — catches any tqdm already imported
import io as _io
try:
    import tqdm as _tqdm
    _tqdm.tqdm.__init__.__defaults__ = tuple(
        _io.StringIO() if i == 5 else d
        for i, d in enumerate(_tqdm.tqdm.__init__.__defaults__ or [])
    )
except Exception:
    pass

import shutil
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import (
    DEFAULT_INDEX_DIR,
    ingest_multiple_files,
    load_vector_store,
)
from rag import query_rag

st.set_page_config(
    page_title="Mini AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_load_vector_store(index_dir: str):
    """Load FAISS vector store + embedding model once per server process.

    @st.cache_resource persists the result across Streamlit reruns and across
    multiple user sessions — the heavy model is loaded from disk only once.
    Call _cached_load_vector_store.clear() before re-indexing to invalidate.
    """
    _idx = Path(index_dir) / "index.faiss"
    if not _idx.exists():
        return None
    return load_vector_store(index_dir)


@st.cache_data(show_spinner=False)
def _get_secret(key_name: str) -> str:
    """Cache secret lookups so st.secrets is not queried on every rerun."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, "")


# ── Session State Initialization ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    os.makedirs(DEFAULT_INDEX_DIR, exist_ok=True)
    # Fast path: skip heavy model load if no FAISS index exists yet
    if (Path(DEFAULT_INDEX_DIR) / "index.faiss").exists():
        st.session_state.vector_store = _cached_load_vector_store(DEFAULT_INDEX_DIR)
    else:
        st.session_state.vector_store = None

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _do_index(file_paths: list, doc_names: list) -> None:
    """Build FAISS index, update session state, and clear the resource cache."""
    vs = ingest_multiple_files(file_paths, index_dir=DEFAULT_INDEX_DIR)
    _cached_load_vector_store.clear()   # invalidate cache so next load picks up new index
    st.session_state.vector_store = vs
    st.session_state.indexed_docs = doc_names


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings & Ingestion")

    provider_options = ["Offline Grounded Engine (No API Key Required)", "Gemini", "Groq"]
    default_gemini_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
    default_groq_key = _get_secret("GROQ_API_KEY")

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
            options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"],
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
            options=["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0,
        )
        if not api_key:
            st.info("💡 Don't have an API key? Switch to 'Offline Grounded Engine' to test without a key.")
    else:
        api_key = None
        model_name = "offline-grounded"
        st.success("🟢 Local CPU Grounded Engine active. Zero external API calls.")

    # ── Upload Section ────────────────────────────────────────────────────────
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
                saved_paths, doc_names = [], []
                try:
                    for up_file in uploaded_files:
                        tmp_path = os.path.join(temp_dir, up_file.name)
                        with open(tmp_path, "wb") as f:
                            f.write(up_file.getbuffer())
                        saved_paths.append(tmp_path)
                        doc_names.append(up_file.name)
                    _do_index(saved_paths, doc_names)
                    st.success(f"✅ Indexed {len(doc_names)} document(s)!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    if load_samples_clicked:
        with st.spinner("Indexing sample PDFs (Apollo & Security)..."):
            sample_dir = Path(__file__).parent / "sample_docs"
            if not (sample_dir / "project_apollo.pdf").exists():
                from eval import setup_sample_documents
                setup_sample_documents()
            samples = [
                str(sample_dir / "project_apollo.pdf"),
                str(sample_dir / "security_policy.pdf"),
            ]
            try:
                _do_index(samples, ["project_apollo.pdf", "security_policy.pdf"])
                st.success("Sample documents loaded and indexed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    # ── Local Path Indexing ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Index from Local Path")
    st.caption("Enter a file path or folder. All supported files in a folder are indexed recursively.")

    local_path_input = st.text_input(
        "File or Folder Path",
        placeholder=r"e.g. C:\Users\You\Documents\reports  or  C:\report.pdf",
        help="Supports PDF, DOCX, DOC, PPTX, PPT.",
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
                                st.error(f"Unsupported type: `{target.suffix}`. Use PDF, DOCX, or PPTX.")
                                file_paths = []
                            else:
                                file_paths = [str(target)]
                                doc_names = [target.name]
                        else:
                            file_paths = sorted(
                                str(p) for p in target.rglob("*")
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                            )
                            doc_names = [Path(p).name for p in file_paths]

                        if file_paths:
                            _do_index(file_paths, doc_names)
                            st.success(f"✅ Indexed {len(doc_names)} document(s) from `{target.name}`!")
                            st.rerun()
                        else:
                            st.warning("No supported documents found at that path.")
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")

    # ── Index Status ──────────────────────────────────────────────────────────
    st.divider()
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

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Interface ───────────────────────────────────────────────────────

st.title("🤖 Mini AI Knowledge Assistant")
st.caption(
    "Strictly grounded RAG assistant powered by FAISS, "
    "`all-MiniLM-L6-v2`, and verifiable page-level citations."
)

# Render chat history
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

# Chat input
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

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

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": citations}
    )
