# 📚 Mini AI Knowledge Assistant (RAG)

An end-to-end, strictly grounded Retrieval-Augmented Generation (RAG) knowledge assistant built with Streamlit, LangChain, FAISS, `sentence-transformers`, and Google Gemini / Groq free tiers.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multi-Document Ingestion (ingest.py)"]
        A[Multiple PDF Uploads] --> B[PyPDFLoader]
        B -->|Extract text & page metadata| C[RecursiveCharacterTextSplitter]
        C -->|chunk_size=800, overlap=100| D[Chunks with source + page metadata]
        D --> E[all-MiniLM-L6-v2 Local Embeddings]
        E --> F[FAISS Vector Store]
        F -->|Persist to disk| G[(faiss_index/)]
    end

    subgraph Retrieval ["2. Similarity Retrieval (rag.py)"]
        H[User Query] --> I[FAISS Top-k Search (k=4)]
        G --> I
        I --> J[Formatted Context + Citations]
    end

    subgraph Generation ["3. Grounded Generation (rag.py)"]
        J --> K[Strict Grounded Prompt Template]
        K --> L{Provider Choice}
        L -->|Gemini API| M[Google Gemini 1.5 Flash]
        L -->|Groq API| N[Groq Llama 3.3 70B]
        M --> O[Verifiable Answer]
        N --> O
    end

    subgraph UI ["4. Interactive Experience (app.py)"]
        O --> P[Streamlit Chat Window]
        I -->|Filename + Page Number| Q[Expandable Source Citations]
        P --> R[st.session_state Multi-turn History]
    end
```

---

## 💡 Tech Choices & Why

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **PDF Extraction** | `PyPDFLoader` (`langchain-community`) | Industry-standard extractor that natively extracts page-level metadata and raw text without heavyweight binaries. |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` | Preserves semantic paragraph and sentence boundaries with `chunk_size=800` and `chunk_overlap=100` to prevent losing cross-chunk context. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | 100% free, runs locally on CPU, highly efficient (384 dimensions), and produces high-quality semantic representations without external API rate limits. |
| **Vector Database** | `FAISS` (`faiss-cpu`) | High-performance similarity search with zero database infrastructure; persists directly to disk (`index.faiss`) and reloads instantly without re-embedding. |
| **LLM Providers** | Google Gemini & Groq | Free tier access, high token throughput, strong instruction-following for grounded constraints. |
| **User Interface** | `Streamlit` | Rapid, production-grade web chat interface with native support for file uploads, session state persistence, and responsive layout. |

---

## 🚀 Setup Steps

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- (Optional) `uv` or standard Python `venv`

### 2. Clone & Create Environment
```bash
git clone <your-repo-url>
cd mini-ai-knowledge-assistant

# Using standard venv
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Or using uv (recommended for speed)
uv venv .venv --python 3.11
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or: uv pip install -r requirements.txt
```

### 4. Configure API Keys
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Provide either a Gemini or Groq key:
```env
GEMINI_API_KEY="your-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"
```
*(Alternatively, you can type your API key directly into the Streamlit sidebar at runtime).*

---

## 💻 How to Run

### Run the Web Assistant
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

1. **Upload Documents**: Select one or more PDF documents in the sidebar.
2. **Index**: Click **"🚀 Process & Index Documents"**.
3. **Ask Questions**: Ask questions in the chat window.
4. **Inspect Citations**: Click the expandable citation drawer under any assistant response to view the exact document filename, page number, and source snippet.

---

## 🧪 Evaluation Suite (`eval.py`)

Run automated retrieval and grounding tests against sample multi-document benchmarks:
```bash
python eval.py
```

### What `eval.py` Tests:
1. **Multi-Doc Retrieval Quality**: Verifies top-4 search accurately retrieves the target document and page for specific queries across distinct files.
2. **Keyword Match**: Asserts presence of expected domain facts and numerical answers.
3. **Strict Grounding Enforcement**: Passes out-of-domain / negative questions (e.g. "Who won the 2022 World Cup?") and verifies the assistant strictly outputs:
   ```
   Not found in the provided documents.
   ```
4. **Summary Report**: Prints a formatted test grid with retrieval status, grounding status, and pass percentage.

---

## ☁️ Deployment to Streamlit Community Cloud

This repository is pre-configured for one-click deployment:

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Set:
   - **Repository**: `your-username/mini-ai-knowledge-assistant`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Under **Advanced Settings > Secrets**, add your secrets in TOML format:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **Deploy!**

---

## ⚠️ Known Limitations

1. **Scanned PDFs & OCR**: `PyPDFLoader` extracts digital text streams. Scanned image-only PDFs require OCR preprocessing (e.g. `pytesseract` or Google Cloud Vision).
2. **Complex Tables & Multi-column Layouts**: Plain character splitters can occasionally split table rows across chunk boundaries. Layout-aware chunking (e.g., Unstructured or Markdown conversion) can improve tabular precision.
3. **Vector Distance Calibration**: Fixed `k=4` retrieval passes top semantic candidates; queries with completely irrelevant keywords rely on the LLM grounding prompt to recognize lack of relevant context.

---

## 📸 Sample Interface Preview

```
+-------------------------------------------------------------------------------+
| 🤖 Mini AI Knowledge Assistant                                                |
| Strictly grounded RAG assistant powered by FAISS and all-MiniLM-L6-v2        |
+-------------------------------------------------------------------------------+
|                                                                               |
|  👤 User: What is the orbital cruise velocity of Project Apollo?              |
|                                                                               |
|  🤖 Assistant: According to the documents, the orbital cruise velocity of     |
|     Project Apollo is exactly 25,000 mph.                                     |
|                                                                               |
|     ▼ 📚 View Sources & Citations (1 chunks)                                  |
|       Citation 1: 📄 project_apollo.pdf — Page 1                              |
|       > Project Apollo Specification: The orbital cruise velocity is exactly  |
|         25,000 mph. The command module payload capacity is 12,000...          |
|                                                                               |
|  👤 User: Who won the 2022 FIFA World Cup?                                    |
|                                                                               |
|  🤖 Assistant: Not found in the provided documents.                           |
|                                                                               |
+-------------------------------------------------------------------------------+
| [ Ask a question about your uploaded documents...                          ]  |
+-------------------------------------------------------------------------------+
```
