"""Retrieval and Grounded Generation module for Mini AI Knowledge Assistant."""

import os
from typing import Any, Dict, List, Optional, Tuple


GROUNDED_SYSTEM_PROMPT = (
    "You are a strict, grounded knowledge assistant. "
    "Answer ONLY using the provided context. If the answer isn't in the context, "
    "respond: 'Not found in the provided documents.' "
    "Do NOT use outside knowledge, extrapolate, speculate, or fabricate any facts."
)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
Answer ONLY using the provided context above. If the answer isn't in the context, respond: 'Not found in the provided documents.'

Grounded Answer:"""


def retrieve_documents(vector_store: Any, query: str, k: int = 4) -> List[Any]:
    """Retrieve top-k similar chunks from FAISS vector store."""
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)


def format_context(docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Format retrieved document chunks and extract source citations."""
    if not docs:
        return "", []

    context_parts = []
    sources = []

    for doc in docs:
        src = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        text = doc.page_content.strip()

        context_parts.append(f"[Document: {src} | Page: {page}]\n{text}")
        sources.append(
            {
                "source": src,
                "page": page,
                "snippet": text[:200] + ("..." if len(text) > 200 else ""),
            }
        )

    full_context = "\n\n---\n\n".join(context_parts)
    return full_context, sources


def call_gemini(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    system_instruction: str = GROUNDED_SYSTEM_PROMPT,
) -> str:
    """Generate answer using Google Gemini API with fallback model handling."""
    models_to_try = [model]
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = ""

    # Try via google-genai SDK first
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
                return "Not found in the provided documents."
            except Exception as e:
                last_error = str(e)
                continue
    except ImportError:
        pass

    # Fallback to legacy google.generativeai SDK
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        for m in models_to_try:
            try:
                gemini_model = legacy_genai.GenerativeModel(
                    model_name=m,
                    system_instruction=system_instruction,
                    generation_config={"temperature": 0.0},
                )
                response = gemini_model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                return "Not found in the provided documents."
            except Exception as e:
                last_error = str(e)
                continue
    except Exception as e:
        last_error = str(e)

    return (
        f"Error: Unable to generate response via Gemini API ({last_error}). "
        "Please check your API key and model selection, or switch to the Offline Grounded Engine."
    )


def call_groq(
    prompt: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    system_instruction: str = GROUNDED_SYSTEM_PROMPT,
) -> str:
    """Generate answer using Groq API with graceful error and model fallback handling."""
    models_to_try = [model]
    for fallback in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        for m in models_to_try:
            try:
                chat_completion = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                if chat_completion.choices and chat_completion.choices[0].message.content:
                    return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                last_error = str(e)
                continue
    except Exception as e:
        last_error = str(e)

    return (
        f"Error: Unable to generate response via Groq API ({last_error}). "
        "Please check your API key and model selection, or switch to the Offline Grounded Engine."
    )


def call_offline_grounded(query: str, docs: List[Any]) -> str:
    """Deterministic local grounded answering without requiring external API keys."""
    if not docs:
        return "Not found in the provided documents."

    import re

    # Extract non-trivial query terms (3+ letters)
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "can", "could", "should", "would", "may",
        "might", "must", "about", "for", "with", "from", "and", "or", "not",
        "according", "policy", "project", "document", "documents", "tell", "give",
    }
    raw_terms = re.findall(r"[a-zA-Z0-9_\$%,]+", query.lower())
    query_terms = [t for t in raw_terms if len(t) >= 2 and t not in stop_words]

    if not query_terms:
        query_terms = raw_terms

    best_sentence = None
    best_score = 0

    for doc in docs:
        text = doc.page_content
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_lower = s.lower()
            matched = sum(1 for term in query_terms if term in s_lower)
            if matched > best_score:
                best_score = matched
                best_sentence = s.strip()

    # If insufficient keyword overlap with retrieved content, reject as ungrounded
    min_required = max(1, len(query_terms) // 2)
    if best_score < min_required or not best_sentence:
        return "Not found in the provided documents."

    return f"Based on the provided documents: {best_sentence}"


def query_rag(
    query: str,
    vector_store: Any,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    k: int = 4,
) -> Dict[str, Any]:
    """Execute complete RAG pipeline: Retrieve (k=4) -> Format -> Prompt -> Generate."""
    if not query.strip():
        return {"answer": "Please enter a valid question.", "sources": []}

    if vector_store is None:
        return {
            "answer": "No documents have been indexed yet. Please upload and process PDFs first.",
            "sources": [],
        }

    # Top-k similarity retrieval
    retrieved_docs = retrieve_documents(vector_store, query, k=k)
    if not retrieved_docs:
        return {"answer": "Not found in the provided documents.", "sources": []}

    context_str, sources = format_context(retrieved_docs)
    provider_lower = provider.lower()

    # Offline Grounded Engine mode (No external API required)
    if "offline" in provider_lower:
        answer = call_offline_grounded(query, retrieved_docs)
        active_sources = sources if "not found in the provided documents" not in answer.lower() else []
        return {
            "answer": answer,
            "sources": active_sources,
            "context": context_str,
        }

    user_prompt = PROMPT_TEMPLATE.format(context=context_str, question=query)

    # Determine API key from parameter or environment
    active_key = api_key
    if not active_key:
        if "gemini" in provider_lower:
            active_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        else:
            active_key = os.getenv("GROQ_API_KEY")

    if not active_key:
        return {
            "answer": (
                f"⚠️ API key missing for provider '{provider}'. "
                "Please configure your API key in the sidebar, or switch to "
                "'Offline Grounded Engine' to test without an API key."
            ),
            "sources": [],
        }

    # Generate response
    if "gemini" in provider_lower:
        active_model = model or "gemini-1.5-flash"
        answer = call_gemini(user_prompt, api_key=active_key, model=active_model)
    elif "groq" in provider_lower:
        # Default to a free-tier Groq model if none specified
        active_model = model or "llama-3.1-8b-instant"
        answer = call_groq(user_prompt, api_key=active_key, model=active_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    active_sources = sources if "not found in the provided documents" not in answer.lower() else []
    return {
        "answer": answer,
        "sources": active_sources,
        "context": context_str,
    }
