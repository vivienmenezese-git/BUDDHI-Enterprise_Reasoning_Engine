"""
Streamlit Local RAG App
Features:
- Upload CSV or PDF (with OCR for PDFs)
- Chunking of documents
- Create embeddings (tries sentence-transformers locally, falls back to hash-based)
- In-memory vector store (FAISS if available, else NumPy linear scan)
- Local LLM inference via Ollama CLI (model_id = "meta-llama/Llama-3.1-8B-Instruct")

Requirements (install locally):
pip install streamlit pandas numpy scikit-learn faiss-cpu sentence-transformers pdf2image pytesseract pillow
# Ollama: https://ollama.com - ensure `ollama` CLI is installed and model pulled
# Tesseract OCR: install system binary (e.g., apt-get install tesseract-ocr) and poppler for pdf2image

Run:
streamlit run streamlit_local_rag_app.py

Note: This app runs fully locally. Adjust paths/permissions for Tesseract & Poppler on your OS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import tempfile
import hashlib
import subprocess
from typing import List, Dict, Any, Optional, Tuple
def textwrap_short(text, max_len=500):
    """Return only the first max_len characters with '...'."""
    if not isinstance(text, str):
        text = str(text)
    return text[:max_len] + ("..." if len(text) > max_len else "")

def build_prompt(user_query: str, context_texts: List[str]) -> str:
    """Assemble the system message + context + user query into a single LLM prompt."""
    system = (
        "You are a concise, factual financial analytics assistant. "
        "Use ONLY the provided context snippets when answering. "
        "If data is insufficient, clearly say what is missing. "
        "Give numeric answers, short reasoning, and 3 actionable recommendations."
    )

    context_block = "\n\n-----\n\n".join(context_texts)

    prompt = (
        f"SYSTEM:\n{system}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUERY:\n{user_query}\n\n"
        "INSTRUCTIONS:\n"
        "1. Provide a brief executive summary.\n"
        "2. Use numbers if available in the context.\n"
        "3. State assumptions if needed.\n"
        "4. Give 3 actionable recommendations.\n"
    )
    return prompt

# OCR imports (optional)
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    import pytesseract

    # ---- Windows Tesseract Path Fix ----
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    # ------------------------------------

    OCR_AVAILABLE = True

except Exception as e:
    OCR_AVAILABLE = False
    print("OCR not available:", e)

# Embedding / vector store imports
try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    EMB_MODEL = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

# ------------------------- Configuration -------------------------
MODEL_ID = "llama3.2:latest"  # Ollama model id to use
TOP_K = 5
CHUNK_SIZE = 2000  # characters per chunk
EMB_DIM = 384 if SENTENCE_TRANSFORMERS_AVAILABLE else 128

st.set_page_config(page_title="Local RAG + LLM (Offline)", layout="wide")
st.title("🔒 Local RAG Financial Assistant — CSV & PDF (OCR) Support")

# Session state initialization
if 'docs' not in st.session_state:
    st.session_state.docs = []  # list of dicts: {id, source, text, meta}
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None  # numpy array (n, d)
if 'ids' not in st.session_state:
    st.session_state.ids = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# ------------------------- Helper Functions -------------------------

def simple_hash_embedding(text: str) -> np.ndarray:
    """Hash-based embedding fallback (deterministic, privacy-friendly)."""
    vec = np.zeros(EMB_DIM, dtype=np.float32)
    for i, ch in enumerate(text.lower()):
        idx = ord(ch) % EMB_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    # mix in md5 bits to add variability
    h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    for i in range(EMB_DIM):
        vec[i] += ((h >> (i % 64)) & 1) * 0.001
    return vec


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts, using sentence-transformers if available, else hash-based."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        emb = EMB_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return emb.astype(np.float32)
    else:
        return np.vstack([simple_hash_embedding(t) for t in texts]).astype(np.float32)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Naive chunker splitting by characters while trying to respect line breaks."""
    text = text.replace('\r\n', '\n')
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        # try to extend to next newline for readability
        if end < n:
            nxt = text.rfind('\n', start, end)
            if nxt > start:
                end = nxt
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def add_documents(source_name: str, text: str, meta: Dict[str, Any]):
    """Process text into chunks, embed, and add to session vector store."""
    chunks = chunk_text(text)
    new_ids = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{source_name}__{len(st.session_state.docs)}__{i}"
        st.session_state.docs.append({'id': doc_id, 'source': source_name, 'text': chunk, 'meta': meta})
        st.session_state.ids.append(doc_id)
        new_ids.append(doc_id)

    # create embeddings for new chunks
    embs = embed_texts(chunks)

    if st.session_state.embeddings is None:
        st.session_state.embeddings = embs
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, embs])

    # update FAISS index if available
    if FAISS_AVAILABLE:
        d = st.session_state.embeddings.shape[1]
        if st.session_state.faiss_index is None:
            index = faiss.IndexFlatIP(d)
            st.session_state.faiss_index = index
            # normalize vectors for IP
            faiss.normalize_L2(st.session_state.embeddings)
            st.session_state.faiss_index.add(st.session_state.embeddings)
        else:
            # add only newly computed embeddings
            faiss.normalize_L2(embs)
            st.session_state.faiss_index.add(embs)


def search(query: str, top_k: int = TOP_K) -> List[Tuple[float, Dict]]:
    """Search vector store for top_k relevant chunks. Returns list of (score, doc).
    If FAISS available, uses it. Otherwise does linear cosine similarity scan."""
    if st.session_state.embeddings is None:
        return []

    q_emb = embed_texts([query])[0]

    if FAISS_AVAILABLE and st.session_state.faiss_index is not None:
        faiss.normalize_L2(q_emb.reshape(1, -1))
        D, I = st.session_state.faiss_index.search(q_emb.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(st.session_state.docs):
                continue
            results.append((float(score), st.session_state.docs[idx]))
        return results
    else:
        # linear cosine similarity
        db = st.session_state.embeddings
        q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        db_norms = np.linalg.norm(db, axis=1) + 1e-12
        sims = (db @ q) / db_norms
        top_idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), st.session_state.docs[i]) for i in top_idx]


def call_local_ollama(prompt: str, model_id: str = MODEL_ID, timeout: int = 60) -> str:
    """Call Ollama CLI to get model completion. Requires ollama installed locally."""
    try:
        # Build command
        # we send prompt via stdin to avoid shell issues
        cmd = ["ollama", "query", model_id, "--system", "You are a helpful financial assistant. Use provided context."]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(prompt, timeout=timeout)
        if proc.returncode != 0:
            st.warning(f"Ollama returned non-zero exit code. stderr: {err}")
            return ""
        return out.strip()
    except FileNotFoundError:
        st.error("Ollama CLI not found. Please install Ollama and pull the model locally. See https://ollama.com")
        return ""
    except Exception as e:
        st.error(f"Error calling Ollama: {e}")
        return ""

# ------------------------- UI: Upload & Processing -------------------------

with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader("Upload CSV or PDF files", accept_multiple_files=True, type=['csv', 'pdf', 'xlsx', 'txt'])
    if uploaded:
        for file in uploaded:
            name = file.name
            st.info(f"Processing {name} ...")
            if name.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    text = df.to_csv(index=False)
                    add_documents(name, text, {'type': 'csv', 'shape': df.shape, 'columns': df.columns.tolist()})
                    st.success(f"Loaded CSV: {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

            elif name.lower().endswith('.xlsx'):
                try:
                    df = pd.read_excel(file)
                    text = df.to_csv(index=False)
                    add_documents(name, text, {'type': 'excel', 'shape': df.shape, 'columns': df.columns.tolist()})
                    st.success(f"Loaded Excel: {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
                except Exception as e:
                    st.error(f"Failed to read Excel: {e}")

            elif name.lower().endswith('.pdf'):
                if not OCR_AVAILABLE:
                    st.error("OCR libraries not available (pdf2image/pytesseract). Install them to enable PDF OCR.")
                else:
                    try:
                        raw = file.read()
                        pages = convert_from_bytes(raw)
                        full_text = []
                        for p in pages:
                            txt = pytesseract.image_to_string(p)
                            full_text.append(txt)
                        text = "\n\n".join(full_text)
                        add_documents(name, text, {'type': 'pdf', 'pages': len(pages)})
                        st.success(f"Loaded PDF with OCR: {name} ({len(pages)} pages)")
                    except Exception as e:
                        st.error(f"Failed OCR on PDF: {e}")

            else:
                # txt
                try:
                    content = file.read().decode('utf-8')
                    add_documents(name, content, {'type': 'text'})
                    st.success(f"Loaded text file: {name}")
                except Exception as e:
                    st.error(f"Failed to read text file: {e}")

    st.markdown("---")
    st.metric("Documents", len(st.session_state.docs))
    if st.button("Clear Index"):
        st.session_state.docs = []
        st.session_state.embeddings = None
        st.session_state.ids = []
        st.session_state.faiss_index = None
        st.success("Cleared documents and embeddings")

# ------------------------- UI: Querying -------------------------

st.header("Chat & Query")
col1, col2 = st.columns([3,1])

with col1:
    user_query = st.text_area("Enter your question about the uploaded documents:", height=120)
    if st.button("Ask") and user_query.strip():
        st.session_state.conversation.append({'role': 'user', 'text': user_query})
        # search
        with st.spinner("Searching relevant context…"):
            results = search(user_query, top_k=TOP_K)

        if not results:
            st.warning("No indexed documents found. Upload PDFs/CSVs first.")
        else:
            st.markdown("**Top context snippets retrieved:**")
            context_texts = []
            for score, doc in results:
                st.write(f"- (score: {score:.3f}) {doc['source']} — {doc['text'][:250].replace('\n',' ')}...")
                context_texts.append(f"Source: {doc['source']}\n{textwrap_short(doc['text'], 500)}")

            # Build prompt for LLM
            prompt = build_prompt(user_query, context_texts)
            st.markdown("---")
            st.markdown("**LLM Prompt Preview (trimmed):**")
            st.code(prompt[:4000])

            # Call local LLM
            st.info("Calling local LLM (Ollama)...")
            answer = call_local_ollama(prompt)
            if answer:
                st.subheader("Assistant Response")
                st.write(answer)
                st.session_state.conversation.append({'role': 'assistant', 'text': answer})

with col2:
    st.subheader("Conversation")
    for msg in reversed(st.session_state.conversation[-8:]):
        role = msg['role']
        if role == 'user':
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Assistant:** {msg['text']}")

# ------------------------- Utilities -------------------------

def textwrap_short(s: str, n: int = 300) -> str:
    if len(s) <= n:
        return s
    return s[:n].rsplit(' ', 1)[0] + '...'


def build_prompt(user_query: str, context_texts: List[str]) -> str:
    """Assemble a system + instructions + context + user query prompt for the LLM."""
    system = (
        "You are a concise, factual financial analytics assistant. Use ONLY the provided context snippets when answering. "
        "If data is insufficient, be explicit about missing data and what is needed. Provide numeric outputs and short actionable recommendations."
    )
    ctx = "\n\n---\n\n".join(context_texts)
    prompt = f"SYSTEM:\n{system}\n\nCONTEXT:\n{ctx}\n\nUSER QUERY:\n{user_query}\n\nINSTRUCTIONS:\nProvide a short executive summary, exact numeric answers where possible, and 3 actionable recommendations. If calculations were performed, show formulas and results."
    return prompt

# Final note display
st.markdown("---")
st.info("This app uses local models (Ollama required) and local OCR if installed. Adjust CHUNK_SIZE, TOP_K, and MODEL_ID variables at the top of the script as needed.")
