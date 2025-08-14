import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import io
import shutil
import streamlit as st
from rich.console import Console

# Local modules
from config import (
    DATA_DIR,
    FAISS_STORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from config import GEMINI_API_KEY

console = Console()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"  # can expose in UI

# ---------- Ingestion helpers (re-usable from your ingestion.py) ----------
def load_documents_from_dir(directory: str):
    docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_or_reload_vectorstore():
    """Try to load FAISS from disk; if missing, build from DATA_DIR."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        vs = FAISS.load_local(
            FAISS_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception:
        # Build fresh
        docs = load_documents_from_dir(DATA_DIR)
        if not docs:
            return None
        chunks = chunk_documents(docs)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(FAISS_STORE_PATH)
        return vs


def rebuild_index_from_dir(directory: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs = load_documents_from_dir(directory)
    if not docs:
        return None
    chunks = chunk_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_STORE_PATH)
    return vs


# ---------- Retrieval + Generation ----------

def retrieve(vs: FAISS, query: str, k: int = 4):
    return vs.similarity_search(query, k=k)


def generate_answer(context: str, query: str, temperature: float = 0.2):
    prompt = f"""
You are a helpful AI assistant.
Answer the user's question using only the following context. If the answer is not present, say you don't have enough information.

Context:
{context}

Question:
{query}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt, generation_config={"temperature": temperature})
    return resp.text.strip() if hasattr(resp, "text") else "(No response text)"


# ---------- Streamlit UI ----------

st.set_page_config(page_title="RAG — Gemini + FAISS", page_icon="🛸", layout="wide")
st.title("🛸 RAG Shuttle — Chat with your PDFs")
st.caption("Local embeddings (SentenceTransformers) + FAISS + Gemini Flash ✨")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    k = st.slider("Top-k chunks", 1, 10, 4)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    st.divider()

    st.subheader("📥 Upload & Reindex")
    uploaded_files = st.file_uploader(
        "Add PDFs / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join(DATA_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.read())
        st.success(f"Saved {len(uploaded_files)} file(s) to data/ ✅")

    if st.button("🔄 Rebuild Index"):
        vs = rebuild_index_from_dir(DATA_DIR)
        if vs is not None:
            st.success("Index rebuilt and saved ✅")
            st.session_state["vs"] = vs
        else:
            st.warning("No documents found in data/ to index.")

# Vector store cache in session
if "vs" not in st.session_state:
    st.session_state["vs"] = build_or_reload_vectorstore()

vs = st.session_state["vs"]
if vs is None:
    st.info("No index found. Upload files in the sidebar and click **Rebuild Index**.")

# Chat state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome aboard, captain! Upload docs and ask me anything."}
    ]

# Render chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

# Input box
user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve + generate
    with st.chat_message("assistant"):
        if vs is None:
            st.warning("No index available. Please upload documents and rebuild the index from the sidebar.")
        else:
            docs = retrieve(vs, user_input, k=k)
            context = "\n\n".join([d.page_content for d in docs])
            with st.spinner("Consulting the ship's knowledge base…"):
                answer = generate_answer(context, user_input, temperature=temperature)
            st.markdown(answer)

            # Show sources in an expander
            with st.expander("📚 Sources (top chunks)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**\n\n{d.page_content}")

            # Persist answer in history
            st.session_state["messages"].append({"role": "assistant", "content": answer})
