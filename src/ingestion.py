import os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from config import DATA_DIR, FAISS_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from rich.console import Console

console = Console()

def load_documents():
    """Load all supported documents from the data folder."""
    docs = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if filename.lower().endswith(".pdf"):
            console.print(f"[green]Loading PDF:[/green] {filename}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            console.print(f"[green]Loading TXT:[/green] {filename}")
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        elif filename.lower().endswith(".docx"):
            console.print(f"[green]Loading DOCX:[/green] {filename}")
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        else:
            console.print(f"[yellow]Skipping unsupported file:[/yellow] {filename}")
    return docs

def chunk_documents(documents):
    """Split documents into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    """Create FAISS vector store from document chunks."""
    console.print("[blue]Generating embeddings...[/blue]")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_STORE_PATH)
    console.print(f"[bold green]✅ Vector store saved to:[/bold green] {FAISS_STORE_PATH}")

if __name__ == "__main__":
    console.print("[bold cyan]🚀 Starting document ingestion...[/bold cyan]")
    documents = load_documents()
    if not documents:
        console.print("[red]No documents found in data/ folder![/red]")
        exit()

    chunks = chunk_documents(documents)
    console.print(f"[green]Total chunks created:[/green] {len(chunks)}")
    create_vectorstore(chunks)
