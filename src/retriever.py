from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import FAISS_STORE_PATH, EMBEDDING_MODEL
from rich.console import Console

console = Console()

def load_vectorstore():
    """Load existing FAISS vector store from disk."""
    console.print("[blue]Loading FAISS index...[/blue]")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve(query, k=3):
    """Retrieve top-k relevant chunks for a given query."""
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs

if __name__ == "__main__":
    console.print("[bold cyan]🔍 Testing retriever...[/bold cyan]")
    query = "What are the eight planets?"
    results = retrieve(query)
    for i, doc in enumerate(results, 1):
        console.print(f"[green]Result {i}:[/green] {doc.page_content}\n")
