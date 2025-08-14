import google.generativeai as genai
from config import GEMINI_API_KEY
from retriever import retrieve
from rich.console import Console

console = Console()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def generate_answer(query):
    """Retrieve context and use Gemini Flash to answer."""
    docs = retrieve(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI assistant.
Answer the following question based only on the provided context.

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

if __name__ == "__main__":
    console.print("[bold cyan]🤖 Testing Generator with Gemini Flash...[/bold cyan]")
    query = "What are the eight planets?"
    answer = generate_answer(query)
    console.print(f"[bold green]Answer:[/bold green] {answer}")
