import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==== API Keys ====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("🚨 GEMINI_API_KEY is missing! Please set it in .env")

# ==== Embedding Model ====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==== Chunking Settings ====
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks

# ==== Paths ====
DATA_DIR = os.path.join(os.getcwd(), "data")
VECTORSTORE_DIR = os.path.join(os.getcwd(), "vectorstore")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
FAISS_STORE_PATH = VECTORSTORE_DIR  # folder to save metadata

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
