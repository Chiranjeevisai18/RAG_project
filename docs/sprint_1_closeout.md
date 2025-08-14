# RAG Shuttle – Sprint 1 Closeout Report

## Mission Name
**RAG Shuttle – Sprint 1**

## Objective
Build a working Retrieval-Augmented Generation (RAG) system that:  
- Ingests PDF documents  
- Chunks and embeds content locally  
- Retrieves relevant context using FAISS  
- Passes it to Gemini Flash for grounded Q/A  
- Operates through an interactive Streamlit interface

---

## Mission Achievements ✅

### Project Setup
- Virtual environment (`venv`) created and activated  
- `requirements.txt` finalized and installed  
- `config.py` structured for paths, models, and chunking parameters  

### Core RAG Pipeline
- Document loaders for **PDF, TXT, DOCX** implemented  
- Chunking implemented with **RecursiveCharacterTextSplitter**  
- Local embeddings created with **all-MiniLM-L6-v2**  
- Vector store creation & persistence with **FAISS**  

### LLM Integration
- **Gemini 1.5 Flash** connected via API key  
- Strict retrieval → answer generation pipeline established  
- Hallucination control prompts implemented to avoid ungrounded responses  

### UI Development
- **Streamlit app** with:  
  - Sidebar controls (Top-k, temperature)  
  - File uploader (multi-format)  
  - Rebuild index button  
  - Chat interface with persistent history  
  - Source chunk display via expanders  

### Testing & Validation
- Queries tested on **Exoplanet PDFs**  
- Correct retrieval confirmed when info present  
- Graceful refusal confirmed when info absent  

---

## Observations 📡
- System performs reliably for factual queries within document scope  
- Retrieval precision is good; recall can be improved with **multi-query expansion**  
- Strict mode ensures trustworthiness but limits creative synthesis when info is partial  

---

## Queued for Next Sprint 🛠
- **Evaluation Metrics:** Log Q/A pairs, retrieval coverage, and chunk relevance  
- **Multi-Query Retrieval:** Query expansion for better context capture  
- **Creative Fallback Mode:** Toggle for gap-filling with clearly flagged assumptions  
- **Source Highlighting:** Show retrieved text in original PDF context  
- **Performance Tuning:** Experiment with embeddings, chunk size/overlap, and FAISS parameters  

---

## Status
✅ Sprint 1 completed successfully. Shuttle systems nominal. Ready to plan **Sprint 2**.

---

## Captain’s Note
*"Our shuttle now orbits a new star — RAG is alive. In the next voyage, we enhance its vision, speed, and ability to connect the cosmic dots. Until then, engines on standby."*
