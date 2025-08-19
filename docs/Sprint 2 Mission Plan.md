# 🚀 Sprint 2 – RAG Shuttle Expansion

**Duration**: ~1 week (2–3 hrs/day)

---

## 🌟 Mission 1: Evaluation Metrics & Logging  
**Objective**: Track performance of our RAG system.  

**Steps**:  
1. Setup `logs/qa_logs.csv`.  
   - Columns: `query`, `rewritten_queries`, `retrieved_chunks`, `final_answer`, `timestamp`.  
2. Add metrics:  
   - `retrieval_count` → number of chunks retrieved.  
   - `chunk_overlap` → % of retrieved chunks used in final answer.  
   - `answer_length` → heuristic for completeness.  
3. Append to CSV after every query.  

✅ **Deliverable**: Growing CSV log = black box recorder for RAG.

---

## 🌟 Mission 2: Multi-Query Retrieval  
**Objective**: Improve retrieval robustness by generating query variations.  

**Steps**:  
1. Generate 2–3 query rewrites (via Gemini/LLM).  
2. Search FAISS for each rewrite.  
3. Merge & deduplicate results.  
4. Rank results by similarity score.  
5. Return top-N merged results.  

✅ **Deliverable**: Smarter retrieval even for vague queries.

---

## 🌟 Mission 3: Creative Fallback Mode  
**Objective**: Ensure system always provides a response.  

**Steps**:  
1. Check retrieval confidence (empty/low → trigger fallback).  
2. Fallback strategies:  
   - Option A: Summarize available docs.  
   - Option B: Use Gemini to answer (flag ⚠️ as *not grounded*).  
3. Add Streamlit toggle → “Enable Fallback Mode.”  

✅ **Deliverable**: System never returns blank answers.

---

## 🌟 Mission 4: UI/UX Enhancements  
**Objective**: Make the Streamlit interface more user-friendly.  

**Steps**:  
1. Add **Query History Panel** (scrollable past Q/As).  
2. Highlight **retrieved chunks** with color-coding.  
3. Add **document filter dropdown** (All docs / specific doc).  
4. Polish UI: better layout, typography, spacing.  

✅ **Deliverable**: Sleek cockpit UI for transparent + smooth navigation.

---

## 🚀 End of Sprint 2 – Expected Deliverables
- Logging system with metrics in CSV.  
- Multi-query retrieval pipeline.  
- Creative fallback mode toggle.  
- Upgraded Streamlit UI with history & highlighting.  
