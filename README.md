# 🏢 OnboardIQ — Intelligent Company Onboarding Assistant

> An production-grade Agentic RAG system that helps new employees instantly find answers
> about company policies, benefits, IT setup, and onboarding procedures.

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd OnboardIQ

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key
cp .env.example .env
# Edit .env → paste your GROQ_API_KEY
# Get free key at: https://console.groq.com

# 5. Run the app
streamlit run ui/app.py
```

---

## 📂 Project Structure

```
OnboardIQ/
├── config.py                    # All settings (models, thresholds, paths)
├── llm.py                       # LLM factory (Groq — free & fast)
├── main.py                      # Full pipeline: ties everything together
├── requirements.txt
├── .env                         # Your API keys (never commit this)
│
├── rag/
│   ├── ingestor.py              # Load PDFs/URLs/text → ChromaDB
│   ├── retriever/
│   │   ├── vector_search.py     # Semantic search (ChromaDB embeddings)
│   │   ├── bm25_search.py       # Keyword search (BM25)
│   │   ├── hybrid_search.py     # Combine both with RRF fusion
│   │   └── hyde.py              # HyDE search
│   └── advanced/
│       ├── query_pipeline.py    # Rewrite → MultiQuery → Rerank → Compress
│       ├── query_rewriter.py    # Rewrite vague queries
│       ├── multi_query.py       # Generate query variations
│       ├── reranker.py          # Cross-encoder re-ranking
│       ├── compressor.py        # Trim irrelevant chunk parts
│       ├── hyde.py              # HyDE document generation
│       └── self_rag.py          # Critique & re-retrieve loop
│
├── agent/
│   ├── agent.py                 # Custom ReAct agent loop
│   └── tools.py                 # Tools the agent can call
│
├── guardrails/
│   ├── input_guard.py           # Block toxic/irrelevant queries
│   ├── retrieval_guard.py       # Filter low-relevance chunks
│   └── output_guard.py          # Hallucination + groundedness check
│
├── data/
│   └── sample_docs/             # Drop your HR docs here
│       ├── hr_policy.md
│       ├── onboarding_checklist.md
│       └── it_setup_guide.md
│
└── ui/
    └── app.py                   # Streamlit chat interface
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Groq (Llama 3.1 8B) | Free, extremely fast |
| Framework | LangChain | Industry standard RAG framework |
| Vector DB | ChromaDB | Local, no setup needed |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally |
| Keyword Search | rank-bm25 | Exact term matching |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision scoring |
| UI | Streamlit | Fast prototyping |

---

## ✨ Features

- ✅ PDF, URL, and text document ingestion
- ✅ Hybrid Search (Vector + BM25 + RRF)
- ✅ HyDE (Hypothetical Document Embeddings)
- ✅ Multi-Query Retrieval
- ✅ Cross-Encoder Re-ranking
- ✅ Contextual Compression
- ✅ Parent-Child Chunking
- ✅ Agentic ReAct Loop
- ✅ Self-RAG critique loop
- ✅ Input / Retrieval / Output Guardrails
- ✅ Source citations in every answer
# OnboardIQ
