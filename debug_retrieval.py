"""
debug_retrieval.py — Test retrieval directly for any query
Run: python3 debug_retrieval.py "your question here"
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

query = sys.argv[1] if len(sys.argv) > 1 else "what is the onboarding process"

print(f"\n{'='*60}")
print(f"Query: {query}")
print('='*60)

# ── 1. Vector Search ──────────────────────────────────────────
print("\n📌 VECTOR SEARCH (top 5):")
from rag.ingestor import get_vectorstore
from rag.retriever.vector_search import VectorSearchRetriever

vs  = get_vectorstore()
vsr = VectorSearchRetriever(vs)
vector_results = vsr.search(query, top_k=5)
for i, doc in enumerate(vector_results, 1):
    print(f"\n  [{i}] Score: {doc.metadata.get('vector_score', 'N/A')}")
    print(f"       Source: {doc.metadata.get('file_name', 'unknown')}")
    print(f"       Text: {doc.page_content[:200]}")

# ── 2. BM25 Search ────────────────────────────────────────────
print("\n\n📌 BM25 SEARCH (top 5):")
from rag.retriever.bm25_search import BM25Retriever
from langchain_core.documents import Document

raw      = vs.get(include=["documents", "metadatas"])
bm25_docs = [
    Document(page_content=content, metadata=meta or {})
    for content, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
]
bm25 = BM25Retriever()
bm25.build_index(bm25_docs)
bm25_results = bm25.search(query, top_k=5)
for i, doc in enumerate(bm25_results, 1):
    print(f"\n  [{i}] BM25 Score: {doc.metadata.get('bm25_score', 'N/A')}")
    print(f"       Source: {doc.metadata.get('file_name', 'unknown')}")
    print(f"       Text: {doc.page_content[:200]}")

# ── 3. Hybrid Search ──────────────────────────────────────────
print("\n\n📌 HYBRID SEARCH (top 5 after RRF):")
from rag.retriever.hybrid_search import HybridSearchRetriever
hybrid = HybridSearchRetriever(vsr, bm25)
hybrid_results = hybrid.search(query, top_k=5)
for i, doc in enumerate(hybrid_results, 1):
    print(f"\n  [{i}] RRF Score: {doc.metadata.get('rrf_score', 'N/A')}")
    print(f"       Source: {doc.metadata.get('file_name', 'unknown')}")
    print(f"       Text: {doc.page_content[:300]}")

print(f"\n{'='*60}")
print("✅ Debug complete. Check if relevant chunks appear above.")
print("If chunks look correct but answer is wrong → problem is in LLM prompt.")
print("If chunks look wrong → problem is in retrieval/chunking.")
print('='*60)
