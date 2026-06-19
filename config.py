"""
config.py — Central configuration for OnboardIQ
All settings, paths, and model configs live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data" / "sample_docs"
CHROMA_DIR      = BASE_DIR / "data" / "chroma_db"

# ── Groq ──────────────────────────────────────────────────────────────────────
# Get your FREE API key at: https://console.groq.com (just email signup)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Selection ───────────────────────────────────────────────────────────
# Free Groq models:
#   "llama-3.1-8b-instant"     ← fastest, good quality (recommended)
#   "llama-3.3-70b-versatile"  ← best quality, still free
#   "mixtral-8x7b-32768"       ← good for long context
#   "gemma2-9b-it"             ← Google's Gemma 2

LLM_MODEL       = "llama-3.1-8b-instant"   # fast + free
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 2048

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # local, free, no API needed
EMBEDDING_DIMENSION = 384

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "onboardiq_docs"

# ── Chunking ──────────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE    = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE     = 300
CHILD_CHUNK_OVERLAP  = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR        = 20
TOP_K_BM25          = 20
TOP_K_HYBRID        = 10
TOP_K_RERANKED      = 5
RRF_K               = 60

# ── HyDE ──────────────────────────────────────────────────────────────────────
HYDE_ENABLED          = True
HYDE_NUM_HYPOTHETICAL = 1

# ── Multi-Query ───────────────────────────────────────────────────────────────
MULTI_QUERY_ENABLED = True
MULTI_QUERY_COUNT   = 3

# ── Self-RAG ──────────────────────────────────────────────────────────────────
SELF_RAG_ENABLED        = True
SELF_RAG_MAX_ITERATIONS = 2

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Guardrails ────────────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD    = 0.3
GROUNDEDNESS_THRESHOLD = 0.5
TOXICITY_THRESHOLD     = 0.7

# ── Agent ─────────────────────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS = 5
AGENT_VERBOSE        = True

# ── Query Length ──────────────────────────────────────────────────────────────
MIN_QUERY_LENGTH    = 5
MAX_QUERY_LENGTH    = 1000

# ── Retrieval counts ──────────────────────────────────────────────────────────
INITIAL_RETRIEVAL_K = 20
RERANK_TOP_K        = 5

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are OnboardIQ, an intelligent onboarding assistant for new employees.
Your job is to answer questions about company policies, procedures, benefits, and processes
using ONLY the information provided in the retrieved documents.

Guidelines:
- Always cite the source document for every claim
- If information is not in the provided context, say "I don't have that information in the documents provided"
- Be concise, friendly, and helpful
- For multi-part questions, answer each part clearly
- Never make up policies or procedures
"""
