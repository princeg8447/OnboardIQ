from typing import Optional
from llm import get_llm
"""
rag/advanced/query_pipeline.py — Advanced RAG Techniques
Implements:
  1. Query Rewriter       — fixes vague/short queries
  2. Multi-Query          — generates N query variations
  3. Re-ranker            — cross-encoder scoring
  4. Contextual Compressor— trims irrelevant parts of chunks
  5. Parent-Child Lookup  — swap child chunk for richer parent context
"""

import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from rag.ingestor import get_parent_chunk
import config


# ── Shared LLM ────────────────────────────────────────────────────


# ── 1. Query Rewriter ─────────────────────────────────────────────

REWRITE_SYSTEM = """You are a search query optimizer for a company HR knowledge base.
Given an employee's question, rewrite it into a clear, specific search query
that will retrieve the most relevant HR policy documents.
- Fix grammar and spelling
- Make implicit context explicit
- Use HR/policy terminology
Return ONLY the rewritten query, nothing else."""

def rewrite_query(query: str) -> str:
    """Rewrite a vague/short query into a better search query."""
    if len(query.split()) > 8:
        # Already detailed enough — minor cleanup only
        return query.strip()
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=query),
    ])
    rewritten = response.content.strip()
    print(f"[QueryRewriter] '{query}' → '{rewritten}'")
    return rewritten


# ── 2. Multi-Query Generator ──────────────────────────────────────

MULTI_QUERY_SYSTEM = f"""You are an HR document search specialist.
Given one employee question, generate {config.MULTI_QUERY_COUNT} different search query
variations that capture different angles of the same question.
These will be used to retrieve documents from multiple angles.
Return ONLY a numbered list like:
1. query one
2. query two
3. query three"""

def generate_multi_queries(query: str) -> List[str]:
    """Generate multiple query variations for broader retrieval coverage."""
    if not config.MULTI_QUERY_ENABLED:
        return [query]

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=MULTI_QUERY_SYSTEM),
        HumanMessage(content=query),
    ])
    raw = response.content.strip()

    # Parse numbered list
    queries = []
    for line in raw.split("\n"):
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if line:
            queries.append(line)

    queries = queries[:config.MULTI_QUERY_COUNT]
    if not queries:
        queries = [query]

    print(f"[MultiQuery] Generated {len(queries)} query variations")
    return queries


# ── 3. Cross-Encoder Re-ranker ────────────────────────────────────

_reranker = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Loading cross-encoder: {config.RERANKER_MODEL}")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank_chunks(
    query:   str,
    chunks:  List[Tuple[Document, float]],
    top_k:   int = config.RERANK_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Re-score chunks with a cross-encoder (query + chunk together).
    Cross-encoders are more accurate than bi-encoders for relevance scoring.
    """
    if not chunks:
        return []

    reranker = _get_reranker()
    docs     = [doc for doc, _ in chunks]
    pairs    = [[query, doc.page_content] for doc in docs]
    scores   = reranker.predict(pairs)

    # Sort by cross-encoder score descending
    ranked = sorted(
        zip(docs, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    print(f"[Reranker] Kept top {len(ranked)} of {len(chunks)} chunks")
    return ranked


# ── 4. Contextual Compressor ──────────────────────────────────────

COMPRESS_SYSTEM = """You are a document snippet extractor.
Given a question and a document chunk, extract ONLY the sentences that are
directly relevant to answering the question.
If nothing is relevant, return: [NOT RELEVANT]
Return only the extracted sentences, no explanation."""

def compress_chunk(query: str, chunk: Document) -> Optional[Document]:
    """
    Extract only the relevant sentences from a chunk.
    Returns compressed Document or None if chunk is not relevant.
    """
    llm = get_llm()
    prompt = f"Question: {query}\n\nDocument chunk:\n{chunk.page_content}"
    response = llm.invoke([
        SystemMessage(content=COMPRESS_SYSTEM),
        HumanMessage(content=prompt),
    ])
    compressed = response.content.strip()

    if "[NOT RELEVANT]" in compressed or len(compressed) < 20:
        return None

    return Document(
        page_content=compressed,
        metadata={**chunk.metadata, "compressed": True},
    )


def compress_chunks(
    query:  str,
    chunks: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """Run contextual compression on all chunks."""
    compressed_results = []
    for doc, score in chunks:
        compressed = compress_chunk(query, doc)
        if compressed:
            compressed_results.append((compressed, score))
    print(f"[Compressor] {len(chunks)} → {len(compressed_results)} chunks after compression")
    return compressed_results


# ── 5. Parent-Child Lookup ────────────────────────────────────────

def upgrade_to_parent_chunks(
    chunks: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """
    Swap child chunks for their parent chunks (richer context for LLM).
    Deduplicates so each parent is included only once.
    """
    seen_parent_ids = set()
    upgraded = []

    for doc, score in chunks:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent = get_parent_chunk(parent_id)
            if parent:
                seen_parent_ids.add(parent_id)
                upgraded.append((parent, score))
                continue
        # Fallback: use child chunk if no parent found
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id not in seen_parent_ids:
            seen_parent_ids.add(chunk_id)
            upgraded.append((doc, score))

    print(f"[ParentLookup] Upgraded {len(chunks)} child → {len(upgraded)} parent chunks")
    return upgraded
