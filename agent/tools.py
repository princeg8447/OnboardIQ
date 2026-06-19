"""
agent/tools.py — LangChain Tools for the ReAct Agent
Each tool is a discrete action the agent can take.
"""

from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document


# ── Tool 1: Hybrid Search ─────────────────────────────────────────
@tool
def search_company_docs(query: str) -> str:
    """
    Search company HR and onboarding documents for information.
    Use this to find policies, procedures, guidelines, and any company information.
    Input: a search query string.
    Output: relevant document excerpts with source information.
    """
    from rag.retriever.hybrid_search import hybrid_search
    from rag.advanced.query_pipeline import rerank_chunks, upgrade_to_parent_chunks
    import config

    results = hybrid_search(query, k=config.INITIAL_RETRIEVAL_K)
    results = rerank_chunks(query, results, top_k=config.RERANK_TOP_K)
    results = upgrade_to_parent_chunks(results)

    if not results:
        return "No relevant documents found for this query."

    output_parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("file_name") or doc.metadata.get("url", "Unknown")
        output_parts.append(
            f"[Document {i} | Source: {source} | Relevance: {score:.2f}]\n"
            f"{doc.page_content[:600]}"
        )

    return "\n\n---\n\n".join(output_parts)


# ── Tool 2: HyDE Search ───────────────────────────────────────────
@tool
def hyde_search_docs(query: str) -> str:
    """
    Advanced search using HyDE (Hypothetical Document Embeddings).
    Use this when regular search doesn't return good results,
    or when the query is abstract/vague and needs semantic enrichment.
    Input: a search query string.
    Output: relevant document excerpts.
    """
    from rag.retriever.hyde import hyde_search
    from rag.advanced.query_pipeline import rerank_chunks, upgrade_to_parent_chunks
    import config

    results = hyde_search(query, k=config.INITIAL_RETRIEVAL_K)
    results = rerank_chunks(query, results, top_k=config.RERANK_TOP_K)
    results = upgrade_to_parent_chunks(results)

    if not results:
        return "No relevant documents found even with HyDE search."

    output_parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("file_name") or doc.metadata.get("url", "Unknown")
        output_parts.append(
            f"[Document {i} | Source: {source} | Relevance: {score:.2f}]\n"
            f"{doc.page_content[:600]}"
        )
    return "\n\n---\n\n".join(output_parts)


# ── Tool 3: Multi-Query Search ────────────────────────────────────
@tool
def multi_query_search(query: str) -> str:
    """
    Search using multiple query variations for broader coverage.
    Use this for complex questions that might require searching different aspects.
    Input: the original question.
    Output: combined results from multiple query angles.
    """
    from rag.advanced.query_pipeline import generate_multi_queries, rerank_chunks, upgrade_to_parent_chunks
    from rag.retriever.hybrid_search import hybrid_search
    import config

    queries  = generate_multi_queries(query)
    all_docs: dict = {}   # deduplicate by content hash

    for q in queries:
        results = hybrid_search(q, k=10)
        for doc, score in results:
            key = doc.page_content[:100]
            if key not in all_docs or all_docs[key][1] < score:
                all_docs[key] = (doc, score)

    combined = list(all_docs.values())
    combined = rerank_chunks(query, combined, top_k=config.RERANK_TOP_K)
    combined = upgrade_to_parent_chunks(combined)

    if not combined:
        return "No documents found across multiple query variations."

    output_parts = []
    for i, (doc, score) in enumerate(combined, 1):
        source = doc.metadata.get("file_name") or doc.metadata.get("url", "Unknown")
        output_parts.append(
            f"[Document {i} | Source: {source} | Relevance: {score:.2f}]\n"
            f"{doc.page_content[:600]}"
        )
    return "\n\n---\n\n".join(output_parts)


# ── Tool 4: List Available Documents ─────────────────────────────
@tool
def list_available_documents(query: str = "") -> str:
    """
    List all documents that have been ingested into the knowledge base.
    Use this when the user asks what documents are available,
    or before searching to understand what sources exist.
    Input: ignored (leave empty).
    Output: list of ingested document sources.
    """
    from rag.ingestor import get_vectorstore
    vs = get_vectorstore()
    result = vs.get(include=["metadatas"])
    sources = set()
    for meta in result.get("metadatas", []):
        if meta:
            src = meta.get("file_name") or meta.get("url") or meta.get("source", "Unknown")
            sources.add(src)

    if not sources:
        return "No documents have been ingested yet."

    return "Available documents:\n" + "\n".join(f"- {s}" for s in sorted(sources))


# ── Export all tools ──────────────────────────────────────────────
ALL_TOOLS = [
    search_company_docs,
    hyde_search_docs,
    multi_query_search,
    list_available_documents,
]
