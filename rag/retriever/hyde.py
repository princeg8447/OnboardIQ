from llm import get_llm
"""
rag/retriever/hyde.py — HyDE (Hypothetical Document Embeddings)
Instead of embedding the question, we generate a hypothetical answer
and embed THAT — closing the query-document semantic gap.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.retriever.hybrid_search import hybrid_search
import config


# ── LLM for hypothesis generation ────────────────────────────────


HYDE_SYSTEM = """You are a company HR document expert.
Given an employee question, write a short paragraph (3-5 sentences) that looks like
a passage from an official HR policy document that would ANSWER this question.
Write it as if it were extracted from a real policy — use formal document language.
Do NOT say you're hypothesizing. Just write the document passage."""


def generate_hypothetical_document(query: str) -> str:
    """
    Ask the LLM to generate what a relevant document passage would look like.
    This hypothetical text is then used for embedding-based retrieval.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=HYDE_SYSTEM),
        HumanMessage(content=f"Employee question: {query}"),
    ]
    response = llm.invoke(messages)
    hypothesis = response.content.strip()
    print(f"[HyDE] Generated hypothesis ({len(hypothesis)} chars)")
    return hypothesis


def hyde_search(
    query: str,
    k:     int = config.INITIAL_RETRIEVAL_K,
) -> List[Tuple[Document, float]]:
    """
    HyDE search pipeline:
    1. Generate a hypothetical document for the query
    2. Use that hypothetical document as the search query
    3. Run hybrid search against the hypothetical text
    4. Return results
    """
    if not config.HYDE_ENABLED:
        return hybrid_search(query, k=k)

    hypothesis = generate_hypothetical_document(query)

    # Search using the hypothetical document text (richer semantic signal)
    results = hybrid_search(hypothesis, k=k)
    print(f"[HyDE] '{query[:60]}' → {len(results)} results via hypothesis")
    return results
