"""
guardrails/retrieval_guard.py — Retrieval Guardrail
Filters out chunks that are not relevant enough to the query.
Prevents the LLM from being distracted by low-quality context.
"""

from typing import List, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
import config


@dataclass
class RetrievalGuardResult:
    passed:          bool
    filtered_chunks: List[Tuple[Document, float]]
    removed_count:   int
    reason:          str


def run_retrieval_guard(
    query:  str,
    chunks: List[Tuple[Document, float]],
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> RetrievalGuardResult:
    """
    Filter chunks below the relevance threshold.
    Also ensures we have at least 1 chunk before proceeding.
    """
    original_count = len(chunks)

    # Filter by threshold
    filtered = [
        (doc, score) for doc, score in chunks
        if score >= threshold
    ]

    removed = original_count - len(filtered)

    if removed > 0:
        print(f"[RetrievalGuard] Removed {removed} low-relevance chunks "
              f"(threshold={threshold})")

    if not filtered:
        return RetrievalGuardResult(
            passed=False,
            filtered_chunks=[],
            removed_count=removed,
            reason=(
                "No sufficiently relevant documents were found for your question. "
                "Please ensure the relevant documents have been uploaded, "
                "or try rephrasing your question."
            ),
        )

    return RetrievalGuardResult(
        passed=True,
        filtered_chunks=filtered,
        removed_count=removed,
        reason=f"Kept {len(filtered)} of {original_count} chunks above relevance threshold.",
    )
