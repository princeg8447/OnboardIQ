"""
rag/advanced/reranker.py — Cross-Encoder Re-ranking
Re-scores retrieved chunks using a cross-encoder model for higher precision.
"""

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class Reranker:
    """
    Uses a cross-encoder to re-rank retrieved documents.
    Cross-encoders score (query, document) pairs jointly — much more
    accurate than bi-encoders but too slow to run on all docs.
    """

    def __init__(self):
        logger.info(f"[Reranker] Loading cross-encoder: {config.RERANKER_MODEL}")
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query: str, docs: List[Document], top_k: int = config.TOP_K_RERANKED) -> List[Document]:
        """
        Score all docs against query, return top_k highest scoring.
        """
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)

        reranked = sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)
        top_docs = reranked[:top_k]

        logger.info(
            f"[Reranker] Re-ranked {len(docs)} → kept top {len(top_docs)}. "
            f"Top score: {top_docs[0].metadata['rerank_score'] if top_docs else 'N/A'}"
        )
        return top_docs
