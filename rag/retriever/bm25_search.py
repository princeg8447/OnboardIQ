from typing import Optional
"""
rag/retriever/bm25_search.py — BM25 keyword-based search
Complements vector search by catching exact term matches.
"""

from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from loguru import logger
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    BM25 keyword retriever.
    Must be built/rebuilt after every ingestion.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.docs: List[Document] = []

    def build_index(self, docs: List[Document]):
        """Build BM25 index from a list of Documents."""
        self.docs = docs
        corpus = [tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"[BM25] Index built with {len(docs)} documents.")

    def search(self, query: str, top_k: int = config.TOP_K_BM25) -> List[Document]:
        """Retrieve top-k documents by BM25 score."""
        if self.bm25 is None or not self.docs:
            logger.warning("[BM25] Index not built. Returning empty results.")
            return []

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair docs with scores, sort descending
        scored = sorted(
            zip(self.docs, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for doc, score in scored:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": round(float(score), 4)},
            )
            results.append(doc_copy)

        logger.debug(f"[BM25] Retrieved {len(results)} chunks.")
        return results
