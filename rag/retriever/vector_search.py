"""
rag/retriever/vector_search.py — Semantic vector search using ChromaDB
"""

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class VectorSearchRetriever:
    """Performs semantic similarity search against ChromaDB."""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = config.TOP_K_VECTOR) -> List[Document]:
        """Retrieve top-k semantically similar documents."""
        logger.debug(f"[VectorSearch] Query: '{query}' | top_k={top_k}")
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        docs = []
        for doc, score in results:
            doc.metadata["vector_score"] = round(float(score), 4)
            docs.append(doc)

        logger.debug(f"[VectorSearch] Retrieved {len(docs)} chunks.")
        return docs
