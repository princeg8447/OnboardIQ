from typing import Optional
"""
rag/retriever/hybrid_search.py — Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines vector search (semantic) + BM25 (keyword) results.
"""

from typing import List, Dict
from langchain_core.documents import Document
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

from rag.retriever.vector_search import VectorSearchRetriever
from rag.retriever.bm25_search import BM25Retriever


class HybridSearchRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF).

    RRF Formula: score(d) = sum(1 / (k + rank_i(d)))
    where k=60 is a constant that dampens the impact of high ranks.
    """

    def __init__(
        self,
        vector_retriever: VectorSearchRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = config.RRF_K,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever   = bm25_retriever
        self.rrf_k            = rrf_k

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Fuse two ranked lists using RRF.
        Returns merged list sorted by fused score.
        """
        rrf_scores: Dict[str, float]   = {}
        doc_map:    Dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            """Use content hash as a stable key."""
            return doc.page_content[:100]  # first 100 chars as proxy ID

        # Score from vector results
        for rank, doc in enumerate(vector_results):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key]    = doc

        # Score from BM25 results
        for rank, doc in enumerate(bm25_results):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        # Sort by fused score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
            results.append(doc)

        return results

    def search(self, query: str, top_k: int = config.TOP_K_HYBRID) -> List[Document]:
        """
        Run hybrid search:
          1. Vector search
          2. BM25 search
          3. RRF fusion
        """
        logger.info(f"[HybridSearch] Query: '{query}'")

        vector_results = self.vector_retriever.search(query, top_k=config.TOP_K_VECTOR)
        bm25_results   = self.bm25_retriever.search(query,   top_k=config.TOP_K_BM25)

        fused = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)

        logger.info(
            f"[HybridSearch] Vector={len(vector_results)}, "
            f"BM25={len(bm25_results)}, Fused={len(fused)}"
        )
        return fused


# ── Module-level convenience function ─────────────────────────────────────────
_hybrid_retriever: Optional[HybridSearchRetriever] = None

def hybrid_search(query: str, k: int = config.TOP_K_HYBRID):
    """
    Module-level hybrid search. Lazily initialises retriever on first call.
    Returns List[Tuple[Document, float]] for compatibility with pipeline.
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from rag.ingestor import get_vectorstore, _get_ingestor
        from rag.retriever.vector_search import VectorSearchRetriever
        from rag.retriever.bm25_search import BM25Retriever
        from langchain_core.documents import Document

        vs      = get_vectorstore()
        vec_ret = VectorSearchRetriever(vs)
        bm25    = BM25Retriever()

        # Build BM25 from stored docs
        raw = vs.get(include=["documents", "metadatas"])
        bm25_docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(
                raw.get("documents", []),
                raw.get("metadatas", []),
            )
        ]
        if bm25_docs:
            bm25.build_index(bm25_docs)

        _hybrid_retriever = HybridSearchRetriever(vec_ret, bm25)

    results = _hybrid_retriever.search(query, top_k=k)
    return [(doc, doc.metadata.get("rrf_score", 0.0)) for doc in results]
