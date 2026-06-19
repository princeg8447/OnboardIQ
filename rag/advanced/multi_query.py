from llm import get_llm
"""
rag/advanced/multi_query.py — Multi-Query Retrieval
Generates multiple variations of the user query to improve recall.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant helping to improve document retrieval.
Generate {num_queries} different versions of the following question.
Each version should approach the topic from a slightly different angle
to improve the chances of finding relevant information.

Original question: {question}

Return ONLY the {num_queries} questions, one per line, no numbering or preamble.
""")


class MultiQueryGenerator:
    """
    Generates multiple query variations and merges their retrieval results.
    Deduplicates by content to avoid redundant chunks.
    """

    def __init__(self, retriever):
        self.llm = get_llm()
        self.retriever = retriever  # any retriever with .search(query) method
        self.chain = MULTI_QUERY_PROMPT | self.llm

    def generate_queries(self, query: str, num_queries: int = config.MULTI_QUERY_COUNT) -> List[str]:
        """Generate query variations."""
        response = self.chain.invoke({
            "question": query,
            "num_queries": num_queries,
        })
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        queries = queries[:num_queries]  # safety cap
        logger.debug(f"[MultiQuery] Generated {len(queries)} query variations.")
        return queries

    def search(self, query: str) -> List[Document]:
        """
        1. Generate N query variations
        2. Retrieve for each
        3. Deduplicate and merge
        """
        if not config.MULTI_QUERY_ENABLED:
            return self.retriever.search(query)

        queries = self.generate_queries(query)
        all_docs: List[Document] = []
        seen_content: set = set()

        # Always include original query
        for q in [query] + queries:
            results = self.retriever.search(q)
            for doc in results:
                key = doc.page_content[:150]
                if key not in seen_content:
                    seen_content.add(key)
                    doc.metadata["retrieved_by_query"] = q
                    all_docs.append(doc)

        logger.info(f"[MultiQuery] Retrieved {len(all_docs)} unique chunks across {len(queries)+1} queries.")
        return all_docs
