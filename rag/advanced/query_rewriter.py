from llm import get_llm
"""
rag/advanced/query_rewriter.py — Query Rewriting
Rewrites ambiguous or poorly phrased queries into clearer search queries.
"""

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a search query optimizer for a company HR knowledge base.
Rewrite the following user question into a clear, keyword-rich search query
that will retrieve the most relevant HR documents.

Rules:
- Keep it concise (under 20 words)
- Include specific terms like "policy", "procedure", "benefit", "process" where relevant
- Remove conversational filler
- Expand abbreviations

Original question: {question}

Rewritten query (return ONLY the rewritten query, nothing else):""")


class QueryRewriter:
    """Rewrites user queries for better retrieval performance."""

    def __init__(self):
        self.llm = get_llm()
        self.chain = REWRITE_PROMPT | self.llm

    def rewrite(self, query: str) -> str:
        """Rewrite a query for better retrieval."""
        response = self.chain.invoke({"question": query})
        rewritten = response.content.strip()
        logger.info(f"[QueryRewriter] '{query}' → '{rewritten}'")
        return rewritten
