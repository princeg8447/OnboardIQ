from llm import get_llm
"""
rag/advanced/hyde.py — HyDE (Hypothetical Document Embeddings)
Instead of embedding the question, generate a fake answer and embed that.
This bridges the gap between question and answer embedding spaces.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


HYDE_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant. Write a short, factual paragraph (3-5 sentences) that 
would be found in a company's employee handbook and would directly answer this question:

Question: {question}

Write only the hypothetical answer paragraph, no preamble or explanation.
""")


class HyDEGenerator:
    """
    Generates a hypothetical document for a query,
    then uses that to search the vector store.
    """

    def __init__(self, vectorstore):
        self.llm = get_llm()
        self.vectorstore = vectorstore
        self.chain = HYDE_PROMPT | self.llm

    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical answer document for the query."""
        logger.debug(f"[HyDE] Generating hypothetical doc for: '{query}'")
        response = self.chain.invoke({"question": query})
        hypothetical = response.content
        logger.debug(f"[HyDE] Hypothetical doc: {hypothetical[:100]}...")
        return hypothetical

    def search(self, query: str, top_k: int = config.TOP_K_VECTOR) -> List[Document]:
        """
        1. Generate hypothetical answer
        2. Embed hypothetical answer
        3. Search vector store with that embedding
        """
        if not config.HYDE_ENABLED:
            return []

        hypothetical_doc = self.generate_hypothetical_doc(query)

        results = self.vectorstore.similarity_search(hypothetical_doc, k=top_k)
        for doc in results:
            doc.metadata["retrieved_by"] = "hyde"

        logger.info(f"[HyDE] Retrieved {len(results)} chunks via hypothetical doc.")
        return results
