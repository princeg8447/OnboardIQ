from llm import get_llm
"""
rag/advanced/compressor.py — Contextual Compression
Trims retrieved chunks to only the parts relevant to the query.
Reduces noise sent to the LLM.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


COMPRESS_PROMPT = ChatPromptTemplate.from_template("""
Given the following document chunk and a user question, extract ONLY the parts
of the document that are directly relevant to answering the question.

If the document has no relevant information, respond with: "NOT_RELEVANT"
Do not add any commentary — only return the extracted relevant text.

Question: {question}

Document:
{document}

Relevant extract:""")


class ContextualCompressor:
    """
    Compresses retrieved chunks by extracting only relevant sentences/paragraphs.
    Filters out chunks with no relevant content.
    """

    def __init__(self):
        self.llm = get_llm()
        self.chain = COMPRESS_PROMPT | self.llm

    def compress(self, query: str, docs: List[Document]) -> List[Document]:
        """
        For each doc, extract only the relevant portion.
        Drops docs that are NOT_RELEVANT.
        """
        compressed = []
        for doc in docs:
            response = self.chain.invoke({
                "question": query,
                "document": doc.page_content,
            })
            extract = response.content.strip()

            if extract.upper() == "NOT_RELEVANT" or len(extract) < 20:
                logger.debug(f"[Compressor] Dropped irrelevant chunk.")
                continue

            compressed_doc = Document(
                page_content=extract,
                metadata={**doc.metadata, "compressed": True, "original_length": len(doc.page_content)},
            )
            compressed.append(compressed_doc)

        logger.info(f"[Compressor] {len(docs)} chunks → {len(compressed)} after compression.")
        return compressed
