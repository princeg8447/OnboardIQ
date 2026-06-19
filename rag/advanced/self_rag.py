from llm import get_llm
"""
rag/advanced/self_rag.py — Self-RAG
The LLM critiques its own answer and decides whether to re-retrieve.
Implements a simple critique → re-retrieve loop.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating whether a generated answer adequately addresses the question
based on the provided context.

Question: {question}

Context used:
{context}

Generated answer:
{answer}

Evaluate:
1. Is the answer complete? Does it address all parts of the question?
2. Is the answer grounded in the context?
3. Is important information missing?

Respond with ONLY one of:
- "SUFFICIENT" — answer is complete and well-grounded
- "INSUFFICIENT: <brief reason>" — answer needs improvement

Your evaluation:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
{system_prompt}

Context from company documents:
{context}

Question: {question}

Provide a clear, helpful answer based strictly on the context above.
End with a "Sources:" section listing the document names used.

Answer:""")


class SelfRAG:
    """
    Self-RAG loop:
      1. Generate initial answer
      2. Critique the answer
      3. If insufficient, rewrite query + re-retrieve + regenerate
      4. Repeat up to MAX_ITERATIONS
    """

    def __init__(self, retriever):
        self.llm = get_llm()
        self.retriever      = retriever
        self.critique_chain = CRITIQUE_PROMPT | self.llm
        self.answer_chain   = ANSWER_PROMPT | self.llm

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "Unknown")
            parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, query: str, docs: List[Document]) -> str:
        context = self._format_context(docs)
        response = self.answer_chain.invoke({
            "system_prompt": config.SYSTEM_PROMPT,
            "context": context,
            "question": query,
        })
        return response.content

    def _critique_answer(self, query: str, docs: List[Document], answer: str) -> Tuple[bool, str]:
        """Returns (is_sufficient, reason)."""
        context  = self._format_context(docs)
        response = self.critique_chain.invoke({
            "question": query,
            "context":  context,
            "answer":   answer,
        })
        result = response.content.strip()
        if result.startswith("SUFFICIENT"):
            return True, "Answer is complete and grounded."
        else:
            reason = result.replace("INSUFFICIENT:", "").strip()
            return False, reason

    def _refine_query(self, original: str, reason: str) -> str:
        """Create a follow-up query to fill gaps."""
        return f"{original} {reason}"

    def run(self, query: str, initial_docs: List[Document]) -> Tuple[str, List[Document], int]:
        """
        Run the Self-RAG loop.
        Returns: (final_answer, final_docs, iterations_used)
        """
        if not config.SELF_RAG_ENABLED:
            answer = self._generate_answer(query, initial_docs)
            return answer, initial_docs, 1

        docs    = initial_docs
        answer  = ""
        iterations = 0

        for i in range(config.SELF_RAG_MAX_ITERATIONS):
            iterations = i + 1
            logger.info(f"[SelfRAG] Iteration {iterations}")

            answer = self._generate_answer(query, docs)
            sufficient, reason = self._critique_answer(query, docs, answer)

            if sufficient:
                logger.info(f"[SelfRAG] Answer sufficient after {iterations} iteration(s).")
                break
            else:
                logger.info(f"[SelfRAG] Insufficient: {reason}. Re-retrieving...")
                refined_query = self._refine_query(query, reason)
                new_docs      = self.retriever.search(refined_query)
                # Merge new docs with existing, deduplicate
                seen = {d.page_content[:100] for d in docs}
                for d in new_docs:
                    if d.page_content[:100] not in seen:
                        docs.append(d)
                        seen.add(d.page_content[:100])

        return answer, docs, iterations


# ── Module-level convenience function ─────────────────────────────────────────
def self_rag_loop(
    query:       str,
    answer:      str,
    context:     list,
    retrieve_fn,
    generate_fn,
) -> tuple:
    """
    Module-level self-RAG loop.
    Returns (final_answer, final_context, critique_note).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    import config

    llm = get_llm()

    CRITIQUE_SYSTEM = """Evaluate if this answer adequately addresses the question using the context.
Reply with ONLY "SUFFICIENT" or "INSUFFICIENT: <one-line reason>"."""

    for i in range(config.SELF_RAG_MAX_ITERATIONS):
        context_text = "\n\n".join(d.page_content for d in context[:5])
        prompt = f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer:\n{answer}"
        resp   = llm.invoke([SystemMessage(content=CRITIQUE_SYSTEM), HumanMessage(content=prompt)])
        result = resp.content.strip()

        if result.startswith("SUFFICIENT"):
            return answer, context, result

        # Refine: re-retrieve with extra context from critique
        reason       = result.replace("INSUFFICIENT:", "").strip()
        refined_q    = f"{query} {reason}"
        extra_docs   = retrieve_fn(refined_q)
        seen         = {d.page_content[:80] for d in context}
        for d in extra_docs:
            if d.page_content[:80] not in seen:
                context.append(d)
                seen.add(d.page_content[:80])

        answer = generate_fn(query, context)

    return answer, context, "max_iterations_reached"
