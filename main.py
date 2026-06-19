from llm import get_llm
"""
main.py — OnboardIQ Main Pipeline
Orchestrates the full flow:
  Input Guard → Agent (Retrieve + Reason) → Retrieval Guard
  → LLM Answer → Self-RAG → Output Guard → Final Response
"""

from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from guardrails.input_guard    import run_input_guard, InputGuardResult
from guardrails.retrieval_guard import run_retrieval_guard
from guardrails.output_guard   import run_output_guard
from agent.agent               import run_agent
from rag.advanced.self_rag     import self_rag_loop
from rag.retriever.hybrid_search import hybrid_search
from rag.advanced.query_pipeline import rerank_chunks, upgrade_to_parent_chunks, compress_chunks
import config


# ── Response dataclass ────────────────────────────────────────────
@dataclass
class OnboardIQResponse:
    answer:              str
    sources:             List[str]          = field(default_factory=list)
    warnings:            List[str]          = field(default_factory=list)
    blocked:             bool               = False
    block_reason:        str                = ""
    groundedness_score:  float              = 0.0
    hallucination_risk:  bool               = False
    steps_taken:         List[str]          = field(default_factory=list)


# ── LLM for direct answer generation ─────────────────────────────


def _generate_answer(query: str, context: List[Document]) -> str:
    """Generate an answer given a query and list of context documents."""
    context_text = "\n\n---\n\n".join(
        [f"[Source {i+1}: {doc.metadata.get('file_name') or doc.metadata.get('url', 'Unknown')}]\n"
         f"{doc.page_content}"
         for i, doc in enumerate(context)]
    )
    prompt = (
        f"Context from company documents:\n\n{context_text}\n\n"
        f"Employee Question: {query}\n\n"
        "Please answer the question based strictly on the context above. "
        "Cite which source (Source 1, Source 2, etc.) each piece of information comes from."
    )
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content.strip()


def _retrieve_and_filter(query: str) -> List[Document]:
    """Retrieve and filter documents for Self-RAG re-retrieval."""
    results = hybrid_search(query, k=config.INITIAL_RETRIEVAL_K)
    results = rerank_chunks(query, results, top_k=config.RERANK_TOP_K)
    results = upgrade_to_parent_chunks(results)
    guard   = run_retrieval_guard(query, results)
    docs    = [doc for doc, _ in guard.filtered_chunks]
    return docs


# ── Main Pipeline ─────────────────────────────────────────────────
def ask(query: str) -> OnboardIQResponse:
    """
    Full OnboardIQ pipeline.
    Takes a raw user query, returns a structured response.
    """
    steps  = []
    resp   = OnboardIQResponse(answer="", sources=[], warnings=[], steps_taken=steps)

    # ── Step 1: Input Guardrail ───────────────────────────────────
    steps.append("input_guard")
    input_result = run_input_guard(query)

    if input_result.action == "block":
        resp.blocked     = True
        resp.block_reason = input_result.reason
        resp.answer      = input_result.reason
        return resp

    if input_result.action == "warn":
        resp.warnings.append(input_result.reason)

    clean_query = input_result.clean_query

    # ── Step 2: Agentic Retrieval (ReAct) ─────────────────────────
    steps.append("agent_retrieval")
    agent_result = run_agent(clean_query)
    agent_answer = agent_result["answer"]
    resp.sources = agent_result["sources"]

    # ── Step 3: Direct Retrieval for guard checks ─────────────────
    # (We still need raw Document objects for guardrails)
    steps.append("direct_retrieval")
    raw_results = hybrid_search(clean_query, k=config.INITIAL_RETRIEVAL_K)
    raw_results = rerank_chunks(clean_query, raw_results, top_k=config.RERANK_TOP_K)
    raw_results = upgrade_to_parent_chunks(raw_results)

    # ── Step 4: Retrieval Guardrail ───────────────────────────────
    steps.append("retrieval_guard")
    ret_guard = run_retrieval_guard(clean_query, raw_results)

    if not ret_guard.passed:
        resp.answer   = ret_guard.reason
        resp.warnings.append(ret_guard.reason)
        return resp

    context_docs = [doc for doc, _ in ret_guard.filtered_chunks]

    # ── Step 5: Self-RAG loop ─────────────────────────────────────
    steps.append("self_rag")
    final_answer, final_context, critique = self_rag_loop(
        query=clean_query,
        answer=agent_answer,
        context=context_docs,
        retrieve_fn=_retrieve_and_filter,
        generate_fn=_generate_answer,
    )

    # ── Step 6: Output Guardrail ──────────────────────────────────
    steps.append("output_guard")
    out_guard = run_output_guard(final_answer, final_context)
    resp.warnings.extend(out_guard.warnings)
    resp.groundedness_score = out_guard.groundedness_score
    resp.hallucination_risk = out_guard.hallucination_risk

    # ── Final Response ────────────────────────────────────────────
    resp.answer = out_guard.safe_answer
    return resp


# ── CLI Interface ─────────────────────────────────────────────────
if __name__ == "__main__":
    from rag.ingestor import ingest_documents
    import os

    print("=" * 60)
    print("  OnboardIQ — Company Onboarding Assistant")
    print("=" * 60)

    # Auto-ingest sample docs from data directory
    data_dir = config.DATA_DIR
    pdf_paths, text_paths = [], []

    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            fp = os.path.join(data_dir, f)
            if f.endswith(".pdf"):
                pdf_paths.append(fp)
            elif f.endswith((".txt", ".md")):
                text_paths.append(fp)

    if pdf_paths or text_paths:
        print(f"\nIngesting {len(pdf_paths)} PDFs and {len(text_paths)} text files...")
        ingest_documents(pdf_paths=pdf_paths, text_paths=text_paths)
        print("Ingestion complete!\n")
    else:
        print("\nNo documents found in data/sample_docs/")
        print("Add PDF, TXT, or MD files to get started.\n")

    # Chat loop
    print("Ask me anything about company onboarding (type 'quit' to exit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        response = ask(query)
        print(f"\nOnboardIQ: {response.answer}")

        if response.sources:
            print(f"\n📄 Sources: {', '.join(response.sources)}")

        if response.warnings:
            for w in response.warnings:
                print(f"\n{w}")

        print(f"\n[Groundedness: {response.groundedness_score:.0%} | "
              f"Hallucination Risk: {response.hallucination_risk}]")
        print("-" * 60)
