from llm import get_llm
"""
guardrails/output_guard.py — Output Guardrail
Checks the generated answer for:
  1. Groundedness — is the answer supported by the source chunks?
  2. Hallucination — does the answer contain facts not in the sources?
  3. Toxicity — is the response harmful or inappropriate?
"""

import re
from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from better_profanity import profanity
import config




@dataclass
class OutputGuardResult:
    passed:           bool
    grounded:         bool
    hallucination_risk: bool
    toxic:            bool
    groundedness_score: float
    warnings:         List[str] = field(default_factory=list)
    safe_answer:      str = ""


GROUNDEDNESS_SYSTEM = """You are a fact-verification assistant for a RAG system.

Given an answer and the source context it was generated from, evaluate:
1. groundedness_score: float 0.0-1.0 — fraction of answer claims supported by context
2. hallucination: true/false — does the answer contain specific facts NOT in the context?

Respond with ONLY a JSON object:
{
  "groundedness_score": 0.85,
  "hallucination": false,
  "hallucinated_claims": ["claim1 if any"]
}"""


def _check_groundedness(
    answer:  str,
    context: List[Document],
) -> dict:
    """Check if the answer is grounded in the retrieved context."""
    context_text = "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(context)]
    )
    prompt = f"Context:\n{context_text}\n\nAnswer to verify:\n{answer}"

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=GROUNDEDNESS_SYSTEM),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    try:
        import json
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {
            "groundedness_score": 0.7,
            "hallucination": False,
            "hallucinated_claims": [],
        }


def _check_toxicity(answer: str) -> bool:
    """Returns True if the answer contains toxic/profane content."""
    return profanity.contains_profanity(answer)


def run_output_guard(
    answer:  str,
    context: List[Document],
) -> OutputGuardResult:
    """
    Run all output checks on the generated answer.
    Returns a structured result with warnings and a safe version of the answer.
    """
    warnings      = []
    safe_answer   = answer
    hallucination = False
    toxic         = False

    # 1. Groundedness check
    ground_result = _check_groundedness(answer, context)
    g_score       = ground_result.get("groundedness_score", 0.5)
    hallucination = ground_result.get("hallucination", False)

    if g_score < config.GROUNDEDNESS_THRESHOLD:
        warnings.append(
            f"⚠️ Low groundedness score ({g_score:.0%}). "
            "This answer may not be fully supported by company documents."
        )

    if hallucination:
        claims = ground_result.get("hallucinated_claims", [])
        warnings.append(
            "⚠️ Potential hallucination detected. "
            f"Unverified claims: {', '.join(claims[:2]) if claims else 'unknown'}."
        )
        # Append disclaimer to answer
        safe_answer = (
            answer + "\n\n"
            "_Note: Parts of this answer could not be verified against source documents. "
            "Please confirm with HR directly._"
        )

    # 2. Toxicity check
    toxic = _check_toxicity(answer)
    if toxic:
        warnings.append("⚠️ Response flagged for inappropriate content.")
        safe_answer = (
            "I'm sorry, I was unable to generate an appropriate response. "
            "Please contact HR directly for assistance."
        )

    passed = g_score >= config.GROUNDEDNESS_THRESHOLD and not toxic

    print(f"[OutputGuard] Groundedness={g_score:.2f}, "
          f"Hallucination={hallucination}, Toxic={toxic}, Passed={passed}")

    return OutputGuardResult(
        passed=passed,
        grounded=g_score >= config.GROUNDEDNESS_THRESHOLD,
        hallucination_risk=hallucination,
        toxic=toxic,
        groundedness_score=g_score,
        warnings=warnings,
        safe_answer=safe_answer,
    )
