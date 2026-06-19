from llm import get_llm
"""
guardrails/input_guard.py — Input Guardrail
Validates incoming queries before they hit the RAG pipeline.
Checks:
  1. Length constraints
  2. Profanity / toxic content
  3. Topic relevance to HR/onboarding domain
  4. Prompt injection attempts
"""

import re
from dataclasses import dataclass
from better_profanity import profanity
from langchain_core.messages import HumanMessage, SystemMessage
import config


profanity.load_censor_words()



@dataclass
class InputGuardResult:
    passed:      bool
    action:      str          # "allow" | "warn" | "block"
    reason:      str
    clean_query: str          # sanitized version of the query


RELEVANCE_SYSTEM = """You are a topic classifier for a company HR onboarding assistant.
Determine if the employee's question is relevant to company onboarding topics such as:
- HR policies, leave, attendance, benefits
- Salary, reimbursements, payroll
- IT setup, tools, equipment
- Code of conduct, compliance
- Team structure, org chart
- Onboarding tasks, first-week checklist
- Company culture, values

Respond with ONLY one word: RELEVANT or IRRELEVANT"""


# ── Injection patterns ────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+",
    r"disregard\s+your\s+",
    r"forget\s+everything",
    r"act\s+as\s+(if\s+you\s+are|a\s+)",
    r"jailbreak",
    r"DAN\s+mode",
    r"pretend\s+you\s+are",
]

def _check_injection(query: str) -> bool:
    """Returns True if query looks like a prompt injection."""
    q_lower = query.lower()
    return any(re.search(p, q_lower) for p in INJECTION_PATTERNS)


def _check_relevance(query: str) -> bool:
    """Returns True if query is relevant to HR/onboarding domain."""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=RELEVANCE_SYSTEM),
        HumanMessage(content=query),
    ])
    verdict = response.content.strip().upper()
    return "RELEVANT" in verdict


def run_input_guard(query: str) -> InputGuardResult:
    """
    Run all input checks and return a structured result.
    """
    # 1. Length check
    if len(query.strip()) < config.MIN_QUERY_LENGTH:
        return InputGuardResult(
            passed=False, action="block",
            reason="Query is too short. Please ask a complete question.",
            clean_query=query,
        )

    if len(query) > config.MAX_QUERY_LENGTH:
        return InputGuardResult(
            passed=False, action="block",
            reason=f"Query exceeds {config.MAX_QUERY_LENGTH} characters. Please shorten your question.",
            clean_query=query[:config.MAX_QUERY_LENGTH],
        )

    # 2. Prompt injection check
    if _check_injection(query):
        return InputGuardResult(
            passed=False, action="block",
            reason="Your query appears to contain an injection attempt and cannot be processed.",
            clean_query=query,
        )

    # 3. Profanity / toxicity check
    if profanity.contains_profanity(query):
        return InputGuardResult(
            passed=False, action="block",
            reason="Please rephrase your question without inappropriate language.",
            clean_query=profanity.censor(query),
        )

    # 4. Topic relevance check
    is_relevant = _check_relevance(query)
    if not is_relevant:
        return InputGuardResult(
            passed=True, action="warn",
            reason="Your question may be outside the scope of company onboarding topics. "
                   "I'll try to help, but results may be limited.",
            clean_query=query,
        )

    return InputGuardResult(
        passed=True, action="allow",
        reason="Query passed all checks.",
        clean_query=query,
    )
