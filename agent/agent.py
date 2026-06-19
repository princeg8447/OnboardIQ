"""
agent/agent.py — ReAct Agent (custom loop, no AgentExecutor dependency)
Implements Reason + Act pattern manually for full compatibility.
The agent decides WHEN to search, WHAT to search for, and WHETHER to search again.
"""

from llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from typing import List
from loguru import logger

import config


REACT_SYSTEM_PROMPT = """You are OnboardIQ, an intelligent company onboarding assistant.
You help new employees find answers about HR policies, benefits, IT setup, procedures, and more.

You have access to these tools:
- search_company_docs(query): Search HR and onboarding documents
- hyde_search_docs(query): Advanced semantic search for vague questions
- multi_query_search(query): Broad search using multiple query variations
- list_available_documents(): List all available documents

RULES:
- Always search before answering
- If first search is insufficient, try a different tool or refined query
- Always cite the document source in your answer
- Never make up information not found in documents
- If you cannot find the answer after searching, say so clearly

FORMAT your response as:
Thought: what you're going to do
Action: tool_name
Action Input: your search query
Observation: [tool result will appear here]
... repeat if needed ...
Thought: I have enough information
Final Answer: your complete answer with source citations
"""


def _call_tool(tool_name: str, tool_input: str) -> str:
    """Dispatch a tool call by name."""
    from agent.tools import search_company_docs, hyde_search_docs, multi_query_search, list_available_documents

    tool_map = {
        "search_company_docs":      search_company_docs,
        "hyde_search_docs":         hyde_search_docs,
        "multi_query_search":       multi_query_search,
        "list_available_documents": list_available_documents,
    }

    tool_fn = tool_map.get(tool_name)
    if not tool_fn:
        return f"Unknown tool: {tool_name}. Available tools: {list(tool_map.keys())}"

    try:
        return tool_fn.invoke(tool_input)
    except Exception as e:
        return f"Tool error: {str(e)}"


def run_agent(query: str) -> dict:
    """
    Run the ReAct agent loop on a query.
    Returns dict with answer, sources, and steps taken.
    """
    llm   = get_llm()
    steps = []
    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\nThought:"),
    ]

    import re
    final_answer = ""

    for iteration in range(config.AGENT_MAX_ITERATIONS):
        logger.info(f"[Agent] Iteration {iteration + 1}")

        response  = llm.invoke(messages)
        llm_text  = response.content.strip()

        if config.AGENT_VERBOSE:
            logger.debug(f"[Agent] LLM output:\n{llm_text}")

        # Check for Final Answer
        if "Final Answer:" in llm_text:
            final_answer = llm_text.split("Final Answer:")[-1].strip()
            break

        # Parse Action and Action Input
        action_match = re.search(r"Action:\s*(\w+)", llm_text)
        input_match  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", llm_text, re.DOTALL)

        if not action_match:
            # LLM didn't follow format — treat output as final answer
            final_answer = llm_text
            break

        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else query

        logger.info(f"[Agent] Calling tool: {tool_name}({tool_input[:60]}...)")
        observation = _call_tool(tool_name, tool_input)
        steps.append({"tool": tool_name, "input": tool_input, "output": observation})

        # Append to conversation
        messages.append(HumanMessage(content=llm_text))
        messages.append(HumanMessage(content=f"Observation: {observation[:1500]}\nThought:"))

    if not final_answer:
        final_answer = "I was unable to find a definitive answer. Please try rephrasing your question."

    # Extract sources from steps
    sources = []
    for step in steps:
        import re
        matches = re.findall(r"Source:\s*([^\|^\]]+)", step.get("output", ""))
        sources.extend([m.strip() for m in matches])
    sources = list(set(sources))

    return {
        "answer":  final_answer,
        "sources": sources,
        "steps":   steps,
    }
