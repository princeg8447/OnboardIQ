"""
llm.py — Shared LLM factory using Groq (free, fast, no credit card needed)
Get your free API key at: https://console.groq.com
"""

from langchain_groq import ChatGroq
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

import config

_llm_cache: dict = {}


def get_llm(temperature: float = config.LLM_TEMPERATURE, max_tokens: int = config.LLM_MAX_TOKENS) -> ChatGroq:
    """
    Return a cached ChatGroq instance.
    Groq is free, extremely fast, and requires no credit card.
    """
    key = (temperature, max_tokens)
    if key not in _llm_cache:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "1. Go to https://console.groq.com\n"
                "2. Sign up for free (just email, no credit card)\n"
                "3. Create an API key\n"
                "4. Add to your .env file: GROQ_API_KEY=gsk_..."
            )
        _llm_cache[key] = ChatGroq(
            model=config.LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        logger.info(f"[LLM] Initialised Groq model: {config.LLM_MODEL}")
    return _llm_cache[key]
