"""Prompt templates — structured prompt definitions for LLM interactions.

This package contains prompt templates and composition utilities.
Prompts are kept separate from LLM clients to allow independent
iteration on messaging strategy.
"""

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.merchant_prompt import build_merchant_prompt
from app.prompts.customer_prompt import build_customer_prompt

__all__ = ["SYSTEM_PROMPT", "build_merchant_prompt", "build_customer_prompt"]
