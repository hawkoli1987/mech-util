"""
Shared utilities for mechanical design agents.

- llm_client: LLM/VLM client management
- llm_config: LLM configuration helper
- prompt_manager: Prompt template management
"""

from mech_util.tools.llm_client import (
    get_available_model,
    create_llm_client,
    create_vlm_client,
    get_llm_for_tests,
    check_llm_availability,
)
from mech_util.tools.llm_config import get_llm_config
from mech_util.tools.prompt_manager import (
    PromptManager,
    PromptMetadata,
    PromptTemplate,
    get_prompt_manager,
)

__all__ = [
    # LLM client
    "get_available_model",
    "create_llm_client",
    "create_vlm_client",
    "get_llm_for_tests",
    "check_llm_availability",
    # LLM config
    "get_llm_config",
    # Prompt manager
    "PromptManager",
    "PromptMetadata",
    "PromptTemplate",
    "get_prompt_manager",
]

