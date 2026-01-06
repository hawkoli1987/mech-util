"""
LLM Configuration Utility

Extracts get_llm_config() function from agent_client.py for shared use.
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_llm_config() -> Dict[str, str]:
    """
    Get LLM configuration to pass to external agents.
    
    Returns:
        Dict with api_base and model keys
    """
    api_base = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    
    # Import here to avoid circular dependency
    # Note: This will be imported as mech_util.tools.llm_client after package installation
    try:
        from mech_util.tools.llm_client import get_available_model
    except ImportError:
        # Fallback for relative import during development
        from .llm_client import get_available_model
    
    try:
        model = get_available_model(api_base)
    except Exception as e:
        logger.warning(f"Failed to detect model from {api_base}: {e}")
        model = "default"
    
    return {
        "api_base": api_base,
        "model": model
    }

