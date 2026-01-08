"""
LLM Client Management

Provides utilities for discovering and connecting to vLLM servers.
Used by both tests and production code.

Following rules from .cursor/rules/03_prompts_and_provenance.md:
- No hardcoded model names (query server)
- Explicit error messages for misconfigurations
- Environment variable configuration
"""

import os
from typing import Optional
import requests
from langchain_openai import ChatOpenAI


def get_available_model(api_base: str) -> str:
    """
    Query vLLM server to get the actual model name.
    
    This ensures we use whatever model is actually loaded on the server,
    rather than hardcoding model names.
    
    Args:
        api_base: Base URL of the vLLM server (e.g., "http://hopper-34:8001")
    
    Returns:
        Model name string (e.g., "Qwen/Qwen3-8B")
    
    Raises:
        RuntimeError: If model cannot be detected from server
    
    Example:
        >>> model = get_available_model("http://localhost:8001")
        >>> print(f"Detected model: {model}")
        Detected model: Qwen/Qwen3-8B
    """
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_name = data["data"][0]["id"]
                print(f"  → Detected vLLM model: {model_name}")
                return model_name
            else:
                raise RuntimeError(
                    f"vLLM server at {api_base} returned no models in response: {data}\n"
                    f"Check that vLLM server is properly configured and has a model loaded."
                )
        else:
            raise RuntimeError(
                f"vLLM server at {api_base} returned status code {response.status_code}\n"
                f"Response: {response.text}\n"
                f"Check that vLLM server is running and accessible."
            )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not connect to vLLM server at {api_base}\n"
            f"Error: {e}\n"
            f"Troubleshooting:\n"
            f"  1. Check if vLLM server is running: curl {api_base}/v1/models\n"
            f"  2. Verify OPENAI_API_BASE environment variable\n"
            f"  3. Check network connectivity to server"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to query vLLM server at {api_base} for model name\n"
            f"Error: {e}"
        )


def create_llm_client(
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 300,
    max_retries: int = 2,
    max_tokens: int = 4096
) -> ChatOpenAI:
    """
    Create LangChain ChatOpenAI client connected to vLLM server.
    
    This function handles:
    - Environment variable configuration
    - Automatic model discovery from server
    - OpenAI-compatible API setup for vLLM
    
    Args:
        api_base: Base URL of vLLM server. If None, reads OPENAI_API_BASE env var.
                  Default: "http://hopper-34:8001"
        model_name: Model name to use. If None, queries server to detect.
        temperature: Sampling temperature (0.0-1.0)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        max_tokens: Maximum tokens to generate (default: 4096, prevents infinite generation)
    
    Returns:
        Configured ChatOpenAI client
    
    Raises:
        RuntimeError: If server is unreachable or misconfigured
    
    Environment Variables:
        OPENAI_API_BASE: vLLM server URL (e.g., "http://hopper-34:8001")
        LLM_MODEL_NAME: Optional override for model name
    
    Example:
        >>> # Automatic discovery
        >>> llm = create_llm_client()
        >>> response = llm.invoke([HumanMessage(content="Hello")])
        
        >>> # Explicit configuration
        >>> llm = create_llm_client(
        ...     api_base="http://localhost:8001",
        ...     model_name="Qwen/Qwen3-8B",
        ...     temperature=0.5,
        ...     max_tokens=2048
        ... )
    """
    # Get API base URL
    if api_base is None:
        api_base = os.environ.get("OPENAI_API_BASE", "http://hopper-34:8001")
    
    # Get model name (query server if not provided)
    if model_name is None:
        model_name = os.environ.get("LLM_MODEL_NAME")
        if model_name is None:
            # Auto-discover from server
            model_name = get_available_model(api_base)
    
    print(f"  → Creating LLM client: {api_base} (model: {model_name})")
    
    # Check if this is a Qwen3 model - disable thinking mode if so
    extra_body = {}
    if "qwen3" in model_name.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        print(f"  → Detected Qwen3 model, disabling thinking mode")
    
    # Create ChatOpenAI client
    # Note: vLLM provides OpenAI-compatible API
    # max_tokens is critical for large models like DeepSeek to avoid infinite generation
    llm = ChatOpenAI(
        base_url=f"{api_base}/v1",
        api_key="dummy",  # vLLM doesn't require real API key
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        model_kwargs={"extra_body": extra_body} if extra_body else {}
    )
    
    return llm


def create_vlm_client(
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 300,
    max_retries: int = 2,
    max_tokens: int = 4096
) -> ChatOpenAI:
    """
    Create LangChain ChatOpenAI client connected to VLM server.
    
    Same as create_llm_client but defaults to OPENAI_API_BASE2 for vision models.
    
    Args:
        api_base: Base URL of VLM server. If None, reads OPENAI_API_BASE2 env var.
                  Default: "http://hopper-34:8002"
        model_name: Model name to use. If None, queries server to detect.
        temperature: Sampling temperature (0.0-1.0)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        max_tokens: Maximum tokens to generate (default: 4096)
    
    Returns:
        Configured ChatOpenAI client for vision tasks
    
    Environment Variables:
        OPENAI_API_BASE2: VLM server URL (e.g., "http://hopper-34:8002")
    
    Example:
        >>> from tools import create_vlm_client
        >>> vlm = create_vlm_client()
        >>> # Use for vision tasks
    """
    if api_base is None:
        api_base = os.environ.get("OPENAI_API_BASE2", "http://hopper-34:8002")
    
    return create_llm_client(
        api_base=api_base,
        model_name=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens
    )


def get_llm_for_tests(temperature: float = 0.7) -> ChatOpenAI:
    """
    Convenience function for creating LLM client in tests.
    
    This is a simplified wrapper around create_llm_client() that:
    - Uses default server (OPENAI_API_BASE or hopper-34:8001)
    - Auto-discovers model name
    - Sets reasonable test defaults
    
    Args:
        temperature: Sampling temperature for tests (default: 0.7)
    
    Returns:
        Configured ChatOpenAI client
    
    Raises:
        RuntimeError: If server is unreachable (test should fail)
    
    Example:
        >>> @pytest.fixture
        ... def real_llm():
        ...     return get_llm_for_tests()
    """
    return create_llm_client(temperature=temperature)


def check_llm_availability(api_base: Optional[str] = None, verbose: bool = True) -> bool:
    """
    Check if vLLM server is available and responding.
    
    Useful for:
    - Skipping tests when server is unavailable
    - Health checks before starting services
    - Debugging connection issues
    
    Args:
        api_base: Base URL of vLLM server. If None, uses OPENAI_API_BASE env var.
        verbose: If True, print status messages
    
    Returns:
        True if server is available, False otherwise
    
    Example:
        >>> if check_llm_availability():
        ...     print("Server is ready!")
        ... else:
        ...     print("Server is down, skipping test")
        
        >>> @pytest.fixture
        ... def llm_if_available():
        ...     if not check_llm_availability(verbose=False):
        ...         pytest.skip("vLLM server not available")
        ...     return get_llm_for_tests()
    """
    if api_base is None:
        api_base = os.environ.get("OPENAI_API_BASE", "http://hopper-34:8001")
    
    try:
        model_name = get_available_model(api_base)
        if verbose:
            print(f"✓ vLLM server available at {api_base}")
            print(f"  Model: {model_name}")
        return True
    except RuntimeError as e:
        if verbose:
            print(f"✗ vLLM server not available at {api_base}")
            print(f"  Error: {e}")
        return False


# Re-export for convenience
__all__ = [
    'get_available_model',
    'create_llm_client',
    'create_vlm_client',
    'get_llm_for_tests',
    'check_llm_availability'
]

