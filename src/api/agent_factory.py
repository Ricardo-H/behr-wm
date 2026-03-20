"""
Agent Factory - Unified agent client creation

Provides a unified interface for creating agent clients with different backends:
- OpenAI API
- vLLM local deployment
"""

from typing import Optional, Union, Any
from openai import OpenAI


def get_agent_client(
    mode: str,
    model: str = "gpt-4o-mini-20240718",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> OpenAI:
    """
    Unified Agent client factory function
    
    Args:
        mode: Client mode - "openai" | "vllm"
        model: Model name/identifier
        api_key: API key (for OpenAI mode)
        base_url: API base URL (for OpenAI/vLLM mode)
        **kwargs: Additional parameters passed to client
    
    Returns:
        OpenAI-compatible client instance
    
    Examples:
        # OpenAI mode
        client = get_agent_client("openai", api_key="sk-...", model="gpt-4o")
        
        # vLLM mode
        client = get_agent_client("vllm", base_url="http://localhost:8000/v1")
    """
    mode = mode.lower()
    
    if mode == "openai":
        if api_key is None:
            raise ValueError("api_key is required for OpenAI mode")
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    elif mode == "vllm":
        # vLLM uses OpenAI-compatible API
        return OpenAI(
            api_key=api_key or "EMPTY",  # vLLM doesn't require a real key
            base_url=base_url or "http://localhost:8000/v1",
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Supported modes: 'openai', 'vllm'"
        )


def detect_api_mode(api_key: str) -> str:
    """
    Auto-detect API mode from api_key string
    
    Args:
        api_key: API key or mode hint
    
    Returns:
        Detected mode: "openai" | "vllm"
    """
    api_key_lower = api_key.lower()
    
    if api_key_lower in ("empty", "local", "vllm"):
        return "vllm"
    elif api_key.startswith("sk-"):
        return "openai"
    else:
        # Default to vLLM for unknown keys
        return "vllm"


def create_chat_completion(
    client: OpenAI,
    messages: list,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    **kwargs
) -> str:
    """
    Helper function for chat completion
    
    Args:
        client: OpenAI-compatible client
        messages: Message list
        model: Model name
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        **kwargs: Additional parameters
    
    Returns:
        Generated text content
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )
    return response.choices[0].message.content
