"""
vLLM HTTP Client for local model deployment

Provides a unified interface for vLLM-deployed models.
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VLLMConfig:
    """vLLM client configuration"""
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "llm_model"
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: float = 120.0


class VLLMClient:
    """
    vLLM HTTP Client for local model inference
    
    Provides OpenAI-compatible API interface for vLLM-deployed models.
    """
    
    def __init__(self, config: Optional[VLLMConfig] = None, **kwargs):
        """
        Initialize vLLM client
        
        Args:
            config: VLLMConfig object, or pass kwargs directly
            **kwargs: Override config parameters
        """
        if config is None:
            config = VLLMConfig(**kwargs)
        else:
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.completions_url = f"{self.base_url}/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._actual_model_name = None
    
    def initialize(self) -> None:
        """Check if vLLM server is available and get model name"""
        if self._initialized:
            return
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    self._actual_model_name = models[0].get("id", self.config.model_name)
                    print(f"[vLLM Client] Connected to {self.base_url}")
                    print(f"[vLLM Client] Using model: {self._actual_model_name}")
                    self._initialized = True
                else:
                    raise RuntimeError("No models available on vLLM server")
            else:
                raise RuntimeError(f"vLLM server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                "Please start the server first."
            )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate response from model
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_tokens: Max tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            top_p: Top-p sampling (default from config)
            **kwargs: Additional parameters passed to API
        
        Returns:
            Generated text content
        """
        self.initialize()
        
        payload = {
            "model": self._actual_model_name or self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.completions_url,
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"vLLM API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"vLLM API timeout after {self.config.timeout}s")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get full chat completion response (OpenAI-compatible format)
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
        
        Returns:
            Full API response dict
        """
        self.initialize()
        
        payload = {
            "model": self._actual_model_name or self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "top_p": kwargs.pop("top_p", self.config.top_p),
            **kwargs
        }
        
        response = requests.post(
            self.completions_url,
            headers=self.headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"vLLM API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_log_probs(
        self,
        prompt: str,
        max_tokens: int = 0,
        echo: bool = True,
        logprobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Get log probabilities for a prompt (completions API)
        
        Args:
            prompt: Input prompt text
            max_tokens: Max tokens to generate (0 for no generation)
            echo: Whether to include prompt tokens in response
            logprobs: Number of logprobs to return per token
        
        Returns:
            API response with logprobs data
        """
        self.initialize()
        
        completions_url = f"{self.base_url}/completions"
        
        payload = {
            "model": self._actual_model_name or self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": logprobs,
            "echo": echo,
        }
        
        response = requests.post(
            completions_url,
            headers=self.headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"vLLM API error: {response.status_code} - {response.text}")
        
        return response.json()


# Convenience function
def get_vllm_client(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "llm_model",
    **kwargs
) -> VLLMClient:
    """
    Create a vLLM client instance
    
    Args:
        base_url: vLLM server URL
        model_name: Model name/identifier
        **kwargs: Additional VLLMConfig parameters
    
    Returns:
        VLLMClient instance
    """
    config = VLLMConfig(base_url=base_url, model_name=model_name, **kwargs)
    return VLLMClient(config)
