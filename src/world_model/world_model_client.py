"""
World Model Client

HTTP client for interacting with vLLM-deployed world models.
"""

import requests
from typing import List, Dict, Any, Optional


class WorldModelClient:
    """
    World Model HTTP Client
    
    Provides interface for world model inference via vLLM HTTP API.
    """
    
    def __init__(
        self,
        port: int = 8001,
        model_name: str = "llm_world_model",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 120.0
    ):
        """
        Initialize World Model Client
        
        Args:
            port: vLLM server port
            model_name: Model name/identifier
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_base = f"http://localhost:{port}/v1"
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._actual_model_name = None
    
    def initialize(self) -> None:
        """Check if server is available and get actual model name"""
        if self._initialized:
            return
        
        try:
            response = requests.get(f"{self.api_base}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    self._actual_model_name = models[0].get("id", self.model_name)
                    print(f"[WM Client] Connected to vLLM, model: {self._actual_model_name}")
                    self._initialized = True
                else:
                    raise RuntimeError("No models available on WM server")
            else:
                raise RuntimeError(f"WM server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to WM server at {self.api_base}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response from world model
        
        Args:
            messages: Conversation history
            max_tokens: Override default max tokens
            temperature: Override default temperature
        
        Returns:
            Generated text
        """
        self.initialize()
        
        payload = {
            "model": self._actual_model_name or self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": 1.0,
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"WM API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"WM API timeout after {self.timeout}s")
    
    def step(
        self,
        history: List[Dict[str, str]],
        action: str
    ) -> tuple:
        """
        Execute one step in world model
        
        Args:
            history: Current conversation history
            action: Action to execute
        
        Returns:
            (observation, done): Next observation and done flag
        """
        messages = history + [{"role": "user", "content": action}]
        observation = self.generate(messages)
        
        # Check for success marker
        done = False
        if " [SUCCESS]" in observation:
            observation = observation.split(" [SUCCESS]")[0]
            done = True
        
        return observation, done
