"""
ReAct Agent Implementation

Agent that follows the ReAct (Reasoning + Acting) paradigm for interacting
with environments and world models.
"""

import os
import sys
import time
import random
from typing import Optional, List, Dict, Tuple, Any
from openai import OpenAI


class WorldModel:
    """
    World Model client for simulating environment transitions
    
    The world model predicts the next state given the current state and action.
    """
    
    def __init__(
        self,
        wm_messages: List[Dict[str, str]],
        client: OpenAI,
        model_name: str = "llm_world_model"
    ):
        """
        Initialize World Model
        
        Args:
            wm_messages: Initial conversation history for the world model
            client: OpenAI-compatible client
            model_name: Model name/identifier
        """
        self.history = wm_messages
        self.client = client
        self.model_name = model_name

    def llm_generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from world model"""
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=512,
            temperature=0,
            top_p=1,
        )
        return response.choices[0].message.content

    def done(self, observation: str) -> Tuple[str, bool]:
        """Check if episode is done based on observation"""
        if " [SUCCESS]" in observation:
            return observation.split(" [SUCCESS]")[0], True
        else:
            return observation, False

    def step(self, action: str) -> Tuple[str, bool]:
        """
        Execute action in world model and return next state
        
        Args:
            action: Action to execute
        
        Returns:
            (observation, done): Next observation and whether episode is done
        """
        self.history.append({"role": "user", "content": action})
        observation = self.llm_generate(self.history)
        observation, done = self.done(observation)
        self.history.append({"role": "assistant", "content": observation})
        return observation, done


class ReactAgent:
    """
    ReAct Agent for interacting with environments
    
    Follows the ReAct paradigm: Reasoning + Acting
    - Generates thoughts based on observations
    - Produces actions based on reasoning
    """
    
    # Default system prompt for WebShop agent
    WEBSHOP_SYSTEM_PROMPT = (
        "You are web shopping.\n"
        "I will give you instructions about what to do.\n"
        "You have to follow the instructions.\n"
        "Every round I will give you an observation and a list of available actions, "
        "you have to respond an action based on the state and instruction.\n"
        "You can use search action if search is available.\n"
        "You can click one of the buttons in clickables.\n"
        "An action should be of the following structure:\n"
        "search[keywords]\n"
        "click[value]\n"
        "If the action is not valid, perform nothing.\n"
        "Keywords in search are up to you, but the value in click must be a value in the list of available actions.\n"
        "Remember that your keywords in search should be carefully designed.\n"
        "Your response should use the following format:\n\n"
        "Action: \n"
        "click[something]"
    )
    
    def __init__(
        self,
        agent_messages: List[Dict[str, str]],
        agent_model_name: str,
        api_key: str,
        api_base_url: str = ""
    ):
        """
        Initialize ReAct Agent
        
        Args:
            agent_messages: Initial conversation history
            agent_model_name: Model name for the agent
            api_key: API key ("EMPTY" for vLLM, or OpenAI key)
            api_base_url: API base URL (for OpenAI/vLLM mode)
        """
        self.history = agent_messages
        self.agent_model_name = agent_model_name
        self.api_key = api_key
        self.api_base_url = api_base_url
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        self.model_name = self.agent_model_name

    def llm_generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from agent model"""
        # Different models may have different parameter requirements
        if "gpt-5" in self.model_name or "claude" in self.model_name:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
        else:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=4096,
                temperature=1,
                top_p=1,
            )
        return response.choices[0].message.content

    def parse_action(self, text: str) -> Tuple[str, str]:
        """
        Parse ReAct format response to extract thought and action
        
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        
        Args:
            text: Raw response text
        
        Returns:
            (thought, action): Parsed thought and action
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        
        if len(_split) == 0:
            _thought, _action = text, ""
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
            else:
                _thought, _action = _split[0], ""
            invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        
        return thought, action

    def react(self, observation: str) -> Tuple[str, str, str]:
        """
        Perform one ReAct step: observe -> think -> act
        
        Args:
            observation: Current observation from environment
        
        Returns:
            (raw_response, thought, action): Full response, extracted thought, and action
        """
        self.history.append({"role": "user", "content": observation})
        
        max_retries = 50
        for i in range(max_retries):
            try:
                react = self.llm_generate(self.history)
                if react is not None:
                    break
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                print(f"Request failed, retrying ({i+1}/{max_retries})... Error: {e}")
                # Exponential backoff
                sleep_time = min(2 * (2 ** i), 60) + random.uniform(0, 1)
                time.sleep(sleep_time)
        
        self.history.append({"role": "assistant", "content": react})
        thought, action = self.parse_action(react)
        return react, thought, action
