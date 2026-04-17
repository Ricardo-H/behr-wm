#!/usr/bin/env python3
"""
TextWorld Pivot-GRPO: BehR-Only Reward Function

=============================================================================
核心设计: 纯 Behavioral Fidelity Reward (BehR Only)
=============================================================================

R_total = behavior_weight × R_behavior

- R_behavior: exp(-α × |mean_log_prob_pred - mean_log_prob_real|)
  * pred: Agent 在 WM 生成的状态下的动作概率
  * real: Agent 在真实环境状态下的动作概率
  * α = behavior_scale_coef (默认 1.0，与 0208 Exp3 对齐)

- 无 R_facts (TextWorld 无 WebShop 式结构化事实)
- 无 R_length_penalty

=============================================================================
与 WebShop reward_function.py 的差异
=============================================================================

1. FormatValidator: TextWorld 版本（检查 TextWorld 协议特征）
2. _build_prompt_with_action: 支持 TextWorld Agent 的对话格式
3. 无 _compute_facts_reward 调用
4. System prompt 适配 TextWorld Agent

=============================================================================
verl 接口
=============================================================================
- compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
- 返回 {"score": float, ...}
"""

import os
import re
import math
import threading
import requests
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from concurrent.futures import Future as _Future

# =============================================================================
# 全局配置
# =============================================================================
DEFAULT_CONFIG = {
    "behavior_scale_coef": 1.0,
    "format_penalty": -1.0,
    "facts_weight": 0.0,         # TextWorld BehR-only
    "behavior_weight": 1.0,
    "judge_model_path": "Qwen/Qwen3-8B",
    "judge_api_url": "http://localhost:8000",
    "api_timeout": 300.0,
}

_http_judge_agent = None
_format_validator = None


@dataclass
class PivotGRPOConfig:
    """TextWorld Pivot-GRPO 配置"""
    reward_mode: str = "exponential"
    behavior_scale_coef: float = 1.0
    format_penalty: float = -1.0
    facts_weight: float = 0.0
    behavior_weight: float = 1.0
    judge_model_path: str = "Qwen/Qwen3-8B"
    judge_api_url: str = "http://localhost:8000"
    api_timeout: float = 300.0
    use_http_judge: bool = True
    use_full_judge: bool = True
    length_penalty_weight: float = 0.0


# =============================================================================
# TextWorld Format Validator
# =============================================================================

class TextWorldFormatValidator:
    """
    验证 World Model 生成的状态是否符合 TextWorld 环境格式
    
    TextWorld 状态特征:
    - 包含房间名称 (e.g., -= Bedroom =-)
    - 包含环境描述 (You are in ...)
    - 包含 AVAILABLE ACTIONS
    - 包含 > 提示符
    - 包含分数标记 (e.g., -= Room =-0/4)
    """
    
    def validate(self, response: str) -> Tuple[bool, str]:
        if not response or not response.strip():
            return False, "Empty response"
        
        response = response.strip()
        
        if len(response) < 5:
            return False, "Response too short"
        
        if len(response) > 30000:
            return False, "Response too long"
        
        # TextWorld 协议特征
        has_room = bool(re.search(r'-=\s*\w+.*=-', response))
        has_prompt = ">" in response
        has_actions = "AVAILABLE ACTIONS:" in response or "available actions" in response.lower()
        has_description = any(kw in response.lower() for kw in [
            "you are in", "you find yourself", "you arrive", "you're now",
            "you open", "you take", "you go", "you put", "you unlock",
            "you scored", "the end"
        ])
        has_score = bool(re.search(r'-=.*=-\d+/\d+', response))
        
        if has_room:
            return True, "Contains TextWorld room marker"
        if has_actions:
            return True, "Contains action list"
        if has_description and has_prompt:
            return True, "Contains description and prompt"
        if has_score:
            return True, "Contains score marker"
        if has_description:
            return True, "Contains TextWorld description"
        
        # TextWorld 状态也可能只是简短的反馈
        if len(response) > 20 and has_prompt:
            return True, "Contains prompt marker with content"
        
        return False, "Missing TextWorld protocol features"


# =============================================================================
# Real-State LogProb 缓存
# =============================================================================

class _RealLogProbCache:
    """线程安全的 real_state logprob 缓存"""
    
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get_or_compute(self, key, compute_fn):
        created = False
        with self._lock:
            if key in self._store:
                self.hits += 1
                fut = self._store[key]
            else:
                self.misses += 1
                fut = _Future()
                self._store[key] = fut
                created = True
        
        if created:
            try:
                result = compute_fn()
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
                with self._lock:
                    self._store.pop(key, None)
                raise
        
        return fut.result(timeout=600)
    
    def clear(self):
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0


# =============================================================================
# HTTP Judge Agent - TextWorld 版本
# =============================================================================

class TextWorldHTTPJudgeAgent:
    """
    TextWorld 版本的 HTTP Judge Agent
    
    与 WebShop 版本的关键差异:
    1. System prompt 使用 TextWorld agent 的提示
    2. WM 格式到 Agent 格式的转换规则不同
    3. 无 "Action: \\n" 前缀（TextWorld 的 action 是自然语言命令）
    """
    
    # TextWorld Agent 系统提示
    # 注意: TextWorld 的 Agent 接收环境观察并输出自然语言命令
    TEXTWORLD_SYSTEM_PROMPT = (
        "You are playing a text adventure game called TextWorld.\n"
        "Each round you will receive an observation describing your surroundings "
        "and a list of available actions.\n"
        "You must choose one of the available actions to proceed.\n"
        "Your response should be a single action command, for example:\n"
        "open chest drawer\ngo east\ntake old key\nput apple on stove\n"
        "Choose the best action to complete your assigned task."
    )
    
    def __init__(self, config: PivotGRPOConfig):
        self.config = config
        self.api_base = config.judge_api_url.rstrip('/')
        self.completions_url = f"{self.api_base}/v1/completions"
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._model_name = None
        self._tokenizer = None
        self._real_logprob_cache = _RealLogProbCache()
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            response = requests.get(f"{self.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    self._model_name = models[0].get("id", self.config.judge_model_path)
                    print(f"[TW-Reward] Connected to vLLM API at {self.api_base}")
                    print(f"[TW-Reward] Model: {self._model_name}, mode={self.config.reward_mode}")
                    
                    try:
                        from transformers import AutoTokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            self.config.judge_model_path,
                            trust_remote_code=True
                        )
                        print(f"[TW-Reward] Tokenizer loaded: {self.config.judge_model_path}")
                    except Exception as e:
                        print(f"[TW-Reward] Warning: Could not load tokenizer: {e}")
                        self._tokenizer = None
                    
                    self._initialized = True
                else:
                    raise RuntimeError("No models available on vLLM server")
            else:
                raise RuntimeError(f"vLLM API error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.api_base}. "
                "Please start the judge server first."
            )
    
    def _build_prompt_with_action(
        self,
        state: str,
        action: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, int]:
        """
        构建 Agent 视角的 prompt + action
        
        TextWorld WM 格式:
        - system: 游戏初始化 (ASCII banner + 任务 + 初始环境 + AVAILABLE ACTIONS)
        - user: 玩家动作 (e.g., "open antique trunk")
        - assistant: 环境响应 (新状态 + AVAILABLE ACTIONS)
        
        需要转换为 Agent 视角:
        - system: TextWorld Agent 系统提示
        - user: 环境观察 (WM 的 system/assistant 内容)
        - assistant: 玩家动作 (WM 的 user 内容)
        """
        
        # 转换 WM 格式的 history 到 Agent 格式
        converted_history = []
        if history and len(history) > 0:
            # TextWorld WM 格式:
            # messages[0] = system (游戏初始化，包含环境状态)
            # messages[1] = user (第一个动作)
            # messages[2] = assistant (环境响应)
            # ...
            
            # 提取 system 中的初始环境观察
            initial_observation = None
            for msg in history:
                if msg.get("role") == "system":
                    initial_observation = msg.get("content", "")
                    break
            
            # 构建 Agent 对话: 将 WM 的 user(action) 和 assistant(state) 反转
            if initial_observation:
                # 第一个 user turn: 初始环境观察
                converted_history.append({"role": "user", "content": initial_observation})
            
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    continue  # 已处理
                elif role == "user":
                    # WM 的 user (action) -> Agent 的 assistant
                    converted_history.append({"role": "assistant", "content": content})
                elif role == "assistant":
                    # WM 的 assistant (state) -> Agent 的 user
                    converted_history.append({"role": "user", "content": content})
        
        if self._tokenizer is not None and hasattr(self._tokenizer, 'apply_chat_template'):
            messages = []
            
            # System prompt
            sys_prompt = system_prompt or self.TEXTWORLD_SYSTEM_PROMPT
            messages.append({"role": "system", "content": sys_prompt})
            
            # Converted history
            if converted_history:
                for msg in converted_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})
            
            # Current state (WM 预测的或真实的环境状态)
            messages.append({"role": "user", "content": state})
            
            try:
                prompt_without_action = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                prompt_without_action = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            if prompt_without_action and not prompt_without_action.endswith('\n'):
                prompt_without_action = prompt_without_action + '\n'
            
            # TextWorld action 是自然语言命令，不需要 "Action: \n" 前缀
            full_prompt = prompt_without_action + action
            action_start_char_pos = len(prompt_without_action)
            
        else:
            # Fallback
            sys_prompt = system_prompt or self.TEXTWORLD_SYSTEM_PROMPT
            parts = [f"System: {sys_prompt}"]
            
            if converted_history:
                for msg in converted_history:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    if content:
                        parts.append(f"{role}: {content}")
            
            parts.append(f"User: {state}")
            prompt_without_action = "\n\n".join(parts) + "\n\nAssistant: "
            
            full_prompt = prompt_without_action + action
            action_start_char_pos = len(prompt_without_action)
        
        return full_prompt, action_start_char_pos
    
    def compute_action_log_prob(
        self,
        state: str,
        action: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[float, int]:
        """
        计算 log π(action | state) via vLLM API
        
        Returns:
            (sum_log_prob, token_count)
        """
        self.initialize()
        
        full_prompt, action_start_char_pos = self._build_prompt_with_action(
            state, action, system_prompt, history
        )
        
        payload = {
            "model": self._model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        
        try:
            response = requests.post(
                self.completions_url,
                headers=self.headers,
                json=payload,
                timeout=self.config.api_timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"vLLM API error: {response.status_code} - {response.text}")
            
            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                return -20.0, 1
            
            logprobs_data = choices[0].get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", [])
            text_offset = logprobs_data.get("text_offset", [])
            
            if not token_logprobs or not text_offset:
                return -20.0, 1
            
            # 定位 action tokens
            action_token_start_idx = -1
            for i, offset in enumerate(text_offset):
                if offset >= action_start_char_pos:
                    action_token_start_idx = i
                    break
                elif i + 1 < len(text_offset) and text_offset[i + 1] > action_start_char_pos:
                    action_token_start_idx = i
                    break
            
            if action_token_start_idx < 0 and text_offset:
                if text_offset[-1] < action_start_char_pos:
                    action_token_start_idx = len(text_offset) - 1
            
            if action_token_start_idx < 0:
                print(f"[TW-Reward] CRITICAL: Failed to locate action start!")
                return -20.0, 1
            
            action_logprobs = []
            for i in range(action_token_start_idx, len(token_logprobs)):
                lp = token_logprobs[i]
                if lp is not None:
                    action_logprobs.append(lp)
            
            if not action_logprobs:
                return -20.0, 1
            
            return sum(action_logprobs), len(action_logprobs)
            
        except requests.exceptions.Timeout:
            print(f"[TW-Reward] API timeout ({self.config.api_timeout}s)")
            return -20.0, 1
        except Exception as e:
            print(f"[TW-Reward] API error: {e}")
            return -20.0, 1
    
    def compute_behavioral_fidelity_reward(
        self,
        predicted_state: str,
        real_state: str,
        expert_action: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        计算 Behavioral Fidelity Reward
        
        R = exp(-α × |mean_log_prob_pred - mean_log_prob_real|)
        """
        result = {
            "score": 0.0,
            "mean_log_prob_pred": 0.0,
            "mean_log_prob_real": 0.0,
            "mean_diff": 0.0,
            "token_count_pred": 0,
            "token_count_real": 0,
            "api_failed": False,
        }
        
        try:
            # 计算 pred state logprob
            sum_lp_pred, n_pred = self.compute_action_log_prob(
                state=predicted_state,
                action=expert_action,
                system_prompt=system_prompt,
                history=history,
            )
            
            # 计算 real state logprob (使用缓存)
            cache_key = hash((real_state, expert_action, str(history)))
            
            def _compute_real():
                return self.compute_action_log_prob(
                    state=real_state,
                    action=expert_action,
                    system_prompt=system_prompt,
                    history=history,
                )
            
            sum_lp_real, n_real = self._real_logprob_cache.get_or_compute(
                cache_key, _compute_real
            )
            
            # Mean Log Prob
            mean_lp_pred = sum_lp_pred / max(n_pred, 1)
            mean_lp_real = sum_lp_real / max(n_real, 1)
            diff = mean_lp_pred - mean_lp_real
            
            # Compute reward
            alpha = self.config.behavior_scale_coef
            mode = self.config.reward_mode
            
            if mode == "exponential":
                score = math.exp(-alpha * abs(diff))
            elif mode == "cauchy":
                score = 1.0 / (1.0 + alpha * abs(diff))
            elif mode == "linear":
                score = max(0.0, 1.0 - alpha * abs(diff))
            elif mode == "negative_l1":
                score = -abs(diff)
            elif mode == "negative_l2":
                score = -(diff ** 2)
            else:
                score = math.exp(-alpha * abs(diff))
            
            result["score"] = score
            result["mean_log_prob_pred"] = mean_lp_pred
            result["mean_log_prob_real"] = mean_lp_real
            result["mean_diff"] = diff
            result["token_count_pred"] = n_pred
            result["token_count_real"] = n_real
            
        except Exception as e:
            print(f"[TW-Reward] BehR computation failed: {e}")
            result["api_failed"] = True
            result["failure_reason"] = str(e)
        
        return result


# =============================================================================
# 单例管理
# =============================================================================

def get_format_validator():
    global _format_validator
    if _format_validator is None:
        _format_validator = TextWorldFormatValidator()
    return _format_validator


def get_http_judge_agent(config: PivotGRPOConfig):
    global _http_judge_agent
    if _http_judge_agent is None:
        _http_judge_agent = TextWorldHTTPJudgeAgent(config)
    return _http_judge_agent


def _compute_similarity_score(pred: str, real: str) -> float:
    """简单字符串相似度 fallback"""
    if not pred or not real:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, pred[:2000], real[:2000]).ratio()


def _similarity_to_behavior_reward(similarity: float, reward_mode: str) -> float:
    """将相似度转换为与 reward_mode 一致的行为奖励"""
    if reward_mode in ("exponential", "cauchy", "linear"):
        return similarity  # [0, 1]
    elif reward_mode == "negative_l1":
        return -(1.0 - similarity)  # (-inf, 0]
    elif reward_mode == "negative_l2":
        return -((1.0 - similarity) ** 2)
    else:
        return similarity


# =============================================================================
# verl 主入口
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    # === BehR 参数 ===
    reward_mode: str = "exponential",
    behavior_scale_coef: float = 1.0,
    format_penalty: float = -1.0,
    # === 权重 (BehR only) ===
    facts_weight: float = 0.0,
    behavior_weight: float = 1.0,
    length_penalty_weight: float = 0.0,
    # === Judge 参数 ===
    judge_model_path: str = "Qwen/Qwen3-8B",
    use_full_judge: bool = True,
    use_http_judge: bool = True,
    judge_api_url: str = "http://localhost:8000",
    api_timeout: float = 300.0,
    **kwargs
) -> Dict[str, Any]:
    """
    TextWorld Pivot-GRPO: BehR-Only Reward Function
    
    R_total = behavior_weight × R_behavior
    
    Args:
        data_source: 数据源标识 ("textworld_grpo")
        solution_str: WM 生成的预测状态
        ground_truth: 真实环境状态
        extra_info: 额外信息 (expert_action, history, etc.)
    """
    extra_info = extra_info or {}
    
    # 解析 ground_truth
    ground_truth_str = ""
    expert_action = extra_info.get("expert_action", "")
    
    if isinstance(ground_truth, dict):
        ground_truth_str = ground_truth.get("ground_truth", "")
        if not expert_action:
            expert_action = ground_truth.get("expert_action", "")
    elif isinstance(ground_truth, str):
        ground_truth_str = ground_truth
    else:
        ground_truth_str = str(ground_truth) if ground_truth else ""
    
    # 提取 history
    history = extra_info.get("history", None)
    if history is not None:
        import numpy as np
        if isinstance(history, np.ndarray):
            history = history.tolist()
        elif not isinstance(history, list):
            history = list(history)
    
    # 从 prompt 字段获取 history
    if "prompt" in extra_info:
        prompt = extra_info.get("prompt")
        if isinstance(prompt, (list, tuple)) and len(prompt) > 0:
            if history is None:
                history = list(prompt)
    
    config = PivotGRPOConfig(
        reward_mode=reward_mode,
        behavior_scale_coef=behavior_scale_coef,
        format_penalty=format_penalty,
        facts_weight=facts_weight,
        behavior_weight=behavior_weight,
        judge_model_path=judge_model_path,
        use_full_judge=use_full_judge,
        use_http_judge=use_http_judge,
        judge_api_url=judge_api_url,
        api_timeout=api_timeout,
    )
    
    validator = get_format_validator()
    
    if use_full_judge and use_http_judge:
        judge = get_http_judge_agent(config)
    else:
        judge = None
    
    # 初始化结果
    result = {
        "score": 0.0,
        "format_valid": True,
        "format_reason": "",
        "mean_log_prob_pred": 0.0,
        "mean_log_prob_real": 0.0,
        "mean_diff": 0.0,
        "token_count_pred": 0,
        "token_count_real": 0,
        "behavior_reward": 0.0,
        "facts_reward": 0.0,
        "asin_match": 0.0,
        "price_match": 0.0,
        "page_match": 0.0,
        "rating_match": 0.0,
        "fallback_similarity": 0.0,
        "used_fallback": False,
        "pred_length": len(solution_str) if solution_str else 0,
        "real_length": len(ground_truth_str) if ground_truth_str else 0,
        "length_ratio": 0.0,
        "length_penalty": 0.0,
        "length_status": "n/a",
        "length_penalty_weight": 0.0,
        "predicted_state": solution_str[:100] if solution_str else "",
        "real_state": ground_truth_str[:100] if ground_truth_str else "",
        "expert_action": expert_action[:50] if expert_action else "",
        "has_history": history is not None and len(history) > 0,
        "reward_mode": reward_mode,
        "behavior_scale_coef": behavior_scale_coef,
        "behavior_weight": behavior_weight,
        "facts_weight": facts_weight,
    }
    
    # Step 1: Format Validation
    if not solution_str:
        result["score"] = format_penalty
        result["format_valid"] = False
        result["format_reason"] = "Empty response"
        return result
    
    is_valid, reason = validator.validate(solution_str)
    result["format_valid"] = is_valid
    result["format_reason"] = reason
    
    if not is_valid:
        result["score"] = format_penalty
        return result
    
    # Step 2: BehR Reward
    behavior_score = 0.0
    
    if use_full_judge and judge is not None and expert_action and ground_truth_str:
        try:
            fidelity_result = judge.compute_behavioral_fidelity_reward(
                predicted_state=solution_str,
                real_state=ground_truth_str,
                expert_action=expert_action,
                history=history,
            )
            
            behavior_score = fidelity_result["score"]
            
            if fidelity_result.get("api_failed", False):
                similarity = _compute_similarity_score(solution_str, ground_truth_str)
                behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
                result["fallback_similarity"] = similarity
                result["used_fallback"] = True
            
            result["behavior_reward"] = behavior_score
            result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
            result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
            result["mean_diff"] = fidelity_result["mean_diff"]
            result["token_count_pred"] = fidelity_result["token_count_pred"]
            result["token_count_real"] = fidelity_result["token_count_real"]
            
        except Exception as e:
            print(f"[TW-Reward] BehR failed: {e}")
            similarity = _compute_similarity_score(solution_str, ground_truth_str)
            behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
            result["behavior_reward"] = behavior_score
            result["fallback_similarity"] = similarity
            result["used_fallback"] = True
    
    elif ground_truth_str:
        similarity = _compute_similarity_score(solution_str, ground_truth_str)
        behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
        result["behavior_reward"] = behavior_score
        result["fallback_similarity"] = similarity
        result["used_fallback"] = True
    
    # Final score: BehR only
    final_score = behavior_weight * behavior_score
    result["score"] = final_score
    
    return result
