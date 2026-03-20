#!/usr/bin/env python3
"""
Pivot-GRPO: Behavior Consistency Reward Function (BehR)

=============================================================================
Core Philosophy (Behavior Consistency)
=============================================================================

A World Model (WM) is functionally equivalent to the real environment if an
agent cannot distinguish between them through its actions.

"As long as the agent can't tell the difference, the WM is real."

=============================================================================
Reward
=============================================================================

R = R_behavior = exp(-coef * |mean_log_prob_pred - mean_log_prob_real|)

  - Range: (0, 1], higher is better
  - mean_log_prob = (1/N) * sum log pi(token_i | context)
  - Using Mean (not Sum) eliminates length bias

  exponential mode advantages:
  - Range (0, 1], avoids magnitude mismatch
  - behavior_scale_coef controls sensitivity to probability differences

=============================================================================
verl 接口说明
=============================================================================
- compute_score 函数被 verl 的 reward manager 调用
- 参数: data_source, solution_str, ground_truth, extra_info
- 返回: dict (包含 "score" 字段和详细诊断信息)
"""

import os
import re
import math
import asyncio
import requests
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# =============================================================================
# 全局配置
# =============================================================================
DEFAULT_CONFIG = {
    # === Behavior Consistency 参数 ===
    # 使用 Mean Log Prob 消除长度偏置
    "behavior_scale_coef": 1.0,  # exponential 模式下的缩放系数
    
    # === Reality Anchor 参数 ===
    "format_penalty": -1.0,      # 格式错误惩罚（与正常奖励 [0, 1] 同量级）
    
    # === 物理事实奖励参数 (disabled by default for BehR-only) ===
    "facts_weight": 0.0,         # 物理事实奖励权重 (set >0 for ablation)
    "behavior_weight": 1.0,      # 行为一致性奖励权重
    
    # === 裁判模型配置 ===
    "reference_agent_model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    
    # === vLLM HTTP API 配置 ===
    "reference_agent_api_url": "http://localhost:8000",
    "api_timeout": 300.0,  # 增加到 300 秒，防止长 context 并发时超时
}

# 全局单例，延迟初始化
_http_reference_agent = None
_format_validator = None


@dataclass
class PivotGRPOConfig:
    """Pivot-GRPO 奖励函数配置"""
    
    # === Behavior Consistency 参数 ===
    reward_mode: str = "exponential"  # 奖励模式: "negative_l1", "negative_l2", "exponential"
    behavior_scale_coef: float = 1.0  # exponential 模式下的缩放系数
    # 使用 Mean Log Prob 消除长度偏置
    
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0  # 与正常奖励 [0, 1] 同量级
    
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5
    behavior_weight: float = 1.0
    
    # === 裁判模型配置 ===
    reference_agent_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # === vLLM HTTP API 配置 ===
    reference_agent_api_url: str = "http://localhost:8000"
    api_timeout: float = 300.0  # 增加到 300 秒，防止长 context 并发时超时
    
    # === 开关 ===
    use_http_reference_agent: bool = True
    use_full_reference_agent: bool = True


# =============================================================================
# Format Validator - Reality Anchor
# =============================================================================

class FormatValidator:
    """
    验证 World Model 生成的状态是否符合 WebShop 环境格式
    
    这是 "Reality Anchor"，用于防止 Agent Hacking：
    - WM 可能学会生成让 Agent 神经网络产生特定响应的对抗样本
    - 这些样本可能无法迁移到真实环境
    - 格式验证确保生成内容至少遵守基本协议
    """
    
    def __init__(self):
        pass
    
    def validate(self, response: str) -> Tuple[bool, str]:
        """
        验证响应格式是否合法
        
        Returns:
            (is_valid, reason)
        """
        if not response or not response.strip():
            return False, "Empty response"
        
        response = response.strip()
        
        # 基本长度检查
        if len(response) < 10:
            return False, "Response too short"
        
        if len(response) > 20000:
            return False, "Response too long"
        
        # [REMOVED] 不再检查 ``` 标记
        # WebShop 的 State 可能包含 HTML/CSS 符号，直接禁止 ``` 可能误伤
        # 改为仅依赖 WebShop 协议关键字进行验证
        
        # 必须包含 WebShop 协议特征
        has_sep = "[SEP]" in response
        has_instruction = "Instruction:" in response
        has_actions = "admissible actions" in response.lower() or "Your admissible actions" in response
        has_shopping_complete = ("Thank you" in response and "shopping" in response.lower()) or \
                                "Your score" in response
        has_asin = bool(re.search(r'B[A-Z0-9]{9}', response))
        
        if has_sep:
            return True, "Contains WebShop [SEP] separator"
        if has_instruction:
            return True, "Contains Instruction marker"
        if has_actions:
            return True, "Contains action list"
        if has_shopping_complete:
            return True, "Shopping completion page"
        if has_asin:
            return True, "Contains product ASIN"
        
        return False, "Missing WebShop protocol features ([SEP], Instruction:, ASIN, or action list)"


# =============================================================================
# HTTP Reference Agent - 3+1 架构
# =============================================================================

class HTTPReferenceAgent:
    """
    通过 HTTP API 调用 vLLM 部署的裁判模型
    
    这是 "3+1" 架构的核心组件：
    - 3 张卡用于 GRPO 训练
    - 1 张卡独立部署 vLLM 推理服务
    """
    
    # WebShop Agent 的默认系统提示
    WEBSHOP_SYSTEM_PROMPT = ("You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nAction: \nclick[something]")
    
    def __init__(self, config: PivotGRPOConfig):
        self.config = config
        self.api_base = config.reference_agent_api_url.rstrip('/')
        self.completions_url = f"{self.api_base}/v1/completions"
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._model_name = None
        self._tokenizer = None
    
    def initialize(self):
        """检查 API 服务是否可用，并加载 tokenizer"""
        if self._initialized:
            return
        
        try:
            response = requests.get(
                f"{self.api_base}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    self._model_name = models[0].get("id", self.config.reference_agent_model_path)
                    print(f"[Reward] Connected to vLLM API at {self.api_base}")
                    print(f"[Reward] Using model: {self._model_name}, mode={self.config.reward_mode}, using Mean Log Prob")
                    
                    # 加载 tokenizer 用于构建正确的 chat template
                    tokenizer_model = self.config.reference_agent_model_path
                    try:
                        from transformers import AutoTokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_model,
                            trust_remote_code=True
                        )
                        print(f"[Reward] Tokenizer loaded from: {tokenizer_model}")
                    except Exception as e:
                        print(f"[Reward] Warning: Could not load tokenizer: {e}")
                        self._tokenizer = None
                    
                    self._initialized = True
                else:
                    raise RuntimeError("No models available on vLLM server")
            else:
                raise RuntimeError(f"vLLM API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.api_base}. "
                "Please start the reference agent server first: bash start_reference_agent_server.sh"
            )
    
    def _build_prompt_with_action(
        self, 
        state: str, 
        action: str, 
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        instruction: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        构建完整的 prompt（包含 action），并返回 action 在 prompt 中的起始字符位置
        
        [CRITICAL FIX] 现在支持历史对话上下文：
        - System: WebShop Agent 的系统提示
        - History: 历史对话轮次 (可选)
        - User: 当前环境状态/观察（包含 Instruction）
        - Assistant: Agent 采取的动作（action）
        
        Args:
            state: 当前环境状态/观察
            action: Agent 采取的动作
            system_prompt: 系统提示 (可选，默认使用 WEBSHOP_SYSTEM_PROMPT)
            history: 历史对话列表，格式 [{"role": "user/assistant", "content": "..."}]
            instruction: WebShop 任务指令 (可选，会被整合到 user content 中)
        
        Returns:
            (full_prompt, action_start_char_pos)
        """
        if self._tokenizer is not None and hasattr(self._tokenizer, 'apply_chat_template'):
            # 使用 chat template
            messages = []
            
            # 1. System prompt
            sys_prompt = system_prompt or self.WEBSHOP_SYSTEM_PROMPT
            messages.append({"role": "system", "content": sys_prompt})
            
            # 2. History (如果有)
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})
            
            # 3. Current state (包含 Instruction)
            # [CRITICAL FIX] 检查 state 中是否已包含 Instruction，避免重复注入
            # WebShop 的 observation 经常自带 "Instruction: [SEP] ..."
            state_has_instruction = "Instruction:" in state or "Instruction [SEP]" in state
            
            if instruction and not state_has_instruction:
                # state 中没有 Instruction，需要添加
                user_content = f"Instruction: {instruction}\n\n{state}"
            else:
                # state 中已有 Instruction，或者没有提供 instruction
                user_content = state
            messages.append({"role": "user", "content": user_content})
            
            prompt_without_action = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # [CRITICAL] 确保 prompt 以换行符结尾，防止 Token 边界合并问题
            if prompt_without_action and not prompt_without_action.endswith('\n'):
                prompt_without_action = prompt_without_action + '\n'
            
            full_prompt = prompt_without_action + action
            action_start_char_pos = len(prompt_without_action)
            
        else:
            # Fallback: 简单拼接格式
            sys_prompt = system_prompt or ""
            parts = []
            
            if sys_prompt:
                parts.append(f"System: {sys_prompt}")
            
            # History
            if history:
                for msg in history:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    if content:
                        parts.append(f"{role}: {content}")
            
            # Current state with instruction
            # [CRITICAL FIX] 检查 state 中是否已包含 Instruction，避免重复注入
            state_has_instruction = "Instruction:" in state or "Instruction [SEP]" in state
            
            if instruction and not state_has_instruction:
                parts.append(f"User: Instruction: {instruction}\n\n{state}")
            else:
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
        instruction: Optional[str] = None
    ) -> Tuple[float, int]:
        """
        通过 vLLM API 计算 log π(action | state)
        
        Args:
            state: 当前环境状态
            action: Agent 动作
            system_prompt: 系统提示
            history: 历史对话 [{"role": "user/assistant", "content": "..."}]
            instruction: WebShop 任务指令
        
        Returns:
            (sum_log_prob, token_count): 用于在外部计算 Mean Log Prob
        """
        self.initialize()
        
        full_prompt, action_start_char_pos = self._build_prompt_with_action(
            state, action, system_prompt, history, instruction
        )
        
        payload = {
            "model": self._model_name,
            "prompt": full_prompt,
            "max_tokens": 0,  # 不生成任何新 token，只计算已有 token 的 logprob
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
                return -20.0, 1  # 避免除以零
            
            logprobs_data = choices[0].get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", [])
            text_offset = logprobs_data.get("text_offset", [])
            
            if not token_logprobs or not text_offset:
                return -20.0, 1  # 避免除以零
            
            # [CRITICAL FIX] 使用 text_offset 找到 action 段的起始 token index
            # 修复 Token 边界合并问题：
            # 如果 Tokenizer 将 Prompt 的最后一个字符和 Action 的第一个字符合并成了一个 Token
            # （例如 "Assistant:" + " Search" -> ": Search"），
            # 该 Token 的 offset 会小于 action_start_char_pos。
            # 
            # 解决方案：找到第一个 offset >= action_start_char_pos 的 Token，
            # 但如果该 Token 的实际内容横跨了 action_start_char_pos，也应该包含它。
            # 因此，我们还需要检查前一个 Token 是否跨越了边界。
            action_token_start_idx = -1  # 使用 -1 作为哨兵值，表示未找到
            
            for i, offset in enumerate(text_offset):
                if offset >= action_start_char_pos:
                    # 找到第一个 offset >= action_start_char_pos 的 Token
                    action_token_start_idx = i
                    break
                elif i + 1 < len(text_offset) and text_offset[i + 1] > action_start_char_pos:
                    # 当前 Token 的 offset < action_start_char_pos，
                    # 但下一个 Token 的 offset > action_start_char_pos，
                    # 说明当前 Token 横跨了边界（Token 边界合并）
                    action_token_start_idx = i
                    break
            
            # 如果循环完成且未找到，检查最后一个 Token
            if action_token_start_idx < 0 and text_offset:
                # 如果最后一个 Token 的 offset 也小于 action_start_char_pos，
                # 说明整个序列都在 action_start_char_pos 之前，这是异常情况
                if text_offset[-1] < action_start_char_pos:
                    # 最后一个 token 可能横跨了边界（终止位置在 action_start_char_pos 之后）
                    action_token_start_idx = len(text_offset) - 1
            
            # [CRITICAL] 检查是否成功找到 action 起始位置
            if action_token_start_idx < 0:
                # 这是一个严重错误，可能的原因：
                # 1. Token 边界合并导致所有 offset 都小于 action_start_char_pos
                # 2. text_offset 数据异常
                # 3. action 为空字符串
                print(f"[Reward] CRITICAL: Failed to locate action start position!")
                print(f"  action_start_char_pos={action_start_char_pos}")
                print(f"  max_text_offset={max(text_offset) if text_offset else 'N/A'}")
                print(f"  action={action[:50]}...")
                # 返回一个明显的惩罚值，而不是静默返回错误结果
                raise RuntimeError(
                    f"Failed to locate action tokens. action_start_char_pos={action_start_char_pos}, "
                    f"max_offset={max(text_offset) if text_offset else 'N/A'}"
                )
            
            # 只累加 action 段 token 的 logprob
            action_logprobs = []
            for i in range(action_token_start_idx, len(token_logprobs)):
                lp = token_logprobs[i]
                if lp is not None:
                    action_logprobs.append(lp)
            
            if not action_logprobs:
                # Action 部分没有有效的 logprob，可能是 action 为空或全是特殊 token
                print(f"[Reward] Warning: No valid logprobs for action tokens")
                return -20.0, 1  # 避免除以零
            
            total_log_prob = sum(action_logprobs)
            token_count = len(action_logprobs)
            
            return total_log_prob, token_count
            
        except requests.exceptions.Timeout:
            # [CRITICAL] 抛出异常，不要返回默认值！让上层回退到 similarity_score
            print(f"[Reward] API timeout ({self.config.api_timeout}s) for prompt len {len(full_prompt)}, state len {len(state)}")
            raise RuntimeError(f"vLLM API Timeout after {self.config.api_timeout}s") 
        except Exception as e:
            # [CRITICAL] 抛出异常，不要静默返回默认值
            print(f"[Reward] API error: {e}")
            raise e
    
    def compute_behavior_consistency_reward(
        self,
        predicted_state: str,
        real_state: str,
        expert_action: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        instruction: Optional[str] = None
    ) -> Dict[str, float]:
        """
        计算行为一致性奖励 (Behavior Consistency Reward)
        
        使用 Mean Log Prob 消除长度偏置（Length Bias）:
        - R ≈ (1/N) Σ(log π_wm - log π_real) ≈ -2.5 (平均误差)
        
        支持三种奖励模式:
        - negative_l1: R = -|diff|, 范围 (-∞, 0]
        - negative_l2: R = -diff², 范围 (-∞, 0]
        - exponential: R = exp(-coef × |diff|), 范围 (0, 1] (推荐，与 R_facts 同量级)
        
        [CRITICAL FIX] 现在支持历史对话上下文:
        - history: 历史对话列表
        - instruction: WebShop 任务指令
        
        exponential 模式的优势:
        - 奖励范围 (0, 1]，与 R_facts 的 [0, 1] 范围对齐
        - behavior_scale_coef 可调节对概率差异的敏感度
        """
        self.initialize()
        
        # 获取 sum log prob 和 token count
        sum_pred, len_pred = self.compute_action_log_prob(
            predicted_state, expert_action, system_prompt, history, instruction
        )
        sum_real, len_real = self.compute_action_log_prob(
            real_state, expert_action, system_prompt, history, instruction
        )
        
        # 计算 Mean Log Prob（消除长度偏置）
        # 注意：通常 len_pred 和 len_real 应该是一样的，因为 action 是同一个字符串
        # 但为了安全，分别除以各自的长度
        mean_prob_pred = sum_pred / max(1, len_pred)
        mean_prob_real = sum_real / max(1, len_real)
        
        diff = mean_prob_pred - mean_prob_real
        
        # 根据配置选择奖励计算方式
        reward_mode = self.config.reward_mode
        behavior_scale_coef = getattr(self.config, 'behavior_scale_coef', 1.0)
        
        if reward_mode == "negative_l2":
            # 平方惩罚，对大误差极其严厉
            # diff 是 0.5 时，reward 是 -0.25；diff 是 2.0 时，reward 是 -4.0
            reward = -(diff ** 2)
        elif reward_mode == "exponential":
            # 指数奖励，范围 (0, 1]，与 R_facts 同量级
            # diff 是 0 时，reward 是 1.0；diff 越大，reward 越接近 0
            # behavior_scale_coef 控制衰减速度：coef 越大，对误差越敏感
            reward = math.exp(-behavior_scale_coef * abs(diff))
        else:  # negative_l1 (default fallback)
            # 直接使用负的绝对误差
            # diff 是 0.5 时，reward 就是 -0.5
            reward = -abs(diff)
        
        return {
            "score": reward,
            "mean_diff": diff,  # Mean Log Prob 差异，用于分析
            "sum_log_prob_pred": sum_pred,
            "sum_log_prob_real": sum_real,
            "mean_log_prob_pred": mean_prob_pred,
            "mean_log_prob_real": mean_prob_real,
            "token_count_pred": len_pred,
            "token_count_real": len_real,
            "reward_mode": reward_mode,
            "behavior_scale_coef": behavior_scale_coef,
        }
    
    # =========================================================================
    # Batch Inference 支持
    # =========================================================================
    
    def _compute_log_prob_single(self, args: Tuple) -> Tuple[int, Optional[float], int, bool]:
        """
        单个请求的 log prob 计算 (用于线程池)
        
        Args:
            args: (index, state, action, system_prompt, history, instruction)
        
        Returns:
            (index, sum_log_prob, token_count, success)
            如果失败，sum_log_prob 为 None，success 为 False
        """
        idx, state, action, system_prompt, history, instruction = args
        try:
            sum_lp, count = self.compute_action_log_prob(
                state, action, system_prompt, history, instruction
            )
            return (idx, sum_lp, count, True)
        except Exception as e:
            print(f"[Reward] Batch item {idx} failed: {e}")
            # [CRITICAL FIX] 不返回 -20.0，而是标记为失败
            # 上层会根据 success=False 来决定如何处理
            return (idx, None, 0, False)
    
    def compute_action_log_probs_batch(
        self,
        states: List[str],
        actions: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        histories: Optional[List[Optional[List[Dict[str, str]]]]] = None,
        instructions: Optional[List[Optional[str]]] = None,
        max_workers: int = 16
    ) -> List[Tuple[Optional[float], int, bool]]:
        """
        批量计算 log π(action | state)
        
        使用线程池并发发送 HTTP 请求到 vLLM 服务器。
        vLLM 服务器会自动进行请求批处理 (continuous batching)。
        
        Args:
            states: 状态列表
            actions: 动作列表
            system_prompts: 系统提示列表 (可选)
            histories: 历史对话列表 (可选)
            instructions: 任务指令列表 (可选)
            max_workers: 最大并发数
        
        Returns:
            [(sum_log_prob, token_count, success), ...] 与输入顺序对应
            如果 success=False，sum_log_prob 为 None，应使用回退逻辑
        """
        self.initialize()
        
        n = len(states)
        if n == 0:
            return []
        
        # 填充可选参数
        if system_prompts is None:
            system_prompts = [None] * n
        if histories is None:
            histories = [None] * n
        if instructions is None:
            instructions = [None] * n
        
        # 构建参数列表
        args_list = [
            (i, states[i], actions[i], system_prompts[i], histories[i], instructions[i])
            for i in range(n)
        ]
        
        # 使用线程池并发执行
        results = [None] * n
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._compute_log_prob_single, args): args[0] 
                       for args in args_list}
            for future in as_completed(futures):
                idx, sum_lp, count, success = future.result()
                results[idx] = (sum_lp, count, success)
        
        return results
    
    def compute_behavior_consistency_rewards_batch(
        self,
        predicted_states: List[str],
        real_states: List[str],
        expert_actions: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        histories: Optional[List[Optional[List[Dict[str, str]]]]] = None,
        instructions: Optional[List[Optional[str]]] = None,
        max_workers: int = 8
    ) -> List[Dict[str, float]]:
        """
        批量计算行为一致性奖励
        
        Args:
            predicted_states: WM 生成的状态列表
            real_states: 真实状态列表
            expert_actions: 专家动作列表
            system_prompts: 系统提示列表 (可选)
            histories: 历史对话列表 (可选)
            instructions: 任务指令列表 (可选)
            max_workers: 最大并发数
        
        Returns:
            [{"score": ..., "mean_diff": ..., ...}, ...] 与输入顺序对应
        """
        n = len(predicted_states)
        if n == 0:
            return []
        
        # 填充可选参数
        if system_prompts is None:
            system_prompts = [None] * n
        if histories is None:
            histories = [None] * n
        if instructions is None:
            instructions = [None] * n
        
        # 批量计算 predicted 和 real 的 log prob
        # 将两者合并成一个批次，减少线程池开销
        all_states = predicted_states + real_states
        all_actions = expert_actions * 2
        all_system_prompts = system_prompts * 2
        all_histories = histories * 2
        all_instructions = instructions * 2
        
        all_results = self.compute_action_log_probs_batch(
            all_states, all_actions, all_system_prompts, 
            all_histories, all_instructions, max_workers
        )
        
        # 拆分结果
        pred_results = all_results[:n]
        real_results = all_results[n:]
        
        # 计算奖励
        reward_mode = self.config.reward_mode
        rewards = []
        
        for i in range(n):
            sum_pred, len_pred, success_pred = pred_results[i]
            sum_real, len_real, success_real = real_results[i]
            
            # [CRITICAL FIX] 检查是否有 API 调用失败
            # 如果 real_state 计算失败，不应产生 Reward（避免错误的正向奖励）
            if not success_real or sum_real is None:
                # real_state 失败是"不可抗力"，返回中性奖励并标记
                rewards.append({
                    "score": 0.0,  # 中性奖励，不影响训练
                    "mean_diff": 0.0,
                    "sum_log_prob_pred": sum_pred,
                    "sum_log_prob_real": None,
                    "mean_log_prob_pred": 0.0,
                    "mean_log_prob_real": 0.0,
                    "token_count_pred": len_pred,
                    "token_count_real": 0,
                    "reward_mode": reward_mode,
                    "api_failed": True,
                    "failure_reason": "real_state_api_failed",
                })
                continue
            
            if not success_pred or sum_pred is None:
                # pred_state 失败，同样返回中性奖励
                rewards.append({
                    "score": 0.0,
                    "mean_diff": 0.0,
                    "sum_log_prob_pred": None,
                    "sum_log_prob_real": sum_real,
                    "mean_log_prob_pred": 0.0,
                    "mean_log_prob_real": 0.0,
                    "token_count_pred": 0,
                    "token_count_real": len_real,
                    "reward_mode": reward_mode,
                    "api_failed": True,
                    "failure_reason": "pred_state_api_failed",
                })
                continue
            
            mean_prob_pred = sum_pred / max(1, len_pred)
            mean_prob_real = sum_real / max(1, len_real)
            diff = mean_prob_pred - mean_prob_real
            
            if reward_mode == "negative_l2":
                reward = -(diff ** 2)
            else:
                reward = -abs(diff)
            
            rewards.append({
                "score": reward,
                "mean_diff": diff,
                "sum_log_prob_pred": sum_pred,
                "sum_log_prob_real": sum_real,
                "mean_log_prob_pred": mean_prob_pred,
                "mean_log_prob_real": mean_prob_real,
                "token_count_pred": len_pred,
                "token_count_real": len_real,
                "reward_mode": reward_mode,
                "api_failed": False,
            })
        
        return rewards


# =============================================================================
# 单例获取函数
# =============================================================================

def get_format_validator() -> FormatValidator:
    """获取格式验证器单例"""
    global _format_validator
    if _format_validator is None:
        _format_validator = FormatValidator()
    return _format_validator


def get_http_reference_agent(config: PivotGRPOConfig) -> HTTPReferenceAgent:
    """获取 HTTP 裁判 Agent 单例"""
    global _http_reference_agent
    if _http_reference_agent is None:
        _http_reference_agent = HTTPReferenceAgent(config)
    return _http_reference_agent


# =============================================================================
# 物理事实奖励 R_Facts
# =============================================================================

def _compute_facts_reward(predicted_state: str, real_state: str) -> Dict[str, float]:
    """
    计算物理事实奖励 R_Facts
    
    核心思想：
    - World Model 不仅要让 Agent 做出正确决策（Behavior Consistency）
    - 还要保证生成的状态包含正确的物理事实（ASIN、价格、页码等）
    
    返回 [0, 1] 范围的事实匹配奖励，以及各分项得分
    """
    pred = predicted_state.strip()
    real = real_state.strip()
    
    if pred == real:
        return {
            "facts_reward": 1.0,
            "asin_match": 1.0,
            "price_match": 1.0,
            "page_match": 1.0,
            "rating_match": 1.0,
        }
    
    pred_lower = pred.lower()
    real_lower = real.lower()
    
    scores = []
    weights = []
    details = {
        "asin_match": 0.0,
        "price_match": 0.0,
        "page_match": 0.0,
        "rating_match": 0.0,
    }
    
    # 1. ASIN 匹配（产品 ID，最高权重）
    pred_asins = set(re.findall(r'B[A-Z0-9]{9}', pred, re.IGNORECASE))
    real_asins = set(re.findall(r'B[A-Z0-9]{9}', real, re.IGNORECASE))
    
    if real_asins:
        matched = len(pred_asins & real_asins)
        precision = matched / len(pred_asins) if pred_asins else 0.0
        recall = matched / len(real_asins)
        if precision + recall > 0:
            asin_f1 = 2 * precision * recall / (precision + recall)
        else:
            asin_f1 = 0.0
        details["asin_match"] = asin_f1
        scores.append(asin_f1)
        weights.append(4.0)
    
    # 2. 价格匹配（关键购物信息）
    pred_prices = set(re.findall(r'\$[\d,]+\.?\d*', pred))
    real_prices = set(re.findall(r'\$[\d,]+\.?\d*', real))
    
    if real_prices:
        matched = len(pred_prices & real_prices)
        precision = matched / len(pred_prices) if pred_prices else 0.0
        recall = matched / len(real_prices)
        if precision + recall > 0:
            price_f1 = 2 * precision * recall / (precision + recall)
        else:
            price_f1 = 0.0
        details["price_match"] = price_f1
        scores.append(price_f1)
        weights.append(3.0)
    
    # 3. 页码匹配（导航状态）
    pred_pages = re.findall(r'Page (\d+)', pred, re.IGNORECASE)
    real_pages = re.findall(r'Page (\d+)', real, re.IGNORECASE)
    
    if real_pages:
        if pred_pages and pred_pages[0] == real_pages[0]:
            details["page_match"] = 1.0
            scores.append(1.0)
        else:
            details["page_match"] = 0.0
            scores.append(0.0)
        weights.append(2.0)
    
    # 4. 评分匹配（产品质量信息）
    pred_ratings = re.findall(r'Rating:\s*([\d\.]+|N\.A\.)', pred, re.IGNORECASE)
    real_ratings = re.findall(r'Rating:\s*([\d\.]+|N\.A\.)', real, re.IGNORECASE)
    
    if real_ratings:
        pred_rating_set = set(pred_ratings)
        real_rating_set = set(real_ratings)
        matched = len(pred_rating_set & real_rating_set)
        precision = matched / len(pred_rating_set) if pred_rating_set else 0.0
        recall = matched / len(real_rating_set)
        if precision + recall > 0:
            rating_f1 = 2 * precision * recall / (precision + recall)
        else:
            rating_f1 = 0.0
        details["rating_match"] = rating_f1
        scores.append(rating_f1)
        weights.append(1.5)
    
    # 5. 加权平均
    if not scores:
        # 如果没有可匹配的物理事实，回退到词级别相似度
        pred_words = set(pred_lower.split())
        real_words = set(real_lower.split())
        if real_words:
            intersection = pred_words & real_words
            jaccard = len(intersection) / len(real_words)
            details["facts_reward"] = jaccard
        else:
            details["facts_reward"] = 0.0
    else:
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        details["facts_reward"] = weighted_score
    
    return details


def _similarity_to_behavior_reward(similarity: float, reward_mode: str, behavior_scale_coef: float = 1.0) -> float:
    """
    将相似度 [0, 1] 转换为与 reward_mode 一致的行为奖励
    
    这是为了避免 Distribution Shock 风险：
    - 正常情况下 negative_l1 模式的 behavior_score 在 (-∞, 0] 范围
    - 如果回退逻辑返回 [0, 1] 范围的相似度，会导致 Critic 的 Value Estimation 出问题
    - 甚至可能误导 Policy 认为"让 Reference Agent 挂掉就能得高分"
    
    转换规则：
    - negative_l1 模式：转换为 -(1-similarity)，即 [-1, 0]
    - negative_l2 模式：转换为 -(1-similarity)²，即 [-1, 0]
    - exponential 模式：直接返回 similarity，因为 exponential 模式本身范围就是 (0, 1]
    
    Args:
        similarity: 文本相似度 [0, 1]
        reward_mode: 奖励模式 ("negative_l1", "negative_l2", "exponential")
        behavior_scale_coef: exponential 模式下的缩放系数 (此处未使用，保持接口一致性)
    
    Returns:
        与 reward_mode 数值范围一致的行为奖励
    """
    if reward_mode == "negative_l2":
        # 相似度 1.0 -> 惩罚 0.0; 相似度 0.0 -> 惩罚 -1.0
        return -((1.0 - similarity) ** 2)
    elif reward_mode == "exponential":
        # exponential 模式范围本身就是 (0, 1]，直接返回相似度
        # 相似度 1.0 -> 奖励 1.0; 相似度 0.0 -> 奖励 0.0
        return similarity
    else:  # negative_l1 (default)
        # 相似度 1.0 -> 惩罚 0.0; 相似度 0.0 -> 惩罚 -1.0
        return -(1.0 - similarity)


def _compute_similarity_score(predicted_state: str, real_state: str) -> float:
    """
    计算预测状态与真实状态的文本相似度
    
    返回 [0, 1] 范围的相似度分数，1 表示完全匹配
    用作裁判模型不可用时的回退
    """
    pred = predicted_state.strip().lower()
    real = real_state.strip().lower()
    
    if pred == real:
        return 1.0
    
    def extract_key_info(text: str) -> Dict[str, set]:
        info = {
            "asins": set(re.findall(r'b[a-z0-9]{9}', text)),
            "prices": set(re.findall(r'\$[\d\.]+', text)),
            "words": set(text.split()),
        }
        return info
    
    pred_info = extract_key_info(pred)
    real_info = extract_key_info(real)
    
    scores = []
    weights = []
    
    # ASIN 匹配 (高权重)
    if real_info["asins"]:
        asin_match = len(pred_info["asins"] & real_info["asins"]) / len(real_info["asins"])
        scores.append(asin_match)
        weights.append(3.0)
    
    # 价格匹配 (中等权重)
    if real_info["prices"]:
        price_match = len(pred_info["prices"] & real_info["prices"]) / len(real_info["prices"])
        scores.append(price_match)
        weights.append(2.0)
    
    # 词级别 Jaccard 相似度 (基础权重)
    if real_info["words"]:
        intersection = pred_info["words"] & real_info["words"]
        union = pred_info["words"] | real_info["words"]
        jaccard = len(intersection) / len(union) if union else 0
        scores.append(jaccard)
        weights.append(1.0)
    
    if not scores:
        return 0.0
    
    weighted_similarity = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return weighted_similarity


# =============================================================================
# 主奖励函数 - verl 接口
# =============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    # === Behavior Consistency 参数 ===
    reward_mode: str = "exponential",  # 奖励模式: "negative_l1", "negative_l2", "exponential"
    behavior_scale_coef: float = 1.0,  # exponential 模式下的缩放系数
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0,  # 与正常奖励 [0, 1] 同量级
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5,
    behavior_weight: float = 1.0,
    # === 裁判模型参数 ===
    reference_agent_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    use_full_reference_agent: bool = True,
    # === HTTP API 参数 ===
    use_http_reference_agent: bool = True,
    reference_agent_api_url: str = "http://localhost:8000",
    api_timeout: float = 300.0,  # 增加到 300 秒，防止长 context 并发时超时
    **kwargs
) -> Dict[str, Any]:
    """
    Pivot-GRPO: 综合奖励函数 (Behavior Consistency + Physical Facts)
    
    这是 verl 调用的主入口函数。
    
    ==========================================================================
    奖励机制
    ==========================================================================
    
    R_total = behavior_weight × R_behavior + facts_weight × R_facts
    
    - R_behavior: 行为一致性奖励，使用 Mean Log Prob 消除长度偏置:
      * negative_l1: R = -|diff|, 范围 (-∞, 0]
      * negative_l2: R = -diff², 范围 (-∞, 0]
      * exponential: R = exp(-coef × |diff|), 范围 (0, 1] (推荐，与 R_facts 同量级)
      
      其中 diff = mean_log_prob_pred - mean_log_prob_real
      
    - R_facts: 物理事实奖励，范围 [0, 1]
      保证生成状态包含正确的 ASIN/价格/页码等
    
    使用 exponential 模式时，R_behavior 和 R_facts 均在 [0, 1] 范围，
    加权求和后最终得分在 [0, behavior_weight + facts_weight] 范围。
    
    ==========================================================================
    参数说明
    ==========================================================================
    
    Args:
        data_source: 数据来源标识 (webshop_grpo)
        solution_str: World Model 生成的预测状态 (模型输出)
        ground_truth: 真实环境状态
        extra_info: 额外信息字典，支持以下字段:
            - expert_action: 专家动作 (必需，用于 Behavior Consistency)
            - system_prompt: 系统提示 (可选)
            - history: 历史对话 [{\"role\": \"user/assistant\", \"content\": \"...\"}] (可选)
            - instruction: WebShop 任务指令 (可选，会被整合到 user content 中)
            - prompt: verl 传递的完整 prompt (可用于提取 instruction)
        
        reward_mode: 奖励模式 ("negative_l1", "negative_l2", "exponential")
        behavior_scale_coef: exponential 模式下的缩放系数，控制对概率差异的敏感度 (默认 1.0)
        behavior_weight: 行为一致性奖励权重 (默认 1.0)
        facts_weight: 物理事实奖励权重 (默认 0.5)
        format_penalty: 格式错误惩罚 (默认 -1.0，与正常奖励 [0, 1] 同量级)
    
    Returns:
        dict: 包含 "score" 和详细诊断信息的字典
    """
    extra_info = extra_info or {}
    
    # 解析 ground_truth
    ground_truth_str: str = ""
    expert_action: str = extra_info.get("expert_action", "")
    
    if isinstance(ground_truth, dict):
        ground_truth_str = ground_truth.get("ground_truth", "")
        if not expert_action:
            expert_action = ground_truth.get("expert_action", "")
    elif isinstance(ground_truth, str):
        ground_truth_str = ground_truth
    else:
        ground_truth_str = str(ground_truth) if ground_truth else ""
    
    # [CRITICAL FIX] 提取历史对话和 Instruction
    # 在 verl 中，extra_info['prompt'] 通常是 List[Dict] (对话历史)，而不是字符串
    history: Optional[List[Dict[str, str]]] = extra_info.get("history", None)
    instruction: Optional[str] = extra_info.get("instruction", None)
    
    # 处理 prompt 字段 - 它可能是 List[Dict] (对话历史) 或字符串
    if "prompt" in extra_info:
        prompt = extra_info.get("prompt")
        
        # Case 1: prompt 是 List[Dict]，这是 verl 的标准格式
        # 此时 prompt 本身就是 history
        if isinstance(prompt, (list, tuple)) and len(prompt) > 0:
            if history is None:
                # 将 prompt 作为 history 使用
                history = list(prompt)
            
            # 从对话历史中提取 Instruction（通常在 system 或第一条消息中）
            if instruction is None:
                for msg in prompt:
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                        if isinstance(content, str) and "Instruction:" in content:
                            import re as _re
                            match = _re.search(r'Instruction:\s*\[SEP\]\s*(.+?)(?:\n\n|\[SEP\]|$)', content, _re.DOTALL)
                            if match:
                                instruction = match.group(1).strip()
                                break
        
        # Case 2: prompt 是字符串（兼容旧格式）
        elif isinstance(prompt, str) and "Instruction:" in prompt:
            import re as _re
            match = _re.search(r'Instruction:\s*(.+?)(?:\n\n|\[SEP\]|$)', prompt, _re.DOTALL)
            if match:
                instruction = match.group(1).strip()
    
    # 构建配置
    config = PivotGRPOConfig(
        reward_mode=reward_mode,
        behavior_scale_coef=behavior_scale_coef,
        format_penalty=format_penalty,
        facts_weight=facts_weight,
        behavior_weight=behavior_weight,
        reference_agent_model_path=reference_agent_model_path,
        use_full_reference_agent=use_full_reference_agent,
        use_http_reference_agent=use_http_reference_agent,
        reference_agent_api_url=reference_agent_api_url,
        api_timeout=api_timeout,
    )
    
    # 获取单例
    validator = get_format_validator()
    
    if use_full_reference_agent and use_http_reference_agent:
        ref_agent = get_http_reference_agent(config)
    else:
        ref_agent = None
    
    # 初始化结果字典
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
        "predicted_state": solution_str[:100] if solution_str else "",
        "real_state": ground_truth_str[:100] if ground_truth_str else "",
        "expert_action": expert_action[:50] if expert_action else "",
        "instruction": instruction[:50] if instruction else "",
        "has_history": history is not None and len(history) > 0,
        "reward_mode": reward_mode,
        "behavior_scale_coef": behavior_scale_coef,
        "behavior_weight": behavior_weight,
        "facts_weight": facts_weight,
    }
    
    # =========================================================================
    # Step 1: Reality Anchor - 格式验证
    # =========================================================================
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
    
    # =========================================================================
    # Step 2: 计算奖励
    # =========================================================================
    
    behavior_score = 0.0
    facts_score = 0.0
    
    # 2a. 行为一致性奖励 (主要驱动力)
    if use_full_reference_agent and ref_agent is not None and expert_action and ground_truth_str:
        system_prompt = extra_info.get("system_prompt", None)
        
        try:
            # [CRITICAL FIX] 传递 history 和 instruction 到 Reference Agent
            fidelity_result = ref_agent.compute_behavior_consistency_reward(
                predicted_state=solution_str,
                real_state=ground_truth_str,
                expert_action=expert_action,
                system_prompt=system_prompt,
                history=history,
                instruction=instruction
            )
            
            behavior_score = fidelity_result["score"]
            result["behavior_reward"] = behavior_score
            result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
            result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
            result["mean_diff"] = fidelity_result["mean_diff"]
            result["token_count_pred"] = fidelity_result["token_count_pred"]
            result["token_count_real"] = fidelity_result["token_count_real"]
            
        except Exception as e:
            print(f"[Reward] Behavior consistency computation failed: {e}")
            # 回退逻辑：将相似度转换为与 reward_mode 一致的行为奖励
            # 避免 Distribution Shock：negative_l1 模式下不能返回 [0,1] 范围的正数
            similarity = _compute_similarity_score(solution_str, ground_truth_str)
            behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
            result["behavior_reward"] = behavior_score
            result["fallback_similarity"] = similarity  # 记录原始相似度用于诊断
    
    elif ground_truth_str:
        # 没有裁判模型或专家动作，使用相似度作为行为奖励
        # 转换为与 reward_mode 一致的数值范围，避免 Distribution Shock
        similarity = _compute_similarity_score(solution_str, ground_truth_str)
        behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
        result["behavior_reward"] = behavior_score
        result["fallback_similarity"] = similarity  # 记录原始相似度用于诊断
    
    # 2b. 物理事实奖励 R_Facts
    if ground_truth_str:
        facts_result = _compute_facts_reward(solution_str, ground_truth_str)
        facts_score = facts_result["facts_reward"]
        result["facts_reward"] = facts_score
        result["asin_match"] = facts_result.get("asin_match", 0.0)
        result["price_match"] = facts_result.get("price_match", 0.0)
        result["page_match"] = facts_result.get("page_match", 0.0)
        result["rating_match"] = facts_result.get("rating_match", 0.0)
    
    # 2c. 组合最终奖励
    # R_total = behavior_weight × R_behavior + facts_weight × R_facts
    # 注意：不做归一化，直接加权求和
    # 当 behavior_weight=1.0, facts_weight=1.0 时，max = 0 + 1 = 1 (negative_l1 模式)
    if ground_truth_str:
        final_score = behavior_weight * behavior_score + facts_weight * facts_score
    else:
        final_score = behavior_score

    result["score"] = final_score
    return result


# =============================================================================
# Batch 计算函数 - 高性能批量接口
# =============================================================================

def compute_scores_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Union[str, Dict[str, Any]]],
    extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    # === Behavior Consistency 参数 ===
    reward_mode: str = "exponential",  # 奖励模式: "negative_l1", "negative_l2", "exponential"
    behavior_scale_coef: float = 1.0,  # exponential 模式下的缩放系数
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0,  # 与正常奖励 [0, 1] 同量级
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5,
    behavior_weight: float = 1.0,
    # === 裁判模型参数 ===
    reference_agent_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    use_full_reference_agent: bool = True,
    # === HTTP API 参数 ===
    use_http_reference_agent: bool = True,
    reference_agent_api_url: str = "http://localhost:8000",
    api_timeout: float = 300.0,
    # === Batch 参数 ===
    max_workers: int = 8,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    批量计算奖励分数 (高性能接口)
    
    使用线程池并发调用 vLLM API，充分利用 vLLM 的 continuous batching 能力。
    
    Args:
        data_sources: 数据来源标识列表
        solution_strs: World Model 生成的预测状态列表
        ground_truths: 真实环境状态列表
        extra_infos: 额外信息列表
        reward_mode: 奖励模式 ("negative_l1", "negative_l2", "exponential")
        behavior_scale_coef: exponential 模式下的缩放系数 (默认 1.0)
        max_workers: 最大并发数 (默认 8)
        其他参数同 compute_score
    
    Returns:
        奖励结果字典列表，与输入顺序对应
    """
    n = len(solution_strs)
    if n == 0:
        return []
    
    # 填充可选参数
    if extra_infos is None:
        extra_infos = [None] * n
    
    # 构建配置
    config = PivotGRPOConfig(
        reward_mode=reward_mode,
        behavior_scale_coef=behavior_scale_coef,
        format_penalty=format_penalty,
        facts_weight=facts_weight,
        behavior_weight=behavior_weight,
        reference_agent_model_path=reference_agent_model_path,
        use_full_reference_agent=use_full_reference_agent,
        use_http_reference_agent=use_http_reference_agent,
        reference_agent_api_url=reference_agent_api_url,
        api_timeout=api_timeout,
    )
    
    # 获取单例
    validator = get_format_validator()
    ref_agent = get_http_reference_agent(config) if use_full_reference_agent and use_http_reference_agent else None
    
    # 预处理所有样本，筛选出需要调用 Reference Agent 的样本
    preprocessed = []
    for i in range(n):
        extra_info = extra_infos[i] or {}
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        
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
        
        # 提取 history 和 instruction
        # 在 verl 中，extra_info['prompt'] 通常是 List[Dict] (对话历史)，而不是字符串
        history = extra_info.get("history", None)
        instruction = extra_info.get("instruction", None)
        
        if "prompt" in extra_info:
            prompt = extra_info.get("prompt")
            
            # Case 1: prompt 是 List[Dict]，这是 verl 的标准格式
            if isinstance(prompt, (list, tuple)) and len(prompt) > 0:
                if history is None:
                    history = list(prompt)
                
                # 从对话历史中提取 Instruction
                if instruction is None:
                    for msg in prompt:
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            if isinstance(content, str) and "Instruction:" in content:
                                import re as _re
                                match = _re.search(r'Instruction:\s*\[SEP\]\s*(.+?)(?:\n\n|\[SEP\]|$)', content, _re.DOTALL)
                                if match:
                                    instruction = match.group(1).strip()
                                    break
            
            # Case 2: prompt 是字符串（兼容旧格式）
            elif isinstance(prompt, str) and "Instruction:" in prompt:
                import re as _re
                match = _re.search(r'Instruction:\s*(.+?)(?:\n\n|\[SEP\]|$)', prompt, _re.DOTALL)
                if match:
                    instruction = match.group(1).strip()
        
        system_prompt = extra_info.get("system_prompt", None)
        
        preprocessed.append({
            "index": i,
            "solution_str": solution_str,
            "ground_truth_str": ground_truth_str,
            "expert_action": expert_action,
            "history": history,
            "instruction": instruction,
            "system_prompt": system_prompt,
        })
    
    # 初始化结果
    results = [None] * n
    
    # 分类样本
    valid_samples = []  # 格式验证通过且有 expert_action 的样本
    
    for item in preprocessed:
        i = item["index"]
        solution_str = item["solution_str"]
        ground_truth_str = item["ground_truth_str"]
        expert_action = item["expert_action"]
        instruction = item["instruction"]
        history = item["history"]
        
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
            "predicted_state": solution_str[:100] if solution_str else "",
            "real_state": ground_truth_str[:100] if ground_truth_str else "",
            "expert_action": expert_action[:50] if expert_action else "",
            "instruction": instruction[:50] if instruction else "",
            "has_history": history is not None and len(history) > 0,
            "reward_mode": reward_mode,
            "behavior_weight": behavior_weight,
            "facts_weight": facts_weight,
        }
        
        # 格式验证
        if not solution_str:
            result["score"] = format_penalty
            result["format_valid"] = False
            result["format_reason"] = "Empty response"
            results[i] = result
            continue
        
        is_valid, reason = validator.validate(solution_str)
        result["format_valid"] = is_valid
        result["format_reason"] = reason
        
        if not is_valid:
            result["score"] = format_penalty
            results[i] = result
            continue
        
        # 计算 R_facts (不需要 Reference Agent)
        if ground_truth_str:
            facts_result = _compute_facts_reward(solution_str, ground_truth_str)
            result["facts_reward"] = facts_result["facts_reward"]
            result["asin_match"] = facts_result.get("asin_match", 0.0)
            result["price_match"] = facts_result.get("price_match", 0.0)
            result["page_match"] = facts_result.get("page_match", 0.0)
            result["rating_match"] = facts_result.get("rating_match", 0.0)
        
        results[i] = result
        
        # 判断是否需要调用 Reference Agent
        if use_full_reference_agent and ref_agent is not None and expert_action and ground_truth_str:
            valid_samples.append(item)
        elif ground_truth_str:
            # 使用回退逻辑
            similarity = _compute_similarity_score(solution_str, ground_truth_str)
            behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
            result["behavior_reward"] = behavior_score
            result["fallback_similarity"] = similarity
            final_score = behavior_weight * behavior_score + facts_weight * result["facts_reward"]
            result["score"] = final_score
    
    # 批量调用 Reference Agent
    if valid_samples and ref_agent is not None:
        try:
            predicted_states = [s["solution_str"] for s in valid_samples]
            real_states = [s["ground_truth_str"] for s in valid_samples]
            expert_actions = [s["expert_action"] for s in valid_samples]
            system_prompts = [s["system_prompt"] for s in valid_samples]
            histories = [s["history"] for s in valid_samples]
            instructions = [s["instruction"] for s in valid_samples]
            
            fidelity_results = ref_agent.compute_behavior_consistency_rewards_batch(
                predicted_states=predicted_states,
                real_states=real_states,
                expert_actions=expert_actions,
                system_prompts=system_prompts,
                histories=histories,
                instructions=instructions,
                max_workers=max_workers
            )
            
            for j, item in enumerate(valid_samples):
                i = item["index"]
                result = results[i]
                fidelity_result = fidelity_results[j]
                
                behavior_score = fidelity_result["score"]
                result["behavior_reward"] = behavior_score
                result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
                result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
                result["mean_diff"] = fidelity_result["mean_diff"]
                result["token_count_pred"] = fidelity_result["token_count_pred"]
                result["token_count_real"] = fidelity_result["token_count_real"]
                
                final_score = behavior_weight * behavior_score + facts_weight * result["facts_reward"]
                result["score"] = final_score
                
        except Exception as e:
            print(f"[Reward] Batch computation failed: {e}")
            # 回退到相似度
            for item in valid_samples:
                i = item["index"]
                result = results[i]
                solution_str = item["solution_str"]
                ground_truth_str = item["ground_truth_str"]
                
                similarity = _compute_similarity_score(solution_str, ground_truth_str)
                behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
                result["behavior_reward"] = behavior_score
                result["fallback_similarity"] = similarity
                final_score = behavior_weight * behavior_score + facts_weight * result["facts_reward"]
                result["score"] = final_score
    
    return results


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Pivot-GRPO: Behavior Consistency Reward Function Test")
    print("=" * 70)
    
    # 测试格式验证
    validator = FormatValidator()
    test_responses = [
        "Instruction: [SEP] Find me a red shirt [SEP] Back to Search [SEP] Page 1 [SEP] B09P5V8ZSK [SEP] $29.99",
        "Thank you for shopping! [SEP] Reward [SEP] Your score [SEP] 0.666",
        "",
        "```python\nprint('hello')```",  # 现在应该通过（如果包含 WebShop 特征）
        "Too short",
        # 带有 HTML/CSS 的 WebShop 状态 (不应被误伤)
        "[SEP] <div class='product'>B09XYZ1234</div> [SEP]",
    ]
    
    print("\n=== Format Validation (Reality Anchor) ===")
    print("注意: 已移除对 ``` 的检查，不再误伤包含代码块的 WebShop 状态")
    for resp in test_responses:
        is_valid, reason = validator.validate(resp)
        display = resp[:50] + "..." if len(resp) > 50 else resp
        print(f"'{display}' -> Valid: {is_valid}, Reason: {reason}")
    
    print("\n=== Reward Comparison: Negative L1 vs Negative L2 vs Exponential (Mean Log Prob) ===")
    print(f"{'mean_diff':<12} {'negative_l1':<15} {'negative_l2':<15} {'exponential (coef=1.0)':<25}")
    print("-" * 70)
    
    test_diffs = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    for diff in test_diffs:
        neg_l1 = -abs(diff)
        neg_l2 = -(diff ** 2)
        exp_reward = math.exp(-1.0 * abs(diff))
        print(f"{diff:<12.1f} {neg_l1:<15.2f} {neg_l2:<15.2f} {exp_reward:<25.4f}")
    
    print("\n说明:")
    print("  - 使用 Mean Log Prob 消除长度偏置（Length Bias）")
    print("  - 旧公式：R ≈ Σ(log π) ≈ -100 (累积误差，越长惩罚越大)")
    print("  - 新公式：R ≈ (1/N) Σ(log π) ≈ -2.5 (平均误差，与长度无关)")
    print("  - Negative L1: 范围 (-∞, 0]，线性惩罚")
    print("  - Negative L2: 范围 (-∞, 0]，对大误差惩罚更重")
    print("  - Exponential: 范围 (0, 1]，与 R_facts 同量级 (推荐)")
    print("  - behavior_scale_coef 控制 exponential 模式对误差的敏感度")
    
    print("\n=== Full Reward Function Test (without reference agent model) ===")
    test_cases = [
        {
            "solution_str": "Instruction: [SEP] Find me a red shirt [SEP] B09P5V8ZSK [SEP] $29.99",
            "ground_truth": "Instruction: [SEP] Find me a red shirt [SEP] B09P5V8ZSK [SEP] $29.99",
            "extra_info": {"expert_action": "click[Buy Now]"},
            "expected": "should be high score (exact match)",
        },
        {
            "solution_str": "",
            "ground_truth": "Instruction: [SEP] Find me a red shirt",
            "extra_info": {},
            "expected": "should be format_penalty (-1.0)",
        },
        {
            # 测试带有 history 和 instruction 的样本
            "solution_str": "[SEP] Page 1 [SEP] B09P5V8ZSK [SEP] $29.99 [SEP] Red Cotton T-Shirt",
            "ground_truth": "[SEP] Page 1 [SEP] B09P5V8ZSK [SEP] $29.99 [SEP] Red Cotton T-Shirt",
            "extra_info": {
                "expert_action": "click[B09P5V8ZSK]",
                "instruction": "Find me a red cotton shirt under $50",
                "history": [
                    {"role": "user", "content": "[SEP] WebShop Search Page"},
                    {"role": "assistant", "content": "search[red cotton shirt]"},
                ]
            },
            "expected": "should be high score with history context",
        },
    ]
    
    for case in test_cases:
        result = compute_score(
            data_source="webshop_grpo",
            solution_str=case["solution_str"],
            ground_truth=case["ground_truth"],
            extra_info=case.get("extra_info", {}),
            use_full_reference_agent=False,
            reward_mode="exponential",
            behavior_scale_coef=1.0,
        )
        print(f"Score: {result['score']:.4f}, Format valid: {result['format_valid']}")
        print(f"Reward mode: {result.get('reward_mode', 'N/A')}")
        print(f"Behavior scale coef: {result.get('behavior_scale_coef', 'N/A')}")
        print(f"Has history: {result.get('has_history', False)}")
        print(f"Instruction: {result.get('instruction', 'N/A')}")
        print(f"Expected: {case['expected']}")
        print()
    
    print("\n=== Batch Computation Test ===")
    batch_solutions = [
        "Instruction: [SEP] Find me a red shirt [SEP] B09P5V8ZSK [SEP] $29.99",
        "[SEP] Page 1 [SEP] B09XYZ1234 [SEP] $19.99",
        "",  # 空响应，应该返回 format_penalty
    ]
    batch_ground_truths = [
        "Instruction: [SEP] Find me a red shirt [SEP] B09P5V8ZSK [SEP] $29.99",
        "[SEP] Page 1 [SEP] B09XYZ1234 [SEP] $19.99",
        "[SEP] Some state",
    ]
    batch_extra_infos = [
        {"expert_action": "click[Buy]", "instruction": "Find red shirt"},
        {"expert_action": "click[B09XYZ1234]"},
        {},
    ]
    
    batch_results = compute_scores_batch(
        data_sources=["webshop_grpo"] * 3,
        solution_strs=batch_solutions,
        ground_truths=batch_ground_truths,
        extra_infos=batch_extra_infos,
        use_full_reference_agent=False,
        reward_mode="exponential",
        behavior_scale_coef=1.0,
    )
    
    for i, result in enumerate(batch_results):
        print(f"Batch[{i}]: Score={result['score']:.4f}, Valid={result['format_valid']}")
    
    print("\n=== Current Configuration (Default) ===")
    print(f"""
    reward_mode: exponential  (范围 (0, 1]，与 R_facts 同量级，推荐)
    behavior_scale_coef: 1.0  (控制对概率差异的敏感度)
    behavior_weight: 1.0
    facts_weight: 0.5
    format_penalty: -1.0  (与正常奖励 [0, 1] 同量级)
    使用 Mean Log Prob 消除长度偏置
    
    新增功能:
    - 支持 history (历史对话上下文)
    - 支持 instruction (WebShop 任务指令)
    - 支持 compute_scores_batch (批量计算，利用线程池并发)
    - 移除了误伤 WebShop 状态的 ``` 检查
    - exponential 模式使 R_behavior 和 R_facts 处于相同量级
    """)
    
    print("=" * 70)
