#!/usr/bin/env python3
"""
Pivot-GRPO: Behavioral Fidelity + Physical Facts Reward Function

=============================================================================
核心哲学 (Behavioral Fidelity)
=============================================================================

如果一个模拟环境（WM）能让同一个 Agent 产生和在真实环境（Real）中
完全一致的动作概率分布，那么这个 WM 在功能上等价于真实世界。

"只要 Agent 分不清真假，WM 就是真的。"

同时，WM 生成的状态也必须包含正确的物理事实（R_facts），
防止 WM 学会生成 "让 Agent 感觉正确但无法迁移到真实环境" 的对抗样本。

=============================================================================
奖励机制
=============================================================================

R_total = behavior_weight × R_behavior + facts_weight × R_facts

- R_behavior: 行为保真度奖励，支持五种模式:
  * negative_l1: R = -|diff|, 范围 (-∞, 0]
  * negative_l2: R = -diff², 范围 (-∞, 0]
  * exponential: R = exp(-coef × |diff|), 范围 (0, 1]
  * cauchy:      R = 1 / (1 + coef × |diff|), 范围 (0, 1] (推荐)
  * linear:      R = max(0, 1 - coef × |diff|), 范围 [0, 1]
  
  其中 diff = mean_log_prob_pred - mean_log_prob_real
       mean_log_prob = (1/N) * Σ log π(token_i | context)
       coef = behavior_scale_coef (控制敏感度)
  
  使用 Mean（均值）而非 Sum（总和）的优势:
  - 消除长度偏置（Length Bias），关注「平均每个 Token 的预测准确度」
  - 短 action 和长 action 具有可比性
  
  cauchy 模式的优势 (推荐):
  - 奖励范围 (0, 1]，与 R_facts 的 [0, 1] 范围对齐
  - 多项式衰减（~1/|Δ|），梯度不饱和: 在 |Δ|=5 处梯度是 exponential 的 4x
  - exponential 在 |Δ|>3 时几乎为 0 (梯度消失)，cauchy 仍为 0.25
  - behavior_scale_coef 参数可调节对概率差异的敏感度
  
- R_facts: 物理事实奖励
  确保 WM 生成的状态包含正确的物理事实 (ASIN/价格/页码/评分)

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
import threading
import requests
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, Future as _Future

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# =============================================================================
# 全局配置
# =============================================================================
DEFAULT_CONFIG = {
    # === Behavioral Fidelity 参数 ===
    # 使用 Mean Log Prob 消除长度偏置
    "behavior_scale_coef": 1.0,  # 缩放系数，控制对概率差异的敏感度
    
    # === Reality Anchor 参数 ===
    "format_penalty": -1.0,      # 格式错误惩罚（与正常奖励 [0, 1] 同量级）
    
    # === 物理事实奖励参数 ===
    "facts_weight": 0.2,         # 物理事实奖励权重
    "behavior_weight": 0.8,      # 行为一致性奖励权重
    
    # === 裁判模型配置 ===
    "judge_model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    
    # === vLLM HTTP API 配置 ===
    "judge_api_url": "http://localhost:8000",
    "api_timeout": 300.0,  # 增加到 300 秒，防止长 context 并发时超时
}

# 全局单例，延迟初始化
_http_judge_agent = None
_format_validator = None


@dataclass
class PivotGRPOConfig:
    """Pivot-GRPO 奖励函数配置"""
    
    # === Behavioral Fidelity 参数 ===
    reward_mode: str = "exponential"  # 奖励模式: "exponential"(默认), "cauchy", "linear", "negative_l1", "negative_l2"
    behavior_scale_coef: float = 1.0  # 缩放系数，控制对概率差异的敏感度
    # 使用 Mean Log Prob 消除长度偏置
    
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0  # 与正常奖励 [0, 1] 同量级
    
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5
    behavior_weight: float = 1.0
    
    # === 裁判模型配置 ===
    judge_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # === vLLM HTTP API 配置 ===
    judge_api_url: str = "http://localhost:8000"
    api_timeout: float = 300.0  # 增加到 300 秒，防止长 context 并发时超时
    
    # === 开关 ===
    use_http_judge: bool = True
    use_full_judge: bool = True


# =============================================================================
# Format Validator - Reality Anchor
# =============================================================================

class FormatValidator:
    """
    验证 World Model 生成的状态是否符合环境格式（支持 WebShop 和 TextWorld）
    
    这是 "Reality Anchor"，用于防止 Agent Hacking：
    - WM 可能学会生成让 Agent 神经网络产生特定响应的对抗样本
    - 这些样本可能无法迁移到真实环境
    - 格式验证确保生成内容至少遵守基本协议
    """
    
    def __init__(self):
        pass
    
    def validate(self, response: str, domain: str = "auto") -> Tuple[bool, str]:
        """
        验证响应格式是否合法
        
        Args:
            response: WM 生成的状态文本
            domain: 环境域标识 ("webshop", "textworld", "auto")
                    "auto" 会根据内容自动检测域
        
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
        
        # 自动检测域
        if domain == "auto":
            domain = self._detect_domain(response)
        
        if domain == "textworld":
            return self._validate_textworld(response)
        else:
            return self._validate_webshop(response)
    
    def _detect_domain(self, response: str) -> str:
        """根据内容自动检测环境域"""
        # TextWorld 特征
        tw_indicators = [
            "AVAILABLE ACTIONS:" in response,
            bool(re.search(r'-=\s+\w+\s+=-', response)),  # -= Room Name =-
            "*** The End ***" in response,
            "TextWorld" in response,
        ]
        if sum(tw_indicators) >= 1:
            return "textworld"
        
        # WebShop 特征
        ws_indicators = [
            "[SEP]" in response,
            "Instruction:" in response,
            bool(re.search(r'B[A-Z0-9]{9}', response)),
            "Thank you" in response and "shopping" in response.lower(),
        ]
        if sum(ws_indicators) >= 1:
            return "webshop"
        
        return "webshop"  # 默认 WebShop
    
    def _validate_textworld(self, response: str) -> Tuple[bool, str]:
        """验证 TextWorld 环境状态格式"""
        # TextWorld 状态的关键特征
        has_available_actions = "AVAILABLE ACTIONS:" in response
        has_room_name = bool(re.search(r'-=\s+\w+\s+=-', response))
        has_game_prompt = ">" in response  # TextWorld 的 > 提示符
        has_end = "*** The End ***" in response
        has_score = "Your score" in response
        has_textworld = "TextWorld" in response
        
        if has_available_actions:
            return True, "Contains TextWorld AVAILABLE ACTIONS"
        if has_room_name:
            return True, "Contains TextWorld room name"
        if has_end or has_score:
            return True, "TextWorld game completion"
        if has_game_prompt and len(response) > 50:
            return True, "Contains TextWorld prompt marker"
        if has_textworld:
            return True, "Contains TextWorld identifier"
        
        return False, "Missing TextWorld protocol features (AVAILABLE ACTIONS, room name, or game prompt)"
    
    def _validate_webshop(self, response: str) -> Tuple[bool, str]:
        """验证 WebShop 环境状态格式"""
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
# Real-State LogProb 缓存 - GRPO 去重优化
# =============================================================================

class _RealLogProbCache:
    """
    线程安全的 real_state logprob 缓存，支持并发去重。
    
    GRPO n=5 时，同一 prompt 的 5 个 rollout 共享相同的 real_state + action，
    但 verl NaiveRewardManager 逐样本调用 compute_score，导致 real_state 的
    logprob 被重复计算 5 次。本缓存使得只有第一个线程实际调用 API，
    其余线程等待并复用结果。
    
    效果: batch_size=32, n=5 → real API 调用从 160 次降至 32 次 (省 80%)
    """
    
    def __init__(self):
        self._store = {}  # key -> _Future
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get_or_compute(self, key, compute_fn):
        """
        获取缓存结果。如果 key 不存在，执行 compute_fn 并缓存结果。
        多个线程同时请求相同 key 时，只有第一个线程实际计算，其余线程等待。
        
        Args:
            key: 缓存键 (hashable)
            compute_fn: 无参数的可调用对象，返回计算结果
        
        Returns:
            compute_fn 的结果 (来自缓存或新计算)
        """
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
            # 我们是第一个请求此 key 的线程，负责计算
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
        """清理缓存"""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0


# =============================================================================
# HTTP Judge Agent - 3+1 架构
# =============================================================================

class HTTPJudgeAgent:
    """
    通过 HTTP API 调用 vLLM 部署的裁判模型
    
    这是 "3+1" 架构的核心组件：
    - 3 张卡用于 GRPO 训练
    - 1 张卡独立部署 vLLM 推理服务
    """
    
    # WebShop Agent 的默认系统提示
    WEBSHOP_SYSTEM_PROMPT = ("You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nAction: \nclick[something]")
    
    # TextWorld Agent 的系统提示（从 agent_instruct_train.json 提取）
    TEXTWORLD_SYSTEM_PROMPT = ("You are playing a text-based interactive fiction game (TextWorld).\n"
                                "You will receive observations describing the current state. "
                                "When available, a list of admissible actions may be provided.\n"
                                "Always output strictly in the following format:\n"
                                "\"Thought:\n<your reasoning>\n\nAction:\n<the single action to take>\"\n"
                                "Guidelines:\n"
                                "- Prefer actions from admissible commands when provided.\n"
                                "- If no list is provided, issue a valid single command "
                                "(e.g., \"look\", \"inventory\", \"open door\", \"go north\", \"take key\").\n"
                                "- Avoid invalid or multiple actions in one step.")
    
    def __init__(self, config: PivotGRPOConfig):
        self.config = config
        # [MULTI-GPU] 支持逗号分隔的多 URL 负载均衡
        urls = [u.strip().rstrip('/') for u in config.judge_api_url.split(',')]
        self._api_bases = urls
        self._rr_counter = 0
        self._rr_lock = threading.Lock()
        self.api_base = urls[0]  # 兼容：初始化/模型检测用第一个
        self.completions_url = f"{self.api_base}/v1/completions"  # 兼容：被 _get_completions_url() 替代
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._model_name = None
        self._model_root = None
        self._tokenizer = None
        # [PERF] GRPO real_state logprob 去重缓存
        self._real_logprob_cache = _RealLogProbCache()
        if len(urls) > 1:
            print(f"[Reward] Multi-URL load balancing enabled: {len(urls)} endpoints")

    def _get_completions_url(self) -> str:
        """Round-robin 选择下一个 completions endpoint"""
        if len(self._api_bases) == 1:
            return f"{self._api_bases[0]}/v1/completions"
        with self._rr_lock:
            idx = self._rr_counter % len(self._api_bases)
            self._rr_counter += 1
        return f"{self._api_bases[idx]}/v1/completions"
    
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
                    model_info = models[0]
                    # id 用于 API 调用，root 是实际模型路径
                    self._model_name = model_info.get("id", self.config.judge_model_path)
                    self._model_root = model_info.get("root", self._model_name)
                    print(f"[Reward] Connected to vLLM API at {self.api_base}")
                    print(f"[Reward] Using model: {self._model_name} (root: {self._model_root}), mode={self.config.reward_mode}, using Mean Log Prob")
                    
                    # 加载 tokenizer 用于构建正确的 chat template
                    tokenizer_model = self.config.judge_model_path
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
                "Please start the judge server first: bash start_judge_server.sh"
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
        - System: Agent 的系统提示（自动根据域选择 WebShop 或 TextWorld）
        - History: 历史对话轮次 (可选)
        - User: 当前环境状态/观察（包含 Instruction）
        - Assistant: Agent 采取的动作（action）
        
        [IMPORTANT] WM 训练数据的格式和 Agent 格式是反的：
        - WM 格式: user=action, assistant=state
        - Agent 格式: user=state, assistant=action
        
        本函数会自动检测并转换格式。
        支持 WebShop（click[]/search[] 动作）和 TextWorld（自然语言动作）。
        
        Args:
            state: 当前环境状态/观察
            action: Agent 采取的动作
            system_prompt: 系统提示 (可选，自动根据域选择)
            history: 历史对话列表，格式 [{"role": "user/assistant", "content": "..."}]
            instruction: 任务指令 (可选，会被整合到 user content 中)
        
        Returns:
            (full_prompt, action_start_char_pos)
        """
        
        # [DOMAIN DETECTION] 检测环境域（WebShop vs TextWorld）
        # 域影响：系统提示选择、WM 格式检测、动作格式前缀
        detected_domain = "webshop"  # 默认
        if history and len(history) > 0:
            for msg in history:
                if msg.get("role") == "system":
                    sys_content = msg.get("content", "")
                    if "TextWorld" in sys_content or "AVAILABLE ACTIONS" in sys_content:
                        detected_domain = "textworld"
                        break
        # 也检查 state 本身
        if detected_domain == "webshop" and ("AVAILABLE ACTIONS:" in state or 
            bool(re.search(r'-=\s+\w+\s+=-', state))):
            detected_domain = "textworld"
        
        # [CRITICAL FIX] 检测并转换 WM 格式的 history 到 Agent 格式
        # WM 格式: user=action, assistant=state
        # Agent 格式: user=state, assistant=action
        converted_history = None
        if history and len(history) > 0:
            # 检测格式：如果第一个 user 消息是 action，说明是 WM 格式
            first_non_system = None
            for msg in history:
                if msg.get("role") in ["user", "assistant"]:
                    first_non_system = msg
                    break
            
            is_wm_format = False
            if first_non_system and first_non_system.get("role") == "user":
                content = first_non_system.get("content", "").strip()
                # WebShop WM 格式: user 消息是 action (click[...] 或 search[...])
                if content.startswith(("click[", "search[", "think[")):
                    is_wm_format = True
                # TextWorld WM 格式: user 消息是短自然语言动作
                # 检测方法：system 包含 TextWorld 标识 + 第一个 user 消息很短（<100 字符）
                elif detected_domain == "textworld" and len(content) < 100:
                    is_wm_format = True
            
            if is_wm_format:
                # 转换：反转 user 和 assistant 角色
                converted_history = []
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        # system 消息保持不变（但我们会用 Agent 的 system prompt 替换）
                        pass
                    elif role == "user":
                        # WM 的 user (action) -> Agent 的 assistant
                        converted_history.append({"role": "assistant", "content": content})
                    elif role == "assistant":
                        # WM 的 assistant (state) -> Agent 的 user
                        converted_history.append({"role": "user", "content": content})
            else:
                # 已经是 Agent 格式，直接使用（过滤掉 system）
                converted_history = [
                    msg for msg in history 
                    if msg.get("role") in ["user", "assistant"]
                ]
        
        # [DOMAIN-AWARE] 选择系统提示和动作格式前缀
        if system_prompt:
            sys_prompt = system_prompt
        elif detected_domain == "textworld":
            sys_prompt = self.TEXTWORLD_SYSTEM_PROMPT
        else:
            sys_prompt = self.WEBSHOP_SYSTEM_PROMPT
        
        # 动作格式前缀（与系统提示中教的格式一致）
        if detected_domain == "textworld":
            # TextWorld 系统提示教模型输出: "Thought:\n...\n\nAction:\n<action>"
            # 为简化 BehR 计算，只用 "Action:\n" 前缀（跳过 Thought 部分）
            action_prefix = "Action:\n"
        else:
            # WebShop 系统提示教模型输出: "Action: \nclick[something]"
            action_prefix = "Action: \n"
        
        if self._tokenizer is not None and hasattr(self._tokenizer, 'apply_chat_template'):
            # 使用 chat template
            messages = []
            
            # 1. System prompt (使用域感知的 Agent system prompt)
            messages.append({"role": "system", "content": sys_prompt})
            
            # 2. Converted History (如果有)
            if converted_history:
                for msg in converted_history:
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
            
            # [FIX] enable_thinking=False: 禁用 Qwen3 的 thinking 模式
            # Qwen3 是 thinking model，默认倾向于先输出 <think> token
            # 这会导致 action 的首 token (如 "Action") 概率极低（与 <think> 竞争）
            # enable_thinking=False 会预填 <think>\n\n</think>\n\n，跳过思考阶段
            # 使模型直接在 "答案区" 评估 action 的概率
            try:
                prompt_without_action = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                # Fallback: 如果 tokenizer 不支持 enable_thinking 参数（非 Qwen3）
                prompt_without_action = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            # [CRITICAL] 确保 prompt 以换行符结尾，防止 Token 边界合并问题
            if prompt_without_action and not prompt_without_action.endswith('\n'):
                prompt_without_action = prompt_without_action + '\n'

            # [FIX] 添加域感知的动作格式前缀
            # WebShop: "Action: \nclick[something]"  TextWorld: "Action:\nopen chest"
            # action_start_char_pos 指向 action 内容（跳过格式前缀）
            # 格式前缀 token 的 logprob 在 pred 和 real 间差异 ≈0，但会稀释 Mean Log Prob 信号
            action_with_format = action_prefix + action
            full_prompt = prompt_without_action + action_with_format
            action_start_char_pos = len(prompt_without_action) + len(action_prefix)
            
        else:
            # Fallback: 简单拼接格式
            parts = []
            
            if sys_prompt:
                parts.append(f"System: {sys_prompt}")
            
            # Converted History (使用转换后的格式)
            if converted_history:
                for msg in converted_history:
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
            # [FIX] 使用域感知的动作格式前缀（与 chat template 分支保持一致）
            action_with_format = action_prefix + action
            full_prompt = prompt_without_action + action_with_format
            action_start_char_pos = len(prompt_without_action) + len(action_prefix)
        
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
                self._get_completions_url(),
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
            action_token_start_idx = -1  # 使用 -1 作为哨兵值，表示未找到
            
            for i, offset in enumerate(text_offset):
                if offset >= action_start_char_pos:
                    action_token_start_idx = i
                    break
                elif i + 1 < len(text_offset) and text_offset[i + 1] > action_start_char_pos:
                    action_token_start_idx = i
                    break
            
            # 如果循环完成且未找到，检查最后一个 Token
            if action_token_start_idx < 0 and text_offset:
                if text_offset[-1] < action_start_char_pos:
                    action_token_start_idx = len(text_offset) - 1
            
            if action_token_start_idx < 0:
                print(f"[Reward] CRITICAL: Failed to locate action start position!")
                print(f"  action_start_char_pos={action_start_char_pos}")
                print(f"  max_text_offset={max(text_offset) if text_offset else 'N/A'}")
                print(f"  action={action[:50]}...")
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
                print(f"[Reward] Warning: No valid logprobs for action tokens")
                return -20.0, 1  # 避免除以零
            
            total_log_prob = sum(action_logprobs)
            token_count = len(action_logprobs)
            
            return total_log_prob, token_count
            
        except requests.exceptions.Timeout:
            print(f"[Reward] API timeout ({self.config.api_timeout}s) for prompt len {len(full_prompt)}, state len {len(state)}")
            raise RuntimeError(f"vLLM API Timeout after {self.config.api_timeout}s") 
        except Exception as e:
            print(f"[Reward] API error: {e}")
            raise e
    
    def compute_behavioral_fidelity_reward(
        self,
        predicted_state: str,
        real_state: str,
        expert_action: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        instruction: Optional[str] = None
    ) -> Dict[str, float]:
        """
        计算行为保真度奖励 (Behavioral Fidelity Reward)
        
        使用 Mean Log Prob 消除长度偏置（Length Bias）:
        - R ≈ (1/N) Σ(log π_wm - log π_real) ≈ -2.5 (平均误差)
        
        支持五种奖励模式:
        - negative_l1: R = -|diff|, 范围 (-∞, 0]
        - negative_l2: R = -diff², 范围 (-∞, 0]
        - exponential: R = exp(-coef × |diff|), 范围 (0, 1]
        - cauchy:      R = 1/(1 + coef × |diff|), 范围 (0, 1] (推荐)
        - linear:      R = max(0, 1 - coef × |diff|), 范围 [0, 1]
        
        [CRITICAL FIX] 现在支持历史对话上下文:
        - history: 历史对话列表
        - instruction: WebShop 任务指令
        
        cauchy 模式的优势 (推荐):
        - 奖励范围 (0, 1]，与 R_facts 的 [0, 1] 范围对齐
        - 多项式衰减 (~1/|Δ|)，避免 exponential 的梯度消失问题
        - 在 |Δ|=5 时仍有 0.167 (exponential 只有 0.007)
        - behavior_scale_coef 可调节对概率差异的敏感度
        
        [PERF] pred 和 real 的 log prob 计算现在并行执行:
        - 两个 API 调用同时发出，vLLM continuous batching 可以合并处理
        - 单样本延迟从 T_pred + T_real 降低到 max(T_pred, T_real) ≈ T_pred
        - 等效 ~2x 吞吐提升
        """
        self.initialize()
        
        # [PERF] real_state logprob 缓存去重
        # GRPO n=5 时，同一 prompt 的 5 个 rollout 共享相同的 real_state + action
        # 只有第一个线程实际调用 API，其余线程等待并复用结果
        # 效果: real API 调用从 160→32/step (batch_size=32, n=5)
        # [BUGFIX 02-27] 加入 history 哈希，避免同轨迹不同步骤但相同 state+action 时的缓存碰撞
        #   例: traj_000016 的 step 009/020/033 均为 click[back to search] + 相同 state，
        #   但 history 不同 → 不应共享缓存。原 bug 影响 19/500 样本 (3.8%)。
        history_hash = hash(tuple(
            msg.get("content", "") if isinstance(msg, dict) else str(msg)
            for msg in (history or [])
        ))
        cache_key = hash((real_state, expert_action, instruction or "", history_hash))
        
        # pred 在前台线程计算，real 通过缓存获取
        # 如果缓存未命中，pred 和 real 仍然并行（第一个线程负责 real 计算）
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_pred = executor.submit(
                self.compute_action_log_prob,
                predicted_state, expert_action, system_prompt, history, instruction
            )
            
            # 通过缓存获取 real logprob（命中时零开销，未命中时在线程池中计算）
            future_real = executor.submit(
                self._real_logprob_cache.get_or_compute,
                cache_key,
                lambda: self.compute_action_log_prob(
                    real_state, expert_action, system_prompt, history, instruction
                )
            )
            
            sum_pred, len_pred = future_pred.result()
            sum_real, len_real = future_real.result()
        
        # [PERF] 定期输出缓存统计
        self._log_cache_stats_if_needed()
        
        # 计算 Mean Log Prob（消除长度偏置）
        # 注意：通常 len_pred 和 len_real 应该是一样的，因为 action 是同一个字符串
        # 但为了安全，分别除以各自的长度
        mean_prob_pred = sum_pred / max(1, len_pred)
        mean_prob_real = sum_real / max(1, len_real)
        
        diff = mean_prob_pred - mean_prob_real
        
        # 根据配置选择奖励计算方式
        reward_mode = self.config.reward_mode
        behavior_scale_coef = getattr(self.config, 'behavior_scale_coef', 1.0)
        
        if reward_mode == "cauchy":
            # Cauchy/Lorentzian 衰减，范围 (0, 1]，多项式衰减 ~1/|Δ|
            # 优势: 不饱和，在 |Δ|=5 时梯度仍是 exponential 的 4x
            # coef=1: |Δ|=0→1.0, |Δ|=1→0.5, |Δ|=3→0.25, |Δ|=5→0.17
            reward = 1.0 / (1.0 + behavior_scale_coef * abs(diff))
        elif reward_mode == "linear":
            # 线性衰减，范围 [0, 1]，梯度恒定直到截断
            # coef=0.2: |Δ|=0→1.0, |Δ|=1→0.8, |Δ|=3→0.4, |Δ|=5→0.0
            reward = max(0.0, 1.0 - behavior_scale_coef * abs(diff))
        elif reward_mode == "negative_l2":
            # 平方惩罚，对大误差极其严厉
            # diff 是 0.5 时，reward 是 -0.25；diff 是 2.0 时，reward 是 -4.0
            reward = -(diff ** 2)
        elif reward_mode == "exponential":
            # 指数奖励，范围 (0, 1]，但在 |Δ|>3 时梯度接近 0 (已不推荐)
            # diff 是 0 时，reward 是 1.0；diff 越大，reward 越接近 0
            # 问题: e^{-3}≈0.05, e^{-5}≈0.007 — 梯度消失严重
            reward = math.exp(-behavior_scale_coef * abs(diff))
        else:  # negative_l1 (default fallback)
            # 直接使用负的绝对误差，梯度恒为 -1
            # 优点: 无饱和; 缺点: 范围 (-∞, 0]，与 R_facts [0,1] 尺度不匹配
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
    
    def _log_cache_stats_if_needed(self):
        """定期输出 real_state logprob 缓存统计"""
        cache = self._real_logprob_cache
        if cache.misses > 0 and cache.misses % 100 == 0:
            total = cache.hits + cache.misses
            hit_rate = cache.hits / total * 100 if total else 0
            print(f"[Reward] Real logprob cache: {cache.hits} hits, {cache.misses} misses, "
                  f"hit_rate={hit_rate:.1f}%, saved {cache.hits} API calls")
    
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
        max_retries = 2  # 最多重试 2 次（共尝试 3 次）
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                sum_lp, count = self.compute_action_log_prob(
                    state, action, system_prompt, history, instruction
                )
                return (idx, sum_lp, count, True)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    import time
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s
                    print(f"[Reward] Batch item {idx} attempt {attempt+1} failed: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        print(f"[Reward] Batch item {idx} failed after {max_retries+1} attempts: {last_error}")
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
    
    def compute_behavioral_fidelity_rewards_batch(
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
        批量计算行为保真度奖励
        
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
        
        # =====================================================================
        # [PERF] real_state 去重优化
        # GRPO n=5 时，同一个 prompt 的 5 个 sample 共享相同的 real_state + action
        # 去重后 API 调用从 n_pred + n_real = 320 减少到 n_pred + n_unique_real = 192
        # (batch_size=32, n=5: 160 pred + 32 unique real vs 原来 160 pred + 160 real)
        # =====================================================================
        
        # 对 real_state 去重：(real_state, action, instruction) 作为 key
        unique_real_map = {}  # key -> index in unique_real_* lists
        real_idx_to_unique = []  # real_results[i] 对应 unique 结果的 index
        unique_real_states = []
        unique_real_actions = []
        unique_real_system_prompts = []
        unique_real_histories = []
        unique_real_instructions = []
        
        for i in range(n):
            # 用 (real_state, action) 作为去重 key
            # instruction 和 history 与 real_state 一一对应，同一 prompt 必然相同
            key = (real_states[i], expert_actions[i])
            if key not in unique_real_map:
                unique_real_map[key] = len(unique_real_states)
                unique_real_states.append(real_states[i])
                unique_real_actions.append(expert_actions[i])
                unique_real_system_prompts.append(system_prompts[i])
                unique_real_histories.append(histories[i])
                unique_real_instructions.append(instructions[i])
            real_idx_to_unique.append(unique_real_map[key])
        
        n_unique_real = len(unique_real_states)
        if n_unique_real < n:
            print(f"[Reward] Real state dedup: {n} -> {n_unique_real} unique ({n - n_unique_real} saved)")
        
        # 合并 pred + unique_real 发送给 Judge
        all_states = predicted_states + unique_real_states
        all_actions = expert_actions + unique_real_actions
        all_system_prompts = system_prompts + unique_real_system_prompts
        all_histories = histories + unique_real_histories
        all_instructions = instructions + unique_real_instructions
        
        all_results = self.compute_action_log_probs_batch(
            all_states, all_actions, all_system_prompts, 
            all_histories, all_instructions, max_workers
        )
        
        # 拆分结果
        pred_results = all_results[:n]
        unique_real_results = all_results[n:]
        
        # 将 unique real 结果映射回原始顺序
        real_results = [unique_real_results[real_idx_to_unique[i]] for i in range(n)]
        
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
            
            # [FIX] 支持所有五种 reward_mode
            behavior_scale_coef = getattr(self.config, 'behavior_scale_coef', 1.0)
            if reward_mode == "cauchy":
                reward = 1.0 / (1.0 + behavior_scale_coef * abs(diff))
            elif reward_mode == "linear":
                reward = max(0.0, 1.0 - behavior_scale_coef * abs(diff))
            elif reward_mode == "negative_l2":
                reward = -(diff ** 2)
            elif reward_mode == "exponential":
                reward = math.exp(-behavior_scale_coef * abs(diff))
            else:  # negative_l1
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


def get_http_judge_agent(config: PivotGRPOConfig) -> HTTPJudgeAgent:
    """获取 HTTP 裁判 Agent 单例"""
    global _http_judge_agent
    if _http_judge_agent is None:
        _http_judge_agent = HTTPJudgeAgent(config)
    return _http_judge_agent


# =============================================================================
# 物理事实奖励 R_Facts
# =============================================================================

def _compute_facts_reward(predicted_state: str, real_state: str) -> Dict[str, float]:
    """
    计算物理事实奖励 R_Facts
    
    核心思想：
    - World Model 不仅要让 Agent 做出正确决策（Behavioral Fidelity）
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
    - 甚至可能误导 Policy 认为"让 Judge 挂掉就能得高分"
    
    转换规则：
    - cauchy 模式：直接返回 similarity，因为 cauchy 模式范围就是 (0, 1]
    - linear 模式：直接返回 similarity，因为 linear 模式范围就是 [0, 1]
    - exponential 模式：直接返回 similarity，因为 exponential 模式本身范围就是 (0, 1]
    - negative_l1 模式：转换为 -(1-similarity)，即 [-1, 0]
    - negative_l2 模式：转换为 -(1-similarity)²，即 [-1, 0]
    
    Args:
        similarity: 文本相似度 [0, 1]
        reward_mode: 奖励模式 ("cauchy", "linear", "exponential", "negative_l1", "negative_l2")
        behavior_scale_coef: exponential 模式下的缩放系数 (此处未使用，保持接口一致性)
    
    Returns:
        与 reward_mode 数值范围一致的行为奖励
    """
    if reward_mode in ("cauchy", "linear", "exponential"):
        # cauchy/linear/exponential 模式范围都是 [0, 1]，直接返回相似度
        # 相似度 1.0 -> 奖励 1.0; 相似度 0.0 -> 奖励 0.0
        return similarity
    elif reward_mode == "negative_l2":
        # 相似度 1.0 -> 惩罚 0.0; 相似度 0.0 -> 惩罚 -1.0
        return -((1.0 - similarity) ** 2)
    else:  # negative_l1 (default)
        # 相似度 1.0 -> 惩罚 0.0; 相似度 0.0 -> 惩罚 -1.0
        return -(1.0 - similarity)


def _compute_length_penalty(
    pred_len: int,
    real_len: int,
    min_ratio: float = 0.3,
    max_ratio: float = 1.5,
    penalty_scale: float = 1.0
) -> Dict[str, float]:
    """
    计算长度惩罚 (Dense Length Penalty)

    防止 WM 生成过短或过长的响应：
    - 过短：WM 可能只生成页面头部，缺少产品信息 (Agent Hacking)
    - 过长：WM 可能生成冗余信息

    惩罚是稠密的，与偏离程度成正比：
    - length_ratio in [min_ratio, max_ratio]: 无惩罚 (penalty = 0)
    - length_ratio < min_ratio: penalty ∝ (min_ratio - ratio) / min_ratio
    - length_ratio > max_ratio: penalty ∝ (ratio - max_ratio) / max_ratio

    Args:
        pred_len: 预测状态长度
        real_len: 真实状态长度
        min_ratio: 最小允许长度比例 (默认 0.3)
        max_ratio: 最大允许长度比例 (默认 1.5)
        penalty_scale: 惩罚缩放系数 (默认 1.0)

    Returns:
        dict: {
            "length_ratio": 长度比例,
            "length_penalty": 惩罚值 [0, penalty_scale], 0 表示无惩罚,
            "length_status": "ok" | "too_short" | "too_long"
        }
    """
    if real_len == 0:
        return {
            "length_ratio": 0.0,
            "length_penalty": 0.0,
            "length_status": "no_reference"
        }

    ratio = pred_len / real_len

    if min_ratio <= ratio <= max_ratio:
        # 在正常范围内，无惩罚
        return {
            "length_ratio": ratio,
            "length_penalty": 0.0,
            "length_status": "ok"
        }
    elif ratio < min_ratio:
        # 过短：线性惩罚
        # ratio = 0 时，deviation = 1.0 (最大惩罚)
        # ratio = min_ratio 时，deviation = 0.0 (无惩罚)
        deviation = (min_ratio - ratio) / min_ratio
        penalty = deviation * penalty_scale
        return {
            "length_ratio": ratio,
            "length_penalty": penalty,
            "length_status": "too_short"
        }
    else:  # ratio > max_ratio
        # 过长：线性惩罚（但有上限）
        # ratio = max_ratio 时，deviation = 0.0 (无惩罚)
        # ratio = 2 * max_ratio 时，deviation = 1.0 (最大惩罚)
        deviation = min(1.0, (ratio - max_ratio) / max_ratio)
        penalty = deviation * penalty_scale
        return {
            "length_ratio": ratio,
            "length_penalty": penalty,
            "length_status": "too_long"
        }


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
    # === Behavioral Fidelity 参数 ===
    reward_mode: str = "exponential",  # 奖励模式: "exponential"(默认), "cauchy", "linear", "negative_l1", "negative_l2"
    behavior_scale_coef: float = 1.0,  # 缩放系数，控制对概率差异的敏感度
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0,  # 与正常奖励 [0, 1] 同量级
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5,
    behavior_weight: float = 1.0,
    # === 长度惩罚参数 ===
    length_penalty_weight: float = 1.0,  # 长度惩罚权重
    length_min_ratio: float = 0.75,  # 最小允许长度比例
    length_max_ratio: float = 1.3,  # 最大允许长度比例
    # === 裁判模型参数 ===
    judge_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    use_full_judge: bool = True,
    # === HTTP API 参数 ===
    use_http_judge: bool = True,
    judge_api_url: str = "http://localhost:8000",
    api_timeout: float = 300.0,  # 增加到 300 秒，防止长 context 并发时超时
    **kwargs
) -> Dict[str, Any]:
    """
    Pivot-GRPO: 综合奖励函数 (Behavioral Fidelity + Physical Facts + Length Penalty)

    这是 verl 调用的主入口函数。

    ==========================================================================
    奖励机制
    ==========================================================================

    R_total = behavior_weight × R_behavior + facts_weight × R_facts - length_penalty_weight × R_length_penalty

    - R_behavior: 行为保真度奖励，使用 Mean Log Prob 消除长度偏置:
      * cauchy:      R = 1/(1 + coef × |diff|), 范围 (0, 1] (推荐，梯度不饱和)
      * exponential: R = exp(-coef × |diff|), 范围 (0, 1]
      * linear:      R = max(0, 1 - coef × |diff|), 范围 [0, 1]
      * negative_l1: R = -|diff|, 范围 (-∞, 0]
      * negative_l2: R = -diff², 范围 (-∞, 0]

      其中 diff = mean_log_prob_pred - mean_log_prob_real

    - R_facts: 物理事实奖励，范围 [0, 1]
      保证生成状态包含正确的 ASIN/价格/页码等

    - R_length_penalty: 长度惩罚，范围 [0, 1]
      防止 WM 生成过短（Agent Hacking）或过长的响应
      惩罚是稠密的，与偏离程度成正比

    使用 exponential 模式时，R_behavior 和 R_facts 均在 [0, 1] 范围，
    加权求和后最终得分在 [-length_penalty_weight, behavior_weight + facts_weight] 范围。

    ==========================================================================
    参数说明
    ==========================================================================

    Args:
        data_source: 数据来源标识 (webshop_grpo)
        solution_str: World Model 生成的预测状态 (模型输出)
        ground_truth: 真实环境状态
        extra_info: 额外信息字典，支持以下字段:
            - expert_action: 专家动作 (必需，用于 Behavioral Fidelity)
            - system_prompt: 系统提示 (可选)
            - history: 历史对话 [{\"role\": \"user/assistant\", \"content\": \"...\"}] (可选)
            - instruction: WebShop 任务指令 (可选，会被整合到 user content 中)
            - prompt: verl 传递的完整 prompt (可用于提取 instruction)

        reward_mode: 奖励模式 ("cauchy", "exponential", "linear", "negative_l1", "negative_l2")
        behavior_scale_coef: 缩放系数，控制对概率差异的敏感度 (默认 1.0)
        behavior_weight: 行为一致性奖励权重 (默认 1.0)
        facts_weight: 物理事实奖励权重 (默认 0.5)
        format_penalty: 格式错误惩罚 (默认 -1.0，与正常奖励 [0, 1] 同量级)
        length_penalty_weight: 长度惩罚权重 (默认 1.0)
        length_min_ratio: 最小允许长度比例 (默认 0.75)
        length_max_ratio: 最大允许长度比例 (默认 1.3)

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
    
    # [FIX] 处理 numpy.ndarray 类型的 history（从 parquet 读取时可能是 ndarray）
    if history is not None:
        import numpy as np
        if isinstance(history, np.ndarray):
            history = history.tolist()
        elif not isinstance(history, list):
            history = list(history)
    
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
        judge_model_path=judge_model_path,
        use_full_judge=use_full_judge,
        use_http_judge=use_http_judge,
        judge_api_url=judge_api_url,
        api_timeout=api_timeout,
    )
    
    # 获取单例
    validator = get_format_validator()
    
    if use_full_judge and use_http_judge:
        judge = get_http_judge_agent(config)
    else:
        judge = None
    
    # 初始化结果字典
    # 注意: 所有可能的 key 都需要在这里初始化，避免 verl agent_loop 中的 KeyError
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
        "fallback_similarity": 0.0,  # 回退相似度（Judge 失败时使用）
        "used_fallback": False,  # 是否使用了回退逻辑（True 表示 Judge 失败）
        # === 长度监控指标 ===
        "pred_length": len(solution_str) if solution_str else 0,
        "real_length": len(ground_truth_str) if ground_truth_str else 0,
        "length_ratio": 0.0,
        "length_penalty": 0.0,
        "length_status": "unknown",
        "length_penalty_weight": length_penalty_weight,
        # === 其他诊断信息 ===
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
    
    # [DOMAIN-AWARE] 从 data_source 推断环境域
    format_domain = "auto"
    if "textworld" in data_source.lower():
        format_domain = "textworld"
    elif "webshop" in data_source.lower():
        format_domain = "webshop"
    
    is_valid, reason = validator.validate(solution_str, domain=format_domain)
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
    
    # 2a. 行为保真度奖励 (主要驱动力)
    if not expert_action:
        print(f"[Reward] Warning: empty expert_action, falling back to similarity")
    
    if use_full_judge and judge is not None and expert_action and ground_truth_str:
        system_prompt = extra_info.get("system_prompt", None)
        
        try:
            # [CRITICAL FIX] 传递 history 和 instruction 到 Judge
            fidelity_result = judge.compute_behavioral_fidelity_reward(
                predicted_state=solution_str,
                real_state=ground_truth_str,
                expert_action=expert_action,
                system_prompt=system_prompt,
                history=history,
                instruction=instruction
            )
            
            behavior_score = fidelity_result["score"]
            
            # [BUG FIX] API 失败时使用 similarity fallback 而非硬编码 0.0
            if fidelity_result.get("api_failed", False):
                similarity = _compute_similarity_score(solution_str, ground_truth_str)
                behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
                result["fallback_similarity"] = similarity
                result["used_fallback"] = True
                result["api_failure_reason"] = fidelity_result.get("failure_reason", "unknown")
            
            result["behavior_reward"] = behavior_score
            result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
            result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
            result["mean_diff"] = fidelity_result["mean_diff"]
            result["token_count_pred"] = fidelity_result["token_count_pred"]
            result["token_count_real"] = fidelity_result["token_count_real"]
            
        except Exception as e:
            print(f"[Reward] Behavioral fidelity computation failed: {e}")
            # 回退逻辑：将相似度转换为与 reward_mode 一致的行为奖励
            # 避免 Distribution Shock：negative_l1 模式下不能返回 [0,1] 范围的正数
            similarity = _compute_similarity_score(solution_str, ground_truth_str)
            behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
            result["behavior_reward"] = behavior_score
            result["fallback_similarity"] = similarity  # 记录原始相似度用于诊断
            result["used_fallback"] = True  # 标记使用了回退逻辑
    
    elif ground_truth_str:
        # 没有裁判模型或专家动作，使用相似度作为行为奖励
        # 转换为与 reward_mode 一致的数值范围，避免 Distribution Shock
        similarity = _compute_similarity_score(solution_str, ground_truth_str)
        behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
        result["behavior_reward"] = behavior_score
        result["fallback_similarity"] = similarity  # 记录原始相似度用于诊断
        result["used_fallback"] = True  # 标记使用了回退逻辑（无 Judge 或无 expert_action）
    
    # 2b. 物理事实奖励 R_Facts
    if ground_truth_str:
        facts_result = _compute_facts_reward(solution_str, ground_truth_str)
        facts_score = facts_result["facts_reward"]
        result["facts_reward"] = facts_score
        result["asin_match"] = facts_result.get("asin_match", 0.0)
        result["price_match"] = facts_result.get("price_match", 0.0)
        result["page_match"] = facts_result.get("page_match", 0.0)
        result["rating_match"] = facts_result.get("rating_match", 0.0)

    # 2c. 长度惩罚 R_length_penalty
    length_penalty_score = 0.0
    if ground_truth_str:
        length_result = _compute_length_penalty(
            pred_len=len(solution_str) if solution_str else 0,
            real_len=len(ground_truth_str),
            min_ratio=length_min_ratio,
            max_ratio=length_max_ratio,
            penalty_scale=1.0  # 惩罚值在 [0, 1]，通过 length_penalty_weight 控制最终影响
        )
        length_penalty_score = length_result["length_penalty"]
        result["length_ratio"] = length_result["length_ratio"]
        result["length_penalty"] = length_penalty_score
        result["length_status"] = length_result["length_status"]

    # 2d. 组合最终奖励
    # R_total = behavior_weight × R_behavior + facts_weight × R_facts - length_penalty_weight × R_length_penalty
    # 注意：不做归一化，直接加权求和
    if ground_truth_str:
        final_score = (
            behavior_weight * behavior_score
            + facts_weight * facts_score
            - length_penalty_weight * length_penalty_score
        )
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
    # === Behavioral Fidelity 参数 ===
    reward_mode: str = "exponential",  # 奖励模式: "exponential"(默认), "cauchy", "linear", "negative_l1", "negative_l2"
    behavior_scale_coef: float = 1.0,  # 缩放系数，控制对概率差异的敏感度
    # === Reality Anchor 参数 ===
    format_penalty: float = -1.0,  # 与正常奖励 [0, 1] 同量级
    # === 物理事实奖励参数 ===
    facts_weight: float = 0.5,
    behavior_weight: float = 1.0,
    # === 长度惩罚参数 ===
    length_penalty_weight: float = 1.0,  # 长度惩罚权重
    length_min_ratio: float = 0.75,  # 最小允许长度比例
    length_max_ratio: float = 1.3,  # 最大允许长度比例
    # === 裁判模型参数 ===
    judge_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    use_full_judge: bool = True,
    # === HTTP API 参数 ===
    use_http_judge: bool = True,
    judge_api_url: str = "http://localhost:8000",
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
        reward_mode: 奖励模式 ("cauchy", "exponential", "linear", "negative_l1", "negative_l2")
        behavior_scale_coef: 缩放系数 (默认 1.0)
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
        judge_model_path=judge_model_path,
        use_full_judge=use_full_judge,
        use_http_judge=use_http_judge,
        judge_api_url=judge_api_url,
        api_timeout=api_timeout,
    )
    
    # 获取单例
    validator = get_format_validator()
    judge = get_http_judge_agent(config) if use_full_judge and use_http_judge else None
    
    # 预处理所有样本，筛选出需要调用 Judge 的样本
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
        # 注意: 所有可能的 key 都需要在这里初始化，避免 verl agent_loop 中的 KeyError
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
            "fallback_similarity": 0.0,  # 回退相似度（Judge 失败时使用）
            "used_fallback": False,  # 是否使用了回退逻辑（True 表示 Judge 失败）
            # === 长度监控指标 ===
            "pred_length": len(solution_str) if solution_str else 0,
            "real_length": len(ground_truth_str) if ground_truth_str else 0,
            "length_ratio": 0.0,
            "length_penalty": 0.0,
            "length_status": "unknown",
            "length_penalty_weight": length_penalty_weight,
            # === 其他诊断信息 ===
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
        
        # 计算 R_facts (不需要 Judge)
        if ground_truth_str:
            facts_result = _compute_facts_reward(solution_str, ground_truth_str)
            result["facts_reward"] = facts_result["facts_reward"]
            result["asin_match"] = facts_result.get("asin_match", 0.0)
            result["price_match"] = facts_result.get("price_match", 0.0)
            result["page_match"] = facts_result.get("page_match", 0.0)
            result["rating_match"] = facts_result.get("rating_match", 0.0)

            # 计算长度惩罚
            length_result = _compute_length_penalty(
                pred_len=len(solution_str) if solution_str else 0,
                real_len=len(ground_truth_str),
                min_ratio=length_min_ratio,
                max_ratio=length_max_ratio,
                penalty_scale=1.0
            )
            result["length_ratio"] = length_result["length_ratio"]
            result["length_penalty"] = length_result["length_penalty"]
            result["length_status"] = length_result["length_status"]
        
        results[i] = result
        
        # 判断是否需要调用 Judge
        if use_full_judge and judge is not None and expert_action and ground_truth_str:
            valid_samples.append(item)
        elif ground_truth_str:
            # 使用回退逻辑（无 Judge 或无 expert_action）
            similarity = _compute_similarity_score(solution_str, ground_truth_str)
            behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
            result["behavior_reward"] = behavior_score
            result["fallback_similarity"] = similarity
            result["used_fallback"] = True  # 标记使用了回退逻辑
            final_score = (
                behavior_weight * behavior_score
                + facts_weight * result["facts_reward"]
                - length_penalty_weight * result["length_penalty"]
            )
            result["score"] = final_score
    
    # 批量调用 Judge
    if valid_samples and judge is not None:
        try:
            predicted_states = [s["solution_str"] for s in valid_samples]
            real_states = [s["ground_truth_str"] for s in valid_samples]
            expert_actions = [s["expert_action"] for s in valid_samples]
            system_prompts = [s["system_prompt"] for s in valid_samples]
            histories = [s["history"] for s in valid_samples]
            instructions = [s["instruction"] for s in valid_samples]
            
            fidelity_results = judge.compute_behavioral_fidelity_rewards_batch(
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
                
                # [BUG FIX] API 失败时使用 similarity fallback 而非硬编码 0.0
                # 硬编码 0.0 远低于正常 Cauchy 均值(~0.68)，给模型错误的惩罚信号
                if fidelity_result.get("api_failed", False):
                    solution_str = item["solution_str"]
                    ground_truth_str = item["ground_truth_str"]
                    similarity = _compute_similarity_score(solution_str, ground_truth_str)
                    behavior_score = _similarity_to_behavior_reward(similarity, reward_mode)
                    result["behavior_reward"] = behavior_score
                    result["fallback_similarity"] = similarity
                    result["used_fallback"] = True
                    result["api_failure_reason"] = fidelity_result.get("failure_reason", "unknown")
                else:
                    behavior_score = fidelity_result["score"]
                    result["behavior_reward"] = behavior_score
                
                result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
                result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
                result["mean_diff"] = fidelity_result["mean_diff"]
                result["token_count_pred"] = fidelity_result["token_count_pred"]
                result["token_count_real"] = fidelity_result["token_count_real"]

                final_score = (
                    behavior_weight * behavior_score
                    + facts_weight * result["facts_reward"]
                    - length_penalty_weight * result["length_penalty"]
                )
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
                result["used_fallback"] = True  # 标记使用了回退逻辑
                final_score = (
                    behavior_weight * behavior_score
                    + facts_weight * result["facts_reward"]
                    - length_penalty_weight * result["length_penalty"]
                )
                result["score"] = final_score

    return results


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Pivot-GRPO: Behavioral Fidelity Reward Function Test")
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
    
    print("\n=== Reward Comparison: All 5 Modes (Mean Log Prob) ===")
    print(f"{'mean_diff':<12} {'neg_l1':<10} {'neg_l2':<10} {'exponential':<14} {'cauchy':<14} {'linear(c=0.2)':<14}")
    print("-" * 80)
    
    test_diffs = [-5.0, -3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    for diff in test_diffs:
        neg_l1 = -abs(diff)
        neg_l2 = -(diff ** 2)
        exp_reward = math.exp(-1.0 * abs(diff))
        cauchy_reward = 1.0 / (1.0 + 1.0 * abs(diff))
        linear_reward = max(0.0, 1.0 - 0.2 * abs(diff))
        print(f"{diff:<12.1f} {neg_l1:<10.4f} {neg_l2:<10.4f} {exp_reward:<14.6f} {cauchy_reward:<14.6f} {linear_reward:<14.6f}")
    
    print("\n说明:")
    print("  - 使用 Mean Log Prob 消除长度偏置（Length Bias）")
    print("  - Cauchy (推荐): 范围 (0, 1], 多项式衰减 ~1/|Δ|, 梯度不饱和")
    print("  - Exponential (旧默认): 范围 (0, 1], 指数衰减, |Δ|>3 时梯度≈0")
    print("  - Linear: 范围 [0, 1], 恒定梯度, 但在 |Δ|>1/coef 时截断为 0")
    print("  - Negative L1: 范围 (-∞, 0], 恒定梯度, 但与 R_facts [0,1] 尺度不匹配")
    print("  - Negative L2: 范围 (-∞, 0], 对大误差极其严厉")
    print("\n  梯度保留 (在 |Δ|=5 处):")
    print(f"    exponential: {math.exp(-5):.6f}")
    print(f"    cauchy:      {1.0/(1+5)**2:.6f}  ({1.0/(1+5)**2 / math.exp(-5):.1f}x better)")
    print(f"    linear:      0.0 (已截断)")
    print(f"    neg_l1:      1.0 (恒定)")
    
    print("\n=== Full Reward Function Test (without judge model) ===")
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
            use_full_judge=False,
            reward_mode="cauchy",
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
        use_full_judge=False,
        reward_mode="cauchy",
        behavior_scale_coef=1.0,
    )
    
    for i, result in enumerate(batch_results):
        print(f"Batch[{i}]: Score={result['score']:.4f}, Valid={result['format_valid']}")
    
    print("\n=== Current Configuration (Default) ===")
    print(f"""
    reward_mode: cauchy  (范围 (0, 1], 多项式衰减, 梯度不饱和, 推荐)
    behavior_scale_coef: 1.0  (控制对概率差异的敏感度)
    behavior_weight: 1.0
    facts_weight: 0.5
    format_penalty: -1.0  (与正常奖励 [0, 1] 同量级)
    使用 Mean Log Prob 消除长度偏置
    
    Cauchy vs Exponential 关键区别:
    - |Δ|=3: cauchy=0.250, exponential=0.050 (5x difference)
    - |Δ|=5: cauchy=0.167, exponential=0.007 (24x difference)
    - 梯度: cauchy 在 |Δ|=5 处梯度是 exponential 的 4x
    - 原因: cauchy 是 O(1/|Δ|) 衰减, exponential 是 O(e^{{-|Δ|}}) 衰减
    """)
    
    print("=" * 70)
