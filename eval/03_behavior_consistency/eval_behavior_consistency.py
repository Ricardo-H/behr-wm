#!/usr/bin/env python3
"""
Behavior Consistency Evaluation for World Models

=============================================================================
核心思想
=============================================================================
使用 Behavior Consistency Reward 作为独立评估指标，测试 World Model 在测试集上
与真实环境的单步动作一致性。

评估流程：
1. 加载测试集（多轮对话格式）
2. 对于每个 step：
   - 使用历史 context + action 让 WM 生成预测状态
   - 使用裁判模型计算 Behavior Consistency：
     * log π(next_action | WM_predicted_state)  vs 
     * log π(next_action | real_state)
   - 差异越小，说明 WM 越能让 Agent 做出与真实环境一致的决策

=============================================================================
使用方法
=============================================================================
1. 启动裁判模型 vLLM 服务（端口 8001）
2. 启动待评估的 World Model vLLM 服务（端口 8000）
3. 运行评估：
   python eval_behavior_consistency.py \
       --test-file /path/to/webshop_test_109.json \
       --wm-port 8000 \
       --ref-agent-port 8001 \
       --output-dir ./eval_results
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# 添加项目根目录到 path，以便导入 src 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 导入奖励函数组件
from src.reward.pivot_reward import (
    PivotGRPOConfig,
    HTTPReferenceAgent,
    FormatValidator,
    _compute_facts_reward,
    _compute_similarity_score,
)


# =============================================================================
# 配置
# =============================================================================
@dataclass
class EvalConfig:
    """评估配置"""
    # 测试集
    test_file: str = ""
    
    # World Model 服务
    wm_port: int = 8000
    wm_model_name: str = "llm_world_model"
    wm_max_tokens: int = 1024
    wm_temperature: float = 0.0
    
    # 裁判模型服务
    reference_agent_port: int = 8001
    reference_agent_model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # 评估参数
    reward_mode: str = "exponential"  # 奖励模式: "negative_l1", "negative_l2", "exponential"
    behavior_scale_coef: float = 10.0  # exponential 模式下的缩放系数
    max_samples: int = -1  # -1 表示全部
    max_steps_per_sample: int = -1  # -1 表示全部
    max_concurrency: int = 16
    api_timeout: float = 120.0
    
    # 输出
    output_dir: str = "./eval_results"
    save_details: bool = True


# =============================================================================
# World Model Client
# =============================================================================
class WorldModelClient:
    """World Model HTTP 客户端"""
    
    def __init__(self, port: int, model_name: str, max_tokens: int = 1024, 
                 temperature: float = 0.0, timeout: float = 120.0):
        self.api_base = f"http://localhost:{port}/v1"
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        self._initialized = False
        self._actual_model_name = None
    
    def initialize(self):
        """检查服务是否可用"""
        if self._initialized:
            return
        
        try:
            response = requests.get(f"{self.api_base}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    self._actual_model_name = models[0].get("id", self.model_name)
                    print(f"[WM Client] Connected to vLLM at port, model: {self._actual_model_name}")
                    self._initialized = True
                else:
                    raise RuntimeError("No models available on WM server")
            else:
                raise RuntimeError(f"WM server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to WM server at {self.api_base}")
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成响应"""
        self.initialize()
        
        payload = {
            "model": self._actual_model_name or self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
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


# =============================================================================
# 评估器
# =============================================================================
class BehavioralFidelityEvaluator:
    """行为一致性评估器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        
        # 初始化 World Model 客户端
        self.wm_client = WorldModelClient(
            port=config.wm_port,
            model_name=config.wm_model_name,
            max_tokens=config.wm_max_tokens,
            temperature=config.wm_temperature,
            timeout=config.api_timeout,
        )
        
        # 初始化裁判模型
        reference_agent_config = PivotGRPOConfig(
            reward_mode=config.reward_mode,
            behavior_scale_coef=config.behavior_scale_coef,
            reference_agent_model_path=config.reference_agent_model_path,
            reference_agent_api_url=f"http://localhost:{config.reference_agent_port}",
            api_timeout=config.api_timeout,
        )
        self.ref_agent = HTTPReferenceAgent(reference_agent_config)
        self.reward_mode = config.reward_mode
        self.behavior_scale_coef = config.behavior_scale_coef
        self.validator = FormatValidator()
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        with open(self.config.test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if self.config.max_samples > 0:
            data = data[:self.config.max_samples]
        
        print(f"[Eval] Loaded {len(data)} samples from {self.config.test_file}")
        return data
    
    def extract_steps(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从样本中提取评估步骤
        
        每个 step 包含：
        - context: 历史对话（用于 WM 生成）
        - action: 当前动作（user 输入到 WM）
        - real_state: 真实状态（assistant 响应）
        - next_action: 下一个动作（用于计算 behavior consistency）
        """
        messages = sample.get("messages", [])
        steps = []
        
        # 找到 system message 作为初始 context
        context = []
        i = 0
        
        # 处理 system message
        if messages and messages[0]["role"] == "system":
            context.append(messages[0])
            i = 1
        
        # 遍历对话
        while i < len(messages) - 2:  # 需要至少 user, assistant, user
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                action = messages[i]["content"]  # 当前动作
                real_state = messages[i+1]["content"]  # 真实状态
                
                # 检查是否有下一个动作（用于计算 behavior consistency）
                if i + 2 < len(messages) and messages[i+2]["role"] == "user":
                    next_action = messages[i+2]["content"]
                    
                    steps.append({
                        "context": context.copy(),
                        "action": action,
                        "real_state": real_state,
                        "next_action": next_action,
                        "step_idx": len(steps),
                    })
                
                # 更新 context：添加当前轮次
                context.append({"role": "user", "content": action})
                context.append({"role": "assistant", "content": real_state})
            
            i += 1
        
        # 限制步数
        if self.config.max_steps_per_sample > 0:
            steps = steps[:self.config.max_steps_per_sample]
        
        return steps
    
    def evaluate_step(
        self,
        context: List[Dict[str, str]],
        action: str,
        real_state: str,
        next_action: str,
    ) -> Dict[str, Any]:
        """
        评估单个步骤的行为一致性
        
        Returns:
            包含评估结果的字典
        """
        result = {
            "success": False,
            "error": None,
            "predicted_state": "",
            "real_state": real_state[:], # 默认[:200]
            "next_action": next_action[:],  # 默认[:100]
            "format_valid": False,
            "behavior_reward": 0.0,
            "mean_diff": 0.0,
            "mean_log_prob_pred": 0.0,
            "mean_log_prob_real": 0.0,
            "token_count_pred": 0,
            "token_count_real": 0,
            "facts_reward": 0.0,
            "reward_mode": self.reward_mode,
            "behavior_scale_coef": self.behavior_scale_coef,
        }
        
        try:
            # 1. 使用 WM 生成预测状态
            wm_messages = context + [{"role": "user", "content": action}]
            predicted_state = self.wm_client.generate(wm_messages)
            result["predicted_state"] = predicted_state[:]   # 默认[:200]
            
            # 2. 格式验证
            is_valid, reason = self.validator.validate(predicted_state)
            result["format_valid"] = is_valid
            result["format_reason"] = reason
            
            if not is_valid:
                result["error"] = f"Format invalid: {reason}"
                return result
            
            # 3. 计算 Behavior Consistency
            fidelity_result = self.ref_agent.compute_behavior_consistency_reward(
                predicted_state=predicted_state,
                real_state=real_state,
                expert_action=next_action,
                system_prompt=None,
            )
            
            result["behavior_reward"] = fidelity_result["score"]
            result["mean_diff"] = fidelity_result["mean_diff"]
            result["mean_log_prob_pred"] = fidelity_result["mean_log_prob_pred"]
            result["mean_log_prob_real"] = fidelity_result["mean_log_prob_real"]
            result["token_count_pred"] = fidelity_result["token_count_pred"]
            result["token_count_real"] = fidelity_result["token_count_real"]
            
            # 4. 计算 Facts Reward
            facts_result = _compute_facts_reward(predicted_state, real_state)
            result["facts_reward"] = facts_result["facts_reward"]
            result["asin_match"] = facts_result.get("asin_match", 0.0)
            result["price_match"] = facts_result.get("price_match", 0.0)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
    ) -> Dict[str, Any]:
        """评估单个样本的所有步骤"""
        steps = self.extract_steps(sample)
        
        sample_result = {
            "sample_idx": sample_idx,
            "num_steps": len(steps),
            "step_results": [],
            "avg_behavior_reward": 0.0,
            "avg_facts_reward": 0.0,
            "avg_mean_diff": 0.0,
            "format_valid_rate": 0.0,
            "success_rate": 0.0,
        }
        
        if not steps:
            return sample_result
        
        # 评估每个步骤
        behavior_rewards = []
        facts_rewards = []
        mean_diffs = []
        format_valid_count = 0
        success_count = 0
        
        for step in steps:
            step_result = self.evaluate_step(
                context=step["context"],
                action=step["action"],
                real_state=step["real_state"],
                next_action=step["next_action"],
            )
            step_result["step_idx"] = step["step_idx"]
            sample_result["step_results"].append(step_result)
            
            if step_result["success"]:
                success_count += 1
                behavior_rewards.append(step_result["behavior_reward"])
                facts_rewards.append(step_result["facts_reward"])
                mean_diffs.append(step_result["mean_diff"])
            
            if step_result["format_valid"]:
                format_valid_count += 1
        
        # 计算平均值
        if behavior_rewards:
            sample_result["avg_behavior_reward"] = sum(behavior_rewards) / len(behavior_rewards)
            sample_result["avg_facts_reward"] = sum(facts_rewards) / len(facts_rewards)
            sample_result["avg_mean_diff"] = sum(mean_diffs) / len(mean_diffs)
        
        sample_result["format_valid_rate"] = format_valid_count / len(steps) if steps else 0.0
        sample_result["success_rate"] = success_count / len(steps) if steps else 0.0
        
        return sample_result
    
    def run(self) -> Dict[str, Any]:
        """运行评估"""
        # 加载数据
        test_data = self.load_test_data()
        
        # 初始化服务
        print("[Eval] Initializing WM client...")
        self.wm_client.initialize()
        print("[Eval] Initializing Reference Agent model...")
        self.ref_agent.initialize()
        
        # 统计
        all_results = [None] * len(test_data)  # 预分配保持顺序
        all_behavior_rewards = []
        all_facts_rewards = []
        all_mean_diffs = []
        total_steps = 0
        success_steps = 0
        format_valid_steps = 0
        
        # 并发评估
        print(f"[Eval] Evaluating {len(test_data)} samples with concurrency={self.config.max_concurrency}...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self.evaluate_sample, sample, idx): idx
                for idx, sample in enumerate(test_data)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                sample_idx = futures[future]
                try:
                    sample_result = future.result()
                except Exception as e:
                    sample_result = {"sample_idx": sample_idx, "error": str(e), "step_results": []}
                all_results[sample_idx] = sample_result
        
        # 汇总统计
        for sample_result in all_results:
            if sample_result is None:
                continue
            for step_result in sample_result.get("step_results", []):
                total_steps += 1
                if step_result["success"]:
                    success_steps += 1
                    all_behavior_rewards.append(step_result["behavior_reward"])
                    all_facts_rewards.append(step_result["facts_reward"])
                    all_mean_diffs.append(step_result["mean_diff"])
                if step_result["format_valid"]:
                    format_valid_steps += 1
        
        # 汇总结果
        summary = {
            "config": {
                "test_file": self.config.test_file,
                "wm_port": self.config.wm_port,
                "reference_agent_port": self.config.reference_agent_port,
                "reward_mode": self.config.reward_mode,
                "behavior_scale_coef": self.config.behavior_scale_coef,
                "num_samples": len(test_data),
            },
            "metrics": {
                "total_steps": total_steps,
                "success_steps": success_steps,
                "format_valid_steps": format_valid_steps,
                "success_rate": success_steps / total_steps if total_steps > 0 else 0.0,
                "format_valid_rate": format_valid_steps / total_steps if total_steps > 0 else 0.0,
            }
        }
        
        if all_behavior_rewards:
            summary["metrics"].update({
                "avg_behavior_reward": sum(all_behavior_rewards) / len(all_behavior_rewards),
                "avg_facts_reward": sum(all_facts_rewards) / len(all_facts_rewards),
                "avg_mean_diff": sum(all_mean_diffs) / len(all_mean_diffs),
                "min_behavior_reward": min(all_behavior_rewards),
                "max_behavior_reward": max(all_behavior_rewards),
                "std_behavior_reward": self._std(all_behavior_rewards),
            })
        
        # 保存结果
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存 summary
        summary_file = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Summary saved to {summary_file}")
        
        # 保存详细结果
        if self.config.save_details:
            details_file = os.path.join(self.config.output_dir, "details.json")
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[Eval] Details saved to {details_file}")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("Behavior Consistency Evaluation Summary")
        print("=" * 60)
        print(f"Total samples: {len(test_data)}")
        print(f"Total steps: {total_steps}")
        print(f"Success steps: {success_steps} ({summary['metrics']['success_rate']*100:.2f}%)")
        print(f"Format valid: {format_valid_steps} ({summary['metrics']['format_valid_rate']*100:.2f}%)")
        if all_behavior_rewards:
            mode_info = f"{self.config.reward_mode} mode"
            if self.config.reward_mode == "exponential":
                mode_info += f", coef={self.config.behavior_scale_coef}"
            print(f"\nBehavior Consistency ({mode_info}, higher is better):")
            print(f"  Mean: {summary['metrics']['avg_behavior_reward']:.4f}")
            print(f"  Std:  {summary['metrics']['std_behavior_reward']:.4f}")
            print(f"  Min:  {summary['metrics']['min_behavior_reward']:.4f}")
            print(f"  Max:  {summary['metrics']['max_behavior_reward']:.4f}")
            print(f"\nMean Log Prob Diff (closer to 0 is better):")
            print(f"  Mean: {summary['metrics']['avg_mean_diff']:.4f}")
            print(f"\nFacts Reward (higher is better):")
            print(f"  Mean: {summary['metrics']['avg_facts_reward']:.4f}")
        print("=" * 60)
        
        return summary
    
    @staticmethod
    def _std(values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Behavior Consistency Evaluation for World Models"
    )
    
    # 测试集
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/llama_factory/webshop_test_109.json",
        help="Path to test data file",
    )
    
    # World Model 服务
    parser.add_argument("--wm-port", type=int, default=8000, help="WM vLLM server port")
    parser.add_argument("--wm-model-name", type=str, default="llm_world_model", help="WM model name")
    parser.add_argument("--wm-max-tokens", type=int, default=1024, help="WM max generation tokens")
    parser.add_argument("--wm-temperature", type=float, default=0.0, help="WM generation temperature")
    
    # 裁判模型服务
    parser.add_argument("--ref-agent-port", type=int, default=8001, help="Reference Agent vLLM server port")
    parser.add_argument(
        "--ref-agent-model-path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Reference Agent model path (for tokenizer)",
    )
    
    # 评估参数
    parser.add_argument("--reward-mode", type=str, default="exponential", 
                       choices=["negative_l1", "negative_l2", "exponential"], help="Reward mode")
    parser.add_argument("--behavior-scale-coef", type=float, default=10.0,
                       help="Scaling coefficient for exponential reward mode (default: 10.0)")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps per sample (-1 for all)")
    parser.add_argument("--api-timeout", type=float, default=120.0, help="API timeout in seconds")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--no-details", action="store_true", help="Don't save detailed results")
    
    args = parser.parse_args()
    
    # 构建配置
    config = EvalConfig(
        test_file=args.test_file,
        wm_port=args.wm_port,
        wm_model_name=args.wm_model_name,
        wm_max_tokens=args.wm_max_tokens,
        wm_temperature=args.wm_temperature,
        reference_agent_port=args.reference_agent_port,
        reference_agent_model_path=args.reference_agent_model_path,
        reward_mode=args.reward_mode,
        behavior_scale_coef=args.behavior_scale_coef,
        max_samples=args.max_samples,
        max_steps_per_sample=args.max_steps,
        api_timeout=args.api_timeout,
        output_dir=args.output_dir,
        save_details=not args.no_details,
    )
    
    # 运行评估
    evaluator = BehavioralFidelityEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
