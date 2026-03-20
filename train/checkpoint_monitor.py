#!/usr/bin/env python3
"""
Checkpoint Monitor - 在保存 checkpoint 时输出详细的训练监控信息

功能：
1. 记录 WM 的输入与输出
2. 记录每个样本的两种 reward（behavior_reward 和 facts_reward）
3. 记录整个 batch 的 reward 分布统计
4. 记录行为一致性奖励的原始数据（真实环境和预测环境下 Agent 对专家动作的输出概率）
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


def compute_reward_distribution(values: np.ndarray) -> Dict[str, float]:
    """计算奖励分布统计"""
    if values is None or len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def _count_true_values(values) -> int:
    """
    安全计算布尔数组/列表中 True 的数量
    
    处理多种情况:
    - numpy array (bool, int, object dtype)
    - Python list
    - None 或空值
    """
    if values is None:
        return 0
    
    # 转换为 numpy array 以统一处理
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    
    if values.size == 0:
        return 0
    
    # 处理 object dtype (可能包含字符串 "True"/"False" 或混合类型)
    if values.dtype == object:
        count = 0
        for v in values.flat:
            if isinstance(v, bool):
                count += int(v)
            elif isinstance(v, str):
                count += int(v.lower() in ('true', '1', 'yes'))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                count += int(bool(v))
            elif isinstance(v, np.ndarray):
                # 嵌套数组，递归处理
                count += _count_true_values(v)
            else:
                count += int(bool(v))
        return count
    
    # 数值/布尔 dtype
    return int(np.sum(values.astype(bool)))


def _count_false_values(values) -> int:
    """
    安全计算布尔数组/列表中 False 的数量
    """
    if values is None:
        return 0
    
    # 转换为 numpy array 以统一处理
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    
    if values.size == 0:
        return 0
    
    total = values.size if values.dtype != object else len(list(values.flat))
    return total - _count_true_values(values)


def _safe_bool(value) -> bool:
    """
    安全转换单个值为布尔类型
    
    处理多种情况:
    - numpy array (取第一个元素或 any())
    - numpy scalar (np.bool_, np.int64 等)
    - Python 原生类型
    - 字符串 "True"/"False"
    """
    if value is None:
        return False
    
    # numpy array
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        if value.size == 1:
            return bool(value.flat[0])
        # 多元素数组，使用 any()
        return bool(value.any())
    
    # numpy scalar
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return bool(value)
    
    # 字符串
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    
    # Python 原生类型
    return bool(value)


def select_representative_samples(scores: np.ndarray) -> List[Dict[str, Any]]:
    """
    选择代表性样本：max, min, median
    
    Args:
        scores: 分数数组
    
    Returns:
        [(index, score, type), ...] - 选中的样本索引、分数和类型
    """
    if scores is None or len(scores) == 0:
        return []
    
    n = len(scores)
    selected = []
    
    # Max sample
    max_idx = int(np.argmax(scores))
    selected.append({"index": max_idx, "score": float(scores[max_idx]), "type": "MAX"})
    
    # Min sample
    min_idx = int(np.argmin(scores))
    if min_idx != max_idx:  # 避免重复
        selected.append({"index": min_idx, "score": float(scores[min_idx]), "type": "MIN"})
    
    # Median sample - 找最接近中位数的样本
    median_val = np.median(scores)
    # 排除已选的 max 和 min
    excluded = {max_idx, min_idx}
    remaining_indices = [i for i in range(n) if i not in excluded]
    
    if remaining_indices:
        # 找最接近中位数的
        distances = [(i, abs(scores[i] - median_val)) for i in remaining_indices]
        distances.sort(key=lambda x: x[1])
        median_idx = distances[0][0]
        selected.append({"index": median_idx, "score": float(scores[median_idx]), "type": "MEDIAN"})
    
    return selected


def log_checkpoint_monitor(
    global_step: int,
    batch,  # DataProto
    reward_extra_infos_dict: Dict[str, Any],
    tokenizer,
    output_dir: str,
    max_samples: int = 10,  # 最多输出多少个样本的详细信息
):
    """
    在保存 checkpoint 时记录详细的训练监控信息
    
    Args:
        global_step: 当前全局步数
        batch: DataProto 对象，包含 prompts, responses 等
        reward_extra_infos_dict: 奖励额外信息字典
        tokenizer: 分词器，用于解码
        output_dir: 输出目录
        max_samples: 最多输出多少个样本的详细信息
    """
    
    # 创建输出目录
    monitor_dir = os.path.join(output_dir, "checkpoint_monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    
    # 输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(monitor_dir, f"step_{global_step}_{timestamp}.json")
    
    # =========================================================================
    # 1. 解码输入输出
    # =========================================================================
    prompts = batch.batch.get("prompts", None)
    responses = batch.batch.get("responses", None)
    
    inputs = []
    outputs = []
    
    if prompts is not None and tokenizer is not None:
        try:
            inputs = tokenizer.batch_decode(prompts, skip_special_tokens=True)
        except Exception as e:
            print(f"[Monitor] Failed to decode prompts: {e}")
            inputs = [f"<decode_error: {e}>"] * len(prompts)
    
    if responses is not None and tokenizer is not None:
        try:
            outputs = tokenizer.batch_decode(responses, skip_special_tokens=True)
        except Exception as e:
            print(f"[Monitor] Failed to decode responses: {e}")
            outputs = [f"<decode_error: {e}>"] * len(responses)
    
    # =========================================================================
    # 2. 提取奖励信息
    # =========================================================================
    n_samples = len(inputs) if inputs else 0
    
    # 从 reward_extra_infos_dict 提取
    behavior_rewards = reward_extra_infos_dict.get("behavior_reward", [])
    facts_rewards = reward_extra_infos_dict.get("facts_reward", [])
    total_scores = reward_extra_infos_dict.get("score", [])
    
    # 行为一致性奖励的原始数据
    mean_log_prob_preds = reward_extra_infos_dict.get("mean_log_prob_pred", [])
    mean_log_prob_reals = reward_extra_infos_dict.get("mean_log_prob_real", [])
    mean_diffs = reward_extra_infos_dict.get("mean_diff", [])
    token_count_preds = reward_extra_infos_dict.get("token_count_pred", [])
    token_count_reals = reward_extra_infos_dict.get("token_count_real", [])
    
    # 格式验证信息
    format_valids = reward_extra_infos_dict.get("format_valid", [])
    format_reasons = reward_extra_infos_dict.get("format_reason", [])
    
    # 物理事实匹配详情
    asin_matches = reward_extra_infos_dict.get("asin_match", [])
    price_matches = reward_extra_infos_dict.get("price_match", [])
    page_matches = reward_extra_infos_dict.get("page_match", [])
    rating_matches = reward_extra_infos_dict.get("rating_match", [])
    
    # 回退相似度
    fallback_similarities = reward_extra_infos_dict.get("fallback_similarity", [])

    # 长度监控
    pred_lengths = reward_extra_infos_dict.get("pred_length", [])
    real_lengths = reward_extra_infos_dict.get("real_length", [])
    length_ratios = reward_extra_infos_dict.get("length_ratio", [])
    length_penalties = reward_extra_infos_dict.get("length_penalty", [])
    length_statuses = reward_extra_infos_dict.get("length_status", [])
    
    # 从 batch.non_tensor_batch 提取 ground_truth 和 expert_action
    ground_truths = []
    expert_actions = []
    
    if hasattr(batch, 'non_tensor_batch'):
        # 尝试从 non_tensor_batch 获取
        if "reward_model" in batch.non_tensor_batch:
            reward_model_data = batch.non_tensor_batch["reward_model"]
            if isinstance(reward_model_data, dict):
                ground_truths = [reward_model_data.get("ground_truth", "")] * n_samples
            elif isinstance(reward_model_data, (list, np.ndarray)):
                ground_truths = [
                    item.get("ground_truth", "") if isinstance(item, dict) else str(item)
                    for item in reward_model_data
                ]
        
        if "extra_info" in batch.non_tensor_batch:
            extra_infos = batch.non_tensor_batch["extra_info"]
            if isinstance(extra_infos, (list, np.ndarray)):
                expert_actions = [
                    item.get("expert_action", "") if isinstance(item, dict) else ""
                    for item in extra_infos
                ]
    
    # 转换为 numpy array 以便计算统计
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            return np.array(x, dtype=float)
        return np.array([])
    
    behavior_rewards_np = to_numpy(behavior_rewards)
    facts_rewards_np = to_numpy(facts_rewards)
    total_scores_np = to_numpy(total_scores)
    mean_log_prob_preds_np = to_numpy(mean_log_prob_preds)
    mean_log_prob_reals_np = to_numpy(mean_log_prob_reals)
    length_ratios_np = to_numpy(length_ratios)
    length_penalties_np = to_numpy(length_penalties)
    pred_lengths_np = to_numpy(pred_lengths)
    real_lengths_np = to_numpy(real_lengths)
    
    # =========================================================================
    # 3. 计算 Batch 级别的奖励分布
    # =========================================================================
    batch_stats = {
        "global_step": global_step,
        "timestamp": timestamp,
        "n_samples": n_samples,

        # 总分数分布
        "total_score": compute_reward_distribution(total_scores_np),

        # 行为奖励分布
        "behavior_reward": compute_reward_distribution(behavior_rewards_np),

        # 事实奖励分布
        "facts_reward": compute_reward_distribution(facts_rewards_np),

        # 行为一致性原始数据分布
        "mean_log_prob_pred": compute_reward_distribution(mean_log_prob_preds_np),
        "mean_log_prob_real": compute_reward_distribution(mean_log_prob_reals_np),

        # 专家动作概率分布 (exp(mean_log_prob) = 平均每个 token 的预测概率)
        "avg_token_prob_pred": compute_reward_distribution(
            np.exp(mean_log_prob_preds_np) if mean_log_prob_preds_np.size > 0 and np.any(mean_log_prob_preds_np != 0)
            else np.array([])
        ),
        "avg_token_prob_real": compute_reward_distribution(
            np.exp(mean_log_prob_reals_np) if mean_log_prob_reals_np.size > 0 and np.any(mean_log_prob_reals_np != 0)
            else np.array([])
        ),

        # 长度监控
        "length_ratio": compute_reward_distribution(length_ratios_np),
        "length_penalty": compute_reward_distribution(length_penalties_np),
        "pred_length": compute_reward_distribution(pred_lengths_np),
        "real_length": compute_reward_distribution(real_lengths_np),
        "length_too_short_count": int(np.sum(length_statuses == "too_short")) if len(length_statuses) > 0 else 0,
        "length_too_long_count": int(np.sum(length_statuses == "too_long")) if len(length_statuses) > 0 else 0,
        "length_ok_count": int(np.sum(length_statuses == "ok")) if len(length_statuses) > 0 else 0,

        # 格式验证统计
        # 注意: format_valids 可能是 numpy array，需要正确处理布尔判断
        "format_valid_count": _count_true_values(format_valids),
        "format_invalid_count": _count_false_values(format_valids),
    }
    
    # =========================================================================
    # 4. 输出样本级别的详细信息
    # =========================================================================
    sample_details = []
    
    # 选择代表性样本：max, min, median（基于 total_score）
    representative_samples = select_representative_samples(total_scores_np)
    representative_indices = {s["index"]: s["type"] for s in representative_samples}
    
    # 收集所有样本的详细信息（用于 JSON 存储）
    for i in range(min(n_samples, max_samples)):
        sample = {
            "index": i,
            "sample_type": representative_indices.get(i, ""),  # MAX, MIN, MEDIAN 或空
            
            # WM 输入输出
            "input": inputs[i] if i < len(inputs) else "",
            "output": outputs[i] if i < len(outputs) else "",
            
            # Ground truth 和 Expert action
            "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
            "expert_action": expert_actions[i] if i < len(expert_actions) else "",
            
            # 奖励
            "total_score": float(total_scores[i]) if i < len(total_scores) else 0.0,
            "behavior_reward": float(behavior_rewards[i]) if i < len(behavior_rewards) else 0.0,
            "facts_reward": float(facts_rewards[i]) if i < len(facts_rewards) else 0.0,
            
            # 行为一致性原始数据（Agent 对专家动作的输出概率）
            "mean_log_prob_pred": float(mean_log_prob_preds[i]) if i < len(mean_log_prob_preds) else 0.0,
            "mean_log_prob_real": float(mean_log_prob_reals[i]) if i < len(mean_log_prob_reals) else 0.0,
            "mean_diff": float(mean_diffs[i]) if i < len(mean_diffs) else 0.0,
            "token_count_pred": int(token_count_preds[i]) if i < len(token_count_preds) else 0,
            "token_count_real": int(token_count_reals[i]) if i < len(token_count_reals) else 0,
            
            # 概率转换（方便理解）
            # exp(mean_log_prob) 表示平均每个 token 的预测概率
            "avg_token_prob_pred": float(np.exp(mean_log_prob_preds[i])) if i < len(mean_log_prob_preds) and mean_log_prob_preds[i] != 0 else 0.0,
            "avg_token_prob_real": float(np.exp(mean_log_prob_reals[i])) if i < len(mean_log_prob_reals) and mean_log_prob_reals[i] != 0 else 0.0,
            
            # 格式验证 (安全处理可能的 numpy array 元素)
            "format_valid": _safe_bool(format_valids[i]) if i < len(format_valids) else True,
            "format_reason": str(format_reasons[i]) if i < len(format_reasons) else "",
            
            # 物理事实匹配
            "asin_match": float(asin_matches[i]) if i < len(asin_matches) else 0.0,
            "price_match": float(price_matches[i]) if i < len(price_matches) else 0.0,
            "page_match": float(page_matches[i]) if i < len(page_matches) else 0.0,
            "rating_match": float(rating_matches[i]) if i < len(rating_matches) else 0.0,

            # 回退相似度
            "fallback_similarity": float(fallback_similarities[i]) if i < len(fallback_similarities) else 0.0,

            # 长度监控
            "pred_length": int(pred_lengths[i]) if i < len(pred_lengths) else 0,
            "real_length": int(real_lengths[i]) if i < len(real_lengths) else 0,
            "length_ratio": float(length_ratios[i]) if i < len(length_ratios) else 0.0,
            "length_penalty": float(length_penalties[i]) if i < len(length_penalties) else 0.0,
            "length_status": str(length_statuses[i]) if i < len(length_statuses) else "unknown",
        }
        sample_details.append(sample)
    
    # =========================================================================
    # 5. 组装并输出
    # =========================================================================
    monitor_data = {
        "batch_stats": batch_stats,
        "sample_details": sample_details,
    }
    
    # 写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(monitor_data, f, ensure_ascii=False, indent=2)
    
    # 同时输出一个简短的控制台摘要
    print("\n" + "=" * 80)
    print(f"[Checkpoint Monitor] Step {global_step}")
    print("=" * 80)
    print(f"Samples: {n_samples}")
    print(f"Total Score:     mean={batch_stats['total_score']['mean']:.4f}, "
          f"std={batch_stats['total_score']['std']:.4f}, "
          f"min={batch_stats['total_score']['min']:.4f}, "
          f"max={batch_stats['total_score']['max']:.4f}")
    print(f"Behavior Reward: mean={batch_stats['behavior_reward']['mean']:.4f}, "
          f"std={batch_stats['behavior_reward']['std']:.4f}, "
          f"min={batch_stats['behavior_reward']['min']:.4f}, "
          f"max={batch_stats['behavior_reward']['max']:.4f}")
    print(f"Facts Reward:    mean={batch_stats['facts_reward']['mean']:.4f}, "
          f"std={batch_stats['facts_reward']['std']:.4f}, "
          f"min={batch_stats['facts_reward']['min']:.4f}, "
          f"max={batch_stats['facts_reward']['max']:.4f}")
    print(f"Format Valid:    {batch_stats['format_valid_count']}/{n_samples} "
          f"({100*batch_stats['format_valid_count']/max(1,n_samples):.1f}%)")

    # 长度监控
    print(f"\n--- Length Monitoring ---")
    print(f"Length Ratio:    mean={batch_stats['length_ratio']['mean']:.4f}, "
          f"std={batch_stats['length_ratio']['std']:.4f}, "
          f"min={batch_stats['length_ratio']['min']:.4f}, "
          f"max={batch_stats['length_ratio']['max']:.4f}")
    print(f"Length Penalty:  mean={batch_stats['length_penalty']['mean']:.4f}, "
          f"std={batch_stats['length_penalty']['std']:.4f}")
    print(f"Length Status:   ok={batch_stats['length_ok_count']}, "
          f"too_short={batch_stats['length_too_short_count']}, "
          f"too_long={batch_stats['length_too_long_count']}")
    print(f"Pred Length:     mean={batch_stats['pred_length']['mean']:.0f}, "
          f"real_mean={batch_stats['real_length']['mean']:.0f}")

    # 行为一致性原始数据
    print(f"\n--- Behavior Consistency Raw Data ---")
    print(f"Mean Log Prob (Predicted Env): mean={batch_stats['mean_log_prob_pred']['mean']:.4f}, "
          f"std={batch_stats['mean_log_prob_pred']['std']:.4f}")
    print(f"Mean Log Prob (Real Env):      mean={batch_stats['mean_log_prob_real']['mean']:.4f}, "
          f"std={batch_stats['mean_log_prob_real']['std']:.4f}")

    # 专家动作概率分布
    print(f"\n--- Expert Action Probability Distribution ---")
    print(f"Avg Token Prob (Predicted Env): mean={batch_stats['avg_token_prob_pred']['mean']:.6f}, "
          f"std={batch_stats['avg_token_prob_pred']['std']:.6f}, "
          f"min={batch_stats['avg_token_prob_pred']['min']:.6f}, "
          f"max={batch_stats['avg_token_prob_pred']['max']:.6f}")
    print(f"Avg Token Prob (Real Env):      mean={batch_stats['avg_token_prob_real']['mean']:.6f}, "
          f"std={batch_stats['avg_token_prob_real']['std']:.6f}, "
          f"min={batch_stats['avg_token_prob_real']['min']:.6f}, "
          f"max={batch_stats['avg_token_prob_real']['max']:.6f}")
    
    # 输出代表性样本示例（MAX, MIN, MEDIAN）
    print(f"\n--- Representative Samples (MAX / MIN / MEDIAN by total_score) ---")
    
    # 按 MAX, MIN, MEDIAN 顺序输出
    for sample_info in representative_samples:
        idx = sample_info["index"]
        sample_type = sample_info["type"]
        
        # 找到对应的 sample_details
        sample = None
        for s in sample_details:
            if s["index"] == idx:
                sample = s
                break
        
        if sample is None:
            continue
        
        print(f"\n[{sample_type}] Sample #{idx}")
        print(f"  Score: {sample['total_score']:.4f} "
              f"(behavior={sample['behavior_reward']:.4f}, facts={sample['facts_reward']:.4f})")
        print(f"  Log Prob: pred={sample['mean_log_prob_pred']:.4f}, real={sample['mean_log_prob_real']:.4f}, "
              f"diff={sample['mean_diff']:.4f}")
        print(f"  Avg Token Prob: pred={sample['avg_token_prob_pred']:.4f}, real={sample['avg_token_prob_real']:.4f}")
        print(f"  Length: pred={sample.get('pred_length', 0)}, real={sample.get('real_length', 0)}, "
              f"ratio={sample.get('length_ratio', 0):.2f}, penalty={sample.get('length_penalty', 0):.4f} ({sample.get('length_status', 'unknown')})")
        print(f"  Format: {'✓' if sample['format_valid'] else '✗'} {sample['format_reason']}")
        # 截断显示输入输出
        input_preview = sample['input'][:200] + "..." if len(sample['input']) > 200 else sample['input']
        output_preview = sample['output'][:200] + "..." if len(sample['output']) > 200 else sample['output']
        print(f"  Input:  {input_preview}")
        print(f"  Output: {output_preview}")
    
    print(f"\nFull details saved to: {output_file}")
    print("=" * 80 + "\n")
    
    return monitor_data, output_file


def log_checkpoint_monitor_to_wandb(
    global_step: int,
    monitor_data: Dict[str, Any],
):
    """
    将 case study 和 reward 分布上传到 WandB 云端
    
    上传内容：
    1. Reward 分布直方图 (behavior_reward, facts_reward, total_score)
    2. 代表性样本表格 (MAX, MIN, MEDIAN)
    3. Batch 级别统计摘要
    
    Args:
        global_step: 当前全局步数
        monitor_data: log_checkpoint_monitor 返回的数据
    """
    try:
        import wandb
        
        if wandb.run is None:
            print("[Monitor] WandB not initialized, skipping upload")
            return
        
        batch_stats = monitor_data.get("batch_stats", {})
        sample_details = monitor_data.get("sample_details", [])
        n_samples = batch_stats.get("n_samples", 0)
        
        # =====================================================================
        # 1. 上传 Reward 分布直方图
        # =====================================================================
        # 从 sample_details 提取数值用于直方图
        behavior_rewards = [s["behavior_reward"] for s in sample_details if "behavior_reward" in s]
        facts_rewards = [s["facts_reward"] for s in sample_details if "facts_reward" in s]
        total_scores = [s["total_score"] for s in sample_details if "total_score" in s]
        mean_diffs = [s["mean_diff"] for s in sample_details if "mean_diff" in s and s["mean_diff"] != 0]
        length_ratios = [s["length_ratio"] for s in sample_details if "length_ratio" in s and s["length_ratio"] > 0]
        length_penalties = [s["length_penalty"] for s in sample_details if "length_penalty" in s]
        # 专家动作概率分布
        avg_token_probs_pred = [s["avg_token_prob_pred"] for s in sample_details
                                if "avg_token_prob_pred" in s and s["avg_token_prob_pred"] > 0]
        avg_token_probs_real = [s["avg_token_prob_real"] for s in sample_details
                                if "avg_token_prob_real" in s and s["avg_token_prob_real"] > 0]

        histograms = {}
        if behavior_rewards:
            histograms["checkpoint/histogram/behavior_reward"] = wandb.Histogram(behavior_rewards)
        if facts_rewards:
            histograms["checkpoint/histogram/facts_reward"] = wandb.Histogram(facts_rewards)
        if total_scores:
            histograms["checkpoint/histogram/total_score"] = wandb.Histogram(total_scores)
        if mean_diffs:
            histograms["checkpoint/histogram/mean_diff"] = wandb.Histogram(mean_diffs)
        if length_ratios:
            histograms["checkpoint/histogram/length_ratio"] = wandb.Histogram(length_ratios)
        if length_penalties:
            histograms["checkpoint/histogram/length_penalty"] = wandb.Histogram(length_penalties)
        # 专家动作概率分布直方图
        if avg_token_probs_pred:
            histograms["checkpoint/histogram/avg_token_prob_pred"] = wandb.Histogram(avg_token_probs_pred)
        if avg_token_probs_real:
            histograms["checkpoint/histogram/avg_token_prob_real"] = wandb.Histogram(avg_token_probs_real)
        
        if histograms:
            wandb.log(histograms, step=global_step)
        
        # =====================================================================
        # 2. 上传代表性样本表格 (MAX / MIN / MEDIAN)
        # =====================================================================
        # 筛选代表性样本
        representative = [s for s in sample_details if s.get("sample_type") in ("MAX", "MIN", "MEDIAN")]
        
        if representative:
            # 创建 WandB Table
            columns = [
                "Type",           # MAX, MIN, MEDIAN
                "Total Score",
                "Behavior Reward",
                "Facts Reward",
                "Length Penalty",  # 长度惩罚
                "Mean Diff",      # log prob 差异
                "Avg Token Prob (Pred)",  # 专家动作在预测环境的概率
                "Avg Token Prob (Real)",  # 专家动作在真实环境的概率
                "Length Ratio",
                "Length Status",
                "Format Valid",
                "Input (preview)",
                "Output (preview)",
            ]

            data = []
            for sample in representative:
                # 截取前 500 字符作为预览
                input_preview = sample.get("input", "")[:500]
                output_preview = sample.get("output", "")[:500]
                if len(sample.get("input", "")) > 500:
                    input_preview += "..."
                if len(sample.get("output", "")) > 500:
                    output_preview += "..."

                data.append([
                    sample.get("sample_type", ""),
                    round(sample.get("total_score", 0), 4),
                    round(sample.get("behavior_reward", 0), 4),
                    round(sample.get("facts_reward", 0), 4),
                    round(sample.get("length_penalty", 0), 4),
                    round(sample.get("mean_diff", 0), 4),
                    round(sample.get("avg_token_prob_pred", 0), 6),
                    round(sample.get("avg_token_prob_real", 0), 6),
                    round(sample.get("length_ratio", 0), 2),
                    sample.get("length_status", "unknown"),
                    "Y" if sample.get("format_valid", True) else "N",
                    input_preview,
                    output_preview,
                ])
            
            table = wandb.Table(columns=columns, data=data)
            wandb.log({f"checkpoint/case_study": table}, step=global_step)
        
        # =====================================================================
        # 3. 上传 Batch 级别统计摘要
        # =====================================================================
        summary_metrics = {
            "checkpoint/n_samples": n_samples,
            "checkpoint/format_valid_ratio": batch_stats.get("format_valid_count", 0) / max(1, n_samples),
        }

        # 各类 reward 的统计
        for key in ["total_score", "behavior_reward", "facts_reward"]:
            if key in batch_stats:
                stats = batch_stats[key]
                summary_metrics[f"checkpoint/{key}/mean"] = stats.get("mean", 0)
                summary_metrics[f"checkpoint/{key}/std"] = stats.get("std", 0)
                summary_metrics[f"checkpoint/{key}/min"] = stats.get("min", 0)
                summary_metrics[f"checkpoint/{key}/max"] = stats.get("max", 0)

        # 行为一致性原始数据
        if "mean_log_prob_pred" in batch_stats:
            summary_metrics["checkpoint/mean_log_prob_pred/mean"] = batch_stats["mean_log_prob_pred"].get("mean", 0)
            summary_metrics["checkpoint/mean_log_prob_pred/std"] = batch_stats["mean_log_prob_pred"].get("std", 0)
        if "mean_log_prob_real" in batch_stats:
            summary_metrics["checkpoint/mean_log_prob_real/mean"] = batch_stats["mean_log_prob_real"].get("mean", 0)
            summary_metrics["checkpoint/mean_log_prob_real/std"] = batch_stats["mean_log_prob_real"].get("std", 0)

        # 专家动作概率分布
        for key in ["avg_token_prob_pred", "avg_token_prob_real"]:
            if key in batch_stats:
                stats = batch_stats[key]
                summary_metrics[f"checkpoint/{key}/mean"] = stats.get("mean", 0)
                summary_metrics[f"checkpoint/{key}/std"] = stats.get("std", 0)
                summary_metrics[f"checkpoint/{key}/min"] = stats.get("min", 0)
                summary_metrics[f"checkpoint/{key}/max"] = stats.get("max", 0)
                summary_metrics[f"checkpoint/{key}/median"] = stats.get("median", 0)

        # 长度监控
        if "length_ratio" in batch_stats:
            summary_metrics["checkpoint/length_ratio/mean"] = batch_stats["length_ratio"].get("mean", 0)
            summary_metrics["checkpoint/length_ratio/std"] = batch_stats["length_ratio"].get("std", 0)
        if "length_penalty" in batch_stats:
            summary_metrics["checkpoint/length_penalty/mean"] = batch_stats["length_penalty"].get("mean", 0)
            summary_metrics["checkpoint/length_penalty/max"] = batch_stats["length_penalty"].get("max", 0)
        summary_metrics["checkpoint/length_too_short_count"] = batch_stats.get("length_too_short_count", 0)
        summary_metrics["checkpoint/length_too_long_count"] = batch_stats.get("length_too_long_count", 0)
        summary_metrics["checkpoint/length_ok_count"] = batch_stats.get("length_ok_count", 0)

        wandb.log(summary_metrics, step=global_step)
        
        print(f"[Monitor] Successfully uploaded to WandB (step {global_step})")
        
    except ImportError:
        print("[Monitor] WandB not installed, skipping upload")
    except Exception as e:
        print(f"[Monitor] Failed to upload to WandB: {e}")
        import traceback
        traceback.print_exc()
