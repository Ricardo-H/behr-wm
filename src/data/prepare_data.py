#!/usr/bin/env python3
"""
WebShop Data Preparation Pipeline for Pivot-GRPO Training

功能：
1. 读取原始 WebShop JSON 轨迹数据
2. 转换为 VERL 兼容的 Parquet 格式
3. 生成不同规模的数据集子集（用于 Scaling Law 实验）

输入:
- webshop_train_70790.json (训练集)
- webshop_test_109.json (测试集)

输出目录结构:
    pivot_grpo/webshop/data/
    ├── train/
    │   ├── debug.parquet      (1k samples)
    │   ├── tiny.parquet       (10k samples)
    │   ├── small.parquet      (50k samples)
    │   ├── medium.parquet     (200k samples)
    │   └── full.parquet       (all samples)
    └── test/
        └── test.parquet       (all test samples)

使用方法:
    python prepare_data.py [--source_dir PATH] [--output_dir PATH]
"""

import os
import re
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Iterator, Optional
import pyarrow as pa
import pyarrow.parquet as pq


# ============ 配置 ============

# 数据集规模配置（包含关系）
TRAIN_SPLITS_CONFIG = {
    "debug": 1000,      # ~1k samples, 代码调试
    "tiny": 10000,      # ~10k samples, 快速实验
    "small": 50000,     # ~50k samples, 核心实验
    "medium": 200000,   # ~200k samples, Scaling 验证
    # "full" 会包含所有样本
}


# ============ 数据读写工具 ============

def read_json(file_path: str) -> List[Dict]:
    """读取 JSON 文件"""
    print(f"  Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} trajectories")
    return data


def write_parquet(records: List[Dict], file_path: str, compression: str = "snappy", row_group_size: int = 10000):
    """
    写入 Parquet 文件
    
    使用较小的 row_group_size 避免大数据集的 chunked array 问题
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    
    if len(records) == 0:
        print(f"  Warning: No records to write to {file_path}")
        return
    
    # 分批写入以处理大数据集
    if len(records) > row_group_size:
        writer = None
        n_batches = (len(records) + row_group_size - 1) // row_group_size
        for i in range(0, len(records), row_group_size):
            batch = records[i:i + row_group_size]
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(file_path, table.schema, compression=compression)
            writer.write_table(table)
        if writer:
            writer.close()
    else:
        table = pa.Table.from_pylist(records)
        pq.write_table(table, file_path, compression=compression, row_group_size=row_group_size)
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Written {len(records):,} samples to {file_path} ({file_size_mb:.2f} MB)")


# ============ 核心逻辑：WebShop 数据适配 ============

def generate_uid(traj_idx: int, step_idx: int, split: str = "train") -> str:
    """生成唯一的样本 ID"""
    return f"{split}_traj_{traj_idx:06d}_step_{step_idx:03d}"


def parse_webshop_state(state: str) -> tuple:
    """
    解析 WebShop 状态
    
    WebShop 结束特征:
    - 包含 "Reward [SEP]" 或 "Thank you for shopping"
    - 示例: "... [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 0.66 [SEP] ..."
    
    Returns:
        (full_state, reward_score, is_terminal)
    """
    is_terminal = False
    reward = 0.0
    
    # WebShop 的标准结束标志
    if "Reward [SEP]" in state or "Thank you for shopping" in state:
        is_terminal = True
        
        # 提取分数
        score_pattern = r"Your score \(min 0.0, max 1.0\) \[SEP\]\s*([0-9\.]+)"
        match = re.search(score_pattern, state)
        if match:
            try:
                reward = float(match.group(1))
            except ValueError:
                reward = 0.0
    
    return state, reward, is_terminal


def extract_single_step_samples(
    trajectory: Dict,
    traj_idx: int,
    split: str = "train",
    skip_last_step: bool = False,
    max_steps: int = -1
) -> List[Dict]:
    """
    从单条轨迹中提取单步样本
    
    每个样本格式:
    - prompt: 包含历史对话的消息列表 (System + User Actions + Assistant States)
    - ground_truth: 真实的下一状态
    - expert_action: 专家在该状态下的动作（用于决策一致性奖励）
    """
    messages = trajectory["messages"]
    samples = []
    
    n_messages = len(messages)
    n_total_steps = (n_messages - 1) // 2
    
    step_idx = 0
    # 遍历对话历史 (User -> Assistant)
    for i in range(1, n_messages - 1, 2):
        if messages[i]["role"] != "user" or messages[i + 1]["role"] != "assistant":
            continue
        
        # Prompt: System + ... + User Action (包含当前动作 A_t)
        prompt_messages = messages[:i + 1]
        
        # Ground Truth: 真实的 Assistant 回复 (S_{t+1})
        ground_truth_state_raw = messages[i + 1]["content"]
        
        # 解析 WebShop 状态
        ground_truth_state, reward, is_terminal = parse_webshop_state(ground_truth_state_raw)
        
        # 获取下一步的 Expert Action (A_{t+1})
        if i + 2 < n_messages and messages[i + 2]["role"] == "user":
            expert_action = messages[i + 2]["content"]
        else:
            expert_action = "[END_OF_EPISODE]"
        
        # 跳过最后一步选项 (如果不需要预测结束状态)
        if skip_last_step and expert_action == "[END_OF_EPISODE]":
            continue
        
        uid = generate_uid(traj_idx, step_idx, split)
        
        sample = {
            "uid": uid,
            "traj_idx": traj_idx,
            "step_idx": step_idx,
            "prompt_messages": prompt_messages,
            "ground_truth_state": ground_truth_state,
            "expert_action": expert_action,
            "reward": reward,
            "is_terminal": is_terminal,
            "n_total_steps": n_total_steps,
        }
        
        samples.append(sample)
        step_idx += 1
        
        if max_steps > 0 and step_idx >= max_steps:
            break
    
    return samples


def convert_to_verl_format(sample: Dict) -> Dict:
    """
    转换为 VERL 兼容的格式
    
    VERL 格式要求:
    - prompt: List[Dict] - 消息列表 [{"role": "user/assistant", "content": "..."}]
    - data_source: str - 数据来源标识
    - reward_model: Dict - 奖励模型信息 {"ground_truth": "...", "style": "rule"}
    - extra_info: Dict - 额外信息（传递给 reward function）
    """
    prompt = sample["prompt_messages"]
    
    # 构造 Reward Model 字段
    reward_model = {
        "ground_truth": sample["ground_truth_state"],
        "style": "rule"
    }
    
    # 构造 Extra Info（传递给 reward_function.py 中的 compute_score）
    extra_info = {
        "index": sample.get("step_idx", 0),
        "item_id": sample.get("uid", ""),
        "expert_action": sample.get("expert_action", ""),
        "is_terminal": sample.get("is_terminal", False),
        "env_reward": sample.get("reward", 0),
        "traj_idx": sample.get("traj_idx", 0),
    }

    return {
        "prompt": prompt,
        "data_source": "webshop_grpo",
        "reward_model": reward_model,
        "extra_info": extra_info,
        "item_id": sample.get("uid", "")
    }


def process_trajectories(
    trajectories: List[Dict],
    split: str = "train",
    skip_last_step: bool = False,
    max_trajs: int = -1,
    max_samples: int = -1,
    max_steps_per_traj: int = -1,
) -> List[Dict]:
    """处理轨迹数据并转换为 VERL 格式"""
    if max_trajs > 0:
        trajectories = trajectories[:max_trajs]
        print(f"  Using first {max_trajs} trajectories")
    
    all_samples = []
    for traj_idx, traj in enumerate(tqdm(trajectories, desc=f"  Extracting {split} samples", leave=False)):
        samples = extract_single_step_samples(
            traj, 
            traj_idx, 
            split=split,
            skip_last_step=skip_last_step,
            max_steps=max_steps_per_traj
        )
        all_samples.extend(samples)
    
    print(f"  Extracted {len(all_samples):,} single-step samples")
    
    if max_samples > 0:
        all_samples = all_samples[:max_samples]
        print(f"  Truncated to first {max_samples:,} samples")
    
    verl_samples = []
    for sample in tqdm(all_samples, desc=f"  Converting to VERL format", leave=False):
        verl_samples.append(convert_to_verl_format(sample))
    
    return verl_samples


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="WebShop Data Preparation for GRPO")
    
    # 默认路径（相对于脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(script_dir, "../../Word2World/data/llama_factory")
    default_output = os.path.join(script_dir, "data")
    
    parser.add_argument("--source_dir", type=str, default=default_source,
                        help="Directory containing source JSON files")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help="Output directory for Parquet files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--skip-last-step", action="store_true",
                        help="Skip the last step of each trajectory")
    
    args = parser.parse_args()
    
    # 规范化路径
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    train_input = os.path.join(source_dir, "webshop_train_70790.json")
    test_input = os.path.join(source_dir, "webshop_test_109.json")
    
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "test")
    
    print("=" * 70)
    print("WebShop Data Preparation Pipeline for Pivot-GRPO")
    print("=" * 70)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # 检查输入文件
    if not os.path.exists(train_input):
        print(f"Error: Training data not found: {train_input}")
        return 1
    if not os.path.exists(test_input):
        print(f"Error: Test data not found: {test_input}")
        return 1
    
    # 创建输出目录
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # ============ 处理训练数据 ============
    print("-" * 70)
    print("[1/2] Processing Training Data")
    print("-" * 70)
    
    train_trajectories = read_json(train_input)
    train_samples = process_trajectories(
        train_trajectories,
        split="train",
        skip_last_step=args.skip_last_step
    )
    
    total_train_samples = len(train_samples)
    print(f"\n  Total training samples: {total_train_samples:,}")
    
    # 打乱数据
    print(f"  Shuffling with seed {args.seed}...")
    np.random.seed(args.seed)
    np.random.shuffle(train_samples)
    
    # 生成不同规模的子集
    print("\n  Generating training splits:")
    train_results = []
    
    for name, size in TRAIN_SPLITS_CONFIG.items():
        if size > total_train_samples:
            print(f"    ⚠️  Skipping {name} (size {size:,} > total {total_train_samples:,})")
            continue
        
        output_path = os.path.join(train_output_dir, f"{name}.parquet")
        subset = train_samples[:size]
        write_parquet(subset, output_path)
        
        train_results.append({
            "name": name,
            "size": size,
            "path": output_path
        })
    
    # 保存完整训练集
    full_output_path = os.path.join(train_output_dir, "full.parquet")
    write_parquet(train_samples, full_output_path)
    train_results.append({
        "name": "full",
        "size": total_train_samples,
        "path": full_output_path
    })
    
    # ============ 处理测试数据 ============
    print()
    print("-" * 70)
    print("[2/2] Processing Test Data")
    print("-" * 70)
    
    test_trajectories = read_json(test_input)
    test_samples = process_trajectories(
        test_trajectories,
        split="test",
        skip_last_step=args.skip_last_step
    )
    
    total_test_samples = len(test_samples)
    print(f"\n  Total test samples: {total_test_samples:,}")
    
    # 保存测试集
    test_output_path = os.path.join(test_output_dir, "test.parquet")
    write_parquet(test_samples, test_output_path)
    
    # ============ 输出总结 ============
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    print("\nTraining Splits:")
    print(f"  {'Split':<10} {'Samples':>12} {'Path':<50}")
    print("  " + "-" * 65)
    for r in train_results:
        print(f"  {r['name']:<10} {r['size']:>12,} {r['path']:<50}")
    
    print("\nTest Data:")
    print(f"  {'test':<10} {total_test_samples:>12,} {test_output_path:<50}")
    
    # 包含关系说明
    print("\n" + "-" * 70)
    print("Data Inclusion Relationship (for controlled experiments):")
    names = [r['name'] for r in train_results]
    print("  " + " ⊂ ".join(names))
    
    # 使用说明
    print("\n" + "-" * 70)
    print("Usage with run_grpo.sh:")
    print("  1. Debug:   bash run_grpo.sh debug   # 快速验证 pipeline")
    print("  2. Tiny:    bash run_grpo.sh tiny    # 快速超参搜索")
    print("  3. Small:   bash run_grpo.sh small   # 核心实验")
    print("  4. Medium:  bash run_grpo.sh medium  # Scaling 验证")
    print("  5. Full:    bash run_grpo.sh full    # 最终模型")
    
    # 保存元数据
    metadata = {
        "seed": args.seed,
        "source_files": {
            "train": train_input,
            "test": test_input
        },
        "total_samples": {
            "train": total_train_samples,
            "test": total_test_samples
        },
        "train_splits": {r["name"]: r["size"] for r in train_results},
        "inclusion_order": names
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")
    
    # 样本预览
    if len(train_samples) > 0:
        print("\n" + "=" * 70)
        print("Sample Preview (first training sample)")
        print("=" * 70)
        s = train_samples[0]
        print(f"ID: {s['item_id']}")
        print(f"Data Source: {s['data_source']}")
        print(f"Prompt (last user msg): {s['prompt'][-1]['content'][:100]}...")
        print(f"Ground Truth (preview): {s['reward_model']['ground_truth'][:100]}...")
        print(f"Expert Action: {s['extra_info']['expert_action'][:80]}...")
        print(f"Is Terminal: {s['extra_info']['is_terminal']}")
        print(f"Env Reward: {s['extra_info']['env_reward']}")
    
    print("\n✅ Data preparation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
