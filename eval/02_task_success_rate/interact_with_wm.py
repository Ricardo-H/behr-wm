import os
import sys
import json
import argparse
import time
import random
import re
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def parse_think_tags(content: str) -> Tuple[str, Optional[str]]:
    """
    Parse <think>...</think> tags from content.
    
    Handles two cases:
    1. Full tags: <think>reasoning</think>answer
    2. Missing opening tag (enable_thinking=True pre-fills <think> in chat template):
       reasoning</think>answer
    
    Returns:
        Tuple of (content_without_think, reasoning_content)
    """
    if content is None:
        return None, None
    
    # Case 1: Full <think>...</think> tags present
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, content, re.DOTALL)
    if match:
        reasoning_content = match.group(1).strip()
        content_without_think = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        return content_without_think, reasoning_content
    
    # Case 2: Only </think> present (opening <think> was pre-filled by chat template)
    if '</think>' in content:
        parts = content.split('</think>', 1)
        reasoning_content = parts[0].strip()
        content_without_think = parts[1].strip() if len(parts) > 1 else ''
        return content_without_think, reasoning_content
    
    # Case 3: Truncated thinking output (max_tokens exhausted before </think>)
    # Detect: no </think> tag AND no valid Action: structure = pure reasoning that got cut off
    if 'Action:' not in content and ('Thought:' not in content):
        # This is truncated thinking content - treat entire content as reasoning
        return '', content.strip()
    
    return content, None

# Add project root to path for cloudgpt import (lazy loaded)
# interact_with_wm.py is at: fe-world/eval/02_task_success_rate/interact_with_wm.py
# So we need 3 levels up to get to fe-world/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Global CloudGPT client - initialized once and reused across all threads
_cloudgpt_client = None

def get_global_cloudgpt_client():
    """Get or create global CloudGPT client (thread-safe via lru_cache in cloudgpt_aoai)"""
    global _cloudgpt_client
    if _cloudgpt_client is None:
        # Lazy import: only load cloudgpt_aoai (which requires azure-identity-broker) when actually needed
        from clodgpt.cloudgpt_aoai import get_openai_client as get_cloudgpt_client
        print("Initializing global CloudGPT client...")
        _cloudgpt_client = get_cloudgpt_client()
        print("CloudGPT client initialized successfully!")
    return _cloudgpt_client


def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

class WorldModel:
    def __init__(self, wm_messages, client, model_name="llm_world_model", max_model_len=32768):
        self.history = wm_messages
        self.client = client
        self.model_name = model_name
        self.max_model_len = max_model_len

    def llm_generate(self, messages):
        # Dynamic max_tokens: conservatively estimate input tokens (chars//3) to avoid overflow
        estimated_input_tokens = sum(len(m.get("content", "") or "") for m in messages) // 3
        remaining = self.max_model_len - estimated_input_tokens - 50
        if remaining < 64:
            raise ValueError(
                f"WM context nearly full: ~{estimated_input_tokens} est. tokens, "
                f"max_model_len={self.max_model_len}, remaining={remaining}"
            )
        dynamic_max_tokens = min(1024, remaining)
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=dynamic_max_tokens,
                temperature=0,
                top_p=1,
            )
        except Exception as e:
            if "400" in str(e) and ("max_tokens" in str(e) or "context length" in str(e)):
                raise ValueError(
                    f"WM context overflow (server-side): {e}"
                )
            raise
        return response.choices[0].message.content

    def done(self, observation):
        if " [SUCCESS]" in observation:
            return observation.split(" [SUCCESS]")[0], True
        else:
            return observation, False

    def step(self, action):
        self.history.append({"role": "user", "content": action})
        observation = self.llm_generate(self.history)
        observation, done = self.done(observation)
        self.history.append({"role": "assistant", "content": observation})
        return observation, done


class ReactAgent:
    def __init__(self, agent_messages, agent_model_name, api_key, api_base_url, temperature=0):
        self.history = agent_messages
        self.agent_model_name = agent_model_name
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.temperature = temperature
        # Detect if using local vLLM (api_key=EMPTY) vs remote third-party API
        self.is_local_vllm = api_key.lower() in ("empty", "local", "vllm")
        # Initialize client once in __init__ instead of every llm_generate call
        if "cloudgpt" in self.api_key:
            # Use global CloudGPT client (thread-safe)
            self.client = get_global_cloudgpt_client()
            self.model_name = self.agent_model_name
        elif "azure" in self.api_key:
            from azure_openai import get_client as get_azure_client
            self.client, self.model_name = get_azure_client(model_name=self.agent_model_name)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
            self.model_name = self.agent_model_name

    def llm_generate(self, messages):
        # Strip reasoning_content from history messages to avoid context explosion
        # reasoning_content is only stored for logging, NOT re-injected into prompt
        clean_messages = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        
        # Client is already initialized in __init__
        if "gpt" in self.model_name or "claude" in self.model_name:
            # Dynamic max_tokens: cap to avoid exceeding Azure/CloudGPT context window
            # Azure GPT-4o context = 32768 (S0 tier); estimate input tokens ~= 4 chars/token
            estimated_input_tokens = sum(len(m.get("content", "")) for m in clean_messages) // 4
            gpt_context_limit = 32768  # Azure S0 GPT-4o context limit
            dynamic_max_tokens = max(256, min(4096, gpt_context_limit - estimated_input_tokens - 100))
            # GPT-5 requires max_completion_tokens instead of max_tokens
            token_key = "max_completion_tokens" if "gpt-5" in self.model_name else "max_tokens"
            api_kwargs = dict(
                messages=clean_messages,
                model=self.model_name,
            )
            api_kwargs[token_key] = dynamic_max_tokens
            # GPT-5 does not support temperature != 1; omit if unsupported
            if "gpt-5" not in self.model_name:
                api_kwargs["temperature"] = self.temperature
            response = self.client.chat.completions.create(**api_kwargs)
        elif self.is_local_vllm:
            # Local vLLM: Qwen3 Best Practices with vLLM-specific extra_body
            # Dynamic max_tokens: conservatively estimate input tokens (chars//3) to avoid overflow
            estimated_input_tokens = sum(len(m.get("content", "") or "") for m in clean_messages) // 3
            max_model_len = agent_max_model_len
            remaining = max_model_len - estimated_input_tokens - 100
            if remaining < 64:
                raise ValueError(
                    f"Agent context nearly full: ~{estimated_input_tokens} est. tokens, "
                    f"max_model_len={max_model_len}, remaining={remaining}"
                )
            dynamic_max_tokens = max(256, min(4096, remaining))
            try:
                response = self.client.chat.completions.create(
                    messages=clean_messages,
                    model=self.model_name,
                    max_tokens=dynamic_max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "top_k": 20,
                        "min_p": 0,
                    }
                )
            except Exception as e:
                if "400" in str(e) and ("max_tokens" in str(e) or "context length" in str(e)):
                    raise ValueError(f"Agent context overflow (server-side): {e}")
                raise
        else:
            # Remote third-party API (e.g., vectorengine.ai) — no vLLM-specific extra_body
            estimated_input_tokens = sum(len(m.get("content", "") or "") for m in clean_messages) // 3
            max_model_len = agent_max_model_len
            remaining = max_model_len - estimated_input_tokens - 100
            if remaining < 64:
                raise ValueError(
                    f"Agent context nearly full: ~{estimated_input_tokens} est. tokens, "
                    f"max_model_len={max_model_len}, remaining={remaining}"
                )
            dynamic_max_tokens = max(256, min(4096, remaining))
            try:
                response = self.client.chat.completions.create(
                    messages=clean_messages,
                    model=self.model_name,
                    max_tokens=dynamic_max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                )
            except Exception as e:
                if "400" in str(e) and ("max_tokens" in str(e) or "context length" in str(e)):
                    raise ValueError(f"Agent context overflow (server-side): {e}")
                raise
        return response.choices[0].message.content

    def parse_action(self, text: str):
        # AgentGym/agentenv/agentenv/controller/utils.py:L118
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
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

    def react(self, observation):
        self.history.append({"role": "user", "content": observation})
        max_retries = 50
        react_raw = None
        react_content = None
        reasoning_content = None
        for i in range(max_retries):
            try:
                react_raw = self.llm_generate(self.history)
                if react_raw is None:
                    continue
                # Parse <think> tags: reasoning goes to reasoning_content, not accumulated in history
                react_content, reasoning_content = parse_think_tags(react_raw)
                # If content is empty (truncated thinking), retry
                if not react_content or not react_content.strip():
                    print(f"Truncated thinking output detected (attempt {i+1}/{max_retries}), retrying...")
                    sleep_time = min(2 * (2 ** i), 60) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
                break
            except ValueError:
                # Context overflow — not retryable, propagate immediately
                raise
            except Exception as e:
                # Context overflow — retrying won't help, fail fast
                if "400" in str(e) and ("max_tokens" in str(e) or "context length" in str(e)):
                    raise ValueError(f"Agent context overflow: {e}")
                if i == max_retries - 1:
                    raise e
                print(f"Request failed, retrying ({i+1}/{max_retries})... Error: {e}")
                # 增加指数退避 (Exponential Backoff)
                sleep_time = min(2 * (2 ** i), 60) + random.uniform(0, 1)
                time.sleep(sleep_time)
        
        # Fallback if all retries produced truncated output
        if react_content is None or not react_content.strip():
            react_content = react_content or ""
        
        # History only stores content WITHOUT <think> tags (to avoid context pollution)
        self.history.append({
            "role": "assistant",
            "content": react_content,
            "reasoning_content": reasoning_content,  # Store separately for logging
        })
        thought, action = self.parse_action(react_content)  # Parse from clean content
        return react_raw, thought, action  # Return raw for debugging



def process_single_sample(
    agent_messages,
    wm_messages,
    id,
    max_steps=50,
    output_file=".outputs/alfworld_X.json",
    agent_model: str = "gpt-4o",
    wm_port: int = 8000,
    wm_name: str = "llm_world_model",
    wm_max_model_len: int = 32768,
):
    # import pdb; pdb.set_trace()
    wm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{wm_port}/v1")
    world_model = WorldModel(wm_messages, wm_client, model_name=wm_name, max_model_len=wm_max_model_len)
    react_agent = ReactAgent(agent_messages[:-1], agent_model, api_key, api_base_url, temperature=agent_temperature)
    observation = agent_messages[-1]["content"]

    try:
        # run interaction loop
        for _ in range(max_steps):
            react, thought, action = react_agent.react(observation)
            observation, done = world_model.step(action)
            if done:
                break

        # save interaction history
        output_data = {
            "conversations": react_agent.history + [{"role": "user", "content": observation}],
            "item_id": f"{TASK}_{id}",
            "reward": 1.0 if done else 0.0,
            "success": 1 if done else 0,
        }
        write_json(output_data, output_file)
        # print(f"Saved interaction history to {output_file}")
        return 1 if done else 0
    except ValueError as ve:
        # Context overflow from Agent or WM — save result file (success=0) but count as api_error
        print(f"Context overflow for sample {id}: {ve}")
        output_data = {
            "conversations": react_agent.history + [{"role": "user", "content": observation}],
            "item_id": f"{TASK}_{id}",
            "reward": 0.0,
            "success": 0,
            "note": f"context_overflow: {ve}",
        }
        write_json(output_data, output_file)
        raise  # propagate so main() counts it as api_error
    except Exception as e:
        output_data = {
            "item_id": f"{TASK}_{id}",
            "conversations": react_agent.history,
            "error": str(e),
        }
        write_json(output_data, f".debug/error_{TASK}_{id}.json")
        print(f"Error occurred during processing sample {id}. Saved debug info.")
        raise


def main():
    global TASK

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Interact with world model using a React agent")
    task_choices = ["alfworld", "sciworld", "textworld", "webshop"]
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choices,
        default="alfworld",
        help="Task to run: alfworld | sciworld | textworld | webshop",
    )
    parser.add_argument(
        "--model",
        type=str,
        # choices=model_choices,
        default="gpt-4o",
        help="Agent model for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="",
        help="API base url for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max interaction steps per item",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Max concurrent items to process",
    )
    parser.add_argument(
        "--wm-port",
        type=int,
        nargs='+',
        default=[8000],
        help="World model server port(s). Multiple ports for load distribution (default: 8000)",
    )
    parser.add_argument(
        "--wm-name",
        type=str,
        default="llm_world_model_0",
        help="World model served name (default: llm_world_model)",
    )
    parser.add_argument(
        "--agent-instruct-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wm-instruct-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./outputs/interact_with_world_model/",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0, deterministic)",
    )
    parser.add_argument(
        "--wm-max-model-len",
        type=int,
        default=32768,
        help="WM server max_model_len for dynamic max_tokens (default: 32768)",
    )
    parser.add_argument(
        "--agent-max-model-len",
        type=int,
        default=40960,
        help="Agent vLLM server max-model-len (for dynamic max_tokens capping)",
    )

    args = parser.parse_args()

    TASK = args.task
    agent_model = args.model
    max_steps = args.max_steps
    max_concurrency = args.max_concurrency
    wm_ports = args.wm_port  # list of ports
    wm_name = args.wm_name
    agent_instruct_file = args.agent_instruct_file
    wm_instruct_file = args.wm_instruct_file
    output_root = args.output_root
    global api_key, api_base_url, agent_temperature, agent_max_model_len
    api_key = args.api_key
    api_base_url = args.api_base_url
    agent_temperature = args.temperature
    agent_max_model_len = args.agent_max_model_len

    # Pre-initialize CloudGPT client before starting concurrent workers
    if "cloudgpt" in api_key:
        print("Pre-initializing CloudGPT client for concurrent requests...")
        get_global_cloudgpt_client()

    agent_instruct_data = read_json(agent_instruct_file)
    wm_instruct_data = read_json(wm_instruct_file)

    def merge_data(agent_instruct_data, wm_instruct_data):
        agent_instruct_data = {item["data_idx"]: item for item in agent_instruct_data}
        wm_instruct_data = {item["id"]: item for item in wm_instruct_data}
        merged_data = []
        for id in agent_instruct_data.keys():
            if id in wm_instruct_data:
                merged_data.append({
                    "id": id,
                    "agent_messages": agent_instruct_data[id]["messages"],
                    "wm_messages": wm_instruct_data[id]["messages"],
                })
            else:
                print(f"WARNING: id {id} is not in wm_instruct_data")
        return merged_data

    test_data = merge_data(agent_instruct_data, wm_instruct_data)
    total_items = len(test_data)
    if args.n_samples > 0:
        test_data = test_data[: args.n_samples]
    print(f"Loaded {total_items} items from {agent_instruct_file} and {wm_instruct_file}")

    pending_items = []
    processed_count = 0
    total_success = 0.0
    for item in test_data:
        id = item["id"]
        agent_messages = item["agent_messages"]
        wm_messages = item["wm_messages"]

        # init_observation = item["messages"][0]["content"]  # with env description
        output_file = os.path.join(output_root, f"{TASK}_{id}.json")
        if os.path.exists(output_file):
            existing = read_json(output_file)
            success_val = existing.get("success")
            if success_val is None:
                success_val = 1 if (existing.get("reward", 0)==1 or existing.get("reward", 0)==100) else 0
            success_val = float(success_val)
            total_success += success_val
            processed_count += 1
        else:
            pending_items.append((agent_messages, wm_messages, id, output_file))

    remaining = len(pending_items)
    if processed_count > 0:
        init_acc = (total_success / processed_count) * 100 if processed_count else 0

    if remaining < total_items:
        print(f"Resuming: {remaining} remaining out of {total_items} items.")
    else:
        print(f"Total items to run: {total_items}.")

    if not pending_items:
        print("All interaction histories already exist. Nothing to run.")
        if processed_count > 0:
            final_acc = (total_success / processed_count) * 100 if processed_count else 0
            print(f"Final accuracy: {final_acc:.2f}% ({total_success}/{processed_count}).")
        return

    # # debug with for loop
    # for agent_messages, wm_messages, item_id, output_file in pending_items:
    #     process_single_sample(agent_messages, wm_messages, item_id, max_steps, output_file, agent_model, wm_port, wm_name)

    if len(wm_ports) > 1:
        print(f"Using {len(wm_ports)} WM ports for load distribution: {wm_ports}")

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                agent_messages,
                wm_messages,
                item_id,
                max_steps,
                output_file,
                agent_model,
                wm_ports[i % len(wm_ports)],  # Round-robin across WM ports
                wm_name,
                args.wm_max_model_len,
            ): (item_id, output_file)
            for i, (agent_messages, wm_messages, item_id, output_file) in enumerate(pending_items)
        }

        error_count = 0
        new_processed_count = 0
        with tqdm(
            total=total_items,
            initial=processed_count,
            desc="Running agents",
        ) as pbar:
            init_acc = (total_success / processed_count * 100) if processed_count else 0
            pbar.set_postfix(acc=f"{init_acc:.2f}%", processed=0, api_errors=0)

            for future in as_completed(futures):
                item_id, output_file = futures[future]
                try:
                    success = future.result()
                    total_success += success
                    processed_count += 1
                    new_processed_count += 1
                except Exception as exc:
                    error_count += 1
                    print(f"Error during interaction for id {item_id}: {exc}")
                finally:
                    pbar.update(1)
                    cur_acc = (total_success / processed_count * 100) if processed_count else 0
                    pbar.set_postfix(
                        acc=f"{cur_acc:.2f}%",
                        processed=new_processed_count,
                        api_errors=error_count,
                    )

    final_acc = (total_success / processed_count * 100) if processed_count else 0
    print(f"\nFinal accuracy: {final_acc:.2f}% ({total_success}/{processed_count}). API errors: {error_count}.")
    print(f"Interaction histories saved to {output_root}")
    # save acc json to output root
    acc_output_file = os.path.join(output_root, f"_metrics.json")
    acc_data = {
        "task": TASK,
        "agent_model": agent_model,
        "total_items": total_items,
        "total_success": total_success,
        "processed_items": processed_count,
        "accuracy": final_acc,
        "api_errors": error_count,
    }
    write_json(acc_data, acc_output_file)
    print(f"Saved overall metrics to {acc_output_file}")


if __name__ == "__main__":
    main()