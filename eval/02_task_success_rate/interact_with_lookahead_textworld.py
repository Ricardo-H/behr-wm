"""
Lookahead Agent for TextWorld: Uses World Model to simulate candidate actions
before acting in the real TextWorld environment.

Flow at each step:
1. Agent observes REAL state s_t (with AVAILABLE ACTIONS list)
2. For each available action a_i, WM predicts next state: s_{t+1}^i = WM(s_t, a_i)
3. Agent sees the REAL state again + all (action, WM-predicted-future) pairs -> picks best
4. Execute the chosen action in REAL environment

This proves: A WM with higher CR (GRPO) gives more reliable lookahead signals
than lower CR (SFT), leading to higher agent task success rate.
"""

import os
import sys
import json
import argparse
import copy
import time
import random
import re
import requests
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from openai import OpenAI

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def parse_think_tags(content: str) -> Tuple[str, Optional[str]]:
    """Parse <think>...</think> tags from content."""
    if content is None:
        return None, None
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, content, re.DOTALL)
    if match:
        reasoning_content = match.group(1).strip()
        content_without_think = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        return content_without_think, reasoning_content
    if '</think>' in content:
        parts = content.split('</think>', 1)
        reasoning_content = parts[0].strip()
        content_without_think = parts[1].strip() if len(parts) > 1 else ''
        if reasoning_content:
            return content_without_think, reasoning_content
    return content, None


def extract_admissible_actions(observation: str) -> List[str]:
    """Extract admissible actions from TextWorld observation text.
    
    TextWorld observations end with: AVAILABLE ACTIONS: action1, action2, ...
    """
    match = re.search(r'AVAILABLE ACTIONS:\s*(.+)', observation)
    if match:
        actions_str = match.group(1).strip()
        actions = [a.strip() for a in actions_str.split(',') if a.strip()]
        return actions
    return []


def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)


def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)


# ============================================================================
# Core Components
# ============================================================================

class WorldModelSimulator:
    """Uses WM to simulate the effect of an action, returning predicted next state."""

    def __init__(self, wm_port: int, model_name: str = "llm_world_model"):
        import httpx
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{wm_port}/v1",
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self.model_name = model_name

    def simulate(self, wm_history: List[Dict], action: str) -> Tuple[str, bool]:
        """
        Simulate an action using the WM (does NOT modify wm_history).

        Returns:
            (predicted_observation, done)
        """
        sim_history = copy.deepcopy(wm_history)
        sim_history.append({"role": "user", "content": action})
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    messages=sim_history,
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0,
                    top_p=1,
                )
                observation = response.choices[0].message.content
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [WM Simulate Error] action={action}: {e}")
                    return "", False
                time.sleep(2)

        if observation and " [SUCCESS]" in observation:
            return observation.split(" [SUCCESS]")[0], True
        return observation or "", False


class TextWorldRealEnvironment:
    """Interface to real TextWorld environment via the TextWorld HTTP server."""

    def __init__(self, env_port: int, games_dir: str = "data/textworld/games"):
        self.env_server_base = f"http://localhost:{env_port}"
        self.games_dir = games_dir
        self.env_id = None
        self.timeout = 300

    def create(self):
        """Create a new environment instance."""
        resp = requests.post(
            f"{self.env_server_base}/create",
            json={"games_dir": self.games_dir, "max_steps": 50},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Create failed: {data['error']}")
        self.env_id = data["id"]
        return data

    def reset(self, data_idx: int) -> Tuple[str, List[str]]:
        """Reset environment for a specific game. Returns (observation, available_actions)."""
        if self.env_id is None:
            self.create()

        resp = requests.post(
            f"{self.env_server_base}/reset",
            json={"id": self.env_id, "data_idx": data_idx},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Reset failed: {data['error']}")
        obs = data.get("observation", "")
        actions = data.get("available_actions", [])
        return obs, actions

    def step(self, action: str) -> dict:
        """Execute an action. Returns dict with observation/reward/done/won/available_actions."""
        resp = requests.post(
            f"{self.env_server_base}/step",
            json={"id": self.env_id, "action": action},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Step failed: {data['error']}")
        return {
            "observation": data.get("observation", ""),
            "reward": data.get("reward", 0),
            "done": data.get("done", False),
            "won": data.get("won", False),
            "lost": data.get("lost", False),
            "score": data.get("score", 0),
            "max_score": data.get("max_score", 0),
            "available_actions": data.get("available_actions", []),
        }

    def close(self):
        if self.env_id is not None:
            try:
                requests.post(
                    f"{self.env_server_base}/close",
                    json={"id": self.env_id},
                    timeout=10,
                )
            except Exception:
                pass


class LookaheadAgent:
    """
    Agent that uses WM-assisted lookahead for planning in TextWorld.

    Two LLM call types:
    1. react_step(): Standard ReAct for steps with 0-1 available actions
    2. choose_best_action(): Given WM predictions for all candidates, pick the best action
    
    TextWorld typically has <10 available actions per step, so we simulate ALL
    of them (no need for the propose_candidates step used in WebShop).
    """

    def __init__(self, system_prompt: str, agent_model_name: str, api_key: str,
                 api_base_url: str, temperature: float = 0, enable_thinking: bool = False):
        self.system_prompt = system_prompt
        self.agent_model_name = agent_model_name
        self.temperature = temperature
        self.enable_thinking = enable_thinking

        if enable_thinking:
            self.temperature = 0.6

        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = agent_model_name

        # Conversation history for multi-turn interaction
        self.history = [{"role": "user", "content": system_prompt},
                        {"role": "assistant", "content": "Understood. I will respond with one valid action per turn."}]
        self.task_goal = ""  # Set later when initial observation is known

    def _llm_call(self, messages: List[Dict]) -> str:
        """Low-level LLM call with retry."""
        clean = [{"role": m["role"], "content": m.get("content", "")} for m in messages]

        if "gpt" in self.model_name.lower():
            # GPT models via OpenAI API
            for attempt in range(10):
                try:
                    resp = self.client.chat.completions.create(
                        messages=clean,
                        model=self.model_name,
                        max_tokens=4096,
                        temperature=self.temperature,
                    )
                    raw = resp.choices[0].message.content
                    if raw is not None:
                        return raw
                except Exception as e:
                    if attempt == 9:
                        raise
                    time.sleep(min(2 * (2 ** attempt), 60) + random.uniform(0, 1))
            return ""
        else:
            # vLLM (Qwen3 etc.)
            extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
            if self.enable_thinking:
                extra.update({"top_k": 20, "min_p": 0})
                max_tok, top_p = 32768, 0.95
            else:
                max_tok, top_p = 4096, 1

            for attempt in range(10):
                try:
                    resp = self.client.chat.completions.create(
                        messages=clean,
                        model=self.model_name,
                        max_tokens=max_tok,
                        temperature=self.temperature,
                        top_p=top_p,
                        extra_body=extra,
                    )
                    raw = resp.choices[0].message.content
                    if raw is not None:
                        content, _ = parse_think_tags(raw)
                        return content
                except Exception as e:
                    if attempt == 9:
                        raise
                    time.sleep(min(2 * (2 ** attempt), 60) + random.uniform(0, 1))
            return ""

    def parse_action(self, text: str) -> str:
        """Parse action from ReAct format response."""
        _split = text.rsplit("Action:", 1)
        if len(_split) == 2:
            return _split[1].strip()
        # Fallback: return first line that looks like a command
        return text.strip().split('\n')[-1].strip()

    def react_step(self, observation: str) -> str:
        """Standard ReAct step. Returns the chosen action."""
        self.history.append({"role": "user", "content": observation})
        response = self._llm_call(self.history)
        self.history.append({"role": "assistant", "content": response})
        return self.parse_action(response)

    def propose_candidates(self, observation: str, admissible_actions: List[str],
                           k: int = 5) -> List[str]:
        """
        Ask agent: given the current state and all admissible actions,
        pick the top-K actions most likely to help complete the task.
        """
        if len(admissible_actions) <= k:
            return admissible_actions

        actions_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(admissible_actions))

        task_context = ""
        if self.task_goal:
            task_context = f"\nYour TASK GOAL:\n{self.task_goal}\n"

        prompt = (
            f"You are playing a text-based interactive fiction game (TextWorld).\n"
            f"{task_context}\n"
            f"Your current state:\n{observation}\n\n"
            f"All admissible actions ({len(admissible_actions)} total):\n{actions_str}\n\n"
            f"From these admissible actions, select the {k} actions that are MOST LIKELY "
            f"to advance you toward completing the task. Focus on:\n"
            f"- Moving to required locations (go east/west/north/south)\n"
            f"- Picking up required items\n"
            f"- Opening/unlocking things you need to interact with\n"
            f"- Placing items where the task requires\n\n"
            f"Output EXACTLY {k} actions, one per line, in the format:\n"
            f"1. action_here\n"
            f"2. action_here\n"
            f"...\n"
            f"Only output the numbered list, nothing else."
        )

        messages = self.history[:2] + [{"role": "user", "content": prompt}]
        response = self._llm_call(messages)

        selected = []
        for line in response.strip().split("\n"):
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if line and line in admissible_actions and line not in selected:
                selected.append(line)
            elif line:
                for adm in admissible_actions:
                    if adm.lower() == line.lower() and adm not in selected:
                        selected.append(adm)
                        break

        if len(selected) < 2:
            return admissible_actions[:k]
        return selected[:k]

    def choose_best_action(self, observation: str,
                           predictions: List[Tuple[str, str, bool]]) -> str:
        """
        Ask agent: given the current REAL state and multiple (action -> WM-predicted future),
        which action is best for completing the task?

        Args:
            observation: current real state (including AVAILABLE ACTIONS)
            predictions: list of (action, predicted_next_state, predicted_done)

        Returns:
            The chosen action string.
        """
        # Build the prediction descriptions
        pred_descriptions = []
        for i, (action, pred_state, pred_done) in enumerate(predictions):
            pred_preview = pred_state[:2000] if pred_state else "(empty prediction)"
            done_note = " ** The world model predicts this action will COMPLETE THE TASK! **" if pred_done else ""
            pred_descriptions.append(
                f"--- Option {i+1}: \"{action}\" ---\n"
                f"Predicted next observation:{done_note}\n{pred_preview}\n"
            )

        options_text = "\n".join(pred_descriptions)

        # Extract task goal section for context
        task_context = ""
        if self.task_goal:
            # Get just the task description (after the banner, before the room description)
            task_context = f"\nYour TASK GOAL:\n{self.task_goal}\n"

        prompt = (
            f"You are playing a text-based interactive fiction game (TextWorld).\n"
            f"{task_context}\n"
            f"Your current REAL state:\n"
            f"{observation}\n\n"
            f"A world model has predicted what would happen if you take each of these actions:\n\n"
            f"{options_text}\n"
            f"Which action is BEST to make the most direct progress toward completing your task?\n\n"
            f"Guidelines:\n"
            f"- If any option leads to task completion ([SUCCESS]), ALWAYS choose it.\n"
            f"- Choose actions that DIRECTLY advance the next incomplete step of your task.\n"
            f"- Moving to a new location (go east/west/north/south) is usually better than examining things.\n"
            f"- Avoid repeated actions that don't change the state (examine, look, inventory).\n"
            f"- If you need to go somewhere, go there. Don't examine things on the way.\n\n"
            f"Output ONLY the exact action text (e.g., 'go east', 'take milk from couch'), nothing else."
        )

        # Use fresh context (not accumulated history) for the choice
        messages = self.history[:2] + [{"role": "user", "content": prompt}]
        response = self._llm_call(messages)
        chosen = self.parse_action(response) if "Action:" in response else response.strip()
        # Also strip surrounding quotes if present
        chosen = chosen.strip('"').strip("'").strip()

        # Validate: must be one of the candidate actions
        valid_actions = [a for a, _, _ in predictions]
        if chosen in valid_actions:
            return chosen

        # Fuzzy match (case-insensitive)
        chosen_lower = chosen.lower().strip()
        for action, _, _ in predictions:
            if action.lower().strip() == chosen_lower:
                return action

        # Try to find any valid action substring in response
        for action, _, _ in predictions:
            if action.lower() in response.lower():
                return action

        # Last resort: if agent chose something that looks like a valid game command, try it
        # But prefer fallback to first action
        print(f"  [Warning] Agent chose invalid action '{chosen[:100]}', falling back to first candidate")
        return valid_actions[0] if valid_actions else chosen


# ============================================================================
# Main Processing
# ============================================================================

def process_single_sample(
    agent_messages: List[Dict],
    wm_messages: List[Dict],
    data_idx: int,
    sample_id: int,
    max_steps: int = 50,
    output_file: str = "output.json",
    agent_model: str = "vllm_agent",
    api_key: str = "EMPTY",
    api_base_url: str = "http://localhost:8000/v1",
    wm_port: int = 8001,
    wm_name: str = "llm_world_model",
    env_port: int = 8024,
    temperature: float = 0,
    enable_thinking: bool = False,
    min_actions_for_lookahead: int = 2,
    lookahead_k: int = 5,
):
    """Run one TextWorld sample with lookahead planning."""
    wm_sim = WorldModelSimulator(wm_port, model_name=wm_name)

    # Extract system prompt from agent_messages
    system_prompt = agent_messages[0]["content"]
    initial_obs_full = agent_messages[-1]["content"]  # includes banner + task description

    agent = LookaheadAgent(
        system_prompt=system_prompt,
        agent_model_name=agent_model,
        api_key=api_key,
        api_base_url=api_base_url,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    real_env = TextWorldRealEnvironment(env_port)
    wm_history = copy.deepcopy(wm_messages)  # WM initial messages (system prompt)
    conversation = []
    done = False
    success = False
    score = 0.0
    max_score = 0.0
    recent_actions = []  # Track recent actions for loop detection
    MAX_WM_HISTORY = 20  # Max action-response pairs to keep in WM history

    # Extract task goal from the initial observation (after banner, before room description)
    # The task description usually starts with "Hey, thanks for coming over" or similar
    task_goal_match = re.search(r'(Hey,.*?(?:you\'re the winner!|you win!|you will win!))', initial_obs_full, re.DOTALL | re.IGNORECASE)
    if task_goal_match:
        agent.task_goal = task_goal_match.group(1).strip()
    else:
        # Fallback: use everything between the banner and "-= Room =-"
        room_match = re.search(r'-=\s+\w+\s+=-', initial_obs_full)
        if room_match:
            agent.task_goal = initial_obs_full[:room_match.start()].strip()[-500:]
        else:
            agent.task_goal = initial_obs_full[:500]

    try:
        # Reset real environment to the correct game
        real_obs, avail_actions = real_env.reset(data_idx)

        # Build the initial observation in the same format as agent_instruct
        observation = initial_obs_full

        for step_num in range(max_steps):
            # Extract available actions from current observation
            admissible = extract_admissible_actions(observation)

            # Filter out actions that appeared in recent history to avoid loops
            meaningful = admissible.copy()
            if len(recent_actions) >= 2:
                # Check for 2-cycle: A, B, A, B...
                last_two = recent_actions[-2:]
                filtered = [a for a in meaningful if a not in last_two]
                if len(filtered) >= 2:
                    meaningful = filtered
                elif len(filtered) >= 1:
                    meaningful = filtered

            use_lookahead = len(meaningful) >= min_actions_for_lookahead

            if not use_lookahead:
                # Standard ReAct step (only 0-1 action available)
                chosen_action = agent.react_step(observation)
                method = f"ReAct (actions={len(admissible)})"
                print(f"  [Step {step_num}] ReAct: {chosen_action}")
            else:
                # Lookahead: propose top-K candidates, then simulate via WM
                if len(meaningful) > lookahead_k:
                    candidates = agent.propose_candidates(observation, meaningful, k=lookahead_k)
                else:
                    candidates = meaningful

                predictions = []
                for cand in candidates:
                    pred_obs, pred_done = wm_sim.simulate(wm_history, cand)
                    predictions.append((cand, pred_obs, pred_done))

                # Agent picks best action given WM predictions
                chosen_action = agent.choose_best_action(observation, predictions)
                method = f"Lookahead ({len(candidates)}/{len(meaningful)} actions)"
                print(f"  [Step {step_num}] Lookahead: {chosen_action} (from {len(candidates)} candidates)")

                # Update agent history for multi-turn continuity
                agent.history.append({"role": "user", "content": observation})
                agent.history.append({
                    "role": "assistant",
                    "content": f"Thought: Evaluated {len(candidates)} candidates via world model lookahead.\n\nAction:\n{chosen_action}"
                })

            # Update WM history with chosen action + REAL observation (after env step)
            # We'll add the action now and the real observation after the env step
            wm_history.append({"role": "user", "content": chosen_action})

            # Track for loop detection
            recent_actions.append(chosen_action)
            if len(recent_actions) > 6:
                recent_actions.pop(0)

            # Execute in real environment
            try:
                result = real_env.step(chosen_action)
                real_obs = result["observation"]
                real_done = result["done"]
                real_won = result["won"]
                score = result["score"]
                max_score = result["max_score"]
                avail_actions = result["available_actions"]
            except Exception as e:
                print(f"  [Real Env Error] step {step_num}, action={chosen_action}: {e}")
                real_obs = ""
                real_done = False
                real_won = False
                avail_actions = []

            # Add REAL observation to WM history (not WM prediction)
            # This keeps WM history grounded in reality
            wm_history.append({"role": "assistant", "content": real_obs})

            # Trim WM history to prevent context overflow
            # Keep system prompt (first N messages) + last MAX_WM_HISTORY action-response pairs
            n_system = len(wm_messages)  # Number of initial system messages
            n_interaction = len(wm_history) - n_system
            if n_interaction > MAX_WM_HISTORY * 2:
                # Keep system + last MAX_WM_HISTORY pairs
                wm_history = wm_history[:n_system] + wm_history[-(MAX_WM_HISTORY * 2):]

            step_log = {
                "step": step_num,
                "method": method,
                "chosen_action": chosen_action,
                "score": score,
                "max_score": max_score,
                "done": real_done,
                "won": real_won,
                "real_obs_preview": real_obs[:300] if real_obs else "",
            }
            conversation.append(step_log)

            if real_done:
                done = True
                success = real_won
                break

            # Build next observation in TextWorld format
            if avail_actions:
                observation = f"{real_obs}\nAVAILABLE ACTIONS: {', '.join(avail_actions)}"
            else:
                observation = real_obs

        output_data = {
            "item_id": f"textworld_{sample_id}",
            "data_idx": data_idx,
            "success": 1 if success else 0,
            "score": score,
            "max_score": max_score,
            "total_steps": len(conversation),
            "conversation": conversation,
        }
        write_json(output_data, output_file)
        return 1 if success else 0

    except Exception as e:
        write_json({
            "item_id": f"textworld_{sample_id}",
            "data_idx": data_idx,
            "error": str(e),
            "conversation": conversation,
        }, output_file)
        print(f"Error processing sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        real_env.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TextWorld Lookahead Agent: WM-assisted planning in real TextWorld environment"
    )
    parser.add_argument("--agent-model", type=str, default="vllm_agent")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--api-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--wm-port", type=int, default=8001,
                        help="World model vLLM port")
    parser.add_argument("--wm-name", type=str, default="llm_world_model",
                        help="World model served name")
    parser.add_argument("--env-port", type=int, default=8024,
                        help="TextWorld real environment server port")
    parser.add_argument("--agent-instruct-file", type=str, required=True,
                        help="Agent instruction file (data/init_contexts/textworld/agent_instruct_test.json)")
    parser.add_argument("--wm-instruct-file", type=str, required=True,
                        help="WM instruction file (data/init_contexts/textworld/wm_instruct_test.json)")
    parser.add_argument("--output-root", type=str, default="./outputs/lookahead_textworld/")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max interaction steps per game")
    parser.add_argument("--n-samples", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--min-actions-for-lookahead", type=int, default=2,
                        help="Min available actions to trigger lookahead (else ReAct)")
    parser.add_argument("--lookahead-k", type=int, default=5,
                        help="Max candidate actions to simulate via WM per step")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"TextWorld Lookahead Agent Experiment")
    print(f"{'='*60}")
    print(f"Agent Model:     {args.agent_model}")
    print(f"WM Port:         {args.wm_port} ({args.wm_name})")
    print(f"Env Port:        {args.env_port}")
    print(f"Max Steps:       {args.max_steps}")
    print(f"Temperature:     {args.temperature}")
    print(f"Thinking:        {'ENABLED' if args.enable_thinking else 'DISABLED'}")
    print(f"Min Actions:     {args.min_actions_for_lookahead}")
    print(f"Output:          {args.output_root}")
    print(f"{'='*60}\n")

    agent_instruct_data = read_json(args.agent_instruct_file)
    wm_instruct_data = read_json(args.wm_instruct_file)

    # Merge data by data_idx / id
    agent_by_idx = {item["data_idx"]: item for item in agent_instruct_data}
    wm_by_idx = {item["id"]: item for item in wm_instruct_data}

    # Build test list (all items from agent instruct)
    merged = []
    for item in agent_instruct_data:
        idx = item["data_idx"]
        if idx in wm_by_idx:
            merged.append({
                "id": idx,
                "data_idx": idx,
                "agent_messages": item["messages"],
                "wm_messages": wm_by_idx[idx]["messages"],
            })
        else:
            print(f"WARNING: data_idx {idx} not in WM instruct data, skipping")

    if args.n_samples > 0:
        merged = merged[:args.n_samples]
    print(f"Total samples: {len(merged)}")

    os.makedirs(args.output_root, exist_ok=True)
    pending, total_success, processed_count = [], 0, 0

    for item in merged:
        out = os.path.join(args.output_root, f"textworld_{item['id']}.json")
        if os.path.exists(out):
            existing = read_json(out)
            if "error" not in existing:
                total_success += existing.get("success", 0)
                processed_count += 1
                continue
        pending.append((item, out))

    if processed_count > 0:
        print(f"Resuming: {len(pending)} remaining, {processed_count} done "
              f"(SR={total_success/processed_count*100:.1f}%)")

    if not pending:
        sr = total_success / processed_count * 100 if processed_count else 0
        print(f"\nAll done! SR: {sr:.2f}% ({total_success}/{processed_count})")
        return

    error_count = 0
    with tqdm(total=len(merged), initial=processed_count, desc="Lookahead-TextWorld") as pbar:
        for item, output_file in pending:
            try:
                s = process_single_sample(
                    agent_messages=item["agent_messages"],
                    wm_messages=item["wm_messages"],
                    data_idx=item["data_idx"],
                    sample_id=item["id"],
                    max_steps=args.max_steps,
                    output_file=output_file,
                    agent_model=args.agent_model,
                    api_key=args.api_key,
                    api_base_url=args.api_base_url,
                    wm_port=args.wm_port,
                    wm_name=args.wm_name,
                    env_port=args.env_port,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                    min_actions_for_lookahead=args.min_actions_for_lookahead,
                    lookahead_k=args.lookahead_k,
                )
                total_success += s
                processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error on sample {item['id']}: {e}")
            finally:
                pbar.update(1)
                sr = total_success / processed_count * 100 if processed_count else 0
                pbar.set_postfix(SR=f"{sr:.1f}%", err=error_count)

    final_sr = total_success / processed_count * 100 if processed_count else 0
    print(f"\n{'='*60}")
    print(f"SR: {final_sr:.2f}% ({total_success}/{processed_count}), Errors: {error_count}")
    print(f"{'='*60}")

    write_json({
        "task": "textworld",
        "accuracy": final_sr,
        "success": total_success,
        "total": processed_count,
        "errors": error_count,
        "config": {
            "max_steps": args.max_steps,
            "agent_model": args.agent_model,
            "wm_name": args.wm_name,
            "min_actions_for_lookahead": args.min_actions_for_lookahead,
        },
    }, os.path.join(args.output_root, "_metrics.json"))


if __name__ == "__main__":
    main()
