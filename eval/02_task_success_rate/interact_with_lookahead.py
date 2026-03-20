"""
Lookahead Agent: Uses World Model to simulate candidate actions before acting in real environment.

Flow at each step:
1. Agent observes REAL state s_t
2. Agent selects up to K (default 5) most promising candidate actions from admissible actions
3. WM predicts next state for each candidate: s_{t+1}^i = WM(s_t, a_i)
4. Agent sees the REAL state again + all (action, WM-predicted-future) pairs -> picks the best action
5. Execute the chosen action in REAL environment

This proves: A WM with CR~1.0 (GRPO) gives more reliable lookahead signals than CR<1.0 (SFT),
leading to higher agent task success rate.
"""

import os
import sys
import json
import argparse
import copy
import time
import random
import re
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
    """Extract admissible actions from WebShop observation text."""
    actions = []
    click_pattern = r'click\[([^\]]+)\]'
    for match in re.finditer(click_pattern, observation):
        action = f"click[{match.group(1)}]"
        if action not in actions:
            actions.append(action)
    return actions


# Known WebShop product option category labels
_OPTION_CATEGORY_KEYWORDS = {
    'size', 'color', 'style', 'pattern', 'material', 'pack',
    'count', 'scent', 'flavor', 'fit type', 'type', 'unit count',
}

# Navigation tokens that terminate a category's option list
_NAV_TOKENS = {
    'buy now', 'back to search', '< prev', 'next >',
    'description', 'features', 'reviews', 'search',
}


def parse_option_categories(observation: str) -> Dict[str, str]:
    """Build {click_action → category} mapping from observation text.

    E.g., "... [SEP] color [SEP] red [SEP] blue [SEP] size [SEP] s [SEP] m ..."
    → {"click[red]": "color", "click[blue]": "color",
       "click[s]": "size", "click[m]": "size"}
    """
    parts = [p.strip() for p in observation.split('[SEP]')]
    mapping: Dict[str, str] = {}
    current_category = None
    for p in parts:
        p_lower = p.lower()
        if p_lower in _OPTION_CATEGORY_KEYWORDS:
            current_category = p_lower
        elif current_category:
            if p_lower in _NAV_TOKENS or not p:
                current_category = None  # end of this category's values
            else:
                mapping[f"click[{p}]"] = current_category
                # Also store lowercase variant for fuzzy matching
                if p != p.lower():
                    mapping[f"click[{p.lower()}]"] = current_category
    return mapping


def count_option_categories(observation: str) -> int:
    """Count distinct option categories on a product page from observation text."""
    parts = [p.strip().lower() for p in observation.split('[SEP]')]
    return sum(1 for p in parts if p in _OPTION_CATEGORY_KEYWORDS)


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

    def __init__(self, client, model_name="llm_world_model", max_model_len: int = 32768):
        self.client = client
        self.model_name = model_name
        self.max_model_len = max_model_len

    def simulate(self, wm_history: List[Dict], action: str) -> Tuple[str, bool]:
        """
        Simulate an action using the WM (does NOT modify wm_history).

        Returns:
            (predicted_observation, done)
        """
        sim_history = copy.deepcopy(wm_history)
        sim_history.append({"role": "user", "content": action})
        try:
            # Dynamic max_tokens: estimate input tokens (chars//3) to avoid overflow
            estimated_input_tokens = sum(len(m.get("content", "") or "") for m in sim_history) // 3
            remaining = self.max_model_len - estimated_input_tokens - 100
            if remaining < 64:
                print(f"  [WM Simulate Warning] Context nearly full: ~{estimated_input_tokens} est. tokens, "
                      f"max_model_len={self.max_model_len}, remaining={remaining}")
                return "", False
            dynamic_max_tokens = max(256, min(2048, remaining))
            response = self.client.chat.completions.create(
                messages=sim_history,
                model=self.model_name,
                max_tokens=dynamic_max_tokens,
                temperature=0,
                top_p=1,
            )
            observation = response.choices[0].message.content
        except Exception as e:
            print(f"  [WM Simulate Error] action={action}: {e}")
            return "", False

        if observation and " [SUCCESS]" in observation:
            return observation.split(" [SUCCESS]")[0], True
        return observation or "", False


class RealEnvironment:
    """Interface to real WebShop environment via AgentEnv."""

    def __init__(self, env_port: int):
        self.env_port = env_port
        self._client = None

    def reset(self, data_idx: int) -> str:
        """Reset environment for a new task. Returns initial observation."""
        from agentenv.envs import WebshopEnvClient
        from agentenv.controller import ActionFormat

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._client = WebshopEnvClient(
                    env_server_base=f"http://localhost:{self.env_port}",
                    data_len=10000,
                    timeout=300,
                    action_format=ActionFormat.REACT,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + random.random()
                    print(f"  Env create retry {attempt+1}/{max_retries}: {e}")
                    time.sleep(wait)
                else:
                    raise
        self._client.reset(idx=data_idx)
        return self._client.observe()

    def step(self, action: str) -> dict:
        """Execute an action. Returns dict with observation/reward/done/success."""
        if not ("Action:" in action or "Thought:" in action):
            react_action = f"Thought:\nExecuting action.\n\nAction:\n{action}"
        else:
            react_action = action
        step_output = self._client.step(react_action)
        done = step_output.done
        reward = step_output.reward
        return {
            "observation": step_output.state,
            "reward": reward,
            "done": done,
            "success": (reward == 1 or reward == 100) if done else False,
        }

    def close(self):
        if self._client:
            try:
                self._client.close()
            except:
                pass


class LookaheadAgent:
    """
    Agent that uses WM-assisted lookahead for planning.

    Three LLM call types:
    1. react_step(): Standard ReAct for simple/search-page steps
    2. propose_candidates(): Pick top-K most promising actions from admissible
    3. choose_best_action(): Given WM predictions, pick the best action
    """

    def __init__(self, system_prompt: str, agent_model_name: str, api_key: str,
                 api_base_url: str, temperature: float = 0, enable_thinking: bool = False):
        self.system_prompt = system_prompt
        self.agent_model_name = agent_model_name
        self.temperature = temperature
        self.enable_thinking = enable_thinking

        if enable_thinking:
            self.temperature = 0.6

        self.client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=120)
        self.model_name = agent_model_name
        self.is_remote_api = api_key.lower() not in ("empty", "local", "vllm")

        # Conversation history for multi-turn interaction (standard ReAct)
        self.history = [{"role": "system", "content": system_prompt}]

    def _llm_call(self, messages: List[Dict]) -> str:
        """Low-level LLM call with retry."""
        clean = [{"role": m["role"], "content": m.get("content", "")} for m in messages]

        # chat_template_kwargs is vLLM-only; remote OpenAI rejects it
        if self.is_remote_api:
            extra = {}
            if self.enable_thinking:
                max_tok, top_p = 32768, 0.95
            else:
                # Dynamic max_tokens: cap to avoid exceeding context window
                estimated_input_tokens = sum(len(m.get("content", "")) for m in clean) // 4
                gpt_context_limit = 128000
                max_tok = max(256, min(4096, gpt_context_limit - estimated_input_tokens - 100))
                top_p = 1
        else:
            extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
            if self.enable_thinking:
                extra.update({"top_k": 20, "min_p": 0})
                max_tok, top_p = 32768, 0.95
            else:
                max_tok, top_p = 4096, 1

        # GPT-5 requires max_completion_tokens instead of max_tokens
        tok_key = "max_completion_tokens" if "gpt-5" in self.model_name else "max_tokens"

        for attempt in range(10):
            try:
                kwargs = dict(
                    messages=clean,
                    model=self.model_name,
                    temperature=self.temperature,
                    top_p=top_p,
                )
                kwargs[tok_key] = max_tok
                if extra:
                    kwargs["extra_body"] = extra
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                if raw is not None:
                    content, _ = parse_think_tags(raw)
                    return content
            except Exception as e:
                if attempt == 9:
                    raise
                time.sleep(min(2 * (2 ** attempt), 60) + random.uniform(0, 1))
        return ""

    # ------------------------------------------------------------------
    # Standard ReAct step (for search pages / simple actions)
    # ------------------------------------------------------------------
    def react_step(self, observation: str) -> str:
        """
        Standard ReAct step. Agent sees observation and generates Thought+Action.
        Maintains conversation history.
        Returns the action string.
        """
        self.history.append({"role": "user", "content": observation})
        response = self._llm_call(self.history)
        self.history.append({"role": "assistant", "content": response})

        # Parse action from ReAct format
        _split = response.rsplit("Action:", 1)
        if len(_split) == 2:
            action = _split[1].strip()
        else:
            match = re.search(r'((?:search|click)\[[^\]]+\])', response)
            action = match.group(1) if match else response.strip()

        return action

    # ------------------------------------------------------------------
    # Step A: Agent proposes top-K candidate actions
    # ------------------------------------------------------------------
    def propose_candidates(self, observation: str, admissible_actions: List[str],
                           k: int = 5) -> List[str]:
        """
        Ask agent: given the current state and all admissible actions,
        pick the top-K actions most likely to help complete the task.

        Returns a list of up to K action strings.
        """
        if len(admissible_actions) <= k:
            return admissible_actions

        actions_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(admissible_actions))

        prompt = (
            f"You are currently in this state:\n"
            f"{observation}\n\n"
            f"All admissible actions:\n{actions_str}\n\n"
            f"Your task is described in the instruction above. "
            f"From the admissible actions, select the {k} actions that are MOST LIKELY "
            f"to help you complete the task successfully. "
            f"Consider which products match the requirements, which options need to be selected, etc.\n\n"
            f"Output EXACTLY {k} actions, one per line, in the format:\n"
            f"1. action_here\n"
            f"2. action_here\n"
            f"...\n"
            f"Only output the numbered list, nothing else."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._llm_call(messages)
        # Parse numbered list
        selected = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering like "1. " or "1) "
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if line and line in admissible_actions:
                if line not in selected:
                    selected.append(line)
            elif line:
                # Fuzzy match: try to find the closest admissible action
                for adm in admissible_actions:
                    if adm.lower() == line.lower() and adm not in selected:
                        selected.append(adm)
                        break

        # Fallback: if parsing failed, return first K admissible actions
        if len(selected) < 2:
            return admissible_actions[:k]

        return selected[:k]

    # ------------------------------------------------------------------
    # Step B: Agent chooses best action given WM predictions
    # ------------------------------------------------------------------
    def choose_best_action(self, observation: str,
                           predictions: List[Tuple[str, str, bool]]) -> str:
        """
        Ask agent: given the current REAL state and multiple (action -> WM-predicted future),
        which action is best?  Uses NUMBER-BASED selection for robust parsing.
        """
        # Build the prediction descriptions with numbered options
        pred_descriptions = []
        for i, (action, pred_state, pred_done) in enumerate(predictions):
            pred_preview = pred_state[:1500] if pred_state else "(empty prediction)"
            done_note = " [TASK COMPLETE after this action]" if pred_done else ""
            pred_descriptions.append(
                f"Option {i+1}: {action}{done_note}\n"
                f"  Predicted next state: {pred_preview}\n"
            )

        options_text = "\n".join(pred_descriptions)

        # Use a SIMPLE system prompt (not ReAct) to avoid Thought/Action formatting
        selector_system = (
            "You are a decision-making assistant. You will be given a current state "
            "and multiple action options with their predicted outcomes. "
            "Your job is to select the BEST option by responding with ONLY the option number."
        )

        prompt = (
            f"CURRENT STATE:\n{observation}\n\n"
            f"AVAILABLE OPTIONS (with predicted outcomes from a world model):\n\n"
            f"{options_text}\n"
            f"SELECTION RULES:\n"
            f"- Select the option that best matches the task goal.\n"
            f"- Prefer selecting required product options (color, size) before 'buy now'.\n"
            f"- Avoid 'description', 'features', or 'back to search' unless no relevant option exists.\n\n"
            f"Reply with ONLY the option number (e.g., 1 or 3). Nothing else."
        )

        messages = [
            {"role": "system", "content": selector_system},
            {"role": "user", "content": prompt},
        ]

        response = self._llm_call(messages)
        chosen = response.strip()
        valid_actions = [a for a, _, _ in predictions]
        n = len(valid_actions)

        # --- Parse number ---
        # Try to find a number in the response
        num_match = re.search(r'(\d+)', chosen)
        if num_match:
            idx = int(num_match.group(1)) - 1  # 1-indexed → 0-indexed
            if 0 <= idx < n:
                print(f"  [choose] Selected option {idx+1}: {valid_actions[idx]}")
                return valid_actions[idx]

        # --- Fallback: try to match action string ---
        # Handle "Action: xxx" format
        if "Action:" in chosen:
            action_part = chosen.rsplit("Action:", 1)[1].strip().split("\n")[0].strip()
            m = re.search(r'((?:click|search)\[[^\]]+\])', action_part)
            if m:
                for action in valid_actions:
                    if action.lower() == m.group(1).lower():
                        return action

        # Direct exact/case match
        for action in valid_actions:
            if action == chosen or action.lower() == chosen.lower():
                return action

        # Regex from full response
        all_matches = re.findall(r'((?:click|search)\[[^\]]+\])', chosen)
        for extracted in all_matches:
            for action in valid_actions:
                if action.lower() == extracted.lower():
                    return action

        # Last resort: return first action
        print(f"  [Warning] Could not parse choice '{chosen[:80]}', using option 1: {valid_actions[0]}")
        return valid_actions[0] if valid_actions else chosen


# ============================================================================
# Main Processing
# ============================================================================

def process_single_sample(
    agent_messages: List[Dict],
    wm_messages: List[Dict],
    data_idx: int,
    sample_id: int,
    max_steps: int = 20,
    output_file: str = "output.json",
    agent_model: str = "vllm_agent",
    api_key: str = "EMPTY",
    api_base_url: str = "http://localhost:8000/v1",
    wm_port: int = 8001,
    wm_name: str = "llm_world_model",
    wm_max_model_len: int = 32768,
    env_port: int = 3000,
    lookahead_k: int = 5,
    temperature: float = 0,
    enable_thinking: bool = False,
):
    """
    Run one sample with the lookahead planning flow.
    """
    wm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{wm_port}/v1", timeout=120)
    wm_sim = WorldModelSimulator(wm_client, model_name=wm_name, max_model_len=wm_max_model_len)

    # Extract system prompt from agent_messages (first message)
    system_prompt = agent_messages[0]["content"]
    initial_obs = agent_messages[-1]["content"]

    agent = LookaheadAgent(
        system_prompt=system_prompt,
        agent_model_name=agent_model,
        api_key=api_key,
        api_base_url=api_base_url,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    real_env = RealEnvironment(env_port)
    wm_history = copy.deepcopy(wm_messages)
    conversation = []
    done = False
    success = False
    reward = 0.0

    # Navigation/meta actions that are NOT product options
    NON_OPTION_ACTIONS = {
        'click[buy now]', 'click[back to search]', 'click[< prev]',
        'click[next >]', 'click[description]', 'click[features]',
        'click[reviews]', 'click[search]',
    }

    try:
        real_env.reset(data_idx)
        observation = initial_obs
        prev_actions = []  # Track action history for loop detection

        for step in range(max_steps):
            # ---- Step 1: Extract admissible actions from real observation ----
            admissible = extract_admissible_actions(observation)

            # Only filter out click[search] (search pages need ReAct with search[...])
            meaningful_clicks = [a for a in admissible if a != 'click[search]']

            # ---- Fix 1: Force option selection before buy now ----
            # Product options = click actions that are NOT navigation/meta
            product_options = [a for a in meaningful_clicks
                               if a.startswith('click[') and a not in NON_OPTION_ACTIONS]
            has_buy_now = 'click[buy now]' in meaningful_clicks

            if has_buy_now and len(product_options) > 0:
                # On a product page with selectable options.
                # Build {click_action → category} mapping from observation
                opt_cat_map = parse_option_categories(observation)
                n_categories = len(set(opt_cat_map.values())) if opt_cat_map else 1
                n_categories = max(n_categories, 1)

                # Count how many DISTINCT categories have been covered
                covered_cats = set()
                for pa in prev_actions[-10:]:
                    if pa in opt_cat_map:
                        covered_cats.add(opt_cat_map[pa])
                n_covered = len(covered_cats)

                if n_covered < n_categories:
                    # Not enough categories covered → restrict to product options only
                    meaningful_clicks = product_options[:]
                    print(f"  [Step {step}] Forcing option selection: "
                          f"{n_covered}/{n_categories} categories filled, "
                          f"{len(product_options)} options available")

            # ---- Fix 2: Prevent action loops (consecutive + A-B-A-B) ----
            # Block same action taken 2+ times consecutively
            if len(prev_actions) >= 2 and prev_actions[-1] == prev_actions[-2]:
                repeated = prev_actions[-1]
                if repeated in meaningful_clicks and len(meaningful_clicks) > 1:
                    meaningful_clicks = [a for a in meaningful_clicks if a != repeated]
                    print(f"  [Step {step}] Blocked repeated action: {repeated}")
            # Block A-B-A-B two-step cycle
            if len(prev_actions) >= 4 and (prev_actions[-1] == prev_actions[-3] and
                                            prev_actions[-2] == prev_actions[-4]):
                cycle_a, cycle_b = prev_actions[-1], prev_actions[-2]
                for cyc in [cycle_a, cycle_b]:
                    if cyc in meaningful_clicks and len(meaningful_clicks) > 1:
                        meaningful_clicks = [a for a in meaningful_clicks if a != cyc]
                print(f"  [Step {step}] Blocked A-B cycle: {cycle_a} / {cycle_b}")

            # Use lookahead when there are multiple meaningful choices
            use_lookahead = len(meaningful_clicks) >= 2

            if not use_lookahead:
                # ---- Standard ReAct step (search page, single option, etc.) ----
                chosen_action = agent.react_step(observation)
                thought = f"ReAct step (admissible={len(admissible)})"
                print(f"  [Step {step}] ReAct: {chosen_action}")
            else:
                # ---- Lookahead flow ----
                # Step 2: Agent proposes top-K candidates
                candidates = agent.propose_candidates(observation, meaningful_clicks, k=lookahead_k)

                # Step 3: WM simulates each candidate
                predictions = []
                for cand in candidates:
                    pred_obs, pred_done = wm_sim.simulate(wm_history, cand)
                    predictions.append((cand, pred_obs, pred_done))

                # Step 4: Agent chooses best action given WM predictions
                chosen_action = agent.choose_best_action(observation, predictions)

                thought = f"Lookahead over {len(candidates)} candidates"
                print(f"  [Step {step}] Lookahead: {chosen_action} (from {len(candidates)} candidates)")

                # Update agent history for multi-turn consistency
                agent.history.append({"role": "user", "content": observation})
                agent.history.append({"role": "assistant",
                                      "content": f"Thought: Evaluated {len(candidates)} candidates via world model lookahead.\nAction: {chosen_action}"})

            # ---- Step 5: Update WM history ----
            wm_history.append({"role": "user", "content": chosen_action})
            pred_for_history, _ = wm_sim.simulate(wm_history[:-1], chosen_action)
            wm_history.append({"role": "assistant", "content": pred_for_history})

            # ---- Step 6: Execute in real environment ----
            try:
                result = real_env.step(chosen_action)
                real_obs = result["observation"]
                reward = result["reward"]
                real_done = result["done"]
                real_success = result["success"]
            except Exception as e:
                print(f"  [Real Env Error] step {step}, action={chosen_action}: {e}")
                real_obs = ""
                reward = 0.0
                real_done = False
                real_success = False

            prev_actions.append(chosen_action)

            step_log = {
                "step": step,
                "thought": thought,
                "chosen_action": chosen_action,
                "reward": reward,
                "done": real_done,
                "real_obs_preview": real_obs[:300] if real_obs else "",
            }
            conversation.append(step_log)

            if real_done:
                done = True
                success = real_success
                break

            observation = real_obs

        output_data = {
            "item_id": f"webshop_{sample_id}",
            "data_idx": data_idx,
            "success": 1 if success else 0,
            "reward": reward,
            "score": reward,
            "total_steps": len(conversation),
            "conversation": conversation,
        }
        write_json(output_data, output_file)
        return 1 if success else 0

    except Exception as e:
        write_json({
            "item_id": f"webshop_{sample_id}",
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
        description="Lookahead Agent: WM-assisted planning in real environment"
    )
    parser.add_argument("--agent-model", type=str, default="vllm_agent")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--api-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--wm-port", type=int, default=8001)
    parser.add_argument("--wm-name", type=str, default="llm_world_model")
    parser.add_argument("--env-port", type=int, default=3000)
    parser.add_argument("--agent-instruct-file", type=str, required=True)
    parser.add_argument("--wm-instruct-file", type=str, required=True)
    parser.add_argument("--inference-file", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="./outputs/lookahead/")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lookahead-k", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument(
        "--wm-max-model-len",
        type=int,
        default=32768,
        help="WM server max_model_len for dynamic max_tokens (default: 32768)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Lookahead Agent Experiment")
    print(f"{'='*60}")
    print(f"Agent Model:     {args.agent_model}")
    print(f"WM Port:         {args.wm_port} ({args.wm_name})")
    print(f"Env Port:        {args.env_port}")
    print(f"Lookahead K:     {args.lookahead_k}")
    print(f"Max Steps:       {args.max_steps}")
    print(f"Temperature:     {args.temperature}")
    print(f"WM MaxModelLen:  {args.wm_max_model_len}")
    print(f"Output:          {args.output_root}")
    print(f"{'='*60}\n")

    agent_instruct_data = read_json(args.agent_instruct_file)
    wm_instruct_data = read_json(args.wm_instruct_file)
    test_data = read_json(args.inference_file)

    agent_by_idx = {item["data_idx"]: item for item in agent_instruct_data}
    wm_by_idx = {item["id"]: item for item in wm_instruct_data}
    test_indices = [int(item["item_id"].split("_")[-1]) for item in test_data]

    merged = []
    for idx in test_indices:
        if idx in agent_by_idx and idx in wm_by_idx:
            merged.append({
                "id": idx,
                "data_idx": idx,
                "agent_messages": agent_by_idx[idx]["messages"],
                "wm_messages": wm_by_idx[idx]["messages"],
            })

    if args.n_samples > 0:
        merged = merged[:args.n_samples]
    print(f"Total samples: {len(merged)}")

    os.makedirs(args.output_root, exist_ok=True)
    pending, total_success, processed_count = [], 0, 0
    for item in merged:
        out = os.path.join(args.output_root, f"webshop_{item['id']}.json")
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
    with tqdm(total=len(merged), initial=processed_count, desc="Lookahead") as pbar:
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
                    wm_max_model_len=args.wm_max_model_len,
                    env_port=args.env_port,
                    lookahead_k=args.lookahead_k,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
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
        "accuracy": final_sr,
        "success": total_success,
        "total": processed_count,
        "errors": error_count,
        "config": {
            "lookahead_k": args.lookahead_k,
            "max_steps": args.max_steps,
            "agent_model": args.agent_model,
            "wm_name": args.wm_name,
        },
    }, os.path.join(args.output_root, "_metrics.json"))


if __name__ == "__main__":
    main()
