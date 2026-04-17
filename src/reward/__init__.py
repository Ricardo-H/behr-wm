"""
BehR reward functions for GRPO training of text-based world models.

Two environment-specific implementations are provided:

- ``behr_reward_webshop``: BehR reward for the WebShop environment, combining
  Behavior Consistency Reward with a physical-facts reward (ASIN / price /
  page / rating).
- ``behr_reward_textworld``: BehR reward for TextWorld, using pure behavior
  consistency without physical-facts anchoring.

Both modules expose a ``compute_score`` entry point matching the verl
reward-manager signature ``(data_source, solution_str, ground_truth, extra_info)``.

See ``docs/TRAINING.md`` for the full reward specification and verl integration
instructions.
"""
