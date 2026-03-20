# Data

## Overview

This project uses interaction trajectories from the following environments:

- **WebShop** ([Yao et al., 2022](https://arxiv.org/abs/2207.01206)): E-commerce web navigation with 1.18M real-world products
- **TextWorld** ([Côté et al., 2019](https://arxiv.org/abs/1806.11532)): Text-based game environment with procedurally generated games

## Evaluation Data

Evaluation uses **200 standardized test tasks** from [AgentEnv](https://github.com/WooooDyy/AgentGym) for each environment. The test tasks are included in `data/eval/`.

## System Prompts

The `data/init_contexts/` directory contains system prompts for agents and world models:

```
data/init_contexts/
├── webshop/
│   ├── agent_instruct_test.json    # Agent system prompt (test)
│   ├── wm_instruct_test.json       # World model system prompt (test)
│   ├── agent_instruct_train.json   # Agent system prompt (train)
│   └── wm_instruct_train.json      # World model system prompt (train)
└── textworld/
    ├── agent_instruct_test.json
    ├── wm_instruct_test.json
    ├── agent_instruct_train.json
    └── wm_instruct_train.json
```

## Training Data

Training data, model weights, and full evaluation datasets will be released upon paper acceptance.
