# Data

## Overview

BehR-WM draws on two text-based interactive environments:

- **WebShop** ([Yao et al., 2022](https://arxiv.org/abs/2207.01206)) —
  e-commerce web navigation over 1.18M real-world products.
- **TextWorld** ([Côté et al., 2019](https://arxiv.org/abs/1806.11532)) —
  procedurally generated text adventure games.

Evaluation uses **200 standardized test tasks** per environment, taken from
[AgentGym](https://github.com/WooooDyy/AgentGym).

## How to obtain the init contexts

The evaluation and training pipelines read per-environment system prompts
(agent + world model, train/test splits) from `data/init_contexts/`. Rather
than vendoring the JSON into this repo, we fetch them from HuggingFace Hub:

```bash
python scripts/download_data.py                 # default: all envs, all splits
python scripts/download_data.py --env webshop   # only WebShop
python scripts/download_data.py --env textworld --split test
```

> **Status:** the HuggingFace repository ID is reserved and will be published
> together with the datasets and trained checkpoints (coming soon — see the
> [Release Timeline](../README.md#release-timeline)). Until then
> `download_data.py` prints the upcoming repo ID and exits. If you need the
> init contexts today, request them by opening a GitHub issue.

After a successful download the directory layout is:

```
data/init_contexts/
├── webshop/
│   ├── agent_instruct_train.json
│   ├── agent_instruct_test.json
│   ├── wm_instruct_train.json
│   └── wm_instruct_test.json
└── textworld/
    ├── agent_instruct_train.json
    ├── agent_instruct_test.json
    ├── wm_instruct_train.json
    └── wm_instruct_test.json
```

## Training data

Full training trajectories (parquet, GRPO-ready) will ship under the same
HuggingFace repository (coming soon). They follow the schema expected by
[`src/data/prepare_data.py`](../src/data/prepare_data.py) and can be consumed
directly by the reference verl invocation in
[`docs/TRAINING.md`](../docs/TRAINING.md).

## Model checkpoints

Trained world-model checkpoints will be published on HuggingFace (coming soon).
Each checkpoint card will document the base world model, training data slice,
and the reported CR / CR<sub>pw</sub> / EM numbers.
