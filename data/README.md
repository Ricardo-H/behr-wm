# Data

## Overview

BehR-WM draws on two text-based interactive environments:

- **WebShop** ([Yao et al., 2022](https://arxiv.org/abs/2207.01206)) —
  e-commerce web navigation over 1.18M real-world products.
- **TextWorld** ([Côté et al., 2019](https://arxiv.org/abs/1806.11532)) —
  procedurally generated text adventure games.

Evaluation uses **200 standardized test tasks** per environment, taken from
[AgentGym](https://github.com/WooooDyy/AgentGym).

## What ships in this repository

The test splits required to run the full 3-stage evaluation pipeline are
already bundled in `init_contexts/`:

```
data/init_contexts/
├── webshop/
│   ├── agent_instruct_test.json        (~265 KB)
│   └── wm_instruct_test.json           (~95 KB)
└── textworld/
    ├── agent_instruct_test.json        (~695 KB)
    └── wm_instruct_test.json           (~545 KB)
```

No extra download is needed to reproduce the evaluation numbers reported in
the paper.

## What comes later (via `scripts/download_data.py`)

- **Training-split init contexts** — the `*_train.json` counterparts, used to
  reproduce the GRPO training runs. Considerably larger (~20 MB total) and
  therefore kept out of this git repository.
- **Trained world-model checkpoints** — published separately on HuggingFace.

Both are coming soon on HuggingFace Hub (see the
[Release Timeline](../README.md#release-timeline) in the top-level README).
Until then `scripts/download_data.py` prints an informative message and exits
non-zero. If you need the training split today, please open a GitHub issue.

## Re-producing the data layout

If you download the training split later, the full layout becomes:

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

which matches the structure expected by [`src/data/prepare_data.py`](../src/data/prepare_data.py)
and the reference verl training command in
[`docs/TRAINING.md`](../docs/TRAINING.md).
