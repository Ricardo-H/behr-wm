#!/usr/bin/env python3
"""Fetch BehR-WM data artefacts that are not already shipped in the repo.

What the repo ships (available immediately after ``git clone``):

- ``data/init_contexts/{webshop,textworld}/agent_instruct_test.json``
- ``data/init_contexts/{webshop,textworld}/wm_instruct_test.json``

These test splits are all that is needed to run the evaluation pipeline
(``eval/01_*``, ``eval/02_*``, ``eval/03_*``) end-to-end.

What this script downloads (coming soon on HuggingFace Hub):

- Training splits of the same init_contexts (``agent_instruct_train.json`` /
  ``wm_instruct_train.json``), for reproducing the GRPO training runs.
- Trained world-model checkpoints (separate script once released).

Usage
-----
    python scripts/download_data.py                    # all envs, train split
    python scripts/download_data.py --env webshop      # one environment
    python scripts/download_data.py --env textworld    # TextWorld only

The ``HF_REPO_ID`` constant below will be populated when the HuggingFace
dataset repository goes live. Until then the script prints a clear message and
exits with a non-zero status.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Will be filled in once the HuggingFace dataset repository is published.
HF_REPO_ID: str | None = None  # e.g. "Ricardo-H/behr-wm-data"

ENVS = ("webshop", "textworld")

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "data" / "init_contexts"


def _expected_files(envs: list[str]) -> list[str]:
    files: list[str] = []
    for env in envs:
        files.append(f"init_contexts/{env}/agent_instruct_train.json")
        files.append(f"init_contexts/{env}/wm_instruct_train.json")
    return files


def _download(envs: list[str]) -> int:
    if HF_REPO_ID is None:
        print(
            "[behr-wm] Training-split init contexts are not yet published on\n"
            "         HuggingFace Hub. They are coming soon together with the\n"
            "         trained checkpoints (see README \"Release Timeline\").\n"
            "\n"
            "         The TEST split needed to run evaluation is already bundled\n"
            "         in this repository under data/init_contexts/.\n"
            "\n"
            "         Track progress: https://github.com/Ricardo-H/behr-wm\n"
            "         If you need the training split today, please open a\n"
            "         GitHub issue.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        print(
            "[behr-wm] huggingface_hub is required. Install with:\n"
            "             pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    allow_patterns = _expected_files(envs)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(ROOT / "data"),
        allow_patterns=allow_patterns,
    )
    print(f"[behr-wm] Downloaded {len(allow_patterns)} files into {TARGET_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--env",
        choices=(*ENVS, "all"),
        default="all",
        help="Environment subset to download (default: all).",
    )
    args = parser.parse_args()
    envs = list(ENVS) if args.env == "all" else [args.env]
    return _download(envs)


if __name__ == "__main__":
    sys.exit(main())
