#!/usr/bin/env python3
"""Download BehR-WM init contexts (and, when released, datasets) from HuggingFace.

Usage
-----
    python scripts/download_data.py                        # everything
    python scripts/download_data.py --env webshop          # one environment
    python scripts/download_data.py --env textworld \\
                                    --split test           # one split

Status
------
The HuggingFace repository ID is reserved for BehR-WM release milestone v0.4
(see README Release Timeline). Until the artifacts are published, this script
prints the upcoming repo ID and exits with a non-zero status so that CI and
local runs fail fast with a clear message.

Once the repo goes live the ``HF_REPO_ID`` constant below will be updated and
the script will `snapshot_download` the requested subset into ``data/init_contexts/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# TBA: will be populated when v0.4 artifacts are published on the Hub.
HF_REPO_ID: str | None = None  # e.g. "Ricardo-H/behr-wm-data"

ENVS = ("webshop", "textworld")
SPLITS = ("train", "test")

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "data" / "init_contexts"


def _expected_files(envs: list[str], splits: list[str]) -> list[str]:
    files: list[str] = []
    for env in envs:
        for split in splits:
            files.append(f"init_contexts/{env}/agent_instruct_{split}.json")
            files.append(f"init_contexts/{env}/wm_instruct_{split}.json")
    return files


def _download(envs: list[str], splits: list[str]) -> int:
    if HF_REPO_ID is None:
        print(
            "[behr-wm] Init contexts are not yet published on HuggingFace Hub.\n"
            "         Milestone v0.4 will publish them at: Ricardo-H/behr-wm-data (TBA).\n"
            "         Track progress: https://github.com/Ricardo-H/behr-wm#release-timeline\n"
            "         If you need the data today, please open a GitHub issue.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        print(
            "[behr-wm] huggingface_hub is required. Install with:\n"
            "             pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    allow_patterns = _expected_files(envs, splits)
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
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Split to download (default: all).",
    )
    args = parser.parse_args()

    envs = list(ENVS) if args.env == "all" else [args.env]
    splits = list(SPLITS) if args.split == "all" else [args.split]
    return _download(envs, splits)


if __name__ == "__main__":
    sys.exit(main())
