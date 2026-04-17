#!/usr/bin/env python3
"""Compute Consistency Ratio (CR) and Pairwise CR (CR_pw) across experiments.

Given a real-env baseline directory and one-or-more WM / W2R result directories,
print WM SR, W2R SR, CR, and CR_pw for each.

Each results directory is expected to contain per-task JSON files named
``<env>_<task_id>.json`` (matching the output layout produced by
``eval/02_task_success_rate/``). A task is considered successful when the
stored ``reward`` field is >= 1.0.

Example
-------
    python compute_cr.py \\
        --real-dir  outputs/task_success_rate/real/webshop/gpt5_baseline \\
        --entries  "Qwen BehR=outputs/task_success_rate/wm/webshop/behr_qwen:wm2real" \\
                   "Llama BehR=outputs/task_success_rate/wm/webshop/behr_llama:wm2real" \\
        --num-tasks 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Set, Tuple


def _success_set(dir_path: str, pattern: str) -> Tuple[Set[str], int]:
    succ: Set[str] = set()
    total = 0
    if not os.path.isdir(dir_path):
        return succ, 0
    for f in glob.glob(os.path.join(dir_path, pattern)):
        total += 1
        tid = os.path.splitext(os.path.basename(f))[0]
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if float(data.get("reward", 0)) >= 1.0:
            succ.add(tid)
    return succ, total


def _parse_entry(entry: str) -> Tuple[str, str, str]:
    """Parse ``label=dir[:subdir]`` into ``(label, dir, subdir)``."""
    label, _, rest = entry.partition("=")
    if not label or not rest:
        raise argparse.ArgumentTypeError(
            f"Expected 'label=dir[:subdir]', got: {entry!r}"
        )
    dir_part, _, subdir = rest.partition(":")
    return label.strip(), dir_part.strip(), subdir.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--real-dir",
        required=True,
        help="Directory with real-env rollouts (baseline for CR denominator).",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        required=True,
        help="One or more 'label=wm_dir[:w2r_subdir]' entries. "
             "If ':w2r_subdir' is omitted, W2R is assumed to be in the same dir.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob for per-task JSON files (default: *.json).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=200,
        help="Normaliser for success rate (default: 200 = AgentGym test split).",
    )
    args = parser.parse_args()

    real_set, _ = _success_set(args.real_dir, args.pattern)
    real_sr = len(real_set) / args.num_tasks if args.num_tasks else 0.0
    print(f"Real: {len(real_set)}/{args.num_tasks} = {real_sr:.1%}")

    for entry in args.entries:
        label, wm_dir, w2r_sub = _parse_entry(entry)
        wm_set, wm_total = _success_set(wm_dir, args.pattern)
        w2r_dir = os.path.join(wm_dir, w2r_sub) if w2r_sub else wm_dir
        w2r_set, w2r_total = _success_set(w2r_dir, args.pattern)

        wm_sr = len(wm_set) / args.num_tasks if args.num_tasks else 0.0
        w2r_sr = len(w2r_set) / args.num_tasks if args.num_tasks else 0.0
        cr = w2r_sr / real_sr if real_sr > 0 else 0.0
        cr_pw = (len(real_set & w2r_set) / len(real_set)) if real_set else 0.0

        print(
            f"{label}: "
            f"WM={len(wm_set)}/{wm_total} ({wm_sr:.1%}) "
            f"W2R={len(w2r_set)}/{w2r_total} ({w2r_sr:.1%}) "
            f"CR={cr:.3f} CR_pw={cr_pw:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
