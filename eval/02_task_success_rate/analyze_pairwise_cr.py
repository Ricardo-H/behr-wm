#!/usr/bin/env python3
"""
Pair-wise CR Analysis: Compare aggregate CR vs pair-wise CR.

Aggregate CR (current paper):
    CR = W2R_SR / Real_SR

Pair-wise CR variants:
    CR_pair     = |Real✓ ∩ W2R✓| / |Real✓|
                  (of tasks the agent succeeds on in reality, how many also succeed via W2R?)

    CR_pair_sym = 2·|Real✓ ∩ W2R✓| / (|Real✓| + |W2R✓|)
                  (Dice coefficient: symmetric overlap)

    Cohen's κ   = agreement beyond chance

Usage:
    python analyze_pairwise_cr.py \\
        --w2r-dir outputs/task_success_rate/wm/webshop/EXP_NAME/valid_on_real_env \\
        --real-dir outputs/task_success_rate/real/webshop/AGENT_NAME \\
        --task webshop

    # Or auto-discover all matching experiments:
    python analyze_pairwise_cr.py --auto --task webshop
    python analyze_pairwise_cr.py --auto --task textworld
"""

import argparse
import json
import os
import glob
from collections import defaultdict


def load_results(directory: str, task: str) -> dict:
    """Load per-task success values from a directory.
    
    Returns: {task_id: success_bool}
    """
    results = {}
    pattern = os.path.join(directory, f"{task}_*.json")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        if fname.startswith("_"):
            continue
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract task ID from item_id or filename
            item_id = data.get("item_id", fname.replace(".json", ""))
            # Normalize: "webshop_1022" -> "1022"
            task_id = item_id.split(f"{task}_")[-1] if f"{task}_" in item_id else item_id
            
            # Extract success (handles bool, int, float)
            success = data.get("success", 0)
            if isinstance(success, bool):
                results[task_id] = success
            else:
                results[task_id] = float(success) > 0
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {fpath}: {e}")
    return results


def compute_pairwise_metrics(real: dict, w2r: dict, wm: dict = None):
    """Compute pair-wise CR metrics between Real and W2R results.
    
    Args:
        real: {task_id: success_bool} from real environment
        w2r:  {task_id: success_bool} from W2R replay
        wm:   {task_id: success_bool} from WM interaction (optional)
    
    Returns: dict of metrics
    """
    # Find common task IDs
    common_ids = sorted(set(real.keys()) & set(w2r.keys()))
    n = len(common_ids)
    
    if n == 0:
        return {"error": "No common task IDs found"}
    
    # Build contingency table
    #                  W2R✓     W2R✗
    # Real✓            a        b
    # Real✗            c        d
    a = sum(1 for tid in common_ids if real[tid] and w2r[tid])      # both succeed
    b = sum(1 for tid in common_ids if real[tid] and not w2r[tid])  # real✓, w2r✗
    c = sum(1 for tid in common_ids if not real[tid] and w2r[tid])  # real✗, w2r✓
    d = sum(1 for tid in common_ids if not real[tid] and not w2r[tid])  # both fail
    
    real_success = a + b
    w2r_success = a + c
    
    # Aggregate CR (current paper method)
    real_sr = real_success / n
    w2r_sr = w2r_success / n
    cr_aggregate = w2r_sr / real_sr if real_sr > 0 else float('inf')
    
    # Pair-wise CR: of Real✓ tasks, fraction also W2R✓
    cr_pairwise = a / real_success if real_success > 0 else float('nan')
    
    # Pair-wise CR (symmetric / Dice coefficient)
    cr_dice = 2 * a / (real_success + w2r_success) if (real_success + w2r_success) > 0 else float('nan')
    
    # Agreement rate
    agreement = (a + d) / n
    
    # Cohen's Kappa
    p_o = agreement
    p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else float('nan')
    
    # McNemar's test (asymmetry: b vs c)
    # b = real✓ w2r✗ (WM lost these); c = real✗ w2r✓ (WM gained these)
    
    metrics = {
        "n_common": n,
        "real_success": real_success,
        "w2r_success": w2r_success,
        "real_sr": real_sr,
        "w2r_sr": w2r_sr,
        "cr_aggregate": cr_aggregate,
        "cr_pairwise": cr_pairwise,
        "cr_dice": cr_dice,
        "agreement": agreement,
        "kappa": kappa,
        "contingency": {
            "both_success": a,
            "real_only": b,
            "w2r_only": c,
            "both_fail": d,
        },
    }
    
    # If WM data provided, add WM metrics too
    if wm is not None:
        wm_common = sorted(set(wm.keys()) & set(real.keys()) & set(w2r.keys()))
        wm_success = sum(1 for tid in wm_common if wm[tid])
        wm_sr = wm_success / len(wm_common) if wm_common else 0
        # Triple overlap: all three succeed
        triple = sum(1 for tid in wm_common if real[tid] and w2r[tid] and wm[tid])
        metrics["wm_success"] = wm_success
        metrics["wm_sr"] = wm_sr
        metrics["triple_success"] = triple
    
    return metrics


def print_report(name: str, metrics: dict):
    """Pretty-print a pair-wise CR report."""
    if "error" in metrics:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  ERROR: {metrics['error']}")
        return
    
    ct = metrics["contingency"]
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  N = {metrics['n_common']} common tasks")
    print()
    
    # Contingency table
    print(f"  Contingency Table (Real × W2R):")
    print(f"  {'':>20} {'W2R ✓':>10} {'W2R ✗':>10} {'Total':>10}")
    print(f"  {'Real ✓':>20} {ct['both_success']:>10} {ct['real_only']:>10} {metrics['real_success']:>10}")
    print(f"  {'Real ✗':>20} {ct['w2r_only']:>10} {ct['both_fail']:>10} {metrics['n_common'] - metrics['real_success']:>10}")
    print(f"  {'Total':>20} {metrics['w2r_success']:>10} {metrics['n_common'] - metrics['w2r_success']:>10} {metrics['n_common']:>10}")
    print()
    
    # Metrics comparison
    print(f"  Aggregate Metrics:")
    print(f"    Real SR           = {metrics['real_sr']*100:6.1f}%  ({metrics['real_success']}/{metrics['n_common']})")
    print(f"    W2R  SR           = {metrics['w2r_sr']*100:6.1f}%  ({metrics['w2r_success']}/{metrics['n_common']})")
    print(f"    CR (aggregate)    = {metrics['cr_aggregate']:.4f}  ← current paper definition")
    print()
    print(f"  Pair-wise Metrics:")
    print(f"    CR (pair-wise)    = {metrics['cr_pairwise']:.4f}  (= both✓ / real✓)")
    print(f"    CR (Dice)         = {metrics['cr_dice']:.4f}  (= 2·both✓ / (real✓+w2r✓))")
    print(f"    Agreement         = {metrics['agreement']:.4f}  (= (both✓+both✗) / N)")
    print(f"    Cohen's κ         = {metrics['kappa']:.4f}")
    print()
    
    # Interpretation
    diff = abs(metrics['cr_aggregate'] - metrics['cr_pairwise'])
    print(f"  Δ(aggregate - pairwise) = {metrics['cr_aggregate'] - metrics['cr_pairwise']:+.4f}")
    if diff < 0.03:
        print(f"  → Aggregate and pair-wise CR are very close (Δ < 0.03)")
    else:
        print(f"  → Notable difference between aggregate and pair-wise CR")
    
    # Breakdown
    print(f"\n  Breakdown:")
    print(f"    Real✓ & W2R✓  = {ct['both_success']:>3}  (consistent success)")
    print(f"    Real✓ & W2R✗  = {ct['real_only']:>3}  (WM-induced failures)")
    print(f"    Real✗ & W2R✓  = {ct['w2r_only']:>3}  (WM-induced lucky wins)")
    print(f"    Real✗ & W2R✗  = {ct['both_fail']:>3}  (consistent failure)")
    
    if "wm_success" in metrics:
        print(f"\n  WM SR = {metrics['wm_sr']*100:.1f}% ({metrics['wm_success']}/{metrics['n_common']})")
        print(f"  Triple success (Real✓ & WM✓ & W2R✓) = {metrics['triple_success']}")


def auto_discover(task: str, base_dir: str = "outputs/task_success_rate"):
    """Auto-discover all experiment pairs with W2R + Real data."""
    wm_base = os.path.join(base_dir, "wm", task)
    real_base = os.path.join(base_dir, "real", task)
    
    pairs = []
    
    if not os.path.isdir(wm_base):
        print(f"No WM data found at {wm_base}")
        return pairs
    
    for exp_name in sorted(os.listdir(wm_base)):
        w2r_dir = os.path.join(wm_base, exp_name, "valid_on_real_env")
        wm_dir = os.path.join(wm_base, exp_name)
        
        if not os.path.isdir(w2r_dir):
            continue
        
        # Count W2R files
        w2r_files = glob.glob(os.path.join(w2r_dir, f"{task}_*.json"))
        if len(w2r_files) < 10:
            continue
        
        # Try to find matching Real directory
        # Heuristic: match agent name from experiment name
        real_dir = None
        
        # For table5 experiments: table5_Qwen3-0.6B_sft -> real/table5_Qwen3-0.6B
        if exp_name.startswith("table5_"):
            parts = exp_name.rsplit("_", 1)  # table5_Qwen3-0.6B_sft -> [table5_Qwen3-0.6B, sft]
            candidate = os.path.join(real_base, parts[0])
            if os.path.isdir(candidate):
                real_dir = candidate
        
        # For A_* experiments: try known mappings
        if real_dir is None and "80b" in exp_name.lower():
            candidate = os.path.join(real_base, "qwen3-80b-t_0.6")
            if os.path.isdir(candidate):
                real_dir = candidate
        
        # For TextWorld 8B experiments (baseline, grpo_behr_step*)
        if real_dir is None and task == "textworld":
            # These are 8B agent experiments - Real = 83.5% (from paper Table 4)
            # Check if there's a direct real dir
            for rdir in os.listdir(real_base) if os.path.isdir(real_base) else []:
                if "8b" in rdir.lower() or rdir == "baseline":
                    candidate = os.path.join(real_base, rdir)
                    if os.path.isdir(candidate):
                        real_dir = candidate
                        break
        
        if real_dir is not None:
            pairs.append({
                "name": exp_name,
                "w2r_dir": w2r_dir,
                "wm_dir": wm_dir,
                "real_dir": real_dir,
                "w2r_count": len(w2r_files),
            })
        else:
            print(f"  [SKIP] {exp_name}: W2R found ({len(w2r_files)} files) but no matching Real dir")
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Pair-wise CR Analysis: compare aggregate vs pair-wise CR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", type=str, default="webshop",
                        choices=["webshop", "textworld", "alfworld", "sciworld"],
                        help="Task name (default: webshop)")
    parser.add_argument("--w2r-dir", type=str, default=None,
                        help="W2R results directory (valid_on_real_env/)")
    parser.add_argument("--wm-dir", type=str, default=None,
                        help="WM results directory (optional, for triple overlap)")
    parser.add_argument("--real-dir", type=str, default=None,
                        help="Real env results directory")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover all experiments with W2R+Real data")
    parser.add_argument("--base-dir", type=str, default="outputs/task_success_rate",
                        help="Base directory for auto-discovery")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.auto:
        pairs = auto_discover(args.task, args.base_dir)
        if not pairs:
            print(f"No complete experiment pairs found for task={args.task}")
            return
        
        print(f"\nFound {len(pairs)} experiments with paired data for {args.task}:")
        for p in pairs:
            print(f"  {p['name']} ({p['w2r_count']} W2R files)")
        
        for p in pairs:
            real = load_results(p["real_dir"], args.task)
            w2r = load_results(p["w2r_dir"], args.task)
            wm = load_results(p["wm_dir"], args.task) if os.path.isdir(p["wm_dir"]) else None
            
            metrics = compute_pairwise_metrics(real, w2r, wm)
            label = f"{args.task} | {p['name']} (Real: {os.path.basename(p['real_dir'])})"
            print_report(label, metrics)
            all_results[p["name"]] = metrics
    
    elif args.w2r_dir and args.real_dir:
        real = load_results(args.real_dir, args.task)
        w2r = load_results(args.w2r_dir, args.task)
        wm = load_results(args.wm_dir, args.task) if args.wm_dir else None
        
        metrics = compute_pairwise_metrics(real, w2r, wm)
        label = f"{args.task} | W2R={args.w2r_dir} vs Real={args.real_dir}"
        print_report(label, metrics)
        all_results["manual"] = metrics
    
    else:
        parser.print_help()
        print("\nError: Provide either --auto or both --w2r-dir and --real-dir")
        return
    
    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: {args.task}")
        print(f"{'='*70}")
        print(f"  {'Experiment':<45} {'CR_agg':>8} {'CR_pair':>8} {'Δ':>8} {'κ':>8}")
        print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, m in all_results.items():
            if "error" in m:
                continue
            delta = m["cr_aggregate"] - m["cr_pairwise"]
            print(f"  {name:<45} {m['cr_aggregate']:>8.4f} {m['cr_pairwise']:>8.4f} {delta:>+8.4f} {m['kappa']:>8.4f}")
    
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()
