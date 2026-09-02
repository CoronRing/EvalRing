"""
Convenience wrapper to execute all evaluation modes sequentially.
"""

import argparse
from pathlib import Path

script_dir = Path(__file__).resolve().parent

from run_model_suite import run_local_suite

MODES = [
    #"single-class",
    # "multi-class-chance",
    # "base-vs-rest-binary",
    # "per-class-score",
    "multi-agent-host"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluation modes sequentially via python invocation.")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--models-file", type=str, default="model_list.json")
    parser.add_argument("--cache-mode", type=str, default="both", choices=["runs_only", "cache_file", "both"], help="Global cache operation mode")
    args = parser.parse_args()

    print("=" * 80)
    print("Running ALL execution modes sequentially")
    print("=" * 80)
    
    for mode in MODES:
        print(f"\n" + "=" * 80)
        print(f">>> Starting mode: {mode}")
        print("=" * 80)
        
        code = run_local_suite(
            agent_mode=mode,
            n_samples=args.n_samples,
            max_workers=args.max_workers,
            models_file=args.models_file,
            cache_mode=args.cache_mode
        )
        
        if code != 0:
            print(f"\nWARNING: Suite for mode '{mode}' exited with return code {code}.")
            print("Continuing to the next mode...")
            
    print("\n" + "=" * 80)
    print("Completed evaluating all modes.")
    print("=" * 80)
