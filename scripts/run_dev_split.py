#!/usr/bin/env python3
"""
run_dev_split.py — address the "the levers are tuned on the eval set" threat by
SELECTING the enhanced-panel config on a disjoint dev split, then reporting the
chosen config ONCE on the held-out eval split. LIVE: needs an LLM backend.

Protocol (all seed ranges disjoint):
  train (ML opponent) : seeds     0..299   (base_seed 0)
  demos               : seeds   500..      (demo_seed 500)
  DEV  (lever pick)   : seeds 20_000..     (base_seed 20_000)   <- new, disjoint
  EVAL (report)       : seeds 10_000..     (base_seed 10_000)   <- the paper's eval

Step 1: run a small grid of (k, sc) on the DEV split and pick the argmax
        classification accuracy.
Step 2: run the winning config ONCE on the EVAL split and report it.

The eval-split number is then an honest, non-cherry-picked estimate: the config
was chosen without ever seeing eval scenarios. Compare it to the paper's tuned
0.990 to see how much of the lift survives a blind split.

Usage:
    python scripts/run_dev_split.py --backend litellm --dev-n 60 --eval-n 200
Writes outputs/dev_split.json.
"""
from __future__ import annotations

import argparse
import json
import os

from onnesim import agent_eval as AE


def eval_config(backend, n, base_seed, k, sc, tag):
    cfg = AE.EvalConfig(
        n_scenarios=n, backend=backend, base_seed=base_seed,
        few_shot_k=k, sc_samples=sc, few_shot_mode="roundrobin",
        log_path=f"outputs/devsplit_{tag}_turns.jsonl",
        out_path=f"outputs/devsplit_{tag}_results.json",
    )
    res = AE.run(cfg)
    return res["head_to_head"]["agent_classification_acc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="litellm", choices=["litellm", "gemini", "stub"])
    ap.add_argument("--dev-n", type=int, default=60)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--dev-seed", type=int, default=20_000)
    args = ap.parse_args()

    # Step 1 — select on DEV (seeds 20_000..), never touching eval seeds.
    grid = [(6, 3), (6, 1), (3, 3), (12, 3)]   # candidate lever configs
    dev = []
    for k, sc in grid:
        acc = eval_config(args.backend, args.dev_n, args.dev_seed, k, sc, f"dev_k{k}_sc{sc}")
        dev.append({"k": k, "sc_samples": sc, "dev_acc": acc})
        print(f"[dev] k={k} sc={sc} dev_acc={acc:.3f}")
    best = max(dev, key=lambda r: r["dev_acc"])
    print(f"[dev] selected k={best['k']} sc={best['sc_samples']} (dev_acc={best['dev_acc']:.3f})")

    # Step 2 — report the winner ONCE on the held-out EVAL split (seeds 10_000..).
    eval_acc = eval_config(args.backend, args.eval_n, 10_000,
                           best["k"], best["sc_samples"], "eval_selected")
    print(f"[eval] selected config on held-out eval: cls_acc={eval_acc:.3f}")

    out = {
        "backend": args.backend,
        "protocol": {"train_seeds": "0..299", "demo_seeds": "500..",
                     "dev_seeds": f"{args.dev_seed}..", "eval_seeds": "10000.."},
        "dev_grid": dev,
        "selected": {"k": best["k"], "sc_samples": best["sc_samples"],
                     "dev_acc": best["dev_acc"]},
        "held_out_eval_classification_acc": eval_acc,
        "note": ("Levers selected on a disjoint dev split (20_000..) and reported once "
                 "on the held-out eval split (10_000..). This eval number is free of "
                 "eval-set tuning; compare to the paper's tuned 0.990."),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dev_split.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote outputs/dev_split.json")


if __name__ == "__main__":
    main()
