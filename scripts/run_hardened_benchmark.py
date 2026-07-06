#!/usr/bin/env python3
"""
run_hardened_benchmark.py — push the head-to-head OFF the near-saturation ceiling
so the ML-vs-agent comparison is informative. LIVE: needs an LLM backend.

The default twin renders faults at high SNR, so RF ~0.985 and the enhanced panel
~0.990 both sit at ceiling (see paper Section 9, "parity is parity on a
near-saturated task"). This script re-runs the SAME head-to-head harness at
progressively harder difficulty using the knobs the harness already honors for
BOTH methods (so the comparison stays matched):

  sev_scale     : weaker faults (0.5 default -> 0.35 -> 0.25)
  window_frac   : shorter observation window (1.0 -> 0.5)

Both the agent panel and the RF opponent see the identical stressed window at each
setting (agent_eval regenerates cols once and feeds both), so this remains a paired
comparison. The goal is a regime where NEITHER system saturates -- that is where a
reasoning advantage (or its absence) is actually measurable.

Runs each difficulty cell twice: zero-shot and enhanced (k=6, sc=3).

Usage:
    python scripts/run_hardened_benchmark.py --backend litellm --n 200
Writes outputs/hardened_benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import os

from onnesim import agent_eval as AE

# (sev_scale, window_frac) difficulty cells, easy -> hard
CELLS = [
    ("baseline", 0.5, 1.0),
    ("weaker",   0.35, 1.0),
    ("short",    0.5, 0.5),
    ("hardest",  0.25, 0.5),
]


def run_cell(backend, n, sev, wf, k, sc, tag):
    cfg = AE.EvalConfig(
        n_scenarios=n, backend=backend, sev_scale=sev, window_frac=wf,
        few_shot_k=k, sc_samples=sc, few_shot_mode="roundrobin",
        log_path=f"outputs/hardened_{tag}_turns.jsonl",
        out_path=f"outputs/hardened_{tag}_results.json",
    )
    res = AE.run(cfg)
    h = res["head_to_head"]
    return {
        "agent_cls": h["agent_classification_acc"],
        "agent_det_f1": h["agent_detection_f1"],
        "ml_cls": h["ml_classification_acc"],
        "ml_det_f1": h["ml_detection_f1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="litellm", choices=["litellm", "gemini", "stub"])
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for name, sev, wf in CELLS:
        print(f"[hardened] {name}: sev={sev} window_frac={wf}")
        zs = run_cell(args.backend, args.n, sev, wf, 0, 1, f"{name}_zs")
        en = run_cell(args.backend, args.n, sev, wf, 6, 3, f"{name}_en")
        rows.append({"cell": name, "sev_scale": sev, "window_frac": wf,
                     "zero_shot": zs, "enhanced": en})
        print(f"   zero-shot agent_cls={zs['agent_cls']:.3f} ml_cls={zs['ml_cls']:.3f} | "
              f"enhanced agent_cls={en['agent_cls']:.3f}")

    out = {
        "backend": args.backend, "n_scenarios": args.n,
        "note": ("Head-to-head at increasing difficulty (matched stressors applied to "
                 "BOTH agent and RF via the shared agent_eval harness). Look for a cell "
                 "where neither method saturates -- that is the informative regime for "
                 "a reasoning-vs-supervised comparison."),
        "cells": rows,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hardened_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote outputs/hardened_benchmark.json")


if __name__ == "__main__":
    main()
