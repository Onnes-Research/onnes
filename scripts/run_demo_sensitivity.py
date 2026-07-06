#!/usr/bin/env python3
"""
run_demo_sensitivity.py — the demonstration-sensitivity ablation the paper's
Table (few-shot k-sweep) needs. LIVE: requires an LLM backend (litellm/Claude or
gemini). It reuses the real onnesim.agent_eval harness — nothing re-implemented —
so every number it writes is directly comparable to the head-to-head.

It sweeps two axes reviewers ask about given the "6 demos" headline:
  1. k in {0,1,2,3,6,12}   — how sensitive is the lift to demo COUNT?
  2. curated vs random     — does the confusable-pair weighting matter, or would
                             any 6 demos do? (random = few_shot_mode "retrieval"
                             from a class-balanced bank; curated = round-robin
                             confusable-weighted, the paper's enhanced panel)

All demos are drawn from the demo seed range (500..), disjoint from train (0..)
and eval (10_000..), so there is no test leakage at any k.

Usage (needs API keys via env, see README):
    python scripts/run_demo_sensitivity.py --backend litellm --n 200
    python scripts/run_demo_sensitivity.py --backend gemini  --n 120 --ks 0,1,2,3,6,12

Writes outputs/demo_sensitivity.json. The paper table reads that file; until it
exists, the LaTeX shows an explicit [PENDING RUN] placeholder.
"""
from __future__ import annotations

import argparse
import json
import os

from onnesim import agent_eval as AE


def one_run(backend: str, n: int, k: int, sc: int, mode: str) -> dict:
    tag = f"k{k}_{mode}_sc{sc}"
    cfg = AE.EvalConfig(
        n_scenarios=n,
        backend=backend,
        few_shot_k=k,
        few_shot_mode=mode,       # "roundrobin" = curated; "retrieval" = query-conditioned
        sc_samples=sc,
        # keep eval + demo seed ranges at their disjoint defaults (10_000.. / 500..)
        log_path=f"outputs/demo_sens_{tag}_turns.jsonl",
        out_path=f"outputs/demo_sens_{tag}_results.json",
    )
    res = AE.run(cfg)
    h = res["head_to_head"]
    return {
        "k": k, "sc_samples": sc, "mode": mode, "n": res["n_scenarios"],
        "classification_acc": h["agent_classification_acc"],
        "detection_f1": h["agent_detection_f1"],
        "ml_classification_acc": h["ml_classification_acc"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="litellm", choices=["litellm", "gemini", "stub"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", default="0,1,2,3,6,12")
    ap.add_argument("--sc", type=int, default=3, help="self-consistency vote count for k>0 rows")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    rows = []
    # count sweep, curated demos (the paper's confusable-weighted round-robin)
    for k in ks:
        sc = 1 if k == 0 else args.sc   # k=0 is the zero-shot anchor; no SC there
        print(f"[demo_sens] curated k={k} sc={sc} ...")
        rows.append(one_run(args.backend, args.n, k, sc, "roundrobin"))
    # curated-vs-random contrast at the headline k=6
    if 6 in ks:
        print("[demo_sens] random (retrieval) k=6 sc=%d ..." % args.sc)
        rows.append(one_run(args.backend, args.n, 6, args.sc, "retrieval"))

    out = {
        "backend": args.backend, "n_scenarios": args.n,
        "note": ("Demo-sensitivity ablation. All demos from seed range 500.., disjoint "
                 "from train (0..) and eval (10_000..). 'roundrobin' = curated "
                 "confusable-weighted (paper's enhanced panel); 'retrieval' = "
                 "query-conditioned from a class-balanced bank (the random-selection "
                 "contrast). Numbers directly comparable to the head-to-head."),
        "rows": rows,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/demo_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote outputs/demo_sensitivity.json")
    for r in rows:
        print(f"  {r['mode']:10s} k={r['k']:2d} sc={r['sc_samples']} -> "
              f"cls={r['classification_acc']:.3f} det_f1={r['detection_f1']:.3f}")


if __name__ == "__main__":
    main()
