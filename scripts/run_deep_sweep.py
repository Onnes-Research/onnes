#!/usr/bin/env python3
"""
run_deep_sweep.py — EXTENSIVE severity sweep of the deep-TSAD zoo (the degradation study).

The single-severity run showed deep models (esp. CNN-GRU) saturate the task at sev_scale=0.5.
A reviewer will ask "is the benchmark too easy?". The honest, rigorous answer is a DEGRADATION
CURVE: sweep fault severity down and show every deep model falling out of the trivial regime
into the paper's 0.75-0.85 target band (benchmark.py's own difficulty target). Same rigor as
the main run: multiple seeds per cell -> mean +/- std + Clopper-Pearson CIs.

Grid: severity_scale in {1.0, 0.7, 0.5, 0.3, 0.15} x {cnn_gru, timesnet, anomaly_transformer}
      x SEEDS repeated trainings. Each cell trains on that severity's train seeds (0..) and
      evaluates on that severity's eval seeds (10_000..), so train/test difficulty match.

Writes outputs/deep_sweep.json, checkpointed per (severity, arch) cell so it is resumable.

Usage (B200):
  ONNES_DEVICE=cuda python scripts/run_deep_sweep.py --n-train 4000 --n-test 800 \
      --epochs 300 --seeds 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import deep_tsad as DT
from onnesim import stats as ST
# reuse the trainer/eval from the main runner
from run_deep_tsad import _train_one, _eval  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archs", default="all")
    ap.add_argument("--severities", default="1.0,0.7,0.5,0.3,0.15")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--seq-len", type=int, default=72)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="outputs/deep_sweep.json")
    args = ap.parse_args()
    if args.smoke:
        (args.n_train, args.n_test, args.epochs, args.seeds, args.seq_len,
         args.severities) = 120, 60, 6, 2, 48, "0.5,0.15"

    import torch  # noqa: F401
    dev = DT.get_device()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    archs = (["cnn_gru", "timesnet", "anomaly_transformer"] if args.archs == "all"
             else [a.strip() for a in args.archs.split(",") if a.strip()])
    sevs = [float(s) for s in args.severities.split(",")]
    print(f"[sweep] device={dev} archs={archs} severities={sevs} "
          f"n_train={args.n_train} epochs={args.epochs} seeds={args.seeds}")

    # resume
    cells = {}
    if os.path.exists(args.out):
        try:
            cells = json.load(open(args.out)).get("cells", {})
            print(f"[sweep] resume: {len(cells)} cells already done")
        except Exception:
            cells = {}

    t0 = time.time()
    for sev in sevs:
        # build datasets ONCE per severity (shared across archs)
        print(f"\n[sweep] severity={sev}: building datasets ...")
        Xtr_full, ytr_full = DT.build_dataset(args.n_train, 0, sev, args.seq_len)
        Xte, yte = DT.build_dataset(args.n_test, 10_000, sev, args.seq_len)
        n_val = max(1, int(0.15 * len(Xtr_full)))
        rng = np.random.default_rng(0); idx = rng.permutation(len(Xtr_full))
        Xva, yva = Xtr_full[idx[:n_val]], ytr_full[idx[:n_val]]
        Xtr, ytr = Xtr_full[idx[n_val:]], ytr_full[idx[n_val:]]
        for arch in archs:
            key = f"{sev}|{arch}"
            if key in cells:
                print(f"[sweep] {key}: resume-skip"); continue
            accs = []
            for s in range(args.seeds):
                m = _train_one(arch, Xtr, ytr, Xva, yva, args.epochs, args.batch, s, dev)
                acc, _ = _eval(m, Xte, yte, dev)
                accs.append(acc)
            k = int(round(np.mean(accs) * args.n_test))
            ci = ST.prop_ci(k, args.n_test)
            cells[key] = {"severity": sev, "arch": arch,
                          "acc_mean": round(float(np.mean(accs)), 3),
                          "acc_std": round(float(np.std(accs)), 3),
                          "acc_95ci": [ci["lo"], ci["hi"]],
                          "seeds": args.seeds}
            json.dump({"config": vars(args), "device": dev, "cells": cells,
                       "elapsed_s": round(time.time() - t0, 1)}, open(args.out, "w"), indent=2)
            print(f"   {key:34s} {cells[key]['acc_mean']:.3f}±{cells[key]['acc_std']:.3f} "
                  f"CI{cells[key]['acc_95ci']}  (checkpointed)")

    # tidy per-arch curves for the figure
    curves = {a: [] for a in archs}
    for sev in sevs:
        for a in archs:
            c = cells.get(f"{sev}|{a}")
            if c:
                curves[a].append({"severity": sev, "acc": c["acc_mean"],
                                  "std": c["acc_std"], "ci": c["acc_95ci"]})
    summary = {"config": vars(args), "device": dev, "cells": cells, "curves": curves,
               "target_band": [0.75, 0.85],
               "reference": {"zero_shot_agent": 0.685, "enhanced_agent": 0.990, "rf": 0.985},
               "elapsed_s": round(time.time() - t0, 1),
               "reading": ("Deep-model classification accuracy vs fault severity. As severity "
                           "drops, deep models degrade out of the trivial regime; where they "
                           "enter the 0.75-0.85 band is the benchmark's honest difficulty "
                           "operating point. Compares to agent 0.685->0.990 and RF 0.985.")}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[sweep] DONE in {summary['elapsed_s']}s -> {args.out}")
    for a in archs:
        pts = " ".join(f"{p['severity']}:{p['acc']:.2f}" for p in curves[a])
        print(f"   {a:20s} {pts}")


if __name__ == "__main__":
    main()
