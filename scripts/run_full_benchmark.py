#!/usr/bin/env python3
"""
run_full_benchmark.py — the COMPLETE ML benchmark: every model x severity x seeds, with CIs.

Runs EVERY classifier built in this project under ONE seed-addressed protocol (train seeds
0.., eval seeds 10_000..), across a severity sweep, with multiple repeated seeds per cell so
every number is mean +/- std with a Clopper-Pearson 95% CI. This is the full, rigorous
"where does each method stand, and how does it degrade" artifact for the paper.

MODELS
  Tabular (120 hand-features, onnesim.ml_baseline):
    random_forest, hist_gbdt, logreg   (+ lightgbm, tabpfn if installed)
  Deep (raw windows, onnesim.deep_tsad):
    cnn_gru, timesnet, anomaly_transformer

GRID:  severities x models x seeds.  Each cell: train on that severity's train seeds, eval on
that severity's eval seeds.  Checkpointed per cell -> fully resumable.

Compute: tabular = CPU (sklearn, 20 cores); deep = GPU (cuda/mps).  Designed for a big-GPU
multi-hour run.  --smoke verifies correctness fast.

Usage (B200):
  ONNES_DEVICE=cuda python scripts/run_full_benchmark.py \
      --severities 1.0,0.7,0.5,0.3,0.15 --n-train 3000 --n-test 800 \
      --deep-epochs 250 --seeds 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import cryo_engine as CE
from onnesim import benchmark as BM
from onnesim import ml_baseline as MB
from onnesim import deep_tsad as DT
from onnesim import stats as ST
from run_deep_tsad import _train_one, _eval  # deep trainer/eval


TABULAR = ["random_forest", "hist_gbdt", "logreg"]
DEEP = ["cnn_gru", "timesnet", "anomaly_transformer"]


def _tabular_dataset(n, base_seed, sev, cfg):
    specs = BM.sample_specs(n, base_seed=base_seed)
    X = np.asarray([MB.extract_features(
        CE.simulate(CE.Scenario(s.fault_class, s.severity * sev, s.onset_frac),
                    cfg, hours=6.0, dt_min=5.0, seed=s.seed))[0] for s in specs])
    y = np.asarray([s.fault_class for s in specs], dtype=object)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--severities", default="1.0,0.7,0.5,0.3,0.15")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--seq-len", type=int, default=72)
    ap.add_argument("--deep-epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", default="all", help="all|tabular|deep|comma list")
    ap.add_argument("--out", default="outputs/full_benchmark.json")
    args = ap.parse_args()
    if args.smoke:
        (args.severities, args.n_train, args.n_test, args.deep_epochs,
         args.seeds, args.seq_len) = "0.5,0.15", 150, 80, 6, 2, 48

    import torch  # noqa: F401
    dev = DT.get_device()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sevs = [float(s) for s in args.severities.split(",")]
    extra = MB.available_extra_models()
    tabular = TABULAR + [m for m in ("lightgbm", "tabpfn") if extra.get(m)]
    if args.models == "tabular":
        models = tabular
    elif args.models == "deep":
        models = DEEP
    elif args.models == "all":
        models = tabular + DEEP
    else:
        models = [m.strip() for m in args.models.split(",")]
    print(f"[full] device={dev}  severities={sevs}  seeds={args.seeds}")
    print(f"[full] models: {models}")
    print(f"[full] extra tabular available: {extra}")

    cells = {}
    if os.path.exists(args.out):
        try:
            cells = json.load(open(args.out)).get("cells", {})
            print(f"[full] resume: {len(cells)} cells done")
        except Exception:
            cells = {}

    t0 = time.time()
    for sev in sevs:
        cfg = CE.EngineConfig(realistic=True, imperfections=True)
        # tabular datasets (built lazily, shared by all tabular models at this severity)
        Xtr_t = ytr_t = Xte_t = yte_t = None
        # deep datasets (raw windows, built lazily)
        Xtr_d = ytr_d = Xva_d = yva_d = Xte_d = yte_d = None

        for m in models:
            key = f"{sev}|{m}"
            if key in cells:
                print(f"[full] {key}: resume-skip"); continue
            accs = []
            if m in tabular:
                if Xtr_t is None:
                    print(f"[full] sev={sev}: building tabular data ...")
                    Xtr_t, ytr_t = _tabular_dataset(args.n_train, 0, sev, cfg)
                    Xte_t, yte_t = _tabular_dataset(args.n_test, 10_000, sev, cfg)
                for s in range(args.seeds):
                    est = MB.make_models(s, include_strong=True)[m]()
                    est.fit(Xtr_t, ytr_t)
                    accs.append(float(np.mean(est.predict(Xte_t) == yte_t)))
            else:  # deep
                if Xtr_d is None:
                    print(f"[full] sev={sev}: building deep raw-window data ...")
                    Xtr_d, ytr_d = DT.build_dataset(args.n_train, 0, sev, args.seq_len)
                    Xte_d, yte_d = DT.build_dataset(args.n_test, 10_000, sev, args.seq_len)
                    nv = max(1, int(0.15 * len(Xtr_d)))
                    idx = np.random.default_rng(0).permutation(len(Xtr_d))
                    Xva_d, yva_d = Xtr_d[idx[:nv]], ytr_d[idx[:nv]]
                    Xtr_d, ytr_d = Xtr_d[idx[nv:]], ytr_d[idx[nv:]]
                for s in range(args.seeds):
                    mdl = _train_one(m, Xtr_d, ytr_d, Xva_d, yva_d,
                                     args.deep_epochs, args.batch, s, dev)
                    acc, _ = _eval(mdl, Xte_d, yte_d, dev)
                    accs.append(acc)
            kk = int(round(np.mean(accs) * args.n_test))
            ci = ST.prop_ci(kk, args.n_test)
            cells[key] = {"severity": sev, "model": m, "family": ("tabular" if m in tabular else "deep"),
                          "acc_mean": round(float(np.mean(accs)), 3),
                          "acc_std": round(float(np.std(accs)), 3),
                          "acc_95ci": [ci["lo"], ci["hi"]], "seeds": args.seeds}
            json.dump({"config": vars(args), "device": dev, "cells": cells,
                       "elapsed_s": round(time.time() - t0, 1)}, open(args.out, "w"), indent=2)
            c = cells[key]
            print(f"   {key:34s} {c['acc_mean']:.3f}±{c['acc_std']:.3f} CI{c['acc_95ci']} ({c['family']})")

    # tidy summary table: model x severity
    table = {}
    for m in models:
        table[m] = {sev: cells.get(f"{sev}|{m}", {}).get("acc_mean") for sev in sevs}
    summary = {"config": vars(args), "device": dev, "cells": cells, "table": table,
               "severities": sevs, "target_band": [0.75, 0.85],
               "reference": {"zero_shot_agent": 0.685, "enhanced_agent": 0.990, "rf_paper": 0.985},
               "elapsed_s": round(time.time() - t0, 1),
               "reading": ("Every ML model (tabular + deep) x severity x seeds, on identical "
                           "seed-addressed scenarios. Each cell mean+/-std + Clopper-Pearson "
                           "95% CI. As severity drops, all methods degrade; where they enter "
                           "0.75-0.85 is the honest difficulty operating point. All numbers are "
                           "on the TWIN (no real fault data exists).")}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[full] DONE in {summary['elapsed_s']}s -> {args.out}")
    print(f"{'model':22s} " + " ".join(f"{s:>6}" for s in sevs))
    for m in models:
        print(f"{m:22s} " + " ".join(f"{(table[m][s] if table[m][s] is not None else 0):>6.3f}" for s in sevs))


if __name__ == "__main__":
    main()
