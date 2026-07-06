#!/usr/bin/env python3
"""
run_deep_tsad.py — EXTENSIVE training + testing of the deep-TSAD baseline zoo.

Built for a multi-hour GPU run (B200/H200), not a toy. It trains Anomaly-Transformer,
TimesNet, and CNN-GRU on RAW telemetry windows under the paper's seed-addressed protocol,
with:
  * REPEATED training (multiple seeds per architecture) -> mean +/- std, not one number,
  * held-out EVAL on the agent's eval seeds (10_000..) for a direct head-to-head,
  * an early-stopping validation split, cosine LR, gradient clipping (proper training),
  * per-architecture checkpoints so a long run is resumable,
  * Clopper-Pearson 95% CIs on the mean eval accuracy (same rigor as the paper's tables).

Scale knobs make it fill 2-5 h on a big GPU:
  --n-train 4000 --n-test 800 --epochs 300 --seeds 5 --archs all   (extensive)
  --smoke  (tiny: verify correctness on CPU/MPS in ~1-2 min)

Usage:
  ONNES_DEVICE=cuda python scripts/run_deep_tsad.py --n-train 4000 --epochs 300 --seeds 5
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


def _train_one(arch, Xtr, ytr, Xva, yva, epochs, batch, seed, dev, lr=2e-3, verbose=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    C = Xtr.shape[2]; n_cls = int(max(ytr.max(), yva.max())) + 1
    model = DT.make_model(arch, C, n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    Xv = torch.tensor(Xva, device=dev); yv = torch.tensor(yva, device=dev)
    n = len(Xt); g = torch.Generator().manual_seed(seed)
    best_va, best_state, patience, bad = 0.0, None, max(20, epochs // 6), 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(dev)
            opt.zero_grad()
            out = model(Xt[idx]); out = out[0] if isinstance(out, tuple) else out
            loss = lossf(out, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vo = model(Xv); vo = vo[0] if isinstance(vo, tuple) else vo
            va = float((vo.argmax(1) == yv).float().mean())
        if va > best_va:
            best_va, best_state, bad = va, {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      {arch} ep{ep+1} val_acc {va:.3f} (best {best_va:.3f})")
    if best_state:
        model.load_state_dict(best_state)
    return model


def _eval(model, Xte, yte, dev):
    import torch
    model.eval()
    with torch.no_grad():
        o = model(torch.tensor(Xte, device=dev)); o = o[0] if isinstance(o, tuple) else o
        pred = o.argmax(1).cpu().numpy()
    return float(np.mean(pred == yte)), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archs", default="all", help="all | comma list: cnn_gru,timesnet,anomaly_transformer")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--seq-len", type=int, default=72)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5, help="repeated trainings per arch (mean+/-std)")
    ap.add_argument("--sev-scale", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/deep_tsad_zoo.json")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_test, args.epochs, args.seeds, args.seq_len = 120, 60, 6, 2, 48

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[deep] torch required"); sys.exit(1)

    dev = DT.get_device()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    archs = (["cnn_gru", "timesnet", "anomaly_transformer"] if args.archs == "all"
             else [a.strip() for a in args.archs.split(",") if a.strip()])
    print(f"[deep] device={dev}  archs={archs}  n_train={args.n_train} n_test={args.n_test} "
          f"epochs={args.epochs} seeds={args.seeds}")

    t0 = time.time()
    print("[deep] building raw-window datasets (train seeds 0.., eval seeds 10_000..) ...")
    Xtr_full, ytr_full = DT.build_dataset(args.n_train, 0, args.sev_scale, args.seq_len)
    Xte, yte = DT.build_dataset(args.n_test, 10_000, args.sev_scale, args.seq_len)
    # validation split off the train set (for early stopping)
    n_val = max(1, int(0.15 * len(Xtr_full)))
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xtr_full))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = Xtr_full[tr_idx], ytr_full[tr_idx], Xtr_full[va_idx], ytr_full[va_idx]

    results = {}
    # resume: load existing partial results if present
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out)).get("architectures", {})
            print(f"[deep] resume: {list(results)} already done")
        except Exception:
            results = {}

    for arch in archs:
        if arch in results and results[arch].get("done"):
            print(f"[deep] {arch}: resume-skip"); continue
        print(f"\n[deep] === {arch} : {args.seeds} repeated trainings ===")
        accs, preds_all = [], []
        for s in range(args.seeds):
            ts = time.time()
            model = _train_one(arch, Xtr, ytr, Xva, yva, args.epochs, args.batch,
                               seed=s, dev=dev, verbose=args.smoke)
            acc, pred = _eval(model, Xte, yte, dev)
            accs.append(acc); preds_all.append(pred.tolist())
            print(f"   seed {s}: eval_acc={acc:.3f}  ({time.time()-ts:.0f}s)")
        k = int(round(np.mean(accs) * args.n_test))
        ci = ST.prop_ci(k, args.n_test)
        results[arch] = {
            "done": True, "seeds": args.seeds,
            "eval_acc_mean": round(float(np.mean(accs)), 3),
            "eval_acc_std": round(float(np.std(accs)), 3),
            "eval_acc_all_seeds": [round(a, 3) for a in accs],
            "mean_acc_95ci": [ci["lo"], ci["hi"]],
            "n_test": args.n_test,
        }
        # checkpoint after each architecture
        json.dump({"config": vars(args), "device": dev, "architectures": results,
                   "elapsed_s": round(time.time() - t0, 1)},
                  open(args.out, "w"), indent=2)
        print(f"   -> {arch} mean {results[arch]['eval_acc_mean']}±{results[arch]['eval_acc_std']} "
              f"95%CI {results[arch]['mean_acc_95ci']}  (checkpointed)")

    summary = {"config": vars(args), "device": dev, "architectures": results,
               "elapsed_s": round(time.time() - t0, 1),
               "reference": {"zero_shot_agent_cls": 0.685, "enhanced_agent_cls": 0.990,
                             "supervised_rf_cls": 0.985},
               "reading": ("Deep TS models on RAW windows under the agent's eval seeds. "
                           "Compare eval_acc_mean to the paper's classification numbers "
                           "(zero-shot agent 0.685, enhanced 0.990, RF 0.985). These are the "
                           "deep opponents the tabular-only baseline zoo was missing.")}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[deep] DONE in {summary['elapsed_s']}s -> {args.out}")
    for a, r in results.items():
        print(f"   {a:20s} {r['eval_acc_mean']:.3f}±{r['eval_acc_std']:.3f}  CI{r['mean_acc_95ci']}")
    if args.smoke:
        print("[deep] SMOKE OK — launch extensive with ONNES_DEVICE=cuda --n-train 4000 "
              "--epochs 300 --seeds 5 on the B200.")


if __name__ == "__main__":
    main()
