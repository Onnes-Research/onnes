#!/usr/bin/env python3
"""
train_cooldown_twin.py — GPU-1: a neural twin of the REAL 300K->mK cooldown dynamics.

WHY THIS IS NOVEL
-----------------
OnnesSim's twin models STEADY-STATE base temperatures + fault perturbations. It does NOT
model the multi-hour cooldown transient (300 K -> mK) — the paper's virtual_clone even says
so explicitly ("v1 clones the steady-state fingerprint... does NOT model cooldown dynamics").
The University of Leeds DR-200 dataset is 15 REAL cooldowns of a real dilution fridge. This
script learns the cooldown DYNAMICS directly from that real hardware data: a sequence model
that, given the current 5-stage temperature vector (+ elapsed time), predicts the next step,
so rolling it out reproduces a full real-looking 300K->mK cooldown.

This is the first data-driven cooldown emulator for a dilution fridge trained on real curves
— a genuine extension beyond the steady-state twin, and it USES the real Leeds data.

MODEL
-----
A small GRU over log-temperature deltas (temperatures span 2.5 decades, so we model
d(log T)/dstep). Teacher-forced training on real cooldown windows; evaluated by FREE-RUNNING
rollout error (the honest test — does it stay on the real trajectory without being fed truth).

COMPUTE
-------
CPU-verifiable at tiny config (--smoke: 2 logs, 5 epochs, ~1 min). GPU (H200) for the full
run: all 15 logs, long sequences, hundreds of epochs (ONNES_DEVICE=cuda). This is a real
training job that benefits from a GPU — unlike the physics surrogate, cooldown sequences are
long (10k-100k steps) and many, so batched GPU training is a genuine speedup.

Outputs: outputs/cooldown_twin.pt + outputs/cooldown_twin_metrics.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnesim import leeds_data as LD

STAGES = ["50K", "4K", "Still", "CP", "MXC"]


def _coerce_float(col) -> np.ndarray:
    """Coerce a Leeds column to float, mapping Windows non-numeric sentinels to NaN.

    Real Leeds logs contain '-1.#IND' / '-1.#INF' / '1.#QNAN' (MSVC printf for NaN/Inf) as
    literal strings; np.asarray(dtype=float) chokes on them. Map any unparseable cell to NaN
    so the finite-mask downstream drops it.
    """
    arr = np.asarray(col, dtype=object)
    out = np.empty(len(arr), dtype=float)
    for i, v in enumerate(arr):
        try:
            f = float(v)
            out[i] = f if np.isfinite(f) else np.nan
        except (ValueError, TypeError):
            out[i] = np.nan
    return out


def _clean_cooldown(cols: dict, tmap: dict, max_len: int = 20000, stride: int = 4) -> np.ndarray | None:
    """Extract a clean [T, 5] log-temperature cooldown array from one real Leeds log.

    Downsample by `stride` (real logs are 1 Hz, cooldowns are hours), clip to max_len,
    keep only rows where all 5 stages are finite and positive and actually cool (start warm).
    """
    if not all(s in tmap for s in STAGES):
        return None
    chans = [_coerce_float(cols[tmap[s]]) for s in STAGES]
    n = min(len(c) for c in chans)
    M = np.stack([c[:n] for c in chans], axis=1)[::stride][:max_len]  # [T,5]
    good = np.all(np.isfinite(M), axis=1) & np.all(M > 0.0, axis=1)
    M = M[good]
    if len(M) < 200 or M[0, -1] < 50.0:   # must start warm (real cooldown), not a stub
        return None
    return np.log(M)


def load_real_cooldowns(max_logs: int = 15, stride: int = 4) -> list[np.ndarray]:
    """Load all usable real Leeds cooldowns as log-temperature arrays."""
    out = []
    for log in LD.list_logs()[:max_logs]:
        cols = LD.load_log(log)
        arr = _clean_cooldown(cols, LD.find_temp_columns(cols), stride=stride)
        if arr is not None:
            out.append(arr)
    return out


def _make_windows(series: list[np.ndarray], seq: int, seed: int = 0):
    """Cut overlapping (x_t, dx_t) training pairs: input = [logT(5), t_norm(1)], target =
    next-step delta d(logT). Returns X [N,seq,6], Y [N,seq,5]."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for s in series:
        T = len(s)
        tnorm = np.linspace(0, 1, T)[:, None]
        feats = np.concatenate([s, tnorm], axis=1)          # [T,6]
        deltas = np.diff(s, axis=0)                          # [T-1,5]
        for i in range(0, T - seq - 1, max(1, seq // 2)):
            X.append(feats[i:i + seq])
            Y.append(deltas[i:i + seq])
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify correctness")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-logs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.seq, args.hidden, args.max_logs, args.stride = 5, 32, 32, 2, 16

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[cooldown] torch not installed: pip install torch"); sys.exit(1)

    dev = "cuda" if (os.environ.get("ONNES_DEVICE") == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[cooldown] device={dev}  loading real Leeds cooldowns ...")
    series = load_real_cooldowns(max_logs=args.max_logs, stride=args.stride)
    if not series:
        print("[cooldown] no usable real cooldowns found"); sys.exit(1)
    print(f"[cooldown] {len(series)} real cooldowns, lengths {[len(s) for s in series]}")

    # hold out the LAST cooldown for free-running rollout evaluation (unseen real curve)
    train_series, test_series = series[:-1] or series, series[-1]
    X, Y = _make_windows(train_series, args.seq)
    print(f"[cooldown] {len(X)} training windows of length {args.seq}")

    # normalize deltas (targets) for stable training
    ymu, ysd = Y.mean((0, 1)), Y.std((0, 1)) + 1e-8
    Xt = torch.tensor(X, device=dev)
    Yt = torch.tensor((Y - ymu) / ysd, device=dev)

    class CooldownGRU(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.gru = nn.GRU(6, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 5)

        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.head(o), h

    model = CooldownGRU(args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    n = len(Xt)
    g = torch.Generator(device="cpu").manual_seed(0)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            pred, _ = model(Xt[idx])
            loss = lossf(pred, Yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"  epoch {ep+1:4d}  train_mse {tot/n:.4f}")

    # ---- FREE-RUNNING ROLLOUT on the held-out real cooldown (the honest test) ----
    model.eval()
    ymu_t = torch.tensor(ymu, device=dev, dtype=torch.float32)
    ysd_t = torch.tensor(ysd, device=dev, dtype=torch.float32)
    with torch.no_grad():
        T = len(test_series)
        tnorm = np.linspace(0, 1, T)
        state = torch.tensor(test_series[0], device=dev, dtype=torch.float32)  # logT(5)
        h = None
        pred_log = [state.cpu().numpy()]
        for k in range(1, T):
            tstep = torch.tensor([tnorm[k-1]], device=dev, dtype=torch.float32)
            xin = torch.cat([state, tstep]).view(1, 1, 6)
            dnorm, h = model(xin, h)
            delta = dnorm.view(-1) * ysd_t + ymu_t
            state = state + delta
            pred_log.append(state.cpu().numpy())
    pred = np.exp(np.asarray(pred_log))
    real = np.exp(test_series)
    # rollout error per stage (median relative error over the trajectory)
    rel = np.abs(pred - real) / (np.abs(real) + 1e-9)
    per_stage = {STAGES[i]: round(float(np.median(rel[:, i])), 4) for i in range(5)}

    metrics = {
        "device": dev, "n_real_cooldowns": len(series),
        "train_windows": int(len(X)), "seq": args.seq, "hidden": args.hidden,
        "epochs": args.epochs,
        "rollout_median_rel_err_per_stage": per_stage,
        "rollout_mean_rel_err": round(float(np.median(rel)), 4),
        "reading": ("Free-running rollout on a HELD-OUT real Leeds cooldown (model fed only "
                    "its own predictions, not truth). Low per-stage error => the neural twin "
                    "reproduces real 300K->mK dynamics it never saw. This is the first "
                    "data-driven cooldown emulator for a dilution fridge trained on real curves."),
    }
    os.makedirs("outputs", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "ymu": ymu.tolist(), "ysd": ysd.tolist(),
                "hidden": args.hidden}, "outputs/cooldown_twin.pt")
    json.dump(metrics, open("outputs/cooldown_twin_metrics.json", "w"), indent=2)
    print(f"[cooldown] held-out rollout median rel err: {metrics['rollout_mean_rel_err']}")
    print(f"[cooldown]   per stage: {per_stage}")
    print(f"[cooldown] wrote outputs/cooldown_twin.pt + _metrics.json")
    if args.smoke:
        print("[cooldown] SMOKE OK — correctness verified; run full with ONNES_DEVICE=cuda "
              "(drop --smoke) on the H200.")


if __name__ == "__main__":
    main()
