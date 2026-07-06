"""
ml_surrogate.py — a neural surrogate ("ML simulator") of the real Cryowala
heat-load -> stage-temperature physics.

WHAT THIS IS: a small MLP trained to emulate onnesim.cryowala_physics.all_stage_temps
(the BSD-3 CryowalaCore-calibrated map). Once trained, it:
  - runs fully vectorized / GPU-batched over thousands of fridge configs at once
    (this is where H200s pay off — batch the forward pass),
  - is differentiable end-to-end (enables gradient-based fault inference / control),
  - serves as the fast inner loop for fleet-scale fault-scenario generation.

WHAT IT IS NOT: a replacement for validation. It emulates the Cryowala model, which
is itself calibrated to ONE real fridge. Emulating a model faithfully != validated
against your hardware. Honest framing kept throughout.

Trains on CPU in seconds; set ONNES_DEVICE=cuda to use a GPU.
"""
from __future__ import annotations
import os
import numpy as np

from . import cryowala_physics as CP

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False


# --- physics-grounded training data ----------------------------------------
def sample_loads(n: int, rng: np.random.Generator, max_mult: float = 8.0) -> np.ndarray:
    """Sample n physically-plausible per-stage heat-load vectors [W] around nominal.
    Multipliers 0.5x..max_mult (log-uniform) span normal + fault-scale excursions.
    max_mult is capped where the Cryowala fit stays physical (see make_dataset)."""
    base = CP.NOMINAL_LOADS_W
    mult = np.exp(rng.uniform(np.log(0.5), np.log(max_mult), size=(n, 5)))
    return base[None, :] * mult


def make_dataset(n: int, seed: int = 0, max_mult: float = 8.0):
    """Return (loads[m,5] W, temps[m,5] K) from the REAL Cryowala map, keeping ONLY
    physically-valid samples (positive, warm->cold ordered). The Cryowala polynomial
    fit extrapolates to unphysical (even negative) temps at extreme loads; training on
    those would corrupt the surrogate, so we filter to the fit's valid domain and say so."""
    rng = np.random.default_rng(seed)
    loads = sample_loads(int(n * 1.4), rng, max_mult=max_mult)  # oversample, then filter
    temps = np.array([CP.all_stage_temps(l) for l in loads])
    keep = np.array([CP.is_physical(t) for t in temps])
    loads, temps = loads[keep][:n], temps[keep][:n]
    return loads.astype(np.float32), temps.astype(np.float32)


# --- the surrogate model ----------------------------------------------------
if _HAVE_TORCH:
    class Surrogate(nn.Module):
        """MLP mapping log-heat-loads -> log-temperatures (both span many decades)."""
        def __init__(self, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 5),
            )

        def forward(self, log_loads):
            return self.net(log_loads)

    def _device():
        want = os.environ.get("ONNES_DEVICE", "cpu")
        if want == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def train_surrogate(n_train: int = 40000, n_val: int = 8000, epochs: int = 2000,
                        batch: int = 4096, seed: int = 0, verbose: bool = True) -> dict:
        """Train the surrogate on the real (physically-filtered) Cryowala map.
        Minibatched Adam + cosine LR. Returns model + held-out accuracy in real units."""
        dev = _device()
        Xtr, Ytr = make_dataset(n_train, seed)
        Xva, Yva = make_dataset(n_val, seed + 1)

        def to_log(a):
            return np.log(np.clip(a, 1e-12, None))
        lXtr, lYtr = to_log(Xtr), to_log(Ytr)
        lXva, lYva = to_log(Xva), to_log(Yva)
        xm, xs = lXtr.mean(0), lXtr.std(0) + 1e-9
        ym, ys = lYtr.mean(0), lYtr.std(0) + 1e-9

        def norm(a, m, s):
            return torch.tensor((a - m) / s, device=dev, dtype=torch.float32)
        Xtr_t, Ytr_t = norm(lXtr, xm, xs), norm(lYtr, ym, ys)
        Xva_t, Yva_t = norm(lXva, xm, xs), norm(lYva, ym, ys)

        model = Surrogate().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        n = Xtr_t.shape[0]
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = lossf(model(Xtr_t[idx]), Ytr_t[idx])
                loss.backward(); opt.step()
            sched.step()
            if verbose and (ep + 1) % 400 == 0:
                model.eval()
                with torch.no_grad():
                    vl = lossf(model(Xva_t), Yva_t).item()
                print(f"  epoch {ep+1:4d}  val_mse {vl:.6f}  lr {sched.get_last_lr()[0]:.2e}")

        model.eval()
        with torch.no_grad():
            pred_log = model(Xva_t).cpu().numpy() * ys + ym
        pred_T = np.exp(pred_log)
        rel = np.abs(pred_T - Yva) / (np.abs(Yva) + 1e-12)
        return {
            "model": model, "device": dev, "norm": (xm, xs, ym, ys),
            "median_rel_err_per_stage": np.median(rel, axis=0),
            "p90_rel_err_per_stage": np.percentile(rel, 90, axis=0),
            "mean_rel_err": float(np.mean(rel)),
            "n_train": Xtr.shape[0], "n_val": Xva.shape[0],
        }
