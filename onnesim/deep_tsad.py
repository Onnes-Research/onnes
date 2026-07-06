"""
deep_tsad.py — a zoo of REAL deep time-series models as raw-window fault opponents.

WHY
---
The paper's ML opponents are all TABULAR (RF, TabPFN, GBM over 120 hand-engineered features).
A 2026 reviewer will ask for DEEP time-series models that consume the RAW multi-channel window.
This module implements three, honestly, under the SAME seed-addressed protocol as the agent
and RF (train seeds 0.., eval seeds 10_000..), so their numbers are directly comparable:

  * CNNGRU            — 1D-conv feature extractor + GRU (a strong, standard deep-TS baseline).
  * TimesNet          — FFT finds dominant periods, reshape the 1D series to 2D by period, and
                        apply inception-style 2D convs (Wu et al., ICLR 2023). Real mechanism.
  * AnomalyTransformer— transformer with the series+prior "anomaly attention" and a learned
                        association discrepancy (Xu et al., ICLR 2022). Here the pooled encoder
                        representation feeds a classifier head; the association-discrepancy is
                        also exposed as an anomaly (detection) score.

HONEST FRAMING
--------------
Anomaly-Transformer and TimesNet were designed for anomaly detection / general TS tasks. We
use them as raw-window ENCODERS with a linear classification head for the 6-class task — a
legitimate, standard adaptation — and we say so. The classifier head is trained; the backbones
are trained end-to-end (not frozen), so these are FULL deep baselines, not probes.

COMPUTE
-------
Auto device: cuda (B200/H200) > mps (Apple) > cpu. Designed for EXTENSIVE runs — k-fold CV x
multiple seeds x hundreds of epochs x 3 architectures — which is what fills a multi-hour GPU
session and turns single numbers into mean±std. CPU/MPS-verifiable at tiny config first.

Reuses cryo_engine (twin), benchmark.sample_specs (identical seeds). Standalone training loop
so it has no non-torch deps beyond numpy.
"""
from __future__ import annotations

import numpy as np

from . import cryo_engine as CE
from . import benchmark as BM

# Raw channels fed to every deep model (the model reads these directly, no hand features).
RAW_CHANNELS = ["temp1_T", "temp2_T", "temp3_T", "temp4_T", "temp5_T",
                "temp6_T", "temp7_T", "flowmeter", "p1", "p2", "p3", "p4", "p5", "p6"]
FAULT_CLASSES = list(CE.FAULT_CLASSES)


# --------------------------------------------------------------------------- #
# Data: raw multi-channel windows under the seed-addressed protocol
# --------------------------------------------------------------------------- #
def raw_window(cols: dict, L: int) -> np.ndarray:
    """[L, C] per-channel z-normalized, NaN-filled, resampled-to-L raw window."""
    chans = []
    for ch in RAW_CHANNELS:
        x = np.asarray(cols.get(ch, np.zeros(len(cols["t_s"]))), dtype=float)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0)
        if len(x) != L:
            x = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(x)), x)
        mu, sd = x.mean(), x.std() + 1e-8
        chans.append((x - mu) / sd)
    return np.stack(chans, axis=1)  # [L, C]


def build_dataset(n: int, base_seed: int, sev_scale: float, L: int):
    """Raw windows + integer labels for n seed-addressed scenarios."""
    cfg = CE.EngineConfig(realistic=True, imperfections=True)
    specs = BM.sample_specs(n, base_seed=base_seed)
    X, y = [], []
    for s in specs:
        cols = CE.simulate(CE.Scenario(s.fault_class, s.severity * sev_scale, s.onset_frac),
                           cfg, hours=6.0, dt_min=5.0, seed=s.seed)
        X.append(raw_window(cols, L))
        y.append(FAULT_CLASSES.index(s.fault_class))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def get_device():
    import torch
    import os
    want = os.environ.get("ONNES_DEVICE", "")
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Architectures (built lazily so importing the module needs no torch)
# --------------------------------------------------------------------------- #
def _build_models_module():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CNNGRU(nn.Module):
        """1D-conv stack + GRU + linear head. A strong, standard deep-TS classifier."""
        def __init__(self, c, n_cls, hidden=128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(c, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
            self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(hidden, n_cls))

        def forward(self, x):                       # x [B,L,C]
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            o, _ = self.gru(z)
            return self.head(o.mean(1))             # mean-pool over time

    class InceptionBlock2D(nn.Module):
        """Multi-kernel 2D conv block (the TimesNet inception unit)."""
        def __init__(self, cin, cout, kernels=(1, 3, 5)):
            super().__init__()
            self.branches = nn.ModuleList(
                [nn.Conv2d(cin, cout, k, padding=k // 2) for k in kernels])

        def forward(self, x):
            return sum(b(x) for b in self.branches) / len(self.branches)

    class TimesBlock(nn.Module):
        """One TimesNet block: FFT -> top-k periods -> reshape 1D->2D by period -> 2D incep."""
        def __init__(self, d_model, k_periods=3, d_ff=64):
            super().__init__()
            self.k = k_periods
            self.incep = nn.Sequential(InceptionBlock2D(d_model, d_ff), nn.GELU(),
                                       InceptionBlock2D(d_ff, d_model))

        def forward(self, x):                        # x [B,L,D]
            B, L, D = x.shape
            xf = torch.fft.rfft(x, dim=1)
            amp = xf.abs().mean(-1)                  # [B, L//2+1] period strength
            amp[:, 0] = 0                            # drop DC
            k = min(self.k, amp.shape[1] - 1)
            if k < 1:
                return x
            _, top = torch.topk(amp, k, dim=1)       # [B,k] dominant frequencies
            out = torch.zeros_like(x)
            for i in range(k):
                # use a single period per batch (mode of the top freq) for a clean 2D reshape
                freq = int(torch.clamp(top[:, i].float().mean().round(), 1, L // 2).item())
                period = max(1, L // freq)
                pad = (period - (L % period)) % period
                xp = F.pad(x.transpose(1, 2), (0, pad)).transpose(1, 2)
                Lp = xp.shape[1]
                rows = Lp // period
                grid = xp.reshape(B, rows, period, D).permute(0, 3, 1, 2)   # [B,D,rows,period]
                y = self.incep(grid).permute(0, 2, 3, 1).reshape(B, Lp, D)
                out = out + y[:, :L, :]
            return x + out / k                      # residual

    class TimesNet(nn.Module):
        """Stacked TimesBlocks over an embedded raw window + classifier head (Wu et al. 2023)."""
        def __init__(self, c, n_cls, d_model=64, n_blocks=2):
            super().__init__()
            self.embed = nn.Linear(c, d_model)
            self.blocks = nn.ModuleList([TimesBlock(d_model) for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_cls)

        def forward(self, x):                        # x [B,L,C]
            z = self.embed(x)
            for blk in self.blocks:
                z = self.norm(blk(z))
            return self.head(z.mean(1))

    class AnomalyAttention(nn.Module):
        """Series association (softmax attention) + a learnable Gaussian PRIOR association;
        their discrepancy is the Anomaly-Transformer signal (Xu et al. 2022)."""
        def __init__(self, d_model, n_heads=4):
            super().__init__()
            self.h = n_heads; self.dk = d_model // n_heads
            self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
            self.sigma = nn.Linear(d_model, n_heads)   # per-head prior width

        def forward(self, x):                        # x [B,L,D]
            B, L, D = x.shape
            q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
            k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
            v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)
            series = torch.softmax((q @ k.transpose(-1, -2)) / (self.dk ** 0.5), dim=-1)
            # prior: Gaussian over relative position, width sigma(x) per head
            pos = torch.arange(L, device=x.device).float()
            dist = (pos[None, :] - pos[:, None]).abs()          # [L,L]
            sigma = self.sigma(x).transpose(1, 2).unsqueeze(-1) # [B,h,L,1]
            sigma = torch.clamp(sigma, 1e-2, L).abs() + 1e-2
            prior = torch.exp(-(dist[None, None] ** 2) / (2 * sigma ** 2))
            prior = prior / prior.sum(-1, keepdim=True)
            out = (series @ v).transpose(1, 2).reshape(B, L, D)
            # association discrepancy (per position) — the anomaly score, KL(series||prior)
            assoc = (series * (torch.log(series + 1e-8) - torch.log(prior + 1e-8))).sum(-1).mean(1)
            return self.o(out), assoc                # out [B,L,D], assoc [B,L]

    class AnomalyTransformer(nn.Module):
        """Anomaly-Transformer encoder; pooled repr -> classifier head, plus the mean
        association-discrepancy exposed as a detection score."""
        def __init__(self, c, n_cls, d_model=64, n_layers=2):
            super().__init__()
            self.embed = nn.Linear(c, d_model)
            self.layers = nn.ModuleList([AnomalyAttention(d_model) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Linear(2 * d_model, d_model))
            self.head = nn.Linear(d_model, n_cls)

        def forward(self, x):                        # x [B,L,C]
            z = self.embed(x)
            assocs = []
            for att in self.layers:
                a, assoc = att(self.norm(z))
                z = z + a
                z = z + self.ff(self.norm(z))
                assocs.append(assoc)
            disc = torch.stack(assocs, 0).mean(0).mean(1)  # [B] anomaly score
            return self.head(z.mean(1)), disc

    return {"cnn_gru": CNNGRU, "timesnet": TimesNet, "anomaly_transformer": AnomalyTransformer}


def make_model(name: str, c: int, n_cls: int):
    return _build_models_module()[name](c, n_cls)
